import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Attention as ViTAttention
from timm.models.convnext import ConvNeXtBlock
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
import logging

from .preprocess import get_data_stream

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightConvEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim), # Depthwise
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1), # Pointwise
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # x shape: (batch, channels) -> (batch, channels, 1)
        x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)

class MultiGranularHyperAdapter(nn.Module):
    def __init__(self, backbone_name, backbone_ref, stats_dim=64, encoder_out_dim=256):
        super().__init__()
        self.encoder = LightConvEncoder(input_dim=stats_dim, output_dim=encoder_out_dim)
        self.target_modules_info = self._get_target_modules_info(backbone_ref)
        
        self.heads = nn.ModuleDict()
        
        if self.target_modules_info['norm_params'] > 0:
            self.heads['gamma'] = nn.Linear(encoder_out_dim, self.target_modules_info['norm_params'])
            self.heads['beta'] = nn.Linear(encoder_out_dim, self.target_modules_info['norm_params'])
            # Initialize to identity transformation
            nn.init.zeros_(self.heads['gamma'].weight)
            nn.init.zeros_(self.heads['gamma'].bias)
            nn.init.zeros_(self.heads['beta'].weight)
            nn.init.zeros_(self.heads['beta'].bias)

        if self.target_modules_info['attn_params'] > 0:
            self.heads['delta_alpha'] = nn.Linear(encoder_out_dim, self.target_modules_info['attn_params'])
            nn.init.zeros_(self.heads['delta_alpha'].weight)
            nn.init.zeros_(self.heads['delta_alpha'].bias)

        if self.target_modules_info['conv_params'] > 0:
            self.heads['delta_w'] = nn.Linear(encoder_out_dim, self.target_modules_info['conv_params'])
            nn.init.zeros_(self.heads['delta_w'].weight)
            nn.init.zeros_(self.heads['delta_w'].bias)

    @staticmethod
    def _get_target_modules_info(backbone):
        info = defaultdict(int)
        info['norm_layers'] = []
        info['attn_layers'] = []
        info['conv_layers'] = []

        for name, mod in backbone.named_modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
                info['norm_params'] += mod.weight.numel()
                info['norm_layers'].append(name)
            elif isinstance(mod, ViTAttention):
                # We modulate Q and K projections, which are part of qkv.weight
                # Each has shape (embed_dim, embed_dim), we use one scalar per head.
                num_heads = mod.num_heads
                info['attn_params'] += num_heads
                info['attn_layers'].append(name)
            elif isinstance(mod, ConvNeXtBlock):
                # Target the second pointwise convolution's weight
                info['conv_params'] += 1
                info['conv_layers'].append(f"{name}.conv_dw")
        # Ensure all required keys are present
        info.setdefault('norm_params', 0)
        info.setdefault('attn_params', 0)
        info.setdefault('conv_params', 0)
        
        return dict(info)

    def forward(self, stats):
        features = self.encoder(stats)
        offsets = {}
        for name, head in self.heads.items():
            offsets[name] = head(features)
        return offsets

class Scheduler(nn.Module):
    def __init__(self, state_dim, K):
        super().__init__()
        self.gru = nn.GRU(state_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, K + 1)

    def forward(self, state):
        # state shape: (batch_size, state_dim)
        # GRU expects (batch_size, seq_len, input_size)
        state = state.unsqueeze(1)
        h, _ = self.gru(state)
        logits = self.fc(h[:, -1, :]) # Use last hidden state
        return F.softmax(logits, dim=-1)

class FASTLATTAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_name = config['model']['backbone_name']
        self.device = config['device']
        self.backbone = timm.create_model(self.backbone_name, pretrained=True)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone_features = timm.create_model(self.backbone_name, pretrained=True, features_only=True)
        self.backbone_features.eval()
        for param in self.backbone_features.parameters():
            param.requires_grad = False

        self.adapter = MultiGranularHyperAdapter(self.backbone_name, self.backbone)
        
        self.K = len(self.adapter.target_modules_info.get('norm_layers', [])) + \
                 len(self.adapter.target_modules_info.get('attn_layers', [])) + \
                 len(self.adapter.target_modules_info.get('conv_layers', []))
        self.K = min(self.K, 50) # Cap max adaptation depth
        
        # State vector: 8 inference times, entropy, moving avg acc, gpu util = 11
        self.scheduler = Scheduler(state_dim=11, K=self.K)

        self._cached_target_modules = self._get_target_modules()
        self.stats_projector = nn.Linear(512, 64)

    def _get_target_modules(self):
        modules = []
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm, ViTAttention, ConvNeXtBlock)):
                modules.append((name, mod))
        return modules

    def get_statistics(self, x):
        with torch.no_grad():
            # This is a non-placeholder implementation for statistics extraction.
            features = self.backbone_features(x)
            # Use features from last 2 stages
            stats = []
            for f_map in features[-2:]:
                if f_map.dim() == 4: # CNN features [B, C, H, W]
                    # channel-wise mean and std
                    stats.append(f_map.mean(dim=[2, 3]))
                    stats.append(f_map.std(dim=[2, 3]))
                elif f_map.dim() == 3: # Transformer features [B, N, D]
                    stats.append(f_map.mean(dim=1))
                    stats.append(f_map.std(dim=1))
            
            stats_cat = torch.cat(stats, dim=1)
            # Project to a fixed dimension if necessary
            if stats_cat.shape[1] > 512:
                stats_cat = stats_cat[:, :512] # Truncate
            elif stats_cat.shape[1] < 512:
                padding = torch.zeros(stats_cat.shape[0], 512 - stats_cat.shape[1], device=stats_cat.device)
                stats_cat = torch.cat([stats_cat, padding], dim=1)
            
            projected_stats = self.stats_projector(stats_cat)
            return projected_stats
    
    def forward(self, x):
        return self.backbone(x)

    def apply_offsets(self, offsets, J, variant='full'):
        use_norm = 'norm' in variant or 'full' in variant
        use_attn = 'attn' in variant or 'full' in variant
        use_conv = 'conv' in variant or 'full' in variant
        
        norm_offset = 0
        attn_offset = 0
        conv_offset = 0

        gamma = offsets.get('gamma')
        beta = offsets.get('beta')
        delta_alpha = offsets.get('delta_alpha')
        delta_w = offsets.get('delta_w')
        
        # This ensures we don't try to adapt more layers than exist
        effective_J = min(J, len(self._cached_target_modules))

        with torch.no_grad():
            for i in range(effective_J):
                name, mod = self._cached_target_modules[i]

                if use_norm and isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
                    num_params = mod.weight.numel()
                    mod.weight.data += gamma[0, norm_offset : norm_offset + num_params].view(mod.weight.shape)
                    mod.bias.data += beta[0, norm_offset : norm_offset + num_params].view(mod.bias.shape)
                    norm_offset += num_params
                
                elif use_attn and isinstance(mod, ViTAttention):
                    num_heads = mod.num_heads
                    head_dim = mod.head_dim
                    # (3 * embed_dim, embed_dim) -> (3, num_heads, head_dim, embed_dim)
                    qkv = mod.qkv.weight.data.view(3, num_heads, head_dim, -1)
                    
                    # Get per-head scaling factors
                    scaling = delta_alpha[0, attn_offset : attn_offset + num_heads]
                    # Apply to Q and K
                    qkv[0] *= (1 + scaling.view(num_heads, 1, 1))
                    qkv[1] *= (1 + scaling.view(num_heads, 1, 1))

                    mod.qkv.weight.data.copy_(qkv.view(3 * num_heads * head_dim, -1))
                    attn_offset += num_heads
                
                elif use_conv and isinstance(mod, ConvNeXtBlock):
                    # Apply to depthwise conv weight
                    scaling = delta_w[0, conv_offset : conv_offset + 1]
                    mod.conv_dw.weight.data *= (1 + scaling.item())
                    conv_offset += 1

    def reset_adapters(self):
        # Not a full reset to initial state, but removes current adaptation.
        # A true reset would need storing original params.
        # This is typically handled by re-instantiating the model for each run.
        pass


def train_scheduler(config):
    logging.info("Starting offline meta-training of the scheduler.")
    
    device = torch.device(config['device'])
    model = FASTLATTAModel(config).to(device)
    
    # Use a dummy dataloader for synthetic training data
    train_config = config['training']
    dataset_config = {
        'name': 'synthetic',
        'batch_size': train_config['batch_size'],
        'eta': 1.0, # Not relevant for offline training
        'background_gpu_load': 0.0
    }
    data_loader = get_data_stream(dataset_config, config['model']['backbone_name'])

    optimizer = torch.optim.Adam(list(model.scheduler.parameters()) + list(model.adapter.parameters()), lr=train_config['hyperparameters']['learning_rate'])
    
    # Simplified PPO-like training loop
    for epoch in range(train_config['hyperparameters']['ppo_epochs']):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{train_config['hyperparameters']['ppo_epochs']}")
        total_reward = 0
        for images, _ in pbar:
            images = images.to(device)
            
            # 1. Get model outputs and stats
            stats = model.get_statistics(images)
            with torch.no_grad():
                logits = model(images)
                confidence = logits.softmax(dim=1).max(dim=1)[0]

            # 2. Scheduler decides action J
            # A dummy state for training, in reality this comes from SystemStateMonitor
            dummy_state = torch.randn(images.size(0), 11).to(device) 
            action_probs = model.scheduler(dummy_state)
            dist = torch.distributions.Categorical(action_probs)
            J = dist.sample()

            # 3. Apply adaptation and compute reward
            offsets = model.adapter(stats)
            
            # Simulate latency based on J (simple linear model)
            latency_penalty = train_config['hyperparameters']['lambda'] * (J.float() / model.K)
            reward = confidence.mean() - latency_penalty.mean()
            total_reward += reward.item()

            # 4. PPO-style update
            log_prob = dist.log_prob(J)
            loss = - (log_prob * reward.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item(), 'reward': reward.item()})

    logging.info("Scheduler training finished.")
    
    # Save model artifacts
    artifacts_dir = train_config['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    torch.save(model.adapter.state_dict(), os.path.join(artifacts_dir, 'adapter.pth'))
    torch.save(model.scheduler.state_dict(), os.path.join(artifacts_dir, 'scheduler.pth'))
    logging.info(f"Saved trained adapter and scheduler to {artifacts_dir}")

if __name__ == '__main__':
    # This part is for standalone testing of the module
    class MockConfig:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = {'backbone_name': 'resnet50.a1_in1k'}
        training = {
            'enabled': True,
            'batch_size': 4,
            'artifacts_dir': '.research/iteration1/artifacts',
            'hyperparameters': {
                'learning_rate': 3e-4, 'ppo_epochs': 1, 'lambda': 0.05
            }
        }

    config = MockConfig().__dict__
    train_scheduler(config)
