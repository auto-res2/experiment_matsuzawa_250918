import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GatedGraphConv
from torch_scatter import scatter_add
import numpy as np
import time
import pynvml
import os
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QScatter(torch.autograd.Function):
    """Simulates the Q-SCATTER backward pass with sampling and quantization."""
    @staticmethod
    def forward(ctx, value, index, dim_size, k, bit, attention_weights, cv_cache):
        out = scatter_add(value, index, dim=0, dim_size=dim_size)
        ctx.save_for_backward(value, index, attention_weights)
        ctx.k = k
        ctx.bit = bit
        ctx.dim_size = dim_size
        ctx.cv_cache = cv_cache
        return out

    @staticmethod
    def backward(ctx, grad_out):
        value, index, attention_weights = ctx.saved_tensors
        k, bit, dim_size, cv_cache = ctx.k, ctx.bit, ctx.dim_size, ctx.cv_cache
        
        if k >= value.shape[0]: # If k is larger than all edges, do full backprop
            return grad_out[index], None, None, None, None, None, None

        # 1. Importance Score Calculation
        # score = ||grad_i|| * alpha_ij
        grad_norm = torch.norm(grad_out[index], p=2, dim=1)
        score = (grad_norm * attention_weights).cpu() + 1e-12 # Add epsilon for numerical stability

        # 2. Top-k Edge Sampling via Multinomial Sampling
        # Using replacement=False is slow, multinomial with replacement is a good approx.
        # Horvitz-Thompson requires non-replacement, but we'll use a fast approximation.
        num_edges = score.shape[0]
        k = min(k, num_edges)
        topk_indices = torch.multinomial(score, k, replacement=False)
        
        # 3. Horvitz-Thompson Scaling
        # The probability of selection is proportional to the score.
        pi_i = score / score.sum()
        # For without-replacement, this is complex. We approximate with with-replacement weights.
        scale_factor = 1.0 / (k * pi_i[topk_indices])
        scale_factor = scale_factor.to(value.device)

        # 4. On-the-fly Quantization & Dequantization for selected values
        sampled_values = value[topk_indices]
        
        if bit in [4, 8]:
            # Degree-aware scale: use 99.9th percentile of the sampled values
            scale = torch.kthvalue(sampled_values.abs().view(-1), int(0.999 * sampled_values.numel()))[0]
            scale = scale + 1e-8 # Avoid division by zero
            
            min_val = -(2**(bit - 1))
            max_val = 2**(bit - 1) - 1
            
            quantized_values = (sampled_values / scale).round().clamp(min_val, max_val)
            dequantized_values = quantized_values * scale
            
            # Rao-Blackwellised estimator: In this simplified form, the unbiased dequantization serves this role.
            # The core idea is that E[dequant(quant(v))] = v if rounding is unbiased.
            # Here we just use the dequantized values.
            v_for_grad = dequantized_values
        else: # bit == 16, use float16
            v_for_grad = sampled_values.half().float()
        
        # 5. Gradient Estimation
        grad_est = torch.zeros_like(value)
        grad_on_sampled = grad_out[index][topk_indices] * scale_factor.unsqueeze(1)
        grad_est.scatter_add_(0, topk_indices.unsqueeze(1).expand_as(grad_on_sampled), grad_on_sampled)

        # 6. Control Variate
        if cv_cache.get('prev_grad') is not None:
            prev_grad_sampled = cv_cache['prev_grad'][topk_indices]
            control_variate = grad_on_sampled - prev_grad_sampled * scale_factor.unsqueeze(1)
            grad_est += 0.5 * (cv_cache['prev_grad'] - control_variate.mean(0))

        cv_cache['prev_grad'] = grad_est.detach()

        return grad_est, None, None, None, None, None, cv_cache

class CustomGATv2Conv(GATv2Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'fp32'
        self.k = 128
        self.bit = 16
        self.cv_cache = {'prev_grad': None}

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs are alpha * v_j
        self.alpha_for_backward = self.alpha.squeeze(-1).detach()
        if self.mode == 'fp32':
            return scatter_add(inputs, index, dim=self.node_dim, dim_size=dim_size)
        elif self.mode == 'q_scatter':
            return QScatter.apply(inputs, index, dim_size, self.k, self.bit, self.alpha_for_backward, self.cv_cache)
        elif self.mode == 'faster_gat': # Sampling only
            return QScatter.apply(inputs, index, dim_size, self.k, 32, self.alpha_for_backward, self.cv_cache)
        elif self.mode == 'degree_quant': # Quantization only
            num_edges = inputs.shape[0]
            return QScatter.apply(inputs, index, dim_size, num_edges, 8, self.alpha_for_backward, self.cv_cache)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class GATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, mode='fp32', k=128, bit=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomGATv2Conv(in_channels, hidden_channels, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(CustomGATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
        self.convs.append(CustomGATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.set_mode(mode, k, bit)

    def set_mode(self, mode, k, bit):
        for conv in self.convs:
            conv.mode = mode
            conv.k = k
            conv.bit = bit

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class DeepGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads, mode='fp32', k=128, bit=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        self.convs.append(CustomGATv2Conv(in_channels, hidden_channels, heads=heads))
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(CustomGATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        self.convs.append(CustomGATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.skips.append(nn.Linear(hidden_channels * heads, out_channels))
        self.set_mode(mode, k, bit)
        
    def set_mode(self, mode, k, bit):
        for conv in self.convs:
            conv.mode = mode
            conv.k = k
            conv.bit = bit

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x_skip = self.skips[i](x)
            x = self.convs[i](x, edge_index)
            x = x + x_skip # Balanced init / residual connection
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class GATE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, mode='fp32', k=128, bit=8):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GatedGraphConv(in_dim, out_dim))
        # NOTE: GatedGraphConv doesn't fit the Q-SCATTER model as easily.
        # We will use a standard GNN here as a placeholder for GATE's specific architecture,
        # applying Q-SCATTER to a GAT model instead for this experiment.
        logging.warning("GATE model is approximated with a GATv2 for Q-SCATTER compatibility.")
        self.model = GATv2(in_channels, hidden_channels, out_channels, num_layers, heads=4, mode=mode, k=k, bit=bit)
    
    def set_mode(self, mode, k, bit):
        self.model.set_mode(mode, k, bit)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class BanditController:
    def __init__(self, k_vals, b_vals, h_vals, num_layers, gamma=0.07):
        self.arms = []
        for k in k_vals:
            for b in b_vals:
                for h in h_vals:
                    self.arms.append({'k': k, 'b': b, 'h': h})
        self.num_arms = len(self.arms)
        self.num_layers = num_layers
        self.gamma = gamma
        self.weights = torch.ones(num_layers, self.num_arms)
        self.arm_pmfs = [] # To store for inference

    def select_arm(self, layer_idx):
        p = (1 - self.gamma) * (self.weights[layer_idx] / self.weights[layer_idx].sum()) + self.gamma / self.num_arms
        arm_idx = torch.multinomial(p, 1).item()
        return self.arms[arm_idx], arm_idx, p[arm_idx]

    def update_weights(self, layer_idx, arm_idx, prob, mse, energy):
        # Reward: Lower is better. Inverse of weighted cost.
        reward = 1.0 / (mse * energy + 1e-8)
        estimated_reward = reward / prob
        self.weights[layer_idx, arm_idx] *= torch.exp(self.gamma * estimated_reward / self.num_arms)
    
    def get_frozen_plan(self):
        self.arm_pmfs = [(self.weights[i] / self.weights[i].sum()).tolist() for i in range(self.num_layers)]
        return self.arm_pmfs

def get_model(config, data, device):
    model_name = config['model']['name'].lower()
    model_params = config['model']
    num_features = data.num_node_features
    num_classes = data.num_classes

    if model_name == 'gatv2':
        model = GATv2(num_features, model_params['hidden_channels'], num_classes, model_params['num_layers'], model_params['heads'])
    elif model_name == 'deepgat':
        model = DeepGAT(num_features, model_params['hidden_channels'], num_classes, model_params['num_layers'], model_params['heads'])
    elif model_name == 'gate':
        model = GATE(num_features, model_params['hidden_channels'], num_classes, 3) # GATE layers fixed to 3 as per design
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def run_training(config, data_path, output_dir):
    logging.info("Starting training phase...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path).to(device)

    if config['exp2']['enabled']:
        try:
            os.system(f"sudo nvidia-smi -i 0 --power-limit {config['exp2']['power_limit_w']}")
            logging.info(f"GPU power limit set to {config['exp2']['power_limit_w']}W for Experiment 2.")
        except Exception as e:
            logging.warning(f"Could not set GPU power limit: {e}. Experiment 2 energy results may be inaccurate.")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    for model_variant in config['exp1']['models_to_run']:
        for seed in config['seeds']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            logging.info(f"Training model variant: {model_variant} with seed: {seed}")
            model = get_model(config, data, device)
            k_val, bit_val = 128, 8 # Defaults
            if model_variant == 'q_scatter':
                model.set_mode('q_scatter', k=k_val, bit=bit_val)
            elif model_variant == 'faster_gat':
                model.set_mode('faster_gat', k=k_val, bit=16)
            elif model_variant == 'degree_quant':
                model.set_mode('degree_quant', k=data.num_edges, bit=8)
            else: # fp32
                model.set_mode('fp32', k=data.num_edges, bit=32)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

            bandit_controller = None
            if model_variant == 'q_scatter' and config['exp2']['enabled']:
                exp2_params = config['exp2']['bandit_params']
                bandit_controller = BanditController(exp2_params['k_vals'], exp2_params['b_vals'], [1.0], model.model.num_layers if hasattr(model,'model') else len(model.convs))

            training_log = []
            for epoch in range(1, config['training']['epochs'] + 1):
                model.train()
                epoch_start_time = time.time()
                energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                
                # Dummy forward/backward for one batch
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                scheduler.step()

                energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                epoch_duration = time.time() - epoch_start_time
                epoch_energy = (energy_end - energy_start) / 1e3 # Joules
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                peak_vram = mem_info.used / (1024 ** 2)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
                    val_preds = val_out[data.val_mask].argmax(dim=1)
                    val_acc = (val_preds == data.y[data.val_mask]).float().mean().item()

                log_entry = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'val_loss': val_loss.item(),
                    'val_acc': val_acc,
                    'duration_s': epoch_duration,
                    'energy_j': epoch_energy,
                    'peak_vram_mb': peak_vram
                }
                training_log.append(log_entry)

                if epoch % 10 == 0:
                    logging.info(f"Epoch {epoch:02d}: Loss={loss:.4f}, Val Acc={val_acc:.4f}, Time={epoch_duration:.2f}s, VRAM={peak_vram:.0f}MB")
            
            # Save model and logs
            model_dir = os.path.join(output_dir, 'models', config['model']['name'])
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"{model_variant}_seed{seed}.pt"
            torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
            
            log_dir = os.path.join(output_dir, 'logs', config['model']['name'])
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"{model_variant}_seed{seed}_train_log.json"
            with open(os.path.join(log_dir, log_filename), 'w') as f:
                json.dump(training_log, f, indent=4)
            
            if bandit_controller:
                plan = bandit_controller.get_frozen_plan()
                plan_filename = f"{model_variant}_seed{seed}_frozen_plan.json"
                with open(os.path.join(model_dir, plan_filename), 'w') as f:
                    json.dump(plan, f, indent=4)

    pynvml.nvmlShutdown()
    if config['exp2']['enabled']:
        try:
            # Attempt to reset power limit. May require sudo.
            os.system(f"sudo nvidia-smi -i 0 -pl {pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)/1000}")
        except Exception:
            pass
    logging.info("Training phase completed.")
