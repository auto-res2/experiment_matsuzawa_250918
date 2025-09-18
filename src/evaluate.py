import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import pynvml
import psutil
import numpy as np
import collections
import os
import json
import copy
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from .train import FASTLATTAModel
from .preprocess import get_data_stream, BackgroundLoadSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
# NOTE – All image & JSON artefacts MUST be written to .research/iteration2/…
# -----------------------------------------------------------------------------
BASE_RESULTS_DIR = os.path.join('.research', 'iteration2')
BASE_IMG_DIR = os.path.join(BASE_RESULTS_DIR, 'images')
os.makedirs(BASE_IMG_DIR, exist_ok=True)

class SystemStateMonitor:
    def __init__(self, device_id=0):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.pynvml_available = True
        except pynvml.NVMLError:
            logging.warning("pynvml not available. GPU utilization and energy metrics will be disabled.")
            self.pynvml_available = False
        
        self.inference_times = collections.deque(maxlen=8)
        self.accuracy_history = collections.deque(maxlen=50)

    def get_state(self, current_entropy, current_accuracy):
        # Pad inference times if not full
        times = list(self.inference_times) + [np.mean(self.inference_times) if self.inference_times else 0.0] * (8 - len(self.inference_times))
        
        # GPU utilization
        gpu_util = 0.0
        if self.pynvml_available:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu / 100.0
            except pynvml.NVMLError:
                gpu_util = 0.0 # Handle cases where GPU is lost

        self.accuracy_history.append(current_accuracy)
        moving_avg_acc = np.mean(self.accuracy_history) if self.accuracy_history else 0.0

        state_vector = times + [current_entropy, moving_avg_acc, gpu_util]
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

    def update_inference_time(self, t):
        self.inference_times.append(t)

    def get_energy_consumption(self):
        if self.pynvml_available:
            try:
                return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) # in mJ
            except pynvml.NVMLError:
                return 0
        return 0

    def shutdown(self):
        if self.pynvml_available:
            pynvml.nvmlShutdown()

class MetricsLogger:
    """Collects per-batch metrics and writes complete results to JSON + plots.
    All artefacts are saved inside .research/iteration2 so that CI can pick
    them up irrespective of the user-provided output_dir in the config."""

    def __init__(self, output_dir: str, experiment_name: str):
        # Hard-override to the mandatory research directory
        self.base_dir = BASE_RESULTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self.img_dir = BASE_IMG_DIR  # already created at import time
        self.experiment_name = experiment_name

        self.metrics = collections.defaultdict(list)
        self.window_metrics = collections.defaultdict(list)
        self.window_size = 500
        self.step = 0

    # ------------------------------------------------------------------
    # Per-step logging helpers
    # ------------------------------------------------------------------
    def log_step(self, acc, fps, J, violation, energy_per_img, kl_div):
        self.step += 1
        self.metrics['accuracy'].append(acc)
        self.metrics['fps'].append(fps)
        self.metrics['J'].append(J)
        self.metrics['violations'].append(violation)
        self.metrics['energy_per_image'].append(energy_per_img)
        self.metrics['kl_div'].append(kl_div)
        
        self.window_metrics['accuracy'].append(acc)
        self.window_metrics['fps'].append(fps)
        self.window_metrics['J'].append(J)
        self.window_metrics['violations'].append(violation)
        self.window_metrics['energy_per_image'].append(energy_per_img)

        if self.step % self.window_size == 0:
            self.aggregate_and_log_window()
            self.window_metrics = collections.defaultdict(list)

    def aggregate_and_log_window(self):
        window_num = self.step // self.window_size
        logging.info(f"--- Window {window_num} Summary ---")
        for key, values in self.window_metrics.items():
            if values:
                mean_val = np.mean(values)
                logging.info(f"  Avg {key}: {mean_val:.4f}")

    # ------------------------------------------------------------------
    # Finalisation (writes JSON + plots)
    # ------------------------------------------------------------------
    def finalize(self):
        summary = {}
        logging.info(f"--- Final Experiment Summary: {self.experiment_name} ---")
        for key, values in self.metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                summary[f'mean_{key}'] = mean_val
                summary[f'std_{key}'] = std_val
                logging.info(f"  {key.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

        # AUC calculation
        if self.metrics['accuracy'] and self.metrics['fps']:
            latencies = [1000.0 / f for f in self.metrics['fps']]
            sorted_indices = np.argsort(latencies)
            sorted_latencies = np.array(latencies)[sorted_indices]
            sorted_accuracies = np.array(self.metrics['accuracy'])[sorted_indices]
            auc = np.trapz(sorted_accuracies, sorted_latencies)
            summary['auc_accuracy_vs_latency'] = auc
            logging.info(f"  AUC (Accuracy vs. Latency): {auc:.4f}")

        results_data = {'raw': self.metrics, 'summary': summary}
        
        # --- Persist to JSON (mandatory location) ---
        json_path = os.path.join(self.base_dir, f'{self.experiment_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Results saved to {json_path}")
        # CI visibility (do NOT remove)
        print(json.dumps(results_data, indent=4))

        self.plot_results(results_data)
        return results_data

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def plot_results(self, results_data):
        sns.set_theme(style="whitegrid")

        # Accuracy vs. Latency curve
        plt.figure(figsize=(10, 6))
        latencies = [1000.0 / f for f in results_data['raw']['fps']]
        accuracies = results_data['raw']['accuracy']
        sns.scatterplot(x=latencies, y=accuracies, alpha=0.5,
                        label=f"AUC: {results_data['summary']['auc_accuracy_vs_latency']:.2f}")
        plt.title(f'Accuracy vs. Latency ({self.experiment_name})')
        plt.xlabel('Latency (ms/image)')
        plt.ylabel('Top-1 Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.img_dir, f'{self.experiment_name}_acc_vs_latency.pdf'))
        plt.close()

        # J distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_data['raw']['J'], discrete=True)
        plt.title(f'Distribution of Adaptation Depth J ({self.experiment_name})')
        plt.xlabel('J (Number of adapted layers)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.img_dir, f'{self.experiment_name}_j_dist.pdf'))
        plt.close()

def run_evaluation(config):
    logging.info(f"Starting evaluation for experiment: {config['name']}")
    device = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 1. Setup Model
    model = FASTLATTAModel(config).to(device)
    artifacts_dir = config['training']['artifacts_dir']
    try:
        model.adapter.load_state_dict(torch.load(os.path.join(artifacts_dir, 'adapter.pth')))
        model.scheduler.load_state_dict(torch.load(os.path.join(artifacts_dir, 'scheduler.pth')))
        logging.info("Loaded pre-trained adapter and scheduler weights.")
    except FileNotFoundError:
        logging.warning("Pre-trained weights not found. Using randomly initialized models.")
    model.eval()

    # 2. Setup Data Stream and Background Load
    data_loader = get_data_stream(config['dataset'], config['model']['backbone_name'])
    bg_simulator = None
    if config['dataset']['background_gpu_load'] > 0:
        bg_simulator = BackgroundLoadSimulator(config['dataset']['background_gpu_load'])
        bg_simulator.start()
        logging.info(f"Started background GPU load simulator at {config['dataset']['background_gpu_load']*100}%.")

    # 3. Setup Monitoring and Logging
    state_monitor = SystemStateMonitor()
    metrics_logger = MetricsLogger(config.get('output_dir', BASE_RESULTS_DIR), config['name'])

    # 4. Evaluation Loop
    total_correct = 0
    total_images = 0
    start_energy = state_monitor.get_energy_consumption()

    try:
        pbar = tqdm(data_loader, desc=f"Evaluating {config['name']}")
        for i, (images, labels) in enumerate(pbar):
            if config['eval_steps'] is not None and i >= config['eval_steps']:
                break
            
            start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                # Get pre-adaptation logits and stats
                p_pre = model(images).detach()
                stats = model.get_statistics(images)

                # Get system state
                entropy = -torch.sum(F.softmax(p_pre, dim=1) * F.log_softmax(p_pre, dim=1), dim=1).mean().item()
                current_acc = (p_pre.argmax(1) == labels).float().mean().item()
                system_state = state_monitor.get_state(entropy, current_acc).to(device)
                
                # Scheduler selects action J
                action_probs = model.scheduler(system_state)
                J = torch.multinomial(action_probs, 1).item()
                
                # Store pre-adaptation adapter state for potential rollback
                adapter_state_backup = copy.deepcopy(model.adapter.state_dict())

                # Apply adaptation
                offsets = model.adapter(stats)
                model.apply_offsets(offsets, J, variant=config.get('variant', 'full'))
                p_post = model(images)

                # Safety Envelope Check
                violation = False
                kl_div_val = 0.0
                if config['safety_envelope']['enabled']:
                    kl_div = F.kl_div(F.log_softmax(p_post, dim=1), F.softmax(p_pre, dim=1), reduction='batchmean')
                    kl_div_val = kl_div.item()
                    if kl_div > config['safety_envelope']['epsilon']:
                        violation = True
                        model.adapter.load_state_dict(adapter_state_backup)
                        final_logits = p_pre
                    else:
                        final_logits = p_post
                else:
                    final_logits = p_post

            # Calculate metrics
            end_time = time.time()
            batch_time = end_time - start_time
            state_monitor.update_inference_time(batch_time)
            fps = images.size(0) / batch_time if batch_time > 0 else float('inf')
            
            total_correct += (final_logits.argmax(1) == labels).sum().item()
            total_images += images.size(0)
            current_accuracy = total_correct / total_images

            end_energy = state_monitor.get_energy_consumption()
            energy_per_img = (end_energy - start_energy) / total_images if total_images > 0 else 0

            metrics_logger.log_step(acc=current_accuracy, fps=fps, J=J, violation=int(violation), energy_per_img=energy_per_img, kl_div=kl_div_val)
            pbar.set_postfix({'acc': f'{current_accuracy:.3f}', 'fps': f'{fps:.1f}', 'J': J, 'violations': metrics_logger.metrics['violations'][-1]})

    finally:
        if bg_simulator:
            bg_simulator.stop()
            logging.info("Stopped background GPU load simulator.")
        state_monitor.shutdown()
        results = metrics_logger.finalize()

    return results
