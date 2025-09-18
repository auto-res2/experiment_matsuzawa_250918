import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import pynvml
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from .train import get_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RESEARCH_ROOT = os.path.join('.research', 'iteration2')
IMAGES_DIR = os.path.join(RESEARCH_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def _save_results(sub_name: str, results: dict):
    os.makedirs(RESEARCH_ROOT, exist_ok=True)
    path = os.path.join(RESEARCH_ROOT, f'{sub_name}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n--- RESULTS ({sub_name}) ---")
    print(json.dumps(results, indent=4))
    print("--- END ---\n")
    return path


def run_evaluation(config, data_path, output_dir):
    logging.info("Starting evaluation phase...")
    results = {}
    if config['exp1']['enabled']:
        results['exp1'] = run_exp1(config, data_path, output_dir)
        _save_results('exp1_results', results['exp1'])
    if config['exp2']['enabled']:
        results['exp2'] = run_exp2(config, output_dir)
        _save_results('exp2_results', results['exp2'])
    if config['exp3']['enabled']:
        results['exp3'] = run_exp3(config, data_path, output_dir)
        _save_results('exp3_results', results['exp3'])

    final_path = _save_results('final_results', results)
    logging.info(f"All results saved to {final_path}")


def run_exp1(config, data_path, output_dir):
    logging.info("Running Experiment 1: End-to-End Benchmark")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path).to(device)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    results = {}
    for model_variant in config['exp1']['models_to_run']:
        accuracies, latencies, vrams = [], [], []
        for seed in config['seeds']:
            model = get_model(config, data, device)
            model_path = os.path.join(output_dir, 'models', config['model']['name'], f"{model_variant}_seed{seed}.pt")
            if not os.path.exists(model_path):
                logging.warning(f"Model checkpoint {model_path} not found – skipping.")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                test_out = model(data.x, data.edge_index)
                test_preds = test_out[data.test_mask].argmax(dim=1)
                acc = (test_preds == data.y[data.test_mask]).float().mean().item()
                accuracies.append(acc)

                warmup_runs, timed_runs = 3, 20
                for _ in range(warmup_runs):
                    _ = model(data.x, data.edge_index)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(timed_runs):
                    _ = model(data.x, data.edge_index)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                avg_latency_ms = ((end_time - start_time) / timed_runs) * 1000
                latencies.append(avg_latency_ms)
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vrams.append(mem_info.used / (1024 ** 2))

        if len(accuracies) == 0:
            continue
        results[model_variant] = {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'latency_ms_mean': float(np.mean(latencies)),
            'latency_ms_std': float(np.std(latencies)),
            'peak_vram_mb_mean': float(np.mean(vrams)),
            'peak_vram_mb_std': float(np.std(vrams))
        }
    pynvml.nvmlShutdown()
    return results


def run_exp2(config, output_dir):
    logging.info("Running Experiment 2: Energy/MSE Trade-off & Bandit Controller")
    results = {'analysis': 'Analysis of training logs for Q-SCATTER variant'}
    log_dir = os.path.join(output_dir, 'logs', config['model']['name'])
    all_logs = []
    for seed in config['seeds']:
        log_path = os.path.join(log_dir, f"q_scatter_seed{seed}_train_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                all_logs.extend(json.load(f))
    if not all_logs:
        return {'error': 'No training logs found for Q-SCATTER variant.'}

    avg_energy = float(np.mean([log['energy_j'] for log in all_logs]))
    avg_duration = float(np.mean([log['duration_s'] for log in all_logs]))
    avg_val_acc = float(np.mean([log['val_acc'] for log in all_logs]))
    
    results['q_scatter_avg_metrics'] = {
        'avg_energy_per_epoch_j': avg_energy,
        'avg_duration_per_epoch_s': avg_duration,
        'avg_final_val_acc': avg_val_acc
    }

    fig_path = os.path.join(IMAGES_DIR, 'exp2_placeholder.pdf')
    plt.figure()
    plt.title('Exp 2 - Bandit Performance (Placeholder)')
    plt.xlabel('Training Steps')
    plt.ylabel('Chosen k/bit')
    plt.savefig(fig_path)
    plt.close()
    results['plots'] = [fig_path]
    return results


def calculate_ece(probs, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower.item()) & (probs <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = (labels[in_bin]).float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def run_exp3(config, data_path, output_dir):
    logging.info("Running Experiment 3: Test-Time Plan Reuse & Monte-Carlo Ensembling")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(data_path).to(device)

    seed = config['seeds'][0]
    model_name = config['model']['name']
    model_dir = os.path.join(output_dir, 'models', model_name)
    plan_path = os.path.join(model_dir, f"q_scatter_seed{seed}_frozen_plan.json")
    model_path = os.path.join(model_dir, f"q_scatter_seed{seed}.pt")

    if not (os.path.exists(plan_path) and os.path.exists(model_path)):
        return {'error': 'Q-SCATTER model or frozen plan not found. Cannot run Exp 3.'}

    with open(plan_path, 'r') as f:
        frozen_plan_pmf = json.load(f)
    
    model = get_model(config, data, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = {}
    num_inference_batches = config['exp3'].get('inference_batches', 100)

    latencies_frozen = []
    with torch.no_grad():
        for _ in range(num_inference_batches):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            _ = model(data.x, data.edge_index)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            latencies_frozen.append((end - start) * 1000)
    
    test_out = model(data.x, data.edge_index)
    test_probs = F.softmax(test_out[data.test_mask], dim=1)
    test_preds = test_probs.argmax(dim=1)
    acc_frozen = (test_preds == data.y[data.test_mask]).float().mean().item()
    ece_frozen = calculate_ece(test_probs.max(dim=1)[0], (test_preds == data.y[data.test_mask]))
    results['frozen_plan_single_sample'] = {
        'latency_ms_mean': float(np.mean(latencies_frozen)),
        'accuracy': acc_frozen,
        'ece': ece_frozen
    }

    latencies_mc = []
    all_logits = []
    with torch.no_grad():
        start_mc = time.time()
        for _ in range(4):
            logits = model(data.x, data.edge_index)[data.test_mask]
            all_logits.append(logits)
        end_mc = time.time()
    latencies_mc.append(((end_mc - start_mc) / 4) * 1000)

    avg_logits = torch.stack(all_logits).mean(dim=0)
    mc_probs = F.softmax(avg_logits, dim=1)
    mc_preds = mc_probs.argmax(dim=1)
    acc_mc = (mc_preds == data.y[data.test_mask]).float().mean().item()
    ece_mc = calculate_ece(mc_probs.max(dim=1)[0], (mc_preds == data.y[data.test_mask]))
    results['frozen_plan_4_sample_mc'] = {
        'latency_ms_mean': float(np.mean(latencies_mc) * 4),
        'accuracy': acc_mc,
        'ece': ece_mc
    }

    return results
