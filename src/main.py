import argparse
import yaml
import os
import logging
import torch
import numpy as np
from . import preprocess, train, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run Q-SCATTER experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')

    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        logging.info("--- RUNNING IN SMOKE TEST MODE ---")
    else:
        config_path = 'config/full_experiment.yaml'
        logging.info("--- RUNNING IN FULL EXPERIMENT MODE ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '.research', 'iteration2', 'images'), exist_ok=True)

    seed = config['seeds'][0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    processed_data_path = preprocess.run_preprocessing(config)

    train.run_training(config, processed_data_path, output_dir)

    evaluate.run_evaluation(config, processed_data_path, output_dir)

    logging.info("Experiment run finished successfully.")

if __name__ == '__main__':
    main()
