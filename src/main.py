import yaml
import typer
from pathlib import Path
import logging
import torch
import os

from src import train
from src import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = typer.Typer()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@app.command()
def main(smoke_test: bool = typer.Option(False, "--smoke-test", help="Run a quick smoke test."),
         full_experiment: bool = typer.Option(False, "--full-experiment", help="Run the full experiment.")):

    if not (smoke_test ^ full_experiment):
        logging.error("Please specify either --smoke-test or --full-experiment, but not both.")
        raise typer.Exit(code=1)

    config_path = Path("config/smoke_test.yaml") if smoke_test else Path("config/full_experiment.yaml")
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        raise typer.Exit(code=1)
    
    config = load_config(str(config_path))
    logging.info(f"Loaded configuration from {config_path}")
    
    # Setup environment
    torch.manual_seed(config['seed'])
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA is not available, falling back to CPU.")
        config['device'] = 'cpu'
    
    os.makedirs(config.get('output_dir', '.research/iteration1'), exist_ok=True)

    # --- Phase 1: Offline Training ---
    if config.get('training', {}).get('enabled', False):
        logging.info("=== Starting Training Phase ===")
        # Training needs a specific model config, not the whole experiment list
        # We assume one training run is sufficient for all evaluations
        train_config = config.copy()
        if smoke_test:
             # In smoke test, evaluation section defines the single model
             train_config['model'] = config['evaluation']['experiments'][0]['model']
        else:
             # In full experiment, we can use the first exp's model as reference
             train_config['model'] = config['evaluation']['experiments'][0]['model']

        train.train_scheduler(train_config)
        logging.info("=== Training Phase Finished ===")
    else:
        logging.info("Skipping training phase as per configuration.")

    # --- Phase 2: Online Evaluation ---
    if config.get('evaluation', {}).get('enabled', False):
        logging.info("=== Starting Evaluation Phase ===")
        experiments = config['evaluation'].get('experiments', [])
        if not experiments:
            logging.warning("Evaluation is enabled, but no experiments are defined.")

        for i, exp_config in enumerate(experiments):
            logging.info(f"--- Running Experiment {i+1}/{len(experiments)}: {exp_config.get('name', 'Untitled')} ---")
            # Merge shared config with experiment-specific config
            run_config = config.copy()
            run_config.update(exp_config)
            # Ensure nested dicts like 'training' are available
            run_config['training'] = config.get('training', {})
            evaluate.run_evaluation(run_config)
            logging.info(f"--- Finished Experiment {i+1}/{len(experiments)} ---")

        logging.info("=== Evaluation Phase Finished ===")
    else:
        logging.info("Skipping evaluation phase as per configuration.")

if __name__ == "__main__":
    app()
