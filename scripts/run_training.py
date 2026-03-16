import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from beta_vae_model.main_review import main as run_beta_vae
from config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run Beta-VAE training/testing pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--results", default="RESULTS/beta_vae_results", help="Output folder")
    parser.add_argument("--adnimerge", default=None, help="Optional override for ADNIMERGE CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    adnimerge_path = args.adnimerge or config["loader"]["load_ADNIMERGE"]
    os.makedirs(args.results, exist_ok=True)

    run_beta_vae(
        config_path=args.config,
        results_folder=args.results,
        path_ADNIMERGE=adnimerge_path,
    )


if __name__ == "__main__":
    main()
