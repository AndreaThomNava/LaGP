import argparse
import os
import yaml
from lagp.utils.generate_data import  save_cv_splits_preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation study with models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_run_real.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    models = configs["models"]
    df_names = configs["datasets"]
    n_splits = configs["n_splits"]
    # create splits for all datasets
    for dataset_name in df_names:
        save_cv_splits_preprocessed(dataset_name, n_splits=n_splits, dir="data/real_data", seed=42)
        