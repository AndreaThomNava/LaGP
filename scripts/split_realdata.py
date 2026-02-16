import argparse
import os

import yaml

from lagp.utils.generate_data import prepare_real_data, save_cv_splits_preprocessed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation study with models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_mm_real.yaml",
        help="Path to the YAML config file",
    )

    parser.add_argument(
        "--subsample",
        action="store_true",
        help="If set, subsample the data",
    )

    args = parser.parse_args()
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    df_names = configs["datasets"]
    n_splits = configs["n_splits"]
    # create splits for all datasets
    for dataset_name in df_names:
        prepare_real_data(
            dataset_name, dir="data/real_data_mm", subsample=args.subsample
        )
        save_cv_splits_preprocessed(
            dataset_name, n_splits=n_splits, dir="data/real_data_mm", seed=42
        )
