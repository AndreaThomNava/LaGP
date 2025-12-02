import argparse
import concurrent.futures
import os
import pickle
import re
import numpy as np
import yaml
from scipy.stats import norm

from lagp.models.model import (  # Assuming you have these model functions
    model_gpboost, model_gpytorch, model_viva_gp)
from lagp.utils.generate_data import  load_X_y_preprocessed, load_cv_splits, save_cv_splits_preprocessed
from lagp.utils.metrics import quantile_score


def fit_and_evaluate_replicate(X, group_data, Y, fold,
    configs, models, replicate
):
    """
    Fit the models on a single replicate and compute the evaluation metrics.

    Input:
        - config: dictionary with config information (likelihood, sample_size, input_dim)
        - replicate: replicate index
        - models: list of model names (e.g., ['gpboost', 'gpytorch'])

    Output:
        - metrics: dictionary with model names as keys and metrics as values
    """
    
    train_idx = fold["train_idx"] #[:1000]
    test_idx = fold["test_idx"]# [:1000]
    
    X_train = X[train_idx]
    y_train = Y[train_idx]  
   
    train_mean = np.mean(y_train)
    train_std = np.std(y_train)

    X_test = X[test_idx]    
    y_test = Y[test_idx]

    y_train_scaled = (y_train - train_mean) / train_std
    y_test_scaled = (y_test - train_mean) / train_std
    
    group_train = group_data[train_idx]
    group_test = group_data[test_idx]
    
    # Compute min and max from training data
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    
    # Avoid division by zero in case some feature is constant
    train_range = train_max - train_min
   
    train_min[train_range == 0] =  0.0  # Add this line
    train_range[train_range == 0] = 1.0
    
    # Apply scaling to training and test data
    train_X_scaled = (X_train - train_min) / train_range
    test_X_scaled = (X_test - train_min) / train_range  # use train stats!

    delta_logl = configs["delta_logl"]
    target_quantile = configs["target_quantile"]
    gpb_approxs = configs["gpb_approxs"]
    
   
    model_results = {}
    for model_name in models:
        print(f"model: {model_name}")
        # Fit the model and make predictions
         # Matches models starting with 'gpboost'
        if re.match(r"^gpboost", model_name):
            approx = gpb_approxs[model_name]
            pred, res, elapsed_time, hyper_params = model_gpboost(
                quantile=target_quantile,
                train_X=train_X_scaled,
                group_train=group_train,
                train_y=y_train_scaled,
                test_X=test_X_scaled,
                group_test=group_test,
                test_y=y_test_scaled,
                approx=approx,
                delta_logl=delta_logl,
                estimate_hyper=configs["estimate_hyper"],
            )


            # with fixed effects
            pred_with_fixed_effects = pred["mu"]
            stddev_pred = np.sqrt(pred["var"])
           

        # Compute quantile score
        qs_loss = quantile_score(y = y_test_scaled, preds=pred_with_fixed_effects, quantile=target_quantile)

       
        # store results
            # store results
        model_results[model_name] = {
            "quantile_loss": qs_loss,
            "time": elapsed_time,
            "hyper_params": hyper_params
        }


    return model_results


def fit_models_on_all_datasets_parallel(configs, models):
    
    results = {}
    n_splits = configs["n_splits"]
    # load from config
    DIR =  "data/real_data_mm"
    # Loop over datasets
    for df_name in configs["datasets"]:
        print(f"Dataset: {df_name}")
        # Create splits for the dataset 
        save_cv_splits_preprocessed(df_name, n_splits=n_splits, dir="data/real_data_mm", seed=42)
        X, group_data, y = load_X_y_preprocessed(dataset_name=df_name, dir = DIR)
        folds = load_cv_splits(dataset_name=df_name, dir = DIR, n_splits = n_splits)
        # Store results for this configuration
        config_key = f"{df_name}"
        results[config_key] = {}

        # Use ProcessPoolExecutor to parallelize across replicates
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_splits) as executor: # n_splits
            future_to_replicate = {
                executor.submit(
                    fit_and_evaluate_replicate,
                    X, group_data, y, folds[replicate],
                    configs,
                    models,
                    replicate + 1
                ): replicate
                for replicate  in range(n_splits) #n_splits#
            }

            for future in concurrent.futures.as_completed(future_to_replicate):
                replicate = future_to_replicate[future]
                try:
                    replicate_results = future.result()
                    print(f"results: {replicate_results}")
                    # Store the results for this replicate
                    results[config_key][replicate + 1] = replicate_results
                except Exception as e:
                    print(f"Error with replicate {replicate}: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation study with models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_mm_real.yaml",
        help="Path to the YAML config file",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="001",
        help="version number for the run (default: 001)",
    )

    args = parser.parse_args()

    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    models = configs["models"]
    df_names = configs["datasets"]
    n_splits = configs["n_splits"]

        
    # fit models
    results = fit_models_on_all_datasets_parallel(
        configs, models
    )

    # Save the results
    # Ensure the results directory exists
    OUTPUT_DIR = f"results/real_data_mm/{args.version}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Combine the results and config into one dictionary
    all_results = {
        "config": configs,
        "results": results,
    }

    # Construct the output file path (e.g., "results.pkl")
    output_file = os.path.join(OUTPUT_DIR, "real_data_results_python.pkl")

    # Save the combined dictionary using pickle
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results and config saved to {output_file}")
