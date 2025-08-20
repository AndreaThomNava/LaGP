import argparse
import concurrent.futures
import os
import pickle
import re
import numpy as np
import yaml
from scipy.stats import norm
import matplotlib.pyplot as plt

from lagp.models.model import (  # Assuming you have these model functions
    model_gpboost, model_gpytorch, model_viva_gp, model_boosting_l1)
from lagp.utils.generate_data import  load_X_y_preprocessed, save_cv_splits, load_cv_splits, load_X_y, save_cv_splits_preprocessed
from lagp.utils.metrics import quantile_score


def fit_and_evaluate_replicate(X, y, fold,
    configs, models, version
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
    # standardize the response

    y = (y - y.mean()) / y.std()    
    train_idx = fold["train_idx"][:10000]
    test_idx = fold["test_idx"][:1000]
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Define a tolerance for near-constant features
    TOL = 1e-8
    # Drop constant or nearly constant columns in training set
    is_nonconstant = X_train.std(axis=0) > TOL
    X_train = X_train.loc[:, is_nonconstant]
    X_test = X_test.loc[:, is_nonconstant]  # apply the same mask to test

    # Log or print which features were dropped
    dropped = X.columns[~is_nonconstant]
    if len(dropped) > 0:
        print(f"Dropped constant features: {list(dropped)}")


    # Compute min and max from training data
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)

    # Avoid division by zero in case some feature is constant
    train_range = train_max - train_min
    train_range[train_range == 0] = 1.0

    # Apply scaling to training and test data
    train_X_scaled = (X_train - train_min) / train_range
    test_X_scaled = (X_test - train_min) / train_range  # use train stats!


    delta_logl = configs["delta_logl"]
    n_epochs = configs["n_epochs"]
    lr = configs["lr"]
    threshold_approx = configs["threshold_approximation"]
    inducing_points = configs["inducing_points"]
    target_quantile = configs["target_quantile"]
    gpb_approxs = configs["gpb_approxs"]
    
   
    model_results = {}
    for model_name in models:
        print(f"model: {model_name}")
        # Fit the model and make predictions
         # Matches models starting with 'gpboost'
        if re.match(r"^gpboost", model_name):
            approx = gpb_approxs[model_name]
            pred, pred_train, elapsed_time, hyper_params = model_gpboost(
                quantile=target_quantile,
                train_X=train_X_scaled,
                train_y=y_train,
                test_X=test_X_scaled,
                test_y=y_test,
                approx=approx,
                delta_logl=delta_logl,
                n_vecchia= threshold_approx,
            )

            latent_pred = pred["mu"]
           

        elif model_name == "gpytorch":
            pred, elapsed_time, hyper_params, pred_train = model_gpytorch(
                quantile=target_quantile,
                train_X=train_X_scaled,
                train_y=y_train,
                test_X=test_X_scaled,
                test_y=y_test,
                epochs=n_epochs,
                lr = lr,
                inducing_threshold=threshold_approx,
                inducing_points=inducing_points
            )

            latent_pred = pred.mean.numpy()


        elif model_name == "VIVA":
            latent_pred, latend_std, elapsed_time, hyper_params = model_viva_gp(
                quantile=target_quantile,
                train_X=train_X_scaled,
                train_y=y_train,
                test_X=test_X_scaled,
                test_y=y_test,
                rho = 1.5, # fixed
                lengthscale_init=0.25,
                outputscale_init=0.25,
                epochs=n_epochs,
                use_ic0=True,
                classify=False,
            )

        elif model_name == "boosting":
            latent_pred, pred_train, elapsed_time, hyper_params= model_boosting_l1(
                quantile=target_quantile,
                train_X=train_X_scaled,
                train_y=y_train,
                test_X=test_X_scaled,
                test_y=y_test,
            )
            

        # if X is 2d -> plot countor and save figure
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=latent_pred, s=50, cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, label='Predicted Quantile')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title('Predicted Quantile (No Interpolation)')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            # Save plots
            filename = f"{model_name}"
            for ext in ["png", "pdf"]:
                plt.savefig(f"results/real_data/{version}" / f"{filename}_contour.{ext}", bbox_inches="tight", dpi=300)
            plt.close()
          
        # Compute quantile score
        qs_loss = quantile_score(y = y_test, preds=latent_pred, quantile=target_quantile)

       # compute empirical quantile
        empirical_quantile = np.mean(y_test <= latent_pred)


        # store results
        model_results[model_name] = {
            "quantile_loss": qs_loss,
            "empirical_quantile": empirical_quantile,
            "time": elapsed_time,
            "lengthscale": hyper_params["lengthscale"],
            "signal_variance": hyper_params["signal_variance"],
            "noise_variance": hyper_params["noise_variance"]
        }

    return model_results


def fit_models_on_all_datasets_parallel(configs, models, version):
    
    results = {}
    n_splits = configs["n_splits"]
    # load from config
    DIR =  "data/real_data"
    # Loop over datasets
    for df_name in configs["datasets"]:
        print(f"Dataset: {df_name}")

        # Create splits for the dataset 
        save_cv_splits_preprocessed(df_name, n_splits=n_splits, dir="data/real_data", seed=42)
        X, y = load_X_y_preprocessed(dataset_name=df_name, dir = DIR)
        folds = load_cv_splits(dataset_name=df_name, dir = DIR, n_splits = n_splits)
        # Store results for this configuration
        config_key = f"{df_name}"
        results[config_key] = {}

        # Use ProcessPoolExecutor to parallelize across replicates
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_splits) as executor:
            future_to_replicate = {
                executor.submit(
                    fit_and_evaluate_replicate,
                    X, y, folds[replicate],
                    configs,
                    models,
                    version
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
        default="config_run_real.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="001",
        help="version of experiment",
    )


    args = parser.parse_args()
    version = args.version
    
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    models = configs["models"]
    df_names = configs["datasets"]
    n_splits = configs["n_splits"]

        
    # fit models
    results = fit_models_on_all_datasets_parallel(
        configs, models, version
    )

    # Save the results
    # Ensure the results directory exists
    OUTPUT_DIR = f"results/real_data/{version}"
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
