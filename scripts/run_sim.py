import argparse
import concurrent.futures
import os
import pickle
import re
import numpy as np
import yaml
from scipy.stats import norm

from lagp.models.model import (  # Assuming you have these model functions
    model_gpboost, model_gpytorch)
from lagp.utils.generate_data import load_data, obtain_quantile, load_scale_gp
from lagp.utils.metrics import (coverage_and_width, interval_score,
                                quantile_score)


def fit_and_evaluate_replicate(
    configs, replicate_config, replicate, models
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
    likelihood = replicate_config["likelihood"]
    is_heteroscedastic = re.search("heteroscedastic", likelihood) is not None
    sample_size = replicate_config["sample_size"]
    input_dim = replicate_config["input_dim"]
    
    approx = configs["approximation"]
    delta_logl = configs["delta_logl"]
    alpha = configs["alpha"]
    train_split = configs["train_split"]
    n_epochs = configs["n_epochs"]
    lr = configs["lr"]
    threshold_approx = configs["threshold_approximation"]
    inducing_points = configs["inducing_points"]
    target_quantile = configs["target_quantile"]
    gpb_approxs = configs["gpb_approxs"]

    # Load the dataset for this replicate
    f, train_X, train_y, test_X, test_y = load_data(
        likelihood, sample_size, input_dim, replicate, train_split
    )
    # obtain true latent quantile

    # if heteroscedastic likelihood, load also second GP!
    if is_heteroscedastic:
        g = load_scale_gp(likelihood, sample_size, input_dim, replicate, file_path=None)
        noise = likelihood.split("_")[0]
    else:
        noise = likelihood

    true_latent_quantile = obtain_quantile(
        f=f, noise=noise, pars=configs["simulation"]["pars"], target_quantile=target_quantile, g = g if is_heteroscedastic else None
    )
    # needed for prediction intervals
    normv = norm()
    t = normv.ppf(1 - (1 - alpha) / 2)

    model_results = {}
    for model_name in models:
        # Fit the model and make predictions
        print(f"fitting {model_name}")
        
        # Matches models starting with 'gpboost'
        if re.match(r"^gpboost", model_name):
            approx = gpb_approxs[model_name]
           
            pred, elapsed_time, hyper_params = model_gpboost(
                quantile=target_quantile,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
                approx=approx,
                delta_logl=delta_logl,
                n_vecchia= threshold_approx,
            )

            latent_pred = pred["mu"]
            stddev_pred = np.sqrt(pred["var"])
            low_pred = latent_pred - stddev_pred * t
            up_pred = latent_pred + stddev_pred * t

        elif model_name == "gpytorch":
            pred, elapsed_time, hyper_params = model_gpytorch(
                quantile=target_quantile,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
                epochs=n_epochs,
                lr = lr,
                inducing_threshold=threshold_approx,
                inducing_points=inducing_points
            )

            latent_pred = pred.mean.numpy()

            stddev_pred = pred.stddev.numpy()
            low_pred = latent_pred - stddev_pred * t
            up_pred = latent_pred + stddev_pred * t

        # Compute quantile score
        qs_loss = quantile_score(y=test_y, preds=latent_pred, quantile=target_quantile)

        # Compute interval score
        len_train = len(train_y)
        test_true_latent_quantile = true_latent_quantile[len_train:]
        interval_loss = interval_score(
            y=test_true_latent_quantile,
            pred_low=low_pred,
            pred_up=up_pred,
            alpha=alpha,
        )

        # compute coverage and width
        coverage, width = coverage_and_width(
            y=test_true_latent_quantile, pred_low=low_pred, pred_up=up_pred
        )

        # store results
        model_results[model_name] = {
            "quantile_loss": qs_loss,
            "interval_loss": interval_loss,
            "coverage": coverage,
            "width": width,
            "time": elapsed_time,
            "lengthscale": hyper_params["lengthscale"],
            "signal_variance": hyper_params["signal_variance"],
            "noise_variance": hyper_params["noise_variance"]
        }

    return model_results


def fit_models_on_all_datasets_parallel(configs, models, num_replicates=10):
    results = {}

    # Loop over configurations
    for likelihood in configs["simulation"]["likelihoods"]:
        for sample_size in configs["simulation"]["sample_sizes"]:
            for input_dim in configs["simulation"]["dimensions"]:

                # Prepare the configuration dictionary
                replicate_config = {
                    "likelihood": likelihood,
                    "sample_size": sample_size,
                    "input_dim": input_dim,
                }

                # Store results for this configuration
                config_key = f"{likelihood}_{sample_size}_{input_dim}"
                results[config_key] = {}
                print(config_key)
                # Use ProcessPoolExecutor to parallelize across replicates
                with concurrent.futures.ProcessPoolExecutor(max_workers = num_replicates) as executor:
                    future_to_replicate = {
                        executor.submit(
                            fit_and_evaluate_replicate,
                            configs,
                            replicate_config,
                            replicate,
                            models,
                        ): replicate
                        for replicate  in range(1, num_replicates+1)
                    }

                    for future in concurrent.futures.as_completed(future_to_replicate):
                        replicate = future_to_replicate[future]
                        try:
                            replicate_results = future.result()
                            print(f"results: {replicate_results}")
                            # Store the results for this replicate
                            results[config_key][replicate] = replicate_results
                        except Exception as e:
                            print(f"Error with replicate {replicate}: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation study with models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_run_simulation.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    models = configs["models"]
    num_replicates = configs["simulation"]["replicates"]

    results = fit_models_on_all_datasets_parallel(
        configs, models, num_replicates=num_replicates
    )

    # Save the results
    # Ensure the results directory exists
    OUTPUT_DIR = "results/simulation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Combine the results and config into one dictionary
    all_results = {
        "config": configs,
        "results": results,
    }

    # Construct the output file path (e.g., "results.pkl")
    output_file = os.path.join(OUTPUT_DIR, "simulation_results_python.pkl")

    # Save the combined dictionary using pickle
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results and config saved to {output_file}")
