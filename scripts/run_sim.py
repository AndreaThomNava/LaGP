import argparse
import concurrent.futures
import os
import pickle
import re
import numpy as np
import yaml
from scipy.stats import norm

from lagp.models.model import (  # Assuming you have these model functions
    model_gpboost, model_gpytorch, model_viva_gp, model_gpboost_twostage)
from lagp.utils.generate_data import load_data, obtain_quantile, load_scale_gp, compute_dict_pars, get_true_curvature
from lagp.utils.metrics import (coverage_and_width, interval_score,
                                quantile_score, align_re, compute_rmse)


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
    n_group = replicate_config["n_group"]
    group_size = replicate_config["group_size"]
    
    delta_logl = configs["delta_logl"]
    alpha = configs["alpha"]
    gpb_approxs = configs["gpb_approxs"]
    target_quantile = configs["target_quantile"]
    randeff = configs["randeff"]
    estimate_hyper = configs["estimate_hyper"]
    print(f"estimating hyper-parameters: {estimate_hyper}")
    data = load_data(
        randeff=randeff,  # e.g., "Two_randomly_crossed_random_effects"
        likelihood=likelihood,
        n_groups=n_group,
        group_size=group_size,
        replicate=replicate,
    )

    train_X = data["X_train"]
    train_y = data["y_train"]
    test_X = data["X_test"]
    test_y = data["y_test"]
    group_train = data["group_train"]
    print(group_train.shape)
    group_test = data["group_test"]
    eps_test = data["eps_test"]



    # obtain true latent quantile
    noise = likelihood

    test_true_latent_quantile = obtain_quantile(
        eps = eps_test, noise=noise, pars=configs["data_generation"]["pars"], target_quantile=target_quantile, g = g if is_heteroscedastic else None
    )
    # needed for prediction intervals
    normv = norm()
    t = normv.ppf(1 - (1 - alpha) / 2)

    # true asympt curvature
    true_curvature = get_true_curvature(f = eps_test, true_quantile = test_true_latent_quantile,
                                        noise = noise, pars = configs["data_generation"]["pars"])
    # print(" --------- true curvature ----------", true_curvature)
    # print("likelihood:", noise)
    true_var = target_quantile*(1-target_quantile) / true_curvature**2
    model_results = {}
    for model_name in models:
        # Fit the model and make predictions
        print(f"fitting {model_name}")
        
        # Matches models starting with 'gpboost'
        if re.match(r"^gpboost", model_name):
            approx = gpb_approxs[model_name]
            
            if model_name.endswith("_twostage"):
                print("-- start twostage --")
                pred, res, elapsed_time, hyper_params = model_gpboost_twostage(
                    quantile=target_quantile,
                    train_X=train_X,
                    group_train=group_train,
                    train_y=train_y,
                    test_X=test_X,
                    group_test=group_test,
                    test_y=test_y,
                    approx=approx,
                    delta_logl=delta_logl,
                )
                

            else:
                pred, res, elapsed_time, hyper_params = model_gpboost(
                    quantile=target_quantile,
                    train_X=train_X,
                    group_train=group_train,
                    train_y=train_y,
                    test_X=test_X,
                    group_test=group_test,
                    test_y=test_y,
                    approx=approx,
                    delta_logl=delta_logl,
                    estimate_hyper=estimate_hyper,
                )

            # with fixed effects
            pred_with_fixed_effects = pred["mu"]
            stddev_pred = np.sqrt(pred["var"])
            
    

            if randeff == "One_random_effect":
                # matches predicted random effects from training groups to test groups
                true_re, pred_re, stddev_pred, true_var_aligned = align_re(group_train, group_test, res, 
                                                                           test_true_latent_quantile, 
                                                                           stddev_pred, true_var, group_size)
                low_pred = pred_re - stddev_pred * t
                up_pred = pred_re + stddev_pred * t

                # using true sandwhich variance
                low_true = pred_re - np.sqrt(true_var_aligned) * t
                up_true = pred_re + np.sqrt(true_var_aligned) * t

            
        # Compute quantile score
        qs_loss = quantile_score(y=test_y, preds=pred_with_fixed_effects, quantile=target_quantile)

        if randeff == "One_random_effect":
            interval_loss = interval_score(
                y=true_re,
                pred_low=low_pred,
                pred_up=up_pred,
                alpha=alpha,)
            
            rmse= compute_rmse(f_true=true_re, f_pred=pred_re)

            # compute coverage and width
            coverage, width = coverage_and_width(
            y=true_re, pred_low=low_pred, pred_up=up_pred)

            # compute coverage and width
            coverage_true, width_true = coverage_and_width(
            y=true_re, pred_low=low_true, pred_up=up_true)
            print("coverage true: ", coverage_true)
            print("curvature var:", np.sqrt(true_var_aligned[0]))
            print("model", model_name)
            print("estimated var: ", stddev_pred[0])

        else:
            interval_loss = np.nan
            coverage, width, rmse, coverage_true = np.nan, np.nan, np.nan, np.nan




        # store results
        model_results[model_name] = {
            "quantile_loss": qs_loss,
            "interval_loss": interval_loss,
            "rmse": rmse,
            "coverage": coverage,
            "coverage_true": coverage_true,
            "width": width,
            "time": elapsed_time,
            "hyper_params": hyper_params
        }

    return model_results


def fit_models_on_all_datasets_parallel(configs, models, num_replicates=10):
    results = {}

    # Loop over configurations
    for likelihood in configs["likelihood"]:
        for n_group in configs["n_groups"]:
            for group_size in configs["group_size"]:

                # Prepare the configuration dictionary
                replicate_config = {
                    "likelihood": likelihood,
                    "n_group": n_group,
                    "group_size": group_size,
                }

                # Store results for this configuration
                config_key = f"{likelihood}_{n_group}_{group_size}"
                results[config_key] = {}
                print(config_key)
                # Use ProcessPoolExecutor to parallelize across replicates
                print(os.cpu_count())
                with concurrent.futures.ProcessPoolExecutor(max_workers = 1) as executor: # num replicates
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
        default="config_test.yaml",
        help="Path to the YAML config file",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="001",
        help="Version number",
    )
    args = parser.parse_args()
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)


    # Generate noise parameters if fixed_snr is True
    if configs["data_generation"].get("fixed_snr", True):
        pars = compute_dict_pars(
            signal_variance=configs["data_generation"]["signal_variance"],
            snr=configs["data_generation"]["snr"],
            quantile=configs["data_generation"]["quantile"],
        )
        # Merge noise pars into data generation params
        configs["data_generation"]["pars"].update(pars)
    

    models = configs["models"]
    num_replicates = configs["replicate"]
    randeff = configs["randeff"]

    results = fit_models_on_all_datasets_parallel(
        configs, models, num_replicates=num_replicates
    )

    # Save the results
    # Ensure the results directory exists
    OUTPUT_DIR = os.path.join("results/simulation_mm", f"{randeff}", args.version)
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
