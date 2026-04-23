import argparse
import concurrent.futures
import os
import pickle
import re

import numpy as np
import yaml
from scipy.stats import norm

from lagp.models.model import model_gpboost
from lagp.utils.generate_data import (
    compute_dict_pars,
    get_true_curvature,
    load_data,
    obtain_quantile,
)
from lagp.utils.metrics import (
    compute_rmse,
    coverage_and_width,
    interval_score,
    quantile_score,
)


def threshold_to_key(x: float) -> str:
    return str(x)


def fit_and_evaluate_replicate(configs, replicate_config, replicate):
    likelihood = replicate_config["likelihood"]
    is_heteroscedastic = re.search("heteroscedastic", likelihood) is not None
    g = None

    n_group = replicate_config["n_group"]
    group_size = replicate_config["group_size"]

    alpha = configs["alpha"]
    target_quantile = configs["target_quantile"]
    randeff = configs["randeff"]
    estimate_hyper = configs["estimate_hyper"]

    delta_logl_values = configs["delta_logl_values"]
    approx = configs["gpb_approxs"]["gpboost_cc"]

    data = load_data(
        randeff=randeff,
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
    group_test = data["group_test"]
    eps_test = data["eps_test"]
    fe_test = data["fe_test"]

    noise = likelihood

    test_true_latent_quantile = obtain_quantile(
        eps=eps_test + fe_test,
        noise=noise,
        pars=configs["data_generation"]["pars"],
        target_quantile=target_quantile,
        g=g if is_heteroscedastic else None,
    )

    normv = norm()
    t = normv.ppf(1 - (1 - alpha) / 2)

    true_curvature = get_true_curvature(
        f=eps_test,
        true_quantile=test_true_latent_quantile,
        noise=noise,
        pars=configs["data_generation"]["pars"],
    )
    true_var = target_quantile * (1 - target_quantile) / true_curvature**2

    model_results = {}

    for delta_logl in delta_logl_values:
        model_key = f"gpboost_cc_dll_{threshold_to_key(delta_logl)}"
        print(f"fitting {model_key}")

        pred, res, elapsed_time, hyper_params = model_gpboost(
            quantile=target_quantile,
            train_X=train_X,
            group_train=group_train,
            train_y=train_y,
            test_X=test_X,
            group_test=group_test,
            test_y=test_y,
            approx=approx,
            delta_logl=delta_logl, # 
            estimate_hyper=estimate_hyper,
        )

        pred_with_fixed_effects = pred["mu"]
        stddev_pred = np.sqrt(pred["var"])

        low_pred = pred_with_fixed_effects - stddev_pred * t
        up_pred = pred_with_fixed_effects + stddev_pred * t

        low_true = pred_with_fixed_effects - np.sqrt(true_var) * t
        up_true = pred_with_fixed_effects + np.sqrt(true_var) * t

        qs_loss = quantile_score(
            y=test_y, preds=pred_with_fixed_effects, quantile=target_quantile
        )

        interval_loss = interval_score(
            y=test_true_latent_quantile,
            pred_low=low_pred,
            pred_up=up_pred,
            alpha=alpha,
        )

        rmse = compute_rmse(
            f_true=test_true_latent_quantile, f_pred=pred_with_fixed_effects
        )

        coverage, width = coverage_and_width(
            y=test_true_latent_quantile, pred_low=low_pred, pred_up=up_pred
        )

        coverage_true, width_true = coverage_and_width(
            y=test_true_latent_quantile, pred_low=low_true, pred_up=up_true
        )

        model_results[model_key] = {
            "quantile_loss": qs_loss,
            "interval_loss": interval_loss,
            "rmse": rmse,
            "coverage": coverage,
            "coverage_true": coverage_true,
            "width": width,
            "time": elapsed_time,
            "hyper_params": hyper_params,
            "delta_logl": delta_logl,
        }

    return model_results


def fit_models_on_all_datasets_parallel(configs, num_replicates=10):
    results = {}

    for likelihood in configs["likelihood"]:
        for n_group in configs["n_groups"]:
            for group_size in configs["group_size"]:
                replicate_config = {
                    "likelihood": likelihood,
                    "n_group": n_group,
                    "group_size": group_size,
                }

                config_key = f"{likelihood}_{n_group}_{group_size}"
                results[config_key] = {}
                print(config_key)

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_replicates
                ) as executor:
                    future_to_replicate = {
                        executor.submit(
                            fit_and_evaluate_replicate,
                            configs,
                            replicate_config,
                            replicate,
                        ): replicate
                        for replicate in range(1, num_replicates + 1)
                    }

                    for future in concurrent.futures.as_completed(future_to_replicate):
                        replicate = future_to_replicate[future]
                        try:
                            replicate_results = future.result()
                            results[config_key][replicate] = replicate_results
                        except Exception as e:
                            print(f"Error with replicate {replicate}: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TKC sensitivity study.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_mm_sens.yaml",
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

    if configs["data_generation"].get("fixed_snr", True):
        pars = compute_dict_pars(
            signal_variance=configs["data_generation"]["signal_variance"],
            snr=configs["data_generation"]["snr"],
            quantile=configs["data_generation"]["quantile"],
        )
        configs["data_generation"]["pars"].update(pars)

    num_replicates = configs["replicate"]
    randeff = configs["randeff"]

    results = fit_models_on_all_datasets_parallel(
        configs, num_replicates=num_replicates
    )

    OUTPUT_DIR = os.path.join(
        "results/simulation_mm_sensitivity", f"{randeff}", args.version
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {
        "config": configs,
        "results": results,
    }

    output_file = os.path.join(OUTPUT_DIR, "simulation_results_python.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results and config saved to {output_file}")