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
    load_data,
    load_scale_gp,
    obtain_quantile,
)
from lagp.utils.metrics import (
    compute_rmse,
    coverage_and_width,
    interval_score,
    quantile_score,
)


def threshold_to_key(x: float) -> str:
    return str(float(x))


def fit_and_evaluate_replicate(configs, replicate_config, replicate):
    likelihood = replicate_config["likelihood"]
    is_heteroscedastic = re.search("heteroscedastic", likelihood) is not None
    sample_size = replicate_config["sample_size"]
    input_dim = replicate_config["input_dim"]

    target_coverage = configs["target_coverage"]
    train_split = configs["train_split"]
    threshold_approx = configs["threshold_approximation"]
    target_quantile = configs["target_quantile"]
    gpb_approxs = configs["gpb_approxs"]
    delta_logl_values = configs["delta_logl_values"]

    f, train_X, train_y, test_X, test_y = load_data(
        likelihood, sample_size, input_dim, replicate, train_split
    )

    if is_heteroscedastic:
        g = load_scale_gp(likelihood, sample_size, input_dim, replicate, file_path=None)
        noise = likelihood.split("_")[0]
    else:
        g = None
        noise = likelihood

    true_latent_quantile = obtain_quantile(
        f=f,
        noise=noise,
        pars=configs["simulation"]["pars"],
        target_quantile=target_quantile,
        g=g,
    )

    normv = norm()
    alpha = 1 - target_coverage
    t = normv.ppf(1 - alpha / 2)

    len_train = len(train_y)
    train_true_latent_quantile = true_latent_quantile[:len_train]
    test_true_latent_quantile = true_latent_quantile[len_train:]

    model_results = {}
    approx = gpb_approxs["gpboost_cc"]

    for delta_logl in delta_logl_values:
        model_name = f"gpboost_cc_dll_{threshold_to_key(delta_logl)}"
        print(f"fitting {model_name}")

        pred, pred_train, elapsed_time, hyper_params = model_gpboost(
            quantile=target_quantile,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
            approx=approx,
            delta_logl=delta_logl,
            n_vecchia=threshold_approx,
        )

        latent_pred = pred["mu"]
        stddev_pred = np.sqrt(pred["var"])
        low_pred = latent_pred - stddev_pred * t
        up_pred = latent_pred + stddev_pred * t

        latent_pred_train = pred_train["mu"]
        stddev_pred_train = np.sqrt(pred_train["var"])
        low_pred_train = latent_pred_train - stddev_pred_train * t
        up_pred_train = latent_pred_train + stddev_pred_train * t

        qs_loss = quantile_score(y=test_y, preds=latent_pred, quantile=target_quantile)

        interval_loss = interval_score(
            y=test_true_latent_quantile,
            pred_low=low_pred,
            pred_up=up_pred,
            alpha=alpha,
        )

        rmse = compute_rmse(f_true=test_true_latent_quantile, f_pred=latent_pred)

        coverage, width = coverage_and_width(
            y=test_true_latent_quantile, pred_low=low_pred, pred_up=up_pred
        )

        train_coverage, train_width = coverage_and_width(
            y=train_true_latent_quantile,
            pred_low=low_pred_train,
            pred_up=up_pred_train,
        )

        empirical_quantile = (test_y <= latent_pred).mean()

        true_pinball_loss = quantile_score(
            y=test_y, preds=test_true_latent_quantile, quantile=target_quantile
        )

        bias = (latent_pred - test_true_latent_quantile).mean()

        model_results[model_name] = {
            "delta_logl": delta_logl,
            "quantile_loss": qs_loss,
            "true_pinball_loss": true_pinball_loss,
            "interval_loss": interval_loss,
            "rmse": rmse,
            "bias": bias,
            "empirical_quantile": empirical_quantile,
            "coverage": coverage,
            "width": width,
            "train_coverage": train_coverage,
            "train_width": train_width,
            "time": elapsed_time,
            "lengthscale": hyper_params["lengthscale"],
            "signal_variance": hyper_params["signal_variance"],
            "noise_variance": hyper_params["noise_variance"],
        }

    return model_results


def fit_models_on_all_datasets_parallel(configs, num_replicates=10):
    results = {}

    for likelihood in configs["simulation"]["likelihoods"]:
        for sample_size in configs["simulation"]["sample_sizes"]:
            for input_dim in configs["simulation"]["dimensions"]:
                replicate_config = {
                    "likelihood": likelihood,
                    "sample_size": sample_size,
                    "input_dim": input_dim,
                }

                config_key = f"{likelihood}_{sample_size}_{input_dim}"
                results[config_key] = {}
                print(config_key)
                print(os.cpu_count())

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
    parser = argparse.ArgumentParser(description="Run GP TKC sensitivity study.")
    parser.add_argument(
        "--config",
        type=str,
        default="config_gp_sensitivity.yaml",
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

    snr = configs["simulation"]["snr"]
    fixed_snr = configs["simulation"]["fixed_snr"]
    signal_variance = configs["gp_parameters"]["kernel"]["signal_variance"]

    if fixed_snr:
        pars = compute_dict_pars(
            signal_variance=signal_variance,
            snr=snr,
            quantile=configs["simulation"]["pars"]["ald"]["q"],
        )
        configs["simulation"]["pars"] = pars

    num_replicates = configs["simulation"]["replicates"]

    results = fit_models_on_all_datasets_parallel(
        configs, num_replicates=num_replicates
    )

    OUTPUT_DIR = f"results/simulation_gp_sensitivity/{version}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {
        "config": configs,
        "results": results,
    }

    output_file = os.path.join(OUTPUT_DIR, "simulation_results_python.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(all_results, f)

    print(f"Results and config saved to {output_file}")