from lagp.utils.integration import simulate_grouped_response, compute_aghq, compute_naive_log_marglik, gpboost_nll_min_dec, run_simulation
import argparse
import yaml
from pathlib import Path
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")



def main(config, version):

   
    # Unpack config values
    group_sizes = config["group_sizes"]
    num_groups = config["num_groups"]
    re_mean = config.get("re_mean", 0)
    re_std = config["re_std"]
    scale = config["scale_laplace"]
    quantile = config["quantile"]
    approx_dict = config["approx_dict"]
    min_decrease = config["min_decrease"]
    B = config["B"]
    K = config["integration_nodes"]
    misspecified = config["misspecified"]
    laplace_approximations = config["laplace_approximations"]
    use_pred_var = config["use_pred_var"]
    estimate_scale = config["estimate_scale"]
    results = {}



    for group_size in group_sizes:

        gps, naives, aghqs = run_simulation(
            laplace_approximations=laplace_approximations,
            group_size=group_size,
            num_groups=num_groups,
            re_mean=re_mean,
            re_std=re_std,
            scale=scale,
            quantile=quantile,
            min_decrease=min_decrease,
            K = K,
            B=B,
            misspecified=misspecified,
            estimate_scale=estimate_scale,
            use_pred_var = use_pred_var
        )
        results[group_size] = {"gps": gps, "naives": naives, "aghqs": aghqs}

    # Optionally, save results
    
    OUTPUT_DIR = Path(f"results/numerical/{version}")
    # Ensure the directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/results_numerical.pkl", "wb") as f:
        pickle.dump(results, f)


    all_diffs = []

    for group_size, result in results.items(): #loop over sample_sizes
        gps = result["gps"]
        aghqs = result["aghqs"]
        naives = result["naives"]

        for approx, gp_vals in gps.items(): # loop over approxs
            aghq_vals = aghqs[approx] # since for now I recompute for each min_decrease
            naive_vals = naives[approx] # same here
            diffs = ( np.array(gp_vals) - np.array(aghq_vals) ) / (np.array(aghq_vals))
            all_diffs.extend([
                {
                    "sample_size": group_size,
                    "approx": approx,
                    "value": diff
                }
                for diff in diffs
            ])

        # --- Add Naive (one value) --- the last approx in the dict ----
        naive_val = result["naives"][approx]
        naive_diff = (naive_val - np.array(aghq_vals)) / np.array(aghq_vals)

        all_diffs.extend([
                {
                    "sample_size": group_size,
                    "approx": "naive",
                    "value": diff
                }
                for diff in naive_diff
            ])

    df_long = pd.DataFrame(all_diffs)
    df_long["sample_size"] = df_long["sample_size"].astype(int)
    df_long = df_long.sort_values(by="sample_size")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_long,
        x="sample_size",
        y="value",
        hue="approx",
        palette="tab10",
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        meanline=True
    )
    plt.xlabel("Sample Size")
    plt.ylabel("Relarive Difference w.r.t. adapt. GHQ")
    plt.title(f"Misspecified: {misspecified}. Estimated Scale param: {estimate_scale}", color="black")
    legend = plt.legend(title="Approximation", labelcolor="black")
    legend.get_title().set_color("black")
    plt.tight_layout()

    # Save plots
    
    filename = f"miss{misspecified}_estscale{estimate_scale}"
    for ext in ["png", "pdf"]:
        plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)
        
    plt.show()

    plt.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPBoost simulation study")
    parser.add_argument("--config", type=str,
                        default = "configs/config_run_numerical.yaml"
                        , help="Path to YAML config file")
    
    parser.add_argument("--version", type=str,
                        default = "001",
                        help="version of the experiment")

    args = parser.parse_args()
    version = args.version
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config, version)
