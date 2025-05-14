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



def main(config, approx):

    # set Laplace approximation !
    laplace_approximation = approx

    # Unpack config values
    group_sizes = config["group_sizes"]
    num_groups = config["num_groups"]
    re_mean = config.get("re_mean", 0)
    re_std = config["re_std"]
    scale = config["scale_laplace"]
    quantile = config["quantile"]
    approx_dict = config["approx_dict"]
    min_decreases = config["min_decreases"]
    B = config["B"]
    K = config["integration_nodes"]
    misspecified = config["misspecified"]

    results = {}



    for group_size in group_sizes:
        print(f"Approximation: {approximations}. Sample size: {group_size}")
        gps, naives, aghqs = run_simulation(
            laplace_approximation=laplace_approximation,
            group_size=group_size,
            num_groups=num_groups,
            re_mean=re_mean,
            re_std=re_std,
            scale=scale,
            quantile=quantile,
            min_decreases=min_decreases,
            K = K,
            B=B,
            misspecified=misspecified
        )
        results[group_size] = {"gps": gps, "naives": naives, "aghqs": aghqs}

    # Optionally, save results
    if "save_path" in config:
        save_path = config["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(config["save_path"], "wb") as f:
            pickle.dump(results, f)


    all_diffs = []

    for group_size, result in results.items(): #loop over sample_sizes
        gps = result["gps"]
        aghqs = result["aghqs"]

        for min_dec, gp_vals in gps.items(): # loop over min_decreases
            aghq_vals = aghqs[min_dec] # since for now I recompute for each min_decrease

            diffs = ( np.array(gp_vals) - np.array(aghq_vals) ) / (np.array(aghq_vals))
            all_diffs.extend([
                {
                    "sample_size": group_size,
                    "min_dec_ll": min_dec,
                    "value": diff
                }
                for diff in diffs
            ])

    df_long = pd.DataFrame(all_diffs)
    df_long["sample_size"] = df_long["sample_size"].astype(int)
    df_long = df_long.sort_values(by="sample_size")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_long,
        x="sample_size",
        y="value",
        hue="min_dec_ll",
        palette="tab10",
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        meanline=True
    )
    plt.xlabel("Sample Size")
    plt.ylabel("Relarive Difference w.r.t. adapt. GHQ")
    plt.title(f"{approx_dict[laplace_approximation]}. Misspecified: {misspecified}", color="black")
    legend = plt.legend(title="Min Decrease", labelcolor="black")
    legend.get_title().set_color("black")
    plt.tight_layout()

    # Save plots
    
    OUTPUT_DIR = Path("results/numerical")
    filename = f"{approx_dict[laplace_approximation]}_miss{misspecified}"
    for ext in ["png", "pdf"]:
        plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)
        
    plt.show()

    plt.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPBoost simulation study")
    parser.add_argument("--config", type=str,
                        default = "configs/config_run_numerical.yaml"
                        , help="Path to YAML config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # loop over approximations
    approximations = config["laplace_approximations"]
    for approx in approximations:
        main(config, approx)
