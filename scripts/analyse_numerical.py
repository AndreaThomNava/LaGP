import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle   
import os
import argparse
import yaml

def plot_numerical(DIR_PATH, config, version):
    plt.style.use('seaborn-v0_8-whitegrid')
    # Unpack config values
    misspecified = config["misspecified"]
    estimate_scale = config["estimate_scale"]
    n_groups = config["num_groups"]

    RESULT_PATH = os.path.join(DIR_PATH, "results_numerical.pkl")
    with open(RESULT_PATH, "rb") as f:
        results = pickle.load(f)

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
    
    model_mapping = config["approx_dict"]
    # Map model names if mapping exists
    if model_mapping:
        df_long = df_long.copy()
        df_long["approx_display"] = df_long["approx"].map(
            lambda x: model_mapping.get(x, x)
        )
        hue_col = "approx_display"
    else:
        hue_col = "approx"

    EXTENDED_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df_long,
        x="sample_size",
        y="value",
        hue=hue_col,
        palette=EXTENDED_PALETTE,
        patch_artist=True,
        showmeans=True, ax = ax, 
        meanprops={
                    'marker': 'D',           # Diamond shape
                    'markerfacecolor': 'white', # Red fill
                    'markeredgecolor': 'black', # Black outline
                    'markersize': 6,         # Size
                    'markeredgewidth': 2.5     # Outline thickness
                    },
        #showfliers=False,
        #meanline=True,
    )
    # Increase font sizes
    ax.set_title(f"Marginal Log-Likelihood Approximation Error \n  Misspecified: {misspecified}", 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Group Size", fontsize=18, fontweight='bold')
    ax.set_ylabel("Relative Error", fontsize=18, fontweight='bold')
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    
    # Position legend below plot
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels,
                        # title="Model",
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.12),
                        ncol=min(len(labels), 4),  # Max 4 columns
                        fontsize=16,
                        title_fontsize=18,
                        frameon=True,
                        fancybox=True,
                        shadow=True)
    legend.get_title().set_fontweight('bold')
    
    # Adjust layout for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
            
    # Save plots
    
    filename = f"miss{misspecified}_estscale{estimate_scale}"
    for ext in ["png", "pdf"]:
        plt.savefig(f"{DIR_PATH}/{filename}.{ext}", bbox_inches="tight", dpi=300)
        
    plt.show()

    plt.close()


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type = str,
                        default = "001",
                        help="version of the experiment to analyze")
    
    args = parser.parse_args()
    version = args.version
    DIR_PATH = os.path.join(f"results/numerical/{version}")
    CONFIGS_PATH = "configs/config_run_numerical.yaml"

    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f)
   
    plot_numerical(DIR_PATH, configs, version)