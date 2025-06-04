import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from lagp.utils.generate_data import compute_dict_pars

def replace_missing_with_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_missing_with_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_missing_with_nan(item) for item in obj]
    elif obj is None:
        return np.nan
    else:
        return obj


def flatten_hyper_params(hyper_params):
    flat = {}
    re_idx = 1

    for k, v in hyper_params.items():
        if k == "noise_variance":
            flat["noise_variance"] = float(v)
        else:
            flat[f"re_var_{re_idx}"] = float(v)
            re_idx += 1

    return flat


def make_flattened_df(results: pd.DataFrame) -> pd.DataFrame:
    """
    Make a flattened dataframe out of the pickled results.

    Input:
        - results: (pd.DataFrame) pickled dictionary containting the simulation results.
    
    Output:
        - df: (pd.DataFrame) the flattened dataframe.
    """

    config = results["config"]
    results = results["results"]

    df = pd.DataFrame([
        {
            "model": model,
            "likelihood": "_".join(cfg.split("_")[:-2]), # could have an extra split for heteroscedastic
            "n_groups": int(cfg.split("_")[-2]),
            "group_size": int(cfg.split("_")[-1]),
            "replicate": replicate,
            **{k: v for k, v in metrics.items() if k != "hyper_params"},
            **flatten_hyper_params(metrics.get("hyper_params", {}))
        }
        for cfg, reps in results.items()
        for replicate, rep_result in reps.items()
        for model, metrics in rep_result.items()
        ])
    
    return df

def make_flattened_df_real(results: pd.DataFrame) -> pd.DataFrame:
    """
    Make a flattened dataframe out of the pickled results.

    Input:
        - results: (pd.DataFrame) pickled dictionary containting the simulation results.
    
    Output:
        - df: (pd.DataFrame) the flattened dataframe.
    """

    config = results["config"]
    results = results["results"]

    df = pd.DataFrame([
        {
            "model" : model, 
            "dataset": dataset,
            "replicate": replicate,
            **metrics,
        }
        for dataset, reps in results.items()
        for replicate, rep_result in reps.items()
        for model, metrics in rep_result.items()
        ])
    
    return df

def group_flattened_df_real(flat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a flat dataframe with the results and group by likelihood, sample size and dimension.

    Input:
        - flat_df: (pd.DataFrame)

    Output:
        - grouped_df: (pd.DataFrame) grouped by ...
    """

    # Group by dataset and model_method
    grouped = flat_df.groupby(['model', "dataset"])

    # Assuming 'coverage' and 'width' are columns in your DataFrame
    grouped_df = grouped.agg(
        quantile_loss_mean=('quantile_loss', 'mean'),
        quantile_loss_std=('quantile_loss', 'std'),
        time_mean=("time", "mean" ),
        time_std = ("time", "std")
    ).reset_index()

    return grouped_df


def group_flattened_df(flat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a flat dataframe with the results and group by likelihood, sample size and dimension.

    Input:
        - flat_df: (pd.DataFrame)

    Output:
        - grouped_df: (pd.DataFrame) grouped by ...
    """

    # Group by likelihood, sample_size, dim, and model_method
    grouped = flat_df.groupby(['likelihood', 'n_groups', 'group_size', 'model'])

    def sem(x):
        return x.std() / np.sqrt(len(x))

    # Assuming 'coverage' and 'width' are columns in your DataFrame
    grouped_df = grouped.agg(
        quantile_loss_mean=('quantile_loss', 'mean'),
        quantile_loss_std=('quantile_loss', sem),
        interval_loss_mean=('interval_loss', 'mean'),
        interval_loss_std=('interval_loss', sem),
        coverage_mean=('coverage', 'mean'),
        coverage_std=('coverage', sem),
        width_mean=('width', 'mean'),
        width_std=('width', sem),
        time_mean=("time", "mean" ),
        time_std = ("time", sem)
    ).reset_index()

    return grouped_df


def df_mse_hyper(flat_df, configs):
   
    models_with_hyperparams = configs["models_with_hyperparams"]
    signal_variance = configs["data_generation"]["signal_variance"]
    true_variances = {"re_var_1": signal_variance}
    randeff = configs["randeff"]
    if randeff != "One_random_effect":
        signal_variance_2 = configs["data_generation"]["signal_variance_2"]
        true_variances.update({"re_var_2": signal_variance_2})

    pars = compute_dict_pars(
            signal_variance=signal_variance,
            snr=configs["data_generation"]["snr"],
            quantile=configs["target_quantile"],
        )
    
    # keep only if model has estimated the hyper-parameters
    filtered_df = flat_df[flat_df['model'].isin(models_with_hyperparams)]
    
     # Group by likelihood, sample_size, dim, and model_method
    grouped = filtered_df.groupby(['likelihood', 'n_groups', 'group_size', 'model'])


    # Variance columns and their true values
    variance_cols = [col for col in filtered_df.columns if col.startswith("re_var_")]
    true_vals = [true_variances[f"re_var_{i+1}"] for i in range(len(variance_cols))]

    # Build MSE aggregation functions
    def make_mse_function(true_param: float):
        def mse(estimates: list[float]) -> float:
            est_array = np.stack(estimates)
            return np.mean((est_array - true_param) ** 2)
        return mse

    # Create aggregation dictionary with proper naming
    agg_dict = {
        f"mse_{col}": (col, make_mse_function(true_val))
        for col, true_val in zip(variance_cols, true_vals)
    }

    agg_dict.update({"mse_noise_variance": ("noise_variance", make_mse_function(pars["ald"]["scale"]))})
    # Apply aggregation using named syntax
    grouped_df = grouped.agg(**agg_dict).reset_index()

    return grouped_df, pars



def make_latex_table(config, summary_df, metrics, metric_criteria):
    """
    Make a latex table out of the summarized results.

    Input:
        - config: (dict)
        - summary_df: (pd.DataFrame)
        - metrics: (list)
        - metric_criteria: (dict)

    Output:
        - latex_table: (str)
    
    """
    
    alpha = config["alpha"]
    models = config["all_models"] # eventually qgam
    name_dict = config["name_dict"]
    noise_dict = config["likelihood_dict"]

    # Optional font size
    fontsize = r"\small"  # you can change to \footnotesize, \small, etc.

    # Column format
    n_models = len(models)
    n_metrics = len(metrics)
    col_format = "lll|" + "c" * (n_metrics * n_models)
    col_format = "|l|l|l" + "|Y" * (n_metrics * n_models) + "|"

    # Multi-column headers for metrics
    metric_headers = [fr"\multicolumn{{{n_models}}}{{c|}}{{\textbf{{\scriptsize {label}}}}}" for _, _, label in metrics]

    # Lower headers with the actual method names (one header per method)
    model_headers = []
    for _ in range(n_metrics):
        model_headers.extend([fr"\multicolumn{{1}}{{c}}{{\textbf{{\tiny {name_dict[model]}}}}}" for model in models])


    # Lower headers for fixed columns (likelihood, sample size, dim)
    lower_headers = [r"\scriptsize \textbf{Noise}", r"\scriptsize \textbf{M}", r"\scriptsize \textbf{n}"]
    lower_headers.extend(model_headers)

    # Build the LaTeX header
    header = fr"""
    \begin{{table}}[ht]
    \centering
    {fontsize}
    \begin{{tabularx}}{{1\textwidth}}{{{col_format}}}
    \toprule
    \multicolumn{{3}}{{c|}}{{}} & {' & '.join(metric_headers)} \\
    { ' & '.join(lower_headers) } \\
    \midrule
    """

    # Build table rows
    rows = []
    for _, group in summary_df.groupby(['likelihood', 'n_groups', 'group_size']):
        row = [
            fr"\scriptsize {noise_dict[group['likelihood'].iloc[0]]}", # map likelihood to chosen name
            fr"\scriptsize {group['n_groups'].iloc[0]}",
            fr"\scriptsize {group['group_size'].iloc[0]}"
        ]
        
        # For every metric, look for the best value and later make it bold --> use np.isclose
        best_metrics = {}
        for mean_col, std_col, label in metrics:
            if metric_criteria[label] == "min":
                idx = np.argmin(group[mean_col])
            elif metric_criteria[label] == "check_coverage":
                idx = np.argmin(np.abs(group[mean_col] - alpha))
            else:
                idx = np.argmax(group[mean_col])

            best_value = group[mean_col].iloc[idx]
            best_std = group[std_col].iloc[idx]
            best_metrics[label] = (best_value, best_std)

        # For every metric, loop through methods and add results
        for mean_col, std_col, label in metrics:
            for model in models:
                sub = group[group['model'] == model]
                if not sub.empty:
                    mean = sub[mean_col].values[0]
                    std = sub[std_col].values[0]

                    best_value, best_std = best_metrics[label]
                    if abs(mean - best_value) <= 2 * best_std:
                        row.append(rf"\textbf{{\scriptsize {mean:.2f} \n \tiny ({std:.2f})}}")
                    else:
                        row.append(rf"\scriptsize {mean:.2f} \n \tiny ({std:.2f})")
                else:
                    row.append("–")
        rows.append(" & ".join(row) + r" \\")
                    
    # Footer
    footer = r"""
    \bottomrule
    \end{tabularx}
    \caption{Comparison of models across likelihoods, sample sizes, and dimensions. Each cell shows mean (std).}
    \end{table}
    """

    # Combine all
    latex_table = header + "\n".join(rows) + footer
     
    return latex_table

def make_latex_table_real(config, summary_df, metrics, metric_criteria):
    """
    Make a latex table out of the summarized results.

    Input:
        - config: (dict)
        - summary_df: (pd.DataFrame)
        - metrics: (list)
        - metric_criteria: (dict)

    Output:
        - latex_table: (str)
    
    """
    

    models = config["all_models"] # also eventually Qgam
    name_dict = config["name_dict"]

    # Optional font size
    fontsize = r"\small"  # you can change to \footnotesize, \small, etc.

    # Column format
    n_models = len(models)
    n_metrics = len(metrics)
    col_format = "lll|" + "c" * (n_metrics * n_models)
    col_format = "|l" + "|Y" * (n_metrics * n_models) + "|"

    # Multi-column headers for metrics
    metric_headers = [fr"\multicolumn{{{n_models}}}{{c|}}{{\textbf{{\scriptsize {label}}}}}" for _, _, label in metrics]

    # Lower headers with the actual method names (one header per method)
    model_headers = []
    for _ in range(n_metrics):
        model_headers.extend([fr"\multicolumn{{1}}{{c}}{{\textbf{{\tiny {name_dict[model]}}}}}" for model in models])


    # Lower headers for fixed columns (likelihood, sample size, dim)
    lower_headers = [r"\textbf{dataset}"]
    lower_headers.extend(model_headers)

    # Build the LaTeX header
    header = fr"""
    \begin{{table}}[ht]
    \centering
    {fontsize}
    \begin{{tabularx}}{{1\textwidth}}{{{col_format}}}
    \toprule
    \multicolumn{{1}}{{c|}}{{}} & {' & '.join(metric_headers)} \\
    { ' & '.join(lower_headers) } \\
    \midrule
    """

    # Build table rows
    rows = []
    for _, group in summary_df.groupby(['dataset']):
        row = [
            f"{group['dataset'].iloc[0]}",
        ]
        
        # For every metric, look for the best value and later make it bold --> use np.isclose
        best_metrics = {}
        for mean_col, std_col, label in metrics:
            if metric_criteria[label] == "min":
                best_value = np.min(group[mean_col])
            else:
                best_value = np.max(group[mean_col])
            best_metrics[label] = best_value

        # For every metric, loop through methods and add results
        for mean_col, std_col, label in metrics:
            for model in models:
                sub = group[group['model'] == model]
                if not sub.empty:
                    mean = sub[mean_col].values[0]
                    std = sub[std_col].values[0]
                    if np.isclose(mean, best_metrics[label]):
                        row.append(f"\\textbf{{\\scriptsize {mean:.3f} ({std:.2f})}}")
                    else:
                        row.append(f"\\scriptsize {mean:.3f} ({std:.2f})")
                else:
                    row.append("–")
        rows.append(" & ".join(row) + r" \\")
                    
    # Footer
    footer = r"""
    \bottomrule
    \end{tabularx}
    \caption{Comparison of models across datasets. Each cell shows mean (std).}
    \end{table}
    """

    # Combine all
    latex_table = header + "\n".join(rows) + footer
     
    return latex_table


def make_latex_table_hyperparams(config, summary_df):
    """
    Make a LaTeX table for MSE of estimated GP hyperparameters (lengthscale, signal variance).

    Input:
        - config: (dict)
        - summary_df: (pd.DataFrame) output from df_mse_hyper()
    
    Output:
        - latex_table: (str)
    """
    
    models = config["models_with_hyperparams"]
    name_dict = config["name_dict"]
    noise_dict = config["likelihood_dict"]

    # Metrics to include

    metrics = [(f"mse_{col}", None, f"MSE of RE {col[-1]} Variance") for col in summary_df.columns if col.startswith("re_var_")]

    metric_criteria = {label: "min" for _, _, label in metrics}

    fontsize = r"\small"
    n_models = len(models)
    n_metrics = len(metrics)
    col_format = "|l|l|l" + "|Y" * (n_metrics * n_models) + "|"

    metric_headers = [fr"\multicolumn{{{n_models}}}{{c|}}{{\textbf{{\scriptsize {label}}}}}" for _, _, label in metrics]
    model_headers = []
    for _ in range(n_metrics):
        model_headers.extend([fr"\multicolumn{{1}}{{c}}{{\textbf{{\tiny {name_dict[model]}}}}}" for model in models])

    lower_headers = [r"\scriptsize \textbf{Noise}", r"\scriptsize \textbf{M}", r"\scriptsize \textbf{n}"] + model_headers

    header = fr"""
    \begin{{table}}[ht]
    \centering
    {fontsize}
    \begin{{tabularx}}{{1\textwidth}}{{{col_format}}}
    \toprule
    \multicolumn{{3}}{{c|}}{{}} & {' & '.join(metric_headers)} \\
    { ' & '.join(lower_headers) } \\
    \midrule
    """

    rows = []
    for _, group in summary_df.groupby(['likelihood', 'n_groups', 'group_size']):
        row = [
            fr"\scriptsize {noise_dict[group['likelihood'].iloc[0]]}",
            fr"\scriptsize {group['n_groups'].iloc[0]}",
            fr"\scriptsize {group['group_size'].iloc[0]}"
        ]

        best_metrics = {}
        for mean_col, _, label in metrics:
            col_vals = group[mean_col]
            best_metrics[label] = np.min(col_vals)

        for mean_col, _, label in metrics:
            for model in models:
                sub = group[group['model'] == model]
                if not sub.empty:
                    mean = sub[mean_col].values[0]
                    if np.isclose(mean, best_metrics[label]):
                        row.append(rf"\textbf{{\scriptsize {mean:.4f}}}")
                    else:
                        row.append(rf"\scriptsize {mean:.4f}")
                else:
                    row.append("–")
        rows.append(" & ".join(row) + r" \\")

    footer = r"""
    \bottomrule
    \end{tabularx}
    \caption{MSE of estimated hyperparameters across models and settings. Bold indicates best (lowest) MSE.}
    \end{table}
    """

    latex_table = header + "\n".join(rows) + footer
    return latex_table




def make_plots(df, metrics, configs, OUTPUT_DIR):
    """
    Make plots about results.

    - for each likelihood and dimension combination, show effect of sample size on the metrics (quantile loss, interval score & coverage)
    """

    # Example for plotting with sample_size on x-axis and color by method
    # sns.set(style="whitegrid") 
    OUTPUT_DIR = Path(os.path.join(OUTPUT_DIR, "images"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_groups = df["n_groups"].unique()
    likelihoods = df["likelihood"].unique()

    alpha = configs["alpha"]

    for likelihood in likelihoods:
        for n_g in n_groups:

            df_filtered = df[df["n_groups"] == n_g]
            df_filtered = df_filtered[df_filtered["likelihood"] == likelihood]
            # Assuming 'model_method' is the column representing different methods
            
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_filtered, x="group_size", y=f"{metric}", hue="model", palette="Set2",showmeans=True)
                if metric == "coverage":
                    plt.axhline(y=alpha, color="red", linestyle="--", linewidth=1, label = f"Nominal Coverage: {alpha}")
                elif metric == "time":
                      plt.yscale("log")
                
                # Add titles and labels
                plt.title(f"{metric.replace("_", " ").title()} by group size. Likelihood: {likelihood}. Num. Groups: {n_g}", fontsize=16, c = "black")
                plt.xlabel("Group Size", fontsize=12)
                plt.ylabel(f"{metric.replace("_", " ").title()}", fontsize=12)
                plt.legend(title="Model", title_fontsize="13", fontsize="11", labelcolor = "black")
                plt.tight_layout()
                # Save plots
                filename = f"{metric}_{likelihood}_{n_g}"
                for ext in ["png", "pdf"]:
                    plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)

                plt.close()
                
def make_plots_hypers(df, configs, metrics, pars, OUTPUT_DIR):
    """
    Make plots about results.

    - for each likelihood and dimension combination, show effect of sample size on the metrics (quantile loss, interval score & coverage)
    """

    # Example for plotting with sample_size on x-axis and color by method
    # sns.set(style="whitegrid")  

    print("here")
    OUTPUT_DIR = Path(os.path.join(OUTPUT_DIR, "images"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    randeff = configs["randeff"]
    models_with_hyperparams = configs["models_with_hyperparams"]
    signal_variance = configs["data_generation"]["signal_variance"]
    true_variances = {"re_var_1": signal_variance}

    crossed = False
    if randeff == "Two_completely_crossed_random_effects":
        crossed = True
    if randeff != "One_random_effect":
        signal_variance_2 = configs["data_generation"]["signal_variance_2"]
        n_groups_2 = configs["data_generation"]["n_groups_2"]
        true_variances.update({"re_var_2": signal_variance_2})

    noise_variance = pars["ald"]["scale"]
    true_variances.update({"noise_variance": noise_variance})
    
    # keep only if model has estimated the hyper-parameters
    df = df[df['model'].isin(models_with_hyperparams)]
   
    n_groups = df["n_groups"].unique()
    likelihoods = df["likelihood"].unique()

    for likelihood in likelihoods:
        if likelihood == "ald":
            use_metrics = metrics
        else:
            use_metrics = [metric for metric in metrics if metric != "noise_variance"]
        
        for n_g in n_groups:
            df_filtered = df[df["n_groups"] == n_g]
            df_filtered = df_filtered[df_filtered["likelihood"] == likelihood]
            # Assuming 'model_method' is the column representing different methods
            print(f"{likelihood}_{n_g}")
            for metric in use_metrics:
                print(f"producing plot for metric: {metric}")
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_filtered, x="group_size", y=metric, hue="model", palette="Set2", showmeans=True)
                # Add titles and labels
                plt.title(f"{metric.replace("_", " ").title()} by Group Size. Likelihood: {likelihood}. Num. Groups: {n_g if not crossed else (int(n_g), n_groups_2) }", fontsize=16, c = "black")
                plt.axhline(y=true_variances[metric], color="red", linestyle="--", linewidth=1, label = f"True {metric}") #signal_variance
                plt.xlabel("Group Size", fontsize=12)
                plt.ylabel(f"{metric.replace("_", " ").title()}", fontsize=12)
                plt.legend(title="Model", title_fontsize="13", fontsize="11", labelcolor = "black")
                plt.tight_layout()
                # Save plots
                filename = f"{metric}_{likelihood}_{n_g}"
                for ext in ["png", "pdf"]:
                    plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)

                plt.close()
                


    
def make_plots_real(df, metrics):
    """
    Make plots about results.

    - for each dataset, show effect of sample size on the metrics (quantile loss & time)
    """

    # Example for plotting with sample_size on x-axis and color by method
    # sns.set(style="whitegrid")  

    OUTPUT_DIR = Path("results/real_data/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="dataset", y=f"{metric}", hue="model", palette="Set2", showmeans=True)
        # Add titles and labels
        if metric == "time":
            plt.yscale("log")
        plt.title(f"{metric} by Dataset.", fontsize=16, c = "black")
        plt.xlabel("Dataset", fontsize=12)
        plt.ylabel(f"{metric.replace("_", " ").title()}", fontsize=12)
        plt.legend(title="Model", title_fontsize="13", fontsize="11", labelcolor = "black")
        plt.tight_layout()
        # Save plots
        filename = f"{metric}"
        for ext in ["png", "pdf"]:
            plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)

        plt.close()
            




