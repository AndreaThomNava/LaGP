import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path


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
            "sample_size": int(cfg.split("_")[-2]),
            "dim": int(cfg.split("_")[-1]),
            "replicate": replicate,
            **metrics,
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
    grouped = flat_df.groupby(['likelihood', 'sample_size', 'dim', 'model'])

    # Assuming 'coverage' and 'width' are columns in your DataFrame
    grouped_df = grouped.agg(
        quantile_loss_mean=('quantile_loss', 'mean'),
        quantile_loss_std=('quantile_loss', 'std'),
        interval_loss_mean=('interval_loss', 'mean'),
        interval_loss_std=('interval_loss', 'std'),
        coverage_mean=('coverage', 'mean'),
        coverage_std=('coverage', 'std'),
        width_mean=('width', 'mean'),
        width_std=('width', 'std'),
        time_mean=("time", "mean" ),
        time_std = ("time", "std")
    ).reset_index()

    return grouped_df


def df_mse_hyper(flat_df, configs):
   
    models_with_hyperparams = configs["models_with_hyperparams"]
    lengthscale = configs["gp_parameters"]["kernel"]["lengthscale"]
    signal_variance = configs["gp_parameters"]["kernel"]["signal_variance"]
    # noise_variance = configs["gp_parameters"]["kernel"]["legthscale"]
    
    # keep only if model has estimated the hyper-parameters
    filtered_df = flat_df[flat_df['model'].isin(models_with_hyperparams)]
    
     # Group by likelihood, sample_size, dim, and model_method
    grouped = filtered_df.groupby(['likelihood', 'sample_size', 'dim', 'model'])


    def make_mse_function(true_param: np.ndarray):
        def mse(estimates: list[np.ndarray]) -> float:
            est_array = np.stack(estimates)
            return np.mean((est_array - true_param) ** 2)
        return mse
    
    mse_lengthscale = make_mse_function(lengthscale)
    mse_signal = make_mse_function(signal_variance)
    
    grouped_df = grouped.agg(
       mse_lengthscale =  ("lengthscale", mse_lengthscale),
       mse_signal_variance = ("signal_variance", mse_signal)
    ).reset_index()

    return grouped_df



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
    lower_headers = [r"\scriptsize \textbf{Noise}", r"\scriptsize \textbf{N}", r"\scriptsize \textbf{d}"]
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
    for _, group in summary_df.groupby(['likelihood', 'sample_size', 'dim']):
        row = [
            fr"\scriptsize {noise_dict[group['likelihood'].iloc[0]]}", # map likelihood to chosen name
            fr"\scriptsize {group['sample_size'].iloc[0]}",
            fr"\scriptsize {group['dim'].iloc[0]}"
        ]
        
        # For every metric, look for the best value and later make it bold --> use np.isclose
        best_metrics = {}
        for mean_col, std_col, label in metrics:
            if metric_criteria[label] == "min":
                best_value = np.min(group[mean_col])
            elif metric_criteria[label] == "check_coverage":
                closest_idx = np.argmin(np.abs(group[mean_col] - alpha))
                best_value = group[mean_col].iloc[closest_idx]
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
                    if mean == best_metrics[label]:
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
    metrics = [
        ("mse_lengthscale", None, "Lengthscale MSE"),
        ("mse_signal_variance", None, "Signal Variance MSE"),
    ]
    metric_criteria = {label: "min" for _, _, label in metrics}

    fontsize = r"\small"
    n_models = len(models)
    n_metrics = len(metrics)
    col_format = "|l|l|l" + "|Y" * (n_metrics * n_models) + "|"

    metric_headers = [fr"\multicolumn{{{n_models}}}{{c|}}{{\textbf{{\scriptsize {label}}}}}" for _, _, label in metrics]
    model_headers = []
    for _ in range(n_metrics):
        model_headers.extend([fr"\multicolumn{{1}}{{c}}{{\textbf{{\tiny {name_dict[model]}}}}}" for model in models])

    lower_headers = [r"\scriptsize \textbf{Noise}", r"\scriptsize \textbf{N}", r"\scriptsize \textbf{d}"] + model_headers

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
    for _, group in summary_df.groupby(['likelihood', 'sample_size', 'dim']):
        row = [
            fr"\scriptsize {noise_dict[group['likelihood'].iloc[0]]}",
            fr"\scriptsize {group['sample_size'].iloc[0]}",
            fr"\scriptsize {group['dim'].iloc[0]}"
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




def make_plots(df, metrics):
    """
    Make plots about results.

    - for each likelihood and dimension combination, show effect of sample size on the metrics (quantile loss, interval score & coverage)
    """

    # Example for plotting with sample_size on x-axis and color by method
    # sns.set(style="whitegrid")  

    OUTPUT_DIR = Path("results/simulation/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dims = df["dim"].unique()
    likelihoods = df["likelihood"].unique()

    for likelihood in likelihoods:
        for dim in dims:

            df_filtered = df[df["dim"] == dim]
            df_filtered = df[df["likelihood"] == likelihood]
            # Assuming 'model_method' is the column representing different methods
            
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_filtered, x="sample_size", y=f"{metric}", hue="model", palette="Set2")
                # Add titles and labels
                plt.title(f"{metric} by Sample Size. Likelihood: {likelihood}. Dim: {dim}", fontsize=16, c = "black")
                plt.xlabel("Sample Size", fontsize=12)
                plt.ylabel(f"{metric.replace("_", " ").title()}", fontsize=12)
                plt.legend(title="Model", title_fontsize="13", fontsize="11", labelcolor = "black")
                plt.tight_layout()
                # Save plots
                filename = f"{metric}_{likelihood}_{dim}"
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

    OUTPUT_DIR = Path("results/realdata/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="dataset", y=f"{metric}", hue="model", palette="Set2")
        # Add titles and labels
        plt.title(f"{metric} by Dataset.", fontsize=16, c = "black")
        plt.xlabel("Dataset", fontsize=12)
        plt.ylabel(f"{metric.replace("_", " ").title()}", fontsize=12)
        plt.legend(title="Model", title_fontsize="13", fontsize="11", labelcolor = "black")
        plt.tight_layout()
        # Save plots
        filename = f"{metric}_{metric}"
        for ext in ["png", "pdf"]:
            plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", bbox_inches="tight", dpi=300)

        plt.close()
            




