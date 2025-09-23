import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path


def replace_missing_with_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_missing_with_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_missing_with_nan(item) for item in obj]
    elif obj is None:
        return np.nan
    else:
        return obj


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
        empitrical_quantile=('empirical_quantile', 'mean'),
        empirical_quantile_std=('empirical_quantile', 'std'),
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

    def sem(x):
        return x.std() / np.sqrt(len(x))

    # Assuming 'coverage' and 'width' are columns in your DataFrame
    grouped_df = grouped.agg(
        quantile_loss_mean=('quantile_loss', 'mean'),
        quantile_loss_std=('quantile_loss', sem),
        true_pinball_loss_mean =('true_pinball_loss', 'mean'),
        true_pinball_loss_std=('true_pinball_loss', sem),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', sem),
        bias_mean = ("bias", "mean"),
        bias_std = ("bias", sem),
        interval_loss_mean=('interval_loss', 'mean'),
        interval_loss_std=('interval_loss', sem),
        coverage_mean=('coverage', 'mean'),
        coverage_std=('coverage', sem),
        train_coverage_mean=('train_coverage', 'mean'),
        train_coverage_std=('train_coverage', sem),
        width_mean=('width', 'mean'),
        width_std=('width', sem),
        time_mean=("time", "mean" ),
        time_std = ("time", sem)
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


def make_plots(df, metrics, configs, version):
    """
    Make plots about results.

    - for each likelihood and dimension combination, show effect of sample size on the metrics (quantile loss, interval score & coverage)
    """

    # Set up plotting style and output directory
    plt.style.use('seaborn-v0_8-whitegrid')
    OUTPUT_DIR = Path(f"results/simulation/{version}/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get unique values for iteration
    dims = df["dim"].unique()
    likelihoods = df["likelihood"].unique()

    # Extract configuration parameters
    alpha = configs["alpha"]
    target_quantile = configs["target_quantile"]
    model_mapping = configs.get("name_dict", None)

    # Professional color palettes
    palette = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#D55E00', '#56B4E9']
    EXTENDED_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    for likelihood in likelihoods:
        for dim in dims:
            # Filter data for current likelihood and dimension
            df_filtered = df[df["dim"] == dim]
            df_filtered = df_filtered[df_filtered["likelihood"] == likelihood]
            
            # Map model names if mapping exists
            if model_mapping:
                df_filtered = df_filtered.copy()
                df_filtered["model_display"] = df_filtered["model"].map(
                    lambda x: model_mapping.get(x, x)
                )
                hue_col = "model_display"
            else:
                hue_col = "model"
            
            for metric in metrics:
                # Create figure with professional styling
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Create boxplot with enhanced styling
                sns.boxplot(data=df_filtered, x="sample_size", y=f"{metric}", 
                           hue=hue_col, palette=EXTENDED_PALETTE,
                           showmeans=True, ax=ax,
                           meanprops={
                               'marker': 'D',           # Diamond shape
                               'markerfacecolor': 'white',
                               'markeredgecolor': 'black',
                               'markersize': 6,
                               'markeredgewidth': 2.5
                           })
                
                # Add reference lines based on metric type
                if metric == "coverage" or metric == "train_coverage":
                    ax.axhline(y=alpha, color="red", linestyle="--", linewidth=1, 
                              label=f"Nominal Coverage: {alpha}")
                elif metric == "time":
                    ax.set_yscale("log")
                elif metric == "empirical_quantile":
                    ax.axhline(y=target_quantile, color="red", linestyle="--", linewidth=1,
                              label=f"Target Quantile: {target_quantile}")
                
                # Enhanced title and labels with professional formatting
                ax.set_title(f"{metric.replace('_', ' ').title()} by Sample Size\n"
                            f"Likelihood: {likelihood}, Dimension: {dim}", 
                            fontsize=20, fontweight='bold', pad=20)
                ax.set_xlabel("Sample Size", fontsize=18, fontweight='bold')
                ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='bold')
                
                # Increase tick label sizes
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.tick_params(axis='both', which='minor', labelsize=14)
                
                # Position legend below plot with professional styling
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend(handles, labels,
                                  title="Model",
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
                
                # Save plots in multiple formats
                filename = f"{metric}_{likelihood}_{dim}"
                for ext in ["png", "pdf"]:
                    plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", 
                               bbox_inches="tight", dpi=300)
                plt.close()




def make_plots_hypers(df, configs, metrics, version):
    """
    Make plots about results.

    - for each likelihood and dimension combination, show effect of sample size on the metrics (quantile loss, interval score & coverage)
    """

    # Set up plotting style and output directory
    plt.style.use('seaborn-v0_8-whitegrid')
    OUTPUT_DIR = Path(f"results/simulation/{version}/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration parameters
    models_with_hyperparams = configs["models_with_hyperparams"]
    signal_variance = configs["gp_parameters"]["kernel"]["signal_variance"]
    lengthscale = configs["gp_parameters"]["kernel"]["lengthscale"]
    model_mapping = configs.get("name_dict", None)
    
    true_pars = {"signal_variance": signal_variance,
                 "lengthscale": lengthscale}
    
    # Keep only models that have estimated hyperparameters
    df = df[df['model'].isin(models_with_hyperparams)]
   
    # Get unique values for iteration
    dims = df["dim"].unique()
    likelihoods = df["likelihood"].unique()

    # Professional color palettes
    palette = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#D55E00', '#56B4E9']
    EXTENDED_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    for likelihood in likelihoods:
        for dim in dims:
            # Filter data for current likelihood and dimension
            df_filtered = df[df["dim"] == dim]
            df_filtered = df_filtered[df_filtered["likelihood"] == likelihood]
            
            # Map model names if mapping exists
            if model_mapping:
                df_filtered = df_filtered.copy()
                df_filtered["model_display"] = df_filtered["model"].map(
                    lambda x: model_mapping.get(x, x)
                )
                hue_col = "model_display"
            else:
                hue_col = "model"

            # Calculate effective range for current dimension
            effective_range = lengthscale * np.sqrt(dim/2) if dim > 2 else lengthscale
            true_pars["lengthscale"] = effective_range
            
            for metric in metrics:
                # Create figure with professional styling
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Create boxplot with enhanced styling
                sns.boxplot(data=df_filtered, x="sample_size", y=metric, 
                           hue=hue_col, palette=EXTENDED_PALETTE,
                           showmeans=True, ax=ax,
                           meanprops={
                               'marker': 'D',           # Diamond shape
                               'markerfacecolor': 'white',
                               'markeredgecolor': 'black',
                               'markersize': 6,
                               'markeredgewidth': 2.5
                           })
                
                # Add reference line for true parameter value
                ax.axhline(y=true_pars[metric], color="red", linestyle="--", 
                          linewidth=2, label=f"True {metric.replace('_', ' ').title()}")
                
                # Enhanced title and labels with professional formatting
                ax.set_title(f"{metric.replace('_', ' ').title()} by Sample Size\n"
                            f"Likelihood: {likelihood}, Dimension: {dim}", 
                            fontsize=20, fontweight='bold', pad=20)
                ax.set_xlabel("Sample Size", fontsize=18, fontweight='bold')
                ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='bold')
                
                # Increase tick label sizes
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.tick_params(axis='both', which='minor', labelsize=14)
                
                # Position legend below plot with professional styling
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend(handles, labels,
                                  title="Model",
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
                
                # Save plots in multiple formats
                filename = f"{metric}_{likelihood}_{dim}"
                for ext in ["png", "pdf"]:
                    plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", 
                               bbox_inches="tight", dpi=300)
                plt.close()


    
def make_plots_real(df, metrics, version, configs=None):
    """
    Make plots about results.

    - for each dataset, show effect of sample size on the metrics (quantile loss & time)
    """

    # Set up plotting style and output directory
    plt.style.use('seaborn-v0_8-whitegrid')
    OUTPUT_DIR = Path(f"results/real_data/{version}/images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract configuration parameters if available
    model_mapping = configs.get("name_dict", None) if configs else None
    models_to_plot = configs.get("all_models", None) if configs else None
    if models_to_plot:
        df = df[df['model'].isin(models_to_plot)]
    # Professional color palettes
    palette = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#D55E00', '#56B4E9']
    EXTENDED_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # Map model names if mapping exists
    df_plot = df.copy()
    print(df_plot.columns)
    if model_mapping:
        df_plot["model_display"] = df_plot["model"].map(
            lambda x: model_mapping.get(x, x)
        )
        hue_col = "model_display"
    else:
        hue_col = "model"
    print(df_plot["model_display"].unique())
    for metric in metrics:
        # Create figure with professional styling
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create boxplot with enhanced styling
        sns.boxplot(data=df_plot, x="dataset", y=f"{metric}", 
                   hue=hue_col, palette=EXTENDED_PALETTE,
                   showmeans=True, ax=ax,
                   meanprops={
                       'marker': 'D',           # Diamond shape
                       'markerfacecolor': 'white',
                       'markeredgecolor': 'black',
                       'markersize': 6,
                       'markeredgewidth': 2.5
                   })
        
        # Set log scale for specific metrics
        if metric in ["time", "quantile_loss"]:
            ax.set_yscale("log")
        
        # Enhanced title and labels with professional formatting
        ax.set_title(f"{metric.replace('_', ' ').title()} by Dataset", 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Dataset", fontsize=18, fontweight='bold')
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=18, fontweight='bold')
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        
        # Rotate x-axis labels if there are many datasets
        if len(df_plot["dataset"].unique()) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Position legend below plot with professional styling
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels,
                          title="Model",
                          loc='upper center',
                          bbox_to_anchor=(0.5, -0.12),
                          ncol=min(len(labels), 4),  # Max 4 columns
                          fontsize=16,
                          title_fontsize=18,
                          frameon=True,
                          fancybox=True,
                          shadow=True)
        legend.get_title().set_fontweight('bold')
        
        # Adjust layout for legend and rotated labels
        plt.tight_layout()
        if len(df_plot["dataset"].unique()) > 4:
            plt.subplots_adjust(bottom=0.25)  # More space for rotated labels
        else:
            plt.subplots_adjust(bottom=0.2)
        
        # Save plots in multiple formats
        filename = f"{metric}"
        for ext in ["png", "pdf"]:
            plt.savefig(OUTPUT_DIR / f"{filename}.{ext}", 
                       bbox_inches="tight", dpi=300)
        plt.close()

