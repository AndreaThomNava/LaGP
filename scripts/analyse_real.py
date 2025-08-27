from lagp.utils.analyse_results import make_flattened_df_real, make_latex_table_real, make_plots_real, group_flattened_df_real
import argparse
import os
import pickle
import yaml



def run_analysis(RESULT_PATH, CONFIGS_PATH, version):
    """
    Run analyis: make plots and table of results.
    
    """
    with open(RESULT_PATH, "rb") as f:
        res = pickle.load(f)
    
    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f) 

    metrics = [
        ("quantile_loss_mean", "quantile_loss_std", "Quantile Score"),  # 3rd element is the name/label
        ("time_mean", "time_std", "Time"),
    ]
    # Define metrics and optimality
    metric_criteria = {"Quantile Score": "min",  "Time": "min"}


    df = make_flattened_df_real(results = res)
    summary_df = group_flattened_df_real(flat_df=df)
    
    latex_table = make_latex_table_real(config=configs, summary_df=summary_df, metrics = metrics, metric_criteria= metric_criteria)

    # make and save plots
    metrics_for_plotting = ["quantile_loss", "empirical_quantile", "time"]
    make_plots_real(df = df, metrics = metrics_for_plotting, version = version)
   
    return latex_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_name", type = str,
                        default = "real_data_results.pkl",
                        help="Name of the results.pkl file")
    parser.add_argument("--configs", type = str,
                        default = "config_run_real.yaml",
                        help="Name of the configs file")
    parser.add_argument("--version", type = str,
                        default = "001",
                        help="version of the experiment to analyze")
    
    args = parser.parse_args()
    version = args.version
    RESULT_PATH = os.path.join(f"results/real_data/{version}", args.results_name)
    print(RESULT_PATH)
    CONFIGS_PATH = os.path.join("configs", args.configs)
   
    latex_table = run_analysis(RESULT_PATH, CONFIGS_PATH, version)
    # save it
    OUTPUT_FILE = f"results/real_data/{version}/table_results.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(latex_table)


