from lagp.utils.analyse_results import make_flattened_df_real, make_latex_table_real, make_plots_real, group_flattened_df_real
import argparse
import os
import pickle
import yaml



def run_analysis(RESULT_PATH, CONFIGS_PATH, OUTPUT_DIR):
    """
    Run analyis: make plots and table of results.
    
    """
    with open(RESULT_PATH, "rb") as f:
        res = pickle.load(f)
    
    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f) 

    metrics = [
        ("quantile_loss_mean", "quantile_loss_std", "Quantile Score"),  # 3rd element is the name/label
        ("time_mean", "time_std", "Time")
    ]
    # Define metrics and optimality
    metric_criteria = {"Quantile Score": "min",  "Time": "min"}

    # prepare the data
    df = make_flattened_df_real(results = res)
    summary_df = group_flattened_df_real(flat_df=df)
    
    # make latex table
    latex_table = make_latex_table_real(config=configs, summary_df=summary_df, metrics = metrics, metric_criteria= metric_criteria)

    # make and save plots
    metrics_for_plotting = ["quantile_loss", "time"]
    make_plots_real(df = df, metrics = metrics_for_plotting, OUTPUT_DIR = OUTPUT_DIR)
   
    return latex_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_name", type = str,
                        default = "real_data_results.pkl",
                        help="Name of the results.pkl file")
    
    parser.add_argument("--configs", type = str,
                        default = "config_mm_real.yaml",
                        help="Name of the configs file")
    
    parser.add_argument("--version", type = str,
                        default = "001",
                        help="Version number for the run (default: 001)")
    
    args = parser.parse_args()
    
    OUTPUT_DIR = f"results/real_data_mm/{args.version}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULT_PATH = os.path.join(OUTPUT_DIR, args.results_name)
    CONFIGS_PATH = os.path.join("configs", args.configs)
   
    latex_table = run_analysis(RESULT_PATH, CONFIGS_PATH, OUTPUT_DIR)
    # save it
    OUTPUT_FILE = f"results/real_data_mm/{args.version}/table_results.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(latex_table)


