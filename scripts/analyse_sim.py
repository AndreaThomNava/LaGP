from lagp.utils.analyse_results import make_flattened_df, make_latex_table, group_flattened_df, make_plots
import argparse
import os
import pickle
import yaml



def run_analysis(RESULT_PATH, CONFIGS_PATH):
    """
    Run analyis: make plots and table of results.
    
    """
    with open(RESULT_PATH, "rb") as f:
        res = pickle.load(f)
    
    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f) 

    metrics = [
        ("quantile_loss_mean", "quantile_loss_std", "Quantile Score"),  # 3rd element is the name/label
        ("interval_loss_mean", "interval_loss_std", "Interval Score"),
        ("coverage_mean", "coverage_std", "Coverage")
    ]
    # Define metrics and optimality
    metric_criteria = {"Quantile Score": "min", "Interval Score": "min", "Coverage": "check_coverage"}


    df = make_flattened_df(results = res)
    summary_df = group_flattened_df(flat_df=df)
    
    latex_table = make_latex_table(config=configs, summary_df=summary_df, metrics = metrics, metric_criteria= metric_criteria)

    # make and save plots
    metrics_for_plotting = df.columns[5:]
    make_plots(df = df, metrics = metrics_for_plotting)
   
    return latex_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_name", type = str,
                        default = "simulation_results.pkl",
                        help="Name of the results.pkl file")
    parser.add_argument("--configs", type = str,
                        default = "config_results_simulation.yaml",
                        help="Name of the configs file")
    
    args = parser.parse_args()
    print(args)
    RESULT_PATH = os.path.join("results/simulation", args.results_name)
    CONFIGS_PATH = os.path.join("configs", args.configs)
   
    latex_table = run_analysis(RESULT_PATH, CONFIGS_PATH)
    # save it
    OUTPUT_FILE = "results/simulation/table_results.tex"
    with open(OUTPUT_FILE, "w") as f:
        f.write(latex_table)


