from lagp.utils.analyse_results import (
    make_flattened_df, make_latex_table, group_flattened_df, make_plots, df_mse_hyper,
      make_latex_table_hyperparams, make_plots_hypers)
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
        #("interval_loss_mean", "interval_loss_std", "Interval Score"),
        ("coverage_mean", "coverage_std", "Coverage"),
        ("time_mean", "time_std", "Time")
    ]
    # Define metrics and optimality
    metric_criteria = {"Quantile Score": "min", "Interval Score": "min", "Coverage": "check_coverage", "Time": "min"}


    df = make_flattened_df(results = res)
    summary_df = group_flattened_df(flat_df=df)
    
    # compute MSE on parameter estimation
    df_mse, pars = df_mse_hyper(flat_df = df, configs = configs)
    print(df_mse)
    # make latex table and plots
    hyper_latex_table = make_latex_table_hyperparams(config=configs, summary_df=df_mse)
    latex_table = make_latex_table(config=configs, summary_df=summary_df, metrics = metrics, metric_criteria= metric_criteria)

    # make and save plots
    metrics_for_plotting = ["quantile_loss","time"] # "interval_loss", "coverage", "width", "time"] # df.columns[5:]
    make_plots(df = df, metrics = metrics_for_plotting, configs=configs, OUTPUT_DIR=OUTPUT_DIR)
    metric_for_plotting_hypers = [col for col in df.columns if col.startswith("re_var_")]
    metric_for_plotting_hypers.append("noise_variance")
    make_plots_hypers(df = df, configs=configs, metrics=metric_for_plotting_hypers, pars = pars, OUTPUT_DIR=OUTPUT_DIR) # , "lengthscale"
   
    return latex_table, hyper_latex_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_name", type = str,
                        default = "simulation_results_python.pkl",
                        help="Name of the results.pkl file")
    parser.add_argument("--configs", type = str,
                        default = "configs/config_test.yaml",
                        help="Name of the configs file")
    
    parser.add_argument("--version", type = str,
                        default = "001",
                        help="Versioning of the results")
    
    args = parser.parse_args()
    CONFIGS_PATH = args.configs
    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f)

    randeff = configs["randeff"]
    RESULT_PATH = os.path.join("results/simulation_mm", f"{randeff}",args.version, args.results_name)
    OUTPUT_DIR = f"results/simulation_mm/{randeff}/{args.version}"
    
    latex_table, hyper_latex_table= run_analysis(RESULT_PATH, CONFIGS_PATH, OUTPUT_DIR) # , hyper_latex_table 
    # save it
   
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "table_results.tex")
    with open(OUTPUT_FILE, "w") as f:
        f.write(latex_table)
    OUTPUT_FILE_MSE = os.path.join(OUTPUT_DIR, "table_results_hyper.tex")
    with open(OUTPUT_FILE_MSE, "w") as f:
        f.write(hyper_latex_table)


