### merge R and Python results

import argparse
import os
import pickle

import yaml

from lagp.utils.analyse_results import replace_missing_with_nan


def merge_results(pickle_results_1, pickle_results_2):
    """Merge results from Python (gpboost) and R (lqmm, brsm)."""
    merged_results = pickle_results_1.copy()

    for config_key, replicate_data in pickle_results_2["results"].items():
        if config_key not in merged_results["results"]:
            merged_results["results"][config_key] = replicate_data
        else:
            # Merge replicate results
            for replicate, model_data in replicate_data.items():
                if replicate not in merged_results["results"][config_key]:
                    merged_results["results"][config_key][replicate] = model_data
                else:
                    for model_name, metrics in model_data.items():
                        if (
                            model_name
                            not in merged_results["results"][config_key][replicate]
                        ):
                            existing = merged_results["results"][config_key][
                                replicate
                            ].get(model_name, {})
                            merged_results["results"][config_key][replicate][
                                model_name
                            ] = {**existing, **metrics}
                        else:
                            # Handle if same model already exists (merge or overwrite)
                            merged_results["results"][config_key][replicate][
                                model_name
                            ].update(metrics)

    return merged_results


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Merge results from two pickle files.")
    parser.add_argument(
        "pickle_file_1",
        nargs="?",
        default="real_data_results_python.pkl",
        type=str,
        help="Path to the first pickle file.",
    )
    parser.add_argument(
        "pickle_file_2",
        nargs="?",
        default="real_data_results_R.pkl",
        type=str,
        help="Path to the second pickle file.",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="real_data_results.pkl",
        type=str,
        help="Path to save the merged results.",
    )

    parser.add_argument(
        "configs",
        nargs="?",
        default="configs/config_mm_real.yaml",
        type=str,
        help="Path to experiment config.",
    )

    parser.add_argument(
        "--version", nargs="?", default="001", type=str, help="version number."
    )

    args = parser.parse_args()

    CONFIGS_PATH = args.configs
    with open(CONFIGS_PATH, "r") as f:
        configs = yaml.safe_load(f)

    RESULTS_PATH = os.path.join("results", "real_data_mm", args.version)
    # Load the pickle files
    with open(os.path.join(RESULTS_PATH, args.pickle_file_1), "rb") as f1:
        res_python = pickle.load(f1)

    with open(os.path.join(RESULTS_PATH, args.pickle_file_2), "rb") as f2:
        res_R = pickle.load(f2)

    for config_key, replicate_list in res_R["results"].items():
        res_R["results"][config_key] = {
            int(idx + 1): replicate for idx, replicate in enumerate(replicate_list)
        }

    # Merge the results
    merged_results = merge_results(res_python, res_R)

    ## replace NAs from failed runs with np.nan
    clean_results = replace_missing_with_nan(merged_results)

    OUTPUT_PATH = os.path.join(RESULTS_PATH, args.output_file)
    # Save the merged results to the output file
    with open(OUTPUT_PATH, "wb") as output_f:
        pickle.dump(clean_results, output_f)

    print(f"Results merged and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
