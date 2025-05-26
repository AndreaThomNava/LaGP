### merge R and Python results 

import argparse
import pickle
import os

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
                        if model_name not in merged_results["results"][config_key][replicate]:
                            existing = merged_results["results"][config_key][replicate].get(model_name, {})
                            merged_results["results"][config_key][replicate][model_name] = {**existing, **metrics}
                        else:
                            # Handle if same model already exists (merge or overwrite)
                            merged_results["results"][config_key][replicate][model_name].update(metrics)

    return merged_results



def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Merge results from two pickle files.")
    parser.add_argument('pickle_file_1', nargs='?',
                        default = "simulation_mm/One_random_effect/simulation_results_python.pkl",
                        type=str, help="Path to the first pickle file.")
    parser.add_argument('pickle_file_2', nargs='?',
                        default = "simulation_mm/One_random_effect/simulation_results_R.pkl",
                        type=str, help="Path to the second pickle file.")
    parser.add_argument('output_file', nargs='?',
                        default = "simulation_mm/One_random_effect/simulation_results.pkl",
                        type=str, help="Path to save the merged results.")
    
    args = parser.parse_args()

    RESULTS_PATH = "results"
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
   
    OUTPUT_PATH = os.path.join(RESULTS_PATH, args.output_file)
    # Save the merged results to the output file
    with open(OUTPUT_PATH, "wb") as output_f:
        pickle.dump(merged_results, output_f)

    print(f"Results merged and saved to {args.output_file}")

if __name__ == "__main__":
    main()