import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from lagp.utils.generate_data import make_latents, simulate_response, train_test_split_mixed_data, compute_dict_pars

def gen_mixed_data_from_config(config_file: str):
    # Base directories (modify as needed)
    CONFIGS_DIR =  "configs"
    # Load config YAML
    with open(os.path.join(CONFIGS_DIR, config_file), "r") as f:
        config = yaml.safe_load(f)

    data_gen_conf = config["data_generation"]

    randeff = data_gen_conf["randeff"]
    num_replicates = data_gen_conf["replicate"]
    test_size = data_gen_conf["test_size"]
    OUTPUT_FOLDER = data_gen_conf["output_folder"]
    OUTPUT_FOLDER = os.path.join("data",  OUTPUT_FOLDER )


    # Generate noise parameters if fixed_snr is True
    if data_gen_conf.get("fixed_snr", True):
        pars = compute_dict_pars(
            signal_variance=data_gen_conf["signal_variance"],
            snr=data_gen_conf["snr"],
            quantile=data_gen_conf["quantile"],
        )
        # Merge noise pars into data generation params
        data_gen_conf["pars"].update(pars)
    else:
        pars = data_gen_conf.get("pars", {})

    # Seed RNG for reproducibility
    np.random.seed(data_gen_conf.get("random_state", 42))

    # Loop over sample sizes, input dimensions, likelihoods
    for n_group in data_gen_conf["n_groups"]:
        for group_size in data_gen_conf["group_size"]:
            print(f"Simulating for m={n_group} groups, each of size={group_size}")
            
            latent_replicates = []

            for replicate in range(num_replicates):
                print(f"  Latent replicate {replicate+1}/{num_replicates}")
                
                # Generate fixed latent structure
                X, fe, eps, group_data = make_latents(n_groups=n_group,
                                                      group_size=group_size,
                                                      pars=data_gen_conf)
                latent_replicates.append((X, fe, eps, group_data))


            for likelihood in data_gen_conf["likelihood"]:

                for replicate, (X, fe, eps, group_data) in enumerate(latent_replicates):
                    print(f"  Simulating replicate {replicate+1}/{num_replicates}")

                    result = simulate_response(
                        pars=data_gen_conf,
                        X=X,
                        fe=fe,
                        eps=eps,
                        group_data=group_data,
                        likelihood=likelihood
                    )

                    train_test_split_mixed_data(
                        X=result["X"],
                        y=result["y"],
                        group_data=result["group_data"],
                        fe=result["fe"],
                        eps=result["eps"],
                        n_groups=n_group,
                        group_size=group_size,
                        likelihood=likelihood,
                        randeff=randeff,
                        replicate=replicate + 1, #### starting at index 1!!
                        test_size=test_size,
                        extrapolate=data_gen_conf.get("extrapolate", False),
                        output_folder=str(OUTPUT_FOLDER)
                    )


def main():
    
    parser = argparse.ArgumentParser(description="Simulate mixed effects data over multiple configs.")
    parser.add_argument("--config", type=str, default="config_test.yaml", help="YAML config filename")
    args = parser.parse_args()

    gen_mixed_data_from_config(args.config)


if __name__ == "__main__":
    main()
