import argparse
import os
from pathlib import Path

import gpytorch as gpy
import numpy as np
import torch
import yaml

from lagp.utils.generate_data import (
    generate_input_grid,
    simulate_latentGP,
    simulate_response,
)


def gen_data(name_config_file: str):
    # Get the path to the script (e.g., src/lagp/scripts/your_script.py)
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Go up 3 levels: scripts → lagp → src → project_root
    PROJECT_ROOT = SCRIPT_DIR #.parents[2]

    # Now you can access configs/ and data/ from the root
    CONFIGS_DIR = "configs"
    DATA_DIR = "data"

    # Load the configuration file
    with open(os.path.join(CONFIGS_DIR, name_config_file), "r") as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    gp_config = config["gp_parameters"]

    # Set the seed for reproducibility
    np.random.seed(sim_config["seed"])
    torch.manual_seed(sim_config["seed"])

    # Prepare directories
    output_dir = os.path.join(DATA_DIR, "simulated_data")
    os.makedirs(output_dir, exist_ok=True)

    # Loop over different sample sizes and dimensions
    for sample_size in sim_config["sample_sizes"]:
        for dim in sim_config["dimensions"]:
            print(f"Generating data for sample size: {sample_size}, dimension: {dim}")

            # Prepare the grid (same for all likelihoods)
            X = generate_input_grid(
                d=dim, sample_size=sample_size, lims=sim_config["grid_lims"]
            )  # Grid of input points

            # Generate the latent function (same for all likelihoods)
            nu = gp_config["kernel"]["nu"]
            lengthscale = gp_config["kernel"]["lengthscale"]
            signal_variance = gp_config["kernel"]["signal_variance"]
            base_kernel = gpy.kernels.MaternKernel(nu=nu)
            base_kernel.lengthscale = lengthscale
            kernel = gpy.kernels.ScaleKernel(base_kernel)
            kernel.outputscale = signal_variance

            f = simulate_latentGP(X, kernel, n_samples=1)
            f = f.reshape(sample_size)

            # Loop over different likelihoods
            for likelihood in sim_config["likelihoods"]:
                print(f"Generating data for likelihood: {likelihood}")

                # Output directory for the specific likelihood
                likelihood_dir = os.path.join(
                    output_dir, f"{sample_size}_{dim}", likelihood
                )
                os.makedirs(likelihood_dir, exist_ok=True)

                # Simulate B replicates for each likelihood type
                for replicate in range(sim_config["replicates"]):
                    print(f"  Replicate {replicate + 1}/{sim_config['replicates']}")

                    # Simulate the response for this likelihood
                    y = simulate_response(f, noise=likelihood, pars=sim_config["pars"])

                    # Save the data
                    output_file = os.path.join(
                        likelihood_dir, f"data_replicate_{replicate + 1}.npz"
                    )
                    np.savez(output_file, X=X, f=f, y=y)
                    print(f"  Data saved to {output_file}")


def main():
    # Step 1: Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate simulated data based on configuration."
    )

    # Step 2: Define arguments
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )

    # Step 3: Parse arguments
    args = parser.parse_args()

    # Step 4: Call the function with the parsed arguments
    gen_data(name_config_file=args.config)


if __name__ == "__main__":
    main()
