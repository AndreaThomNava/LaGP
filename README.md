# Repository for the paper: “Laplace Approximations for Mixed-Effects and Gaussian Process Quantile Regression”.

This repository accompanies our paper on Laplace-based inference for (i) Gaussian-process (GP) models and (ii) mixed models. The codebase is organized to support fully reproducible experiments, including synthetic-data generation, method implementations in both Python and R, and scripts to run experiments and reproduce paper figures.

## Branch overview

- **`GP` branch**: Gaussian-process (GP) experiments and GP-based methods.
- **`mixed-model` branch**: mixed-model experiments and mixed-model methods.

If you are looking for GP-related code, use the **`GP`** branch.  
If you are looking for mixed-model code, use the **`mixed-model`** branch.

---

## You are on branch: `GP`

This branch contains the GP experiments and (where relevant) the notebooks used to generate paper plots.

### Repository structure

- **`R/`**  
  R implementations of selected methods and supporting routines used in the experiments.

- **`src/`**  
  Python source code: model implementations and utilities (e.g., data generation helpers, plotting utilities, evaluation metrics).

- **`scripts/`**  
  Experiment entry points and orchestration scripts. Typical tasks include:
  - generating datasets,
  - running Python methods,
  - merging results produced by Python and R,
  - producing summary tables/plots for the paper.

- **`configs/`**  
  Experiment configuration files (e.g., YAML) defining simulation settings, hyperparameters and run grids.

- **`notebooks/`** (GP branch only)  
  Notebooks used to reproduce selected plots/figures for the paper.  
  Only notebooks required for the paper are tracked (typically under `notebooks/paper/`).

- **`data/real_data/`**  
  Real dataset(s) used in the empirical experiments (tracked).  
  See the folder contents for file-level details.

### Reproducing experiments (high level)

1. Set experiment parameters in `configs/`.
2. Run the appropriate entry script(s) from `scripts/` to generate data and fit models.
3. Use the analysis/aggregation scripts in `scripts/` to merge outputs (Python + R) and produce tables/figures.

### Notes on outputs

Intermediate outputs and large result artifacts are typically written to `results/` (often not tracked in Git to keep the repository lightweight).

### Citation

If you use this code, please cite the corresponding paper.