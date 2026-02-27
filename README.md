# Repository for the paper "Laplace Approximations for Mixed-Effects and Gaussian Process Quantile Regression".

This repository accompanies our paper on Laplace-based inference for (i) Gaussian-process (GP) models and (ii) mixed models. The codebase is organized to support fully reproducible experiments, including synthetic-data generation, method implementations in both Python and R, and scripts to run experiments and reproduce paper figures.

## Branch overview

- **`GP` branch**: Gaussian-process (GP) experiments and GP-based methods.
- **`mixed-model` branch**: mixed-model experiments and mixed-model methods.

If you are looking for GP-related code, use the **`GP`** branch.  
If you are looking for mixed-model code, use the **`mixed-model`** branch.

---

## You are on branch: `mixed-model`

This branch contains the mixed-model experiments and mixed-model method implementations. (The plotting notebooks used for GP figures live only on the `GP` branch.)

### Repository structure

- **`R/`**  
  R implementations of mixed-model methods and supporting routines used in the experiments.

- **`src/`**  
  Python source code: utilities for data generation, evaluation, and analysis (and any Python-side method implementations used in the mixed-model pipeline).

- **`scripts/`**  
  Experiment entry points and orchestration scripts. Typical tasks include:
  - generating datasets,
  - running Python methods,
  - merging results produced by Python and R,
  - producing summary tables/plots for the paper.

- **`configs/`**  
  Experiment configuration files (e.g., YAML) defining simulation settings, hyperparameters,and run grids.

- **`data/real_data/`**  
  Real dataset(s) used in the empirical experiments (tracked).  
  See the folder contents for file-level details.

### Reproducing experiments (high level)

1. Set experiment parameters in `configs/`.
2. Run the appropriate entry script(s) from `scripts/` to generate data and fit models.
3. Use the analysis/aggregation scripts in `scripts/` to merge outputs (Python + R) and produce tables/figures.

**Environment.** The experiments were run with **Python 3.13.2** and R 4.4.0.

### Notes on outputs

Intermediate outputs and large result artifacts are typically written to `results/` (often not tracked in Git to keep the repository lightweight).

### Citation

If you use this code, please cite the corresponding paper.