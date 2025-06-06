library(RandomFields)
library(reticulate)
library(yaml)

set.seed(2)

np <- import("numpy")

# SIMULATION SETTINGS
config <- yaml::read_yaml("configs/config_run_simulation.yaml")
sim_config <- config$simulation
gp_parameters <- config$gp_parameters
# Extract config values
dims <- sim_config$dimensions
ns <- sim_config$sample_sizes
B <- sim_config$replicates
sigma2 <- gp_parameters$kernel$signal_variance
rho <- gp_parameters$kernel$lengthscale
nu <- gp_parameters$kernel$nu



for (d in dims) {
  for (n in ns) {
    for (b in 1:B) {
      
      # --- Generate coordinates ---
      set.seed(1000 * n + 100 * d + b)
      coords <- matrix(runif(n * d), ncol = d)
      
      # --- Latent GP (f) ---
      RFmodel_f <- RMmatern(var = sigma2, notinvnu = TRUE, scale = rho, nu = nu)
      sim_f <- RFsimulate(RFmodel_f, x = coords)
      sim_f <- RFspDataFrame2conventional(sim_f)
      f <- sim_f$data
      
      # --- Heteroscedastic GP (g) ---
      RFmodel_g <- RMmatern(var = sigma2, notinvnu = TRUE, scale = rho, nu = nu)  # You can change var here
      sim_g <- RFsimulate(RFmodel_g, x = coords)
      sim_g <- RFspDataFrame2conventional(sim_g)
      g <- sim_g$data
      
      
      # --- Save to .npz ---
      dir_path <- paste0("data/simulated_latent/", n, "_", d, "/")
      dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
      
      file_path <- paste0(dir_path, "/data_replicate_", b, ".npz")
      
      # Convert and save
      np$savez(
        file = file_path,
        X = r_to_py(coords),
        f = r_to_py(f),
        g = r_to_py(g)
      )
      
      cat(sprintf("Saved: %s\n", file_path))
    }
  }
}




