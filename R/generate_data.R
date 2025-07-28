library(RandomFields)
library(reticulate)
library(yaml)
#use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)

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


# disables internal spatial conforming heuristics
# RFoptions(spConform = TRUE)
#RFoptions(spConform = FALSE, pcholesky.ignore = TRUE, modus_operandi = "no", messages = FALSE)


# ns <- 2000
#rho <- 0.01
# sigma_2 <- 1
# dims <- 3
for (d in dims) {
  print(d)
  if (d > 2){
    rho_d <- rho * sqrt(d/2)
  } else {
    rho_d <- rho
  }
  for (n in ns) {
    for (b in 1:B) {
      
      # --- Generate coordinates ---
      set.seed(n + 100 * d + b)
      
      if (d == 3) {
        # Create an approximately cubic grid with ~n points:
        side_len <- ceiling(n^(1/3))
        grid_pts <- seq(0, 1, length.out = side_len)
        coords <- as.matrix(expand.grid(grid_pts, grid_pts, grid_pts))
        # If grid has more points than n, subset
        if (nrow(coords) > n) {
          coords <- coords[1:n, , drop=FALSE]
        }
      } else {
        coords <- matrix(runif(n * d), ncol = d)
      }
     # coords <- matrix(runif(n * d), ncol = d)
      
      # --- Latent GP (f) ---
      RFmodel_f <- RMmatern(var = sigma2, notinvnu = TRUE, scale = rho_d, nu = nu)
      sim_f1 <- RFsimulate(RFmodel_f, x = coords)
      #f <- as.numeric(sim_f)
      sim_f <- RFspDataFrame2conventional(sim_f1)
      f <- sim_f$data
      
      # --- Heteroscedastic GP (g)---
      RFmodel_g <- RMmatern(var = sigma2, notinvnu = TRUE, scale = rho_d, nu = nu)  # You can change var here
      sim_g <- RFsimulate(RFmodel_g, x = coords)
      #g <- as.numeric(sim_g)
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




