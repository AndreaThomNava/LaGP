# Load required libraries

source("R/utils.R")
library(yaml)
library(parallel)
library(reticulate)
#use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)

# Function to fit and evaluate models for a single replicate
fit_and_evaluate_replicate <- function(configs, replicate_config, replicate, models) {
  
  likelihood <- replicate_config$likelihood
  is_heteroscedastic <- grepl("heteroscedastic", likelihood)
  
  sample_size <- replicate_config$sample_size
  input_dim <- replicate_config$input_dim
  print(paste("Likelihood: ", likelihood))
  print(paste("Sample size: ", sample_size))
  print(paste("Dim: ", input_dim))
  print(paste("Replicate", replicate))
  
  delta_logl <- configs$delta_logl
  alpha <- configs$alpha
  train_split <- configs$train_split
  print(train_split)
  n_epochs <- configs$n_epochs
  lr <- configs$lr
  target_quantile <- configs$target_quantile
  
  # Load the dataset for this replicate
  data <- load_data(likelihood, sample_size, input_dim, replicate, train_split)
  f <- data$f
  train_X <- data$X_train
  train_y <- data$y_train
  test_X <- data$X_test
  test_y <- data$y_test
  
  
  if (is_heteroscedastic) {
    # Load the scale GP (g) if heteroscedastic
    g <- load_scale_gp(likelihood, sample_size, input_dim, replicate, file_path = NULL)
    
    # Get the noise model from the likelihood
    noise <- strsplit(likelihood, "_")[[1]][1]
  } else {
    noise <- likelihood
  }
  
  # Obtain the true latent quantile
  true_latent_quantile <- obtain_quantile(
    f = f, 
    noise = noise, 
    pars = configs$simulation$pars, 
    target_quantile = target_quantile, 
    g = if (is_heteroscedastic) g else NULL
  )
  
  # Needed for prediction intervals
  t <- qnorm(1 - (1 - alpha) / 2)
  
  model_results <- list()
  
  for (model_name in models) {
    
    latent_pred <- NULL
    stddev_pred <- NULL
    low_pred <- NULL
    up_pred <- NULL
    fit_time <- NA_real_
    qs_loss <- NA_real_
    interval_loss <- NA_real_
    coverage <- NA_real_
    width <- NA_real_
    
    tryCatch({
      
      if (model_name == "qgam") {
        print("Fitting qgam")
        pred <- model_qgam(train_X, train_y, test_X, target_quantile)
        latent_pred <- pred$predictions
        stddev_pred <- sqrt(pred$se)
        low_pred <- latent_pred - stddev_pred * t
        up_pred <- latent_pred + stddev_pred * t
        fit_time <- pred$fit_time
        
      } else if (model_name == "vecchia_mcmc") {
        print("Sampling via MCMC")
        pred <- model_vecchia_gp(train_X, train_y, test_X,
                                 target_quantile,
                                 stan_model_path = "R/stan/asym_laplace_matern32_noncentered_sparse.stan",
                                 m = 5)
        print("sampled successfully")
        latent_pred <- pred$predictions
        samples <- pred$samples
        low_pred <- apply(samples, 3, quantile, probs = alpha / 2)
        up_pred <- apply(samples, 3, quantile, probs = 1 - alpha / 2)
        fit_time <- pred$fit_time
      }
      
      if (!is.null(latent_pred)) {
        qs_loss <- quantile_score(test_y, latent_pred, target_quantile)
        print(paste("QS loss:", qs_loss))
        
        test_true_latent_quantile <- true_latent_quantile[(length(train_y) + 1):sample_size]
        
        interval_loss <- interval_score(test_true_latent_quantile, low_pred, up_pred, alpha)
        print(paste("IS loss: ", interval_loss))
        
        coverage_and_width_results <- coverage_and_width(test_true_latent_quantile, low_pred, up_pred)
        coverage <- coverage_and_width_results[1]
        width <- coverage_and_width_results[2]
      } else {
        print("No predictions available for QS or IS calculation.")
      }
      
      print(paste("Time:", fit_time))
      
    }, error = function(e) {
      print(paste("Model", model_name, "failed with error:", e$message))
      # All outputs stay NA or NULL
    })
    
    model_results[[model_name]] <- list(
      quantile_loss = qs_loss,
      interval_loss = interval_loss,
      coverage = coverage,
      width = width,
      time = fit_time
    )
  }
  
  return(model_results)
  
}

# Function to fit models on all datasets in parallel
fit_models_on_all_datasets_parallel <- function(configs, models, num_replicates = 10) {
  results <- list()
  
  for (likelihood in configs$simulation$likelihoods) {
    for (sample_size in configs$simulation$sample_sizes) {
      for (input_dim in configs$simulation$dimensions) {
        
        # Prepare the configuration for this dataset
        replicate_config <- list(
          likelihood = likelihood,
          sample_size = sample_size,
          input_dim = input_dim
        )
        
        config_key <- paste(likelihood, sample_size, input_dim, sep = "_")
        results[[config_key]] <- vector("list", num_replicates)
        
        # Use mclapply for parallel execution (requires 'parallel' package)
        replicate_results <- mclapply(1:num_replicates, function(replicate) {
          fit_and_evaluate_replicate(configs, replicate_config, replicate, models)
        }, mc.cores = 1) # detectCores()
        
        # Store results
        for (replicate in 1:num_replicates) {
          results[[config_key]][[replicate]] <- replicate_results[[replicate]]
        }
      }
    }
  }
  
  return(results)
}

configs_sim <- load_config("configs/config_run_simulation.yaml")

# update them!
# Parameter update if fixed SNR is set
if (configs_sim$simulation$fixed_snr) {
  signal_variance <- configs_sim$gp_parameters$kernel$signal_variance
  snr <- configs_sim$simulation$snr
  quantile <- configs_sim$simulation$pars$ald$q
  
  updated_pars <- compute_dict_pars(signal_variance, snr, quantile)
  configs_sim$pars <- updated_pars
}

models <- list("qgam") #, "vecchia_mcmc") #, "vecchia_mcmc")
for (model_name in models){
  print(models)
}
num_replicates <- configs_sim$simulation$replicates
results <- fit_models_on_all_datasets_parallel(configs = configs_sim, models = models,
                                               num_replicates = num_replicates)

# Save the results
version <- "1"
OUTPUT_DIR <- paste0("results/simulation/", version)
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Combine results and config into one list
all_results <- list(
  config = configs_sim,
  results = results
)

# Save the object in Python pickle format
py_run_string("import pickle")
output_file <- file.path(OUTPUT_DIR, "simulation_results_R.pkl")
py_save_object(all_results, output_file)  # Save the R object as a pickle file

# Save using saveRDS instead of pickle
output_file <- file.path(OUTPUT_DIR, "simulation_results_R.rds")
saveRDS(all_results, output_file)

cat("Results and config saved to", output_file, "\n")
