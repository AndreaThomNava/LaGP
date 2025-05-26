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
  
  # target quantile
  target_quantile <- configs$target_quantile
  randeff <- configs$randeff
  
  n_groups <- replicate_config$n_groups
  group_size <- replicate_config$group_size
  print(paste("Likelihood: ", likelihood))
  print(paste("Num. Groups: ", n_groups))
  print(paste("Group Size: ", group_size))
  print(paste("Replicate", replicate))
  
  delta_logl <- configs$delta_logl
  alpha <- configs$alpha
  test_split <- configs$test_size
  print(test_split)

  
  # Load the dataset
  data <- load_data(
    randeff = randeff,
    likelihood = likelihood,
    n_groups = n_groups,
    group_size = group_size,
    replicate = replicate
  )
  
  train_X <- data$X_train
  train_y <- data$y_train
  test_X <- data$X_test
  test_y <- data$y_test
  group_train <- data$group_train
  group_test <- data$group_test
  eps_test <- data$eps_test
  g <- if (is_heteroscedastic) data$g else NULL
  
  
  if (is_heteroscedastic) {
    # Load the scale GP (g) if heteroscedastic
    g <- load_scale_gp(likelihood, sample_size, input_dim, replicate, file_path = NULL)
    
    # Get the noise model from the likelihood
    noise <- strsplit(likelihood, "_")[[1]][1]
  } else {
    noise <- likelihood
  }
  
  # Compute true latent quantile
  test_true_latent_quantile <- obtain_quantile(
    eps = eps_test,
    noise = likelihood,
    pars = configs$data_generation$pars,
    target_quantile = target_quantile,
    g = g
  )
  
  # Needed for prediction intervals
  t <- qnorm(1 - (1 - alpha) / 2)
  
  model_results <- list()

  
  for (model_name in models) {
    if (model_name == "qgam") {
      print("Fitting qgam")
      pred <- model_qgam(train_X, train_y, test_X, target_quantile)
      latent_pred <- pred$predictions
      stddev_pred <- sqrt(pred$se)
      low_pred <- latent_pred - stddev_pred * t
      up_pred <- latent_pred + stddev_pred * t
      fit_time <- pred$fit_time
    } 
    
    else if (model_name == "vecchia_mcmc") {
      print("Sampling via MCMC")
      pred <- model_vecchia_gp(train_X, train_y, test_X,
                              target_quantile,
                              stan_model_path = "R/stan/asym_laplace_matern32_noncentered_sparse.stan",
                              m = 5  # number of neighbors for Vecchia
                              )
      print("sampled successfully")
      latent_pred <- pred$predictions
      samples <- pred$samples
      # symmetric Prediction Interval
      low_pred <- apply(samples, 3, quantile, probs =  alpha/2 )
      up_pred <- apply(samples, 3, quantile, probs =  1 - (alpha/2))
      fit_time <- pred$fit_time
     }
    
    # Compute quantile score
    print("computing scores")
    qs_loss <- quantile_score(test_y, latent_pred, target_quantile)
    print(paste("QS loss:", qs_loss))
    # Compute interval score
    test_true_latent_quantile <- true_latent_quantile[(length(train_y) + 1):sample_size]
    interval_loss <- interval_score(test_true_latent_quantile, low_pred, up_pred, alpha)
    print(paste("IS loss: ", interval_loss))
    
    # Compute coverage and width
    coverage_and_width_results <- coverage_and_width(test_true_latent_quantile, low_pred, up_pred)
    
    model_results[[model_name]] <- list(
      quantile_loss = qs_loss,
      interval_loss = interval_loss,
      coverage = coverage_and_width_results[1],
      width = coverage_and_width_results[2],
      time = fit_time
    )
  }
  
  return(model_results)
}

# Function to fit models on all datasets in parallel
fit_models_on_all_datasets_parallel <- function(configs, models, num_replicates = 10) {
  results <- list()
  
  for (likelihood in configs$likelihood) {
    for (n_groups in configs$n_groups) {
      for (group_size in configs$group_size) {
        
        # Prepare the configuration for this dataset
        replicate_config <- list(
          likelihood = likelihood,
          n_groups = n_groups,
          group_size = group_size
        )
        
        config_key <- paste(likelihood, n_groups, group_size, sep = "_")
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


configs_sim <- load_config("configs/config_test.yaml")
models <- list("qgam", "vecchia_mcmc") #, "vecchia_mcmc")
for (model_name in models){
  print(models)
}
num_replicates <- configs_sim$replicate
results <- fit_models_on_all_datasets_parallel(configs = configs_sim, models = models,
                                               num_replicates = num_replicates)

# Save the results
OUTPUT_DIR <- file.path("results", "simulation_mm", configs$randeff)
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
