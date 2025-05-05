#!/usr/bin/env Rscript
library(parallel)
library(reticulate)
library(yaml)
source("R/utils.R")

fit_and_evaluate_replicate <- function(X, y, fold, configs, models) {
 
  train_idx <- unlist(fold$train_idx)
  test_idx <- unlist(fold$test_idx)
  
  X_train <- X[train_idx, , drop = FALSE]
  X_test  <- X[test_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_test  <- y[test_idx]
  
  
  approx <- configs$approximation
  delta_logl <- configs$delta_logl
  n_epochs <- configs$n_epochs
  lr <- configs$lr
  threshold_approx <- configs$threshold_approximation
  inducing_points <- configs$inducing_points
  target_quantile <- configs$target_quantile
  
  model_results <- list()
  
  for (model_name in models) {
    if (model_name == "qgam") {
      print("Fitting qgam")
      pred <- model_qgam(X_train, y_train, X_test, target_quantile)
      latent_pred <- pred$predictions
      fit_time <- pred$fit_time
    } 
    
    
    # Compute quantile score
    qs_loss <- quantile_score(y = y_test, preds = latent_pred, quantile = target_quantile)
    
    model_results[[model_name]] <- list(
      quantile_loss = qs_loss,
      time = fit_time
    )
  }
  
  return(model_results)
}

fit_models_on_all_datasets_parallel <- function(configs, models) {
  
  n_splits <- configs$n_splits
  results <- list()
  
  DIR <- "data/real_data"
  
  for (df_name in configs$datasets) {
    data <- load_X_y(dataset_name = df_name, dir = DIR)
    X <- data$X
    y <- data$y
    
    folds <- load_cv_splits(dataset_name = df_name, dir = DIR,
                            n_splits = n_splits)
    
    replicate_config <- list(
      df_name = df_name
    )
    
    config_key <- df_name
    results[[config_key]] <- vector("list", n_splits)
    
    replicate_results <- mclapply(seq_len(n_splits), function(replicate) {
      fold <- folds[replicate, ]
      fit_and_evaluate_replicate(X, y, fold, configs, models)
    }, mc.cores = 1)  # Set to detectCores() if you want parallelism
    
    for (replicate in seq_len(n_splits)) {
      results[[config_key]][[replicate]] <- replicate_results[[replicate]]
    }
    
  }
  
  return(results)
}

configs <- load_config("configs/config_run_real.yaml")
models <- list("qgam")
for (model_name in models){
  print(models)
}
results <- fit_models_on_all_datasets_parallel(configs = configs, models = models)

# Save the results
OUTPUT_DIR <- "results/real_data"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Combine results and config into one list
all_results <- list(
  config = configs,
  results = results
)

# Save the object in Python pickle format
py_run_string("import pickle")
output_file <- file.path(OUTPUT_DIR, "real_data_results_R.pkl")
py_save_object(all_results, output_file)  # Save the R object as a pickle file

# Save using saveRDS instead of pickle
output_file <- file.path(OUTPUT_DIR, "real_data_results_R.rds")
saveRDS(all_results, output_file)

cat("Results and config saved to", output_file, "\n")













