#!/usr/bin/env Rscript
library(parallel)
library(reticulate)
use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)
library(yaml)
source("R/utils.R")

fit_and_evaluate_replicate <- function(X, y, fold, configs, models, replicate) {
 
  train_idx <- unlist(fold$train_idx)
  test_idx <- unlist(fold$test_idx)

  # Randomly sample 10k indices (unsorted)
  np <- import("numpy")
  np$random$seed(42L + replicate)
  sampled_idx <- np$random$choice(length(y), size=10000L, replace=FALSE)
  sampled_idx <- as.integer(sampled_idx) + 1L # to match python 0-based indexing
  # Split: first 5k for train, last 1k for test (already shuffled)
  train_idx <- sampled_idx[1:9000]
  test_idx <- sampled_idx[9001:10000]

  X_train <- X[train_idx, , drop = FALSE]
  X_test  <- X[test_idx, , drop = FALSE]
  y_train <- y[train_idx]
  y_test  <- y[test_idx]

  # Standardize using only training data
  y_mean <- mean(y_train)
  y_sd <- sd(y_train)
  y_train <- (y_train - y_mean) / y_sd
  y_test <- (y_test - y_mean) / y_sd
  
  # Define tolerance
  TOL <- 1e-8
  
  # Identify non-constant columns in training set
  is_nonconstant <- apply(X_train, 2, sd) > TOL
  
  # Filter both training and test sets
  X_train <- X_train[, is_nonconstant]
  X_test <- X_test[, is_nonconstant]
  
  # Log dropped features
  dropped <- colnames(X)[!is_nonconstant]
  if (length(dropped) > 0) {
    cat("Dropped constant features:", paste(dropped, collapse=", "), "\n")
  }

  # Compute min and max for each column of the training set
  train_min <- apply(X_train, 2, min)
  train_max <- apply(X_train, 2, max)
  
  # Compute range
  train_range <- train_max - train_min
  
  # Avoid division by zero for constant columns
  train_range[train_range == 0] <- 1.0
  
  # Center and scale training data
  train_X_scaled <- (X_train - matrix(train_min, nrow = nrow(X_train), ncol = ncol(X_train), byrow = TRUE)) /
    matrix(train_range, nrow = nrow(X_train), ncol = ncol(X_train), byrow = TRUE)
  
  # Center and scale test data using training statistics
  test_X_scaled <- (X_test - matrix(train_min, nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)) /
    matrix(train_range, nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)
  
  # configs
  delta_logl <- configs$delta_logl
  n_epochs <- configs$n_epochs
  lr <- configs$lr
  threshold_approx <- configs$threshold_approximation
  inducing_points <- configs$inducing_points
  target_quantile <- configs$target_quantile
  
  model_results <- list()
  
  for (model_name in models) {
    
    latent_pred <- NULL
    fit_time <- NA_real_
    qs_loss <- NA_real_
    
    tryCatch({
      if (model_name == "qgam") {
        print("Fitting qgam")
        pred <- model_qgam(train_X_scaled, y_train, test_X_scaled, target_quantile)
        latent_pred <- pred$predictions
        fit_time <- pred$fit_time
      }
      
      else if (model_name == "qgam_interactions") {
        print("Fitting qgam with interactions")
        pred <- model_qgam_interactions(train_X_scaled, y_train, test_X_scaled, target_quantile)
        latent_pred <- pred$predictions
        fit_time <- pred$fit_time
      }
      
      if (!is.null(latent_pred)) {
        qs_loss <- quantile_score(y = y_test, preds = latent_pred, quantile = target_quantile)
        print(paste("QS loss:", qs_loss))
      } else {
        print("No predictions available for QS loss calculation.")
      }
      print(paste("Time:", fit_time))
      
    }, error = function(e) {
      print(paste("Model", model_name, "failed with error:", e$message))
      # fallback values are already set
    })
    
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
    data <- load_X_y_preprocessed(dataset_name = df_name, dir = DIR)
    X <- data$X
    y <- data$y
    
    folds <- load_cv_splits(dataset_name = df_name, dir = DIR,
                            n_splits = n_splits)
    config_key <- df_name
    results[[config_key]] <- vector("list", n_splits)
    
    replicate_results <- mclapply(seq_len(n_splits), function(replicate) {
      fold <- folds[replicate, ]
      fit_and_evaluate_replicate(X, y, fold, configs, models, replicate)
    }, mc.cores = n_splits)  # Set to detectCores() if you want parallelism
    
    for (replicate in seq_len(n_splits)) {
      results[[config_key]][[replicate]] <- replicate_results[[replicate]]
    }
    
  }
  
  return(results)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
configs <- load_config("configs/config_run_real.yaml")

# Override dataset if provided as argument
if (length(args) > 0) {
  configs$datasets <- list(args[1])  # Use first argument as dataset name
}

models <- list("qgam", "qgam_interactions")
for (model_name in models){
  print(models)
}
results <- fit_models_on_all_datasets_parallel(configs = configs, models = models)

# Save the results
version <- "095"
OUTPUT_DIR <- paste0("results/real_data/", version)
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# Combine results and config into one list
all_results <- list(
  config = configs,
  results = results
)

# --- Python pickle version ---
py_run_string("import pickle")
output_file <- file.path(OUTPUT_DIR, "real_data_results_R.pkl")
if (file.exists(output_file)) {
  # Load existing pickle
  old_results <- py_load_object(output_file)
  # Merge datasets: add new or replace existing
  for (dataset_key in names(all_results$results)) {
    old_results$results[[dataset_key]] <- all_results$results[[dataset_key]]
  }
  # Optionally replace config
  old_results$config <- all_results$config
} else {
  old_results <- all_results
}
py_save_object(old_results, output_file)  # Save the updated object

# --- RDS version ---
output_file_rds <- file.path(OUTPUT_DIR, "real_data_results_R.rds")
if (file.exists(output_file_rds)) {
  old_rds <- readRDS(output_file_rds)
  
  # Merge datasets: add new or replace existing
  for (dataset_key in names(all_results$results)) {
    old_rds$results[[dataset_key]] <- all_results$results[[dataset_key]]
  }
  old_rds$config <- all_results$config
} else {
  old_rds <- all_results
}
saveRDS(old_rds, output_file_rds)

cat("Results and config saved to RDS and pickle in", OUTPUT_DIR, "\n")
