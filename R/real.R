#!/usr/bin/env Rscript
library(parallel)
library(reticulate)
use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)
library(yaml)
source("R/utils.R")

fit_and_evaluate_replicate <- function(X, group_data, y, fold, configs, models, replicate) {
 
  train_idx <- unlist(fold$train_idx)#[1:1000]
  test_idx <- unlist(fold$test_idx)#[1:1000]


  train_X <- X[train_idx, , drop = FALSE]
  test_X  <- X[test_idx, , drop = FALSE]
 
  group_train <- group_data[train_idx, , drop = FALSE]
  group_test <- group_data[test_idx, ,drop = FALSE]
  
  # scale y
  # Compute mean and sd from the training set only
  train_mean <- mean(y[train_idx])
  train_sd   <- sd(y[train_idx])
  
  # Scale
  train_y <- (y[train_idx] - train_mean) / train_sd
  test_y  <- (y[test_idx]  - train_mean) / train_sd

  # Compute min and max for each column of the training set
  train_min <- apply(train_X, 2, min)
  train_max <- apply(train_X, 2, max)
  
  # Compute range
  train_range <- train_max - train_min
 
  # Avoid division by zero for constant columns
  train_min[train_range == 0] <- 0.0  # Add this line
  train_range[train_range == 0] <- 1.0

  # Center and scale training data
  train_X_scaled <- (train_X - matrix(train_min, nrow = nrow(train_X), ncol = ncol(train_X), byrow = TRUE)) /
    matrix(train_range, nrow = nrow(train_X), ncol = ncol(train_X), byrow = TRUE)
  
  # Center and scale test data using training statistics
  test_X_scaled <- (test_X - matrix(train_min, nrow = nrow(test_X), ncol = ncol(test_X), byrow = TRUE)) /
    matrix(train_range, nrow = nrow(test_X), ncol = ncol(test_X), byrow = TRUE)
  
  
  # add intercept
  # Add intercept column
  #train_X_scaled <- cbind(intercept = 1, train_X_scaled)
  #test_X_scaled  <- cbind(intercept = 1, test_X_scaled)
  
  
  delta_logl <- configs$delta_logl
  target_quantile <- configs$target_quantile
  
  model_results <- list()
  
  # For one grouped random effect
  expected_num_cov_pars <- 1
  expected_cov_par_names <- c("Group_1")  # Define expected names
  
  # Define a "safe empty" hyper_params list to use on failures
  empty_hyper_params <- c(
    setNames(rep(NA_real_, expected_num_cov_pars), expected_cov_par_names),
    list(noise_variance = NA_real_)
  )
  
  for (model_name in models) {
    
    latent_pred <- NULL
    hyper_params <- empty_hyper_params
    fit_time <- NA_real_
    qs_loss <- NA_real_
    
    tryCatch({
      if (model_name == "lqmm") {
        print("Fitting lqmm")
        res <- model_lqmm(train_X = train_X_scaled, train_y = train_y,
                          group_train = group_train, test_X = test_X_scaled, group_test = group_test,
                          target_quantile = target_quantile)
        
        latent_pred <- res$predictions
        hyper_params <- res$hyper_params
        fit_time <- res$fit_time
        
      } else if (model_name == "brms") {
        print("BRMS: Sampling via MCMC")
        pred <- model_brms_quantile_v2(train_X = train_X_scaled, train_y = train_y,
                                    group_train = group_train, test_X = test_X_scaled, group_test = group_test,
                                    target_quantile = target_quantile, ndraws = 3000)
        
        print("sampled successfully")
        latent_pred <- pred$predictions
        hyper_params <- pred$hyper_params
        fit_time <- pred$fit_time
      } else if (model_name == "bayesqr") {
        print("Fitting BayesQR (MCMC quantile regression)")
        
        res <- model_bayesqr_v2(train_X = train_X_scaled, train_y = train_y,
                                group_train = group_train,
                                test_X = test_X_scaled, group_test = group_test,
                                target_quantile = target_quantile, ndraws = 3000)
        
        latent_pred <- res$predictions
        hyper_params <- res$hyper_params
        fit_time <- res$fit_time
      }
      
      if (!is.null(latent_pred)) {
        qs_loss <- quantile_score(test_y, latent_pred, target_quantile)
        print(paste("QS loss:", qs_loss))
      } else {
        print("No predictions available for QS loss calculation.")
      }
      
      print(paste("Time:", fit_time))
      
    }, error = function(e) {
      # On any error, fallback to safe empty values
      print(paste("Model", model_name, "failed with error:", e$message))
      latent_pred <<- NULL
      hyper_params <<- empty_hyper_params
      fit_time <<- NA_real_
      qs_loss <<- NA_real_
    })
    
    model_results[[model_name]] <- list(
      quantile_loss = qs_loss,
      hyper_params = hyper_params,
      time = fit_time
    )
  }
  
  return(model_results)
}

fit_models_on_all_datasets_parallel <- function(configs, models) {
  
  n_splits <- configs$n_splits
  results <- list()
  
  DIR <- "data/real_data_mm"
  
  for (df_name in configs$datasets) {
    print(df_name)
    data <- load_X_y_preprocessed(dataset_name = df_name, dir = DIR)
    X <- data$X
    group_data <- data$group_data
    y <- data$Y
    
    folds <- load_cv_splits(dataset_name = df_name, dir = DIR,
                            n_splits = n_splits)
    
    replicate_config <- list(
      df_name = df_name
    )
    
    config_key <- df_name
    results[[config_key]] <- vector("list", n_splits)
    
    replicate_results <- mclapply(seq_len(n_splits), function(replicate) {
      fold <- folds[replicate, ]
      fit_and_evaluate_replicate(X, group_data, y, fold, configs, models, replicate)
    }, mc.cores = n_splits) #n_splits)  # Set to detectCores() if you want parallelism
    
    for (replicate in seq_len(n_splits)) {
      results[[config_key]][[replicate]] <- replicate_results[[replicate]]
    }
    
  }
  
  return(results)
}

configs <- load_config("configs/config_mm_real.yaml")
models <- list("lqmm", "brms", "bayesqr") 
results <- fit_models_on_all_datasets_parallel(configs = configs, models = models)
# Save the results
version <- "paper_group"
OUTPUT_DIR <- paste0("results/real_data_mm/", version)
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