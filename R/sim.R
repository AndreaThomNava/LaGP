# Load required libraries

source("R/utils.R")
library(yaml)
library(parallel)
library(reticulate)
use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)

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
  test_split <- configs$data_generation$test_size
  print(paste("test size", test_split))

  
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
  fe_test <- data$fe_test
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
    eps = eps_test + fe_test,
    noise = likelihood,
    pars = configs$data_generation$pars,
    target_quantile = target_quantile,
    g = g
  )
  
  # Needed for prediction intervals
  t <- qnorm(1 - (1 - alpha) / 2)
    
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
    coverage <- NA_real_
    interval_loss <- NA_real_
    
    tryCatch({
      if (model_name == "lqmm") {
        print("Fitting lqmm")
        
        res <- model_lqmm(train_X = train_X, train_y = train_y,
                          group_train = group_train, test_X = test_X, group_test = group_test,
                          target_quantile = target_quantile)
        
        latent_pred <- res$predictions
        hyper_params <- res$hyper_params
        fit_time <- res$fit_time
        
      } else if (model_name == "brms") {
        print("BRMS: Sampling via MCMC")
        #check_data_structure(train_X, train_y, group_train, test_X, group_test)
        pred <- model_brms_quantile_v2(train_X = train_X, train_y = train_y,
                                    group_train = group_train, test_X = test_X, group_test = group_test,
                                    target_quantile = target_quantile)
        
        print("sampled successfully")
        latent_pred <- pred$predictions
        # pred_std <- pred$se # contains uncertainty also about fixed effects
        hyper_params <- pred$hyper_params
        fit_time <- pred$fit_time
        
       # low_pred <- latent_pred - t * pred_std
       # up_pred <- latent_pred + t * pred_std
        
       # interval_loss <- interval_score(test_true_latent_quantile, low_pred, up_pred, alpha)
       # print(paste("IS loss: ", interval_loss))
        
       # coverage_and_width_results <- coverage_and_width(test_true_latent_quantile, low_pred, up_pred)
       # coverage <- coverage_and_width_results[1]
       # width <- coverage_and_width_results[2]
        
      } else if (model_name == "bayesqr") {
        print("Fitting BayesQR (MCMC quantile regression)")
        
        res <- model_bayesqr_v2(train_X = train_X, train_y = train_y,
                             group_train = group_train,
                             test_X = test_X, group_test = group_test,
                             target_quantile = target_quantile)
        
        latent_pred <- res$predictions
        hyper_params <- res$hyper_params
        fit_time <- res$fit_time
      }
      
      
      if (!is.null(latent_pred)) {
        qs_loss <- quantile_score(test_y, latent_pred, target_quantile)
        print(paste("QS loss:", qs_loss))
        rmse <- sqrt(mean((latent_pred - test_true_latent_quantile)^2))
        
        
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
      rmse <<- NA_real_
    })
    
    model_results[[model_name]] <- list(
      quantile_loss = qs_loss,
      rmse= rmse,
      interval_loss = interval_loss,
      coverage = coverage,
      hyper_params = hyper_params,
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
        }, mc.cores = num_replicates) # detectCores()
        
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

# update them!
# Parameter update if fixed SNR is set
if (configs_sim[["data_generation"]][["fixed_snr"]] %||% TRUE) {
  pars <- compute_dict_pars(
    signal_variance = configs_sim[["data_generation"]][["signal_variance"]],
    snr = configs_sim[["data_generation"]][["snr"]], 
    quantile = configs_sim[["data_generation"]][["quantile"]]
  )
  
  # Update/merge pars (equivalent to Python's .update())
  configs_sim[["data_generation"]][["pars"]] <- modifyList(
    configs_sim[["data_generation"]][["pars"]], 
    pars
  )
}

models <-  list("bayesqr", "lqmm", "brms") #, "brms") #, "brms") 
for (model_name in models){
  print(models)
}

num_replicates <- configs_sim$replicate
results <- fit_models_on_all_datasets_parallel(configs = configs_sim, models = models,
                                               num_replicates = num_replicates)
### select version

version <- "paper_group"

# Save the results
OUTPUT_DIR <- file.path("results", "simulation_mm", configs_sim$randeff, version)
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
