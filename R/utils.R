library(qgam)
library(cmdstanr)
library(gptoolsStan)
library(brms)
library(lqmm)
library(reticulate)
#use_python("/cluster/home/navaan/miniconda3/envs/conda_env/bin/python", required = TRUE)
library(bayesQR)

os <- import("os")
np <- import("numpy")

# Quantile function for the Asymmetric Laplace Distribution
quantile_func_asym_laplace <- function(x, q, scale) {
  # Convert x to a numeric vector
  x <- as.numeric(x)
  
  # Initialize the result vector
  rvs <- numeric(length(x))
  
  # Indices where x is less than or equal to q
  ind <- x <= q
  rvs[ind] <- log(x[ind] / q) * scale / (1 - q)
  
  # Indices where x is greater than q
  ind <- !ind
  rvs[ind] <- -log((1 - x[ind]) / (1 - q)) * scale / q
  
  return(rvs)
}

# Function to generate samples via PIT
ralaplace <- function(n, q, scale) {
  u <- runif(n)  # Step 1: sample from Uniform(0,1)
  samples <- quantile_func_asym_laplace(u, q, scale)  # Step 2: apply quantile function
  return(samples)
}


# Load configuration from a YAML file
load_config <- function(config_path) {
  config <- yaml::read_yaml(config_path)
  return(config)
}

# load data and train and test split
load_data <- function(
    randeff,
    likelihood,
    n_groups,
    group_size,
    replicate,
    file_path = NULL
) {
  folder_name <- file.path(randeff, paste0(n_groups, "_", group_size))
  file_name <- paste0("data_replicate_", replicate, ".npz")
  
  if (is.null(file_path)) {
    file_path <- file.path("data", "simulated_data_mm", folder_name, likelihood, file_name)
  }
  
  np <- reticulate::import("numpy")
  data <- np$load(file_path)
  
  # Define a helper to safely extract optional elements
  safe_get <- function(key) {
    if (key %in% names(data)) data[[key]] else NULL
  }
  
  list(
    X_train = data[["X_train"]],
    y_train = data[["y_train"]],
    X_test = data[["X_test"]],
    y_test = data[["y_test"]],
    group_train = data[["group_train"]],
    group_test = data[["group_test"]],
    fe_train = data[["fe_train"]],
    fe_test = data[["fe_test"]],
    eps_train = data[["eps_train"]],
    eps_test = data[["eps_test"]]
  )
}

# Load the scale GP (g) from a .npz file
load_scale_gp <- function(likelihood, sample_size, input_dim, replicate, file_path = NULL) {
  folder_name <- paste0(sample_size, "_", input_dim)
  file_name <- paste0("data_replicate_", replicate, ".npz")
  
  if (is.null(file_path)) {
    file_path <- file.path("data", "simulated_data", folder_name, likelihood, file_name)
  }
  
  np <- reticulate::import("numpy")
  data <- np$load(file_path)
  
  g <- data[["g"]]
  return(g)
}




load_X_y <- function(dataset_name, dir) {
  path <- file.path(dir, paste0(dataset_name, ".csv"))
  
  if (!file.exists(path)) {
    stop(sprintf("Dataset '%s' not found at %s", dataset_name, path))
  }
  
  df <- read.csv(path)
  
  if (dataset_name == "bike") {
    # Predict 'cnt' from other features
    y <- df$cnt
    X <- df[, !(names(df) %in% c("cnt", "casual", "registered", "dteday"))]
    
  } else if (dataset_name == "house") {
    y <- df$median_house_value
    X <- df[, !(names(df) %in% "median_house_value")]
    
  } else if (dataset_name == "power") {
    y <- df$Global_active_power
    X <- df[, !(names(df) %in% "Global_active_power")]
    
  } else if (dataset_name == "protein") {
    if ("target" %in% names(df)) {
      y <- df$target
    } else {
      y <- df[[ncol(df)]]
    }
    X <- df[, !(names(df) %in% names(y))]
    
  } else if (dataset_name == "elevators") {
    if ("failure" %in% names(df)) {
      y <- df$failure
    } else {
      y <- df[[ncol(df)]]
    }
    X <- df[, !(names(df) %in% names(y))]
    
  } else {
    stop(sprintf("Unknown dataset name: %s", dataset_name))
  }
  
  return(list(X = X, y = y))
}

# Define the R version of load_X_y_preprocessed
load_X_y_preprocessed <- function(dataset_name, dir) {
  # Construct path
  
  path <- file.path(dir, paste0(dataset_name, "_preprocessed.npz"))
  print(path)
  # Check if file exists
  if (!os$path$exists(path)) {
    message(sprintf("DEBUG: File does not exist at %s", path))
    stop(sprintf("Dataset '%s' not found at %s", dataset_name, path))
  }
  
  print("here 2")
  
 
  # Load .npz file using numpy
  data <- np$load(path, allow_pickle = TRUE)
 
  # Extract components
  X <- data[["X"]]
  group_data <- data[["group_data"]]
  Y <- data[["Y"]]
  
  return(list(X = X, group_data = group_data, Y = Y))
}


load_cv_splits <- function(dataset_name, dir = "data/real_data_mm", n_splits = 5) {
  path <- file.path(dir, paste0(dataset_name, "_cv", n_splits, "_splits.json"))
  
  if (!file.exists(path)) {
    stop(sprintf("Split file for dataset '%s' not found at %s", dataset_name, path))
  }
  
  splits <- jsonlite::fromJSON(path)
  
  return(splits)
}


# Recover true quantile
obtain_quantile <- function(eps, noise, pars, target_quantile, g = NULL) {
  n <- length(eps)
  if (!is.null(g)) {
    stopifnot(length(g) == n)
  }
  
  if (noise == "gaussian") {
    scale <- if (!is.null(g)) g else pars$gaussian$scale
    delta <- qnorm(target_quantile) * scale
    
  } else if (noise == "ald") {
    q <- pars$ald$q
    scale <- if (!is.null(g)) g else pars$ald$scale
    delta <- quantile_func_asym_laplace(target_quantile, q, 1) * scale
    
  } else if (noise == "t") {
    df <- pars$t$df
    scale <- if (!is.null(g)) g else pars$t$scale
    delta <- qt(target_quantile, df = df) * scale
    
  } else if (noise == "chi") {
    df <- pars$chi$df
    scale <- if (!is.null(g)) g else pars$chi$scale
    delta <- qchisq(target_quantile, df = df) * scale
    
  } else {
    stop(paste("Unsupported noise model:", noise))
  }
  
  return(eps + delta)
}

compute_dict_pars <- function(signal_variance, snr, quantile) {
  
  target_variance <- signal_variance / snr
  
  # gaussian
  scale_gaussian <- sqrt(target_variance)
  
  # ald
  scale_ald <- sqrt(target_variance * (quantile^2 * (1-quantile)^2 / (1-2*quantile + 2*quantile^2)))
  
  # student t with 3 dfs
  scale_t <- sqrt(0.5 * target_variance)
  
  # chi ?
  
  pars <- list(
    gaussian = list(scale = scale_gaussian),
    ald = list(q = quantile, scale = scale_ald),
    t = list(df = 3, scale = scale_t),
    chi = list(df = 1, scale = 1)  # fixed to 1 for now
  )
  
  return(pars)
}



### ----------------- MODEL FITTING --------------------- ###


### LQMM (Geraci) ###
model_lqmm <- function(train_X, train_y, group_train, 
                       test_X, group_test, target_quantile = 0.5) {
  
  n_predictors <- ncol(train_X) - 1  # Number of non-intercept predictors
  
  if (n_predictors > 0) {
    predictor_names <- c("intercept", paste0("X", 1:n_predictors))
  } else {
    predictor_names <- "intercept"
  }
  #predictor_names <- c("intercept", paste0("X", 1:(ncol(train_X) - 1)))
  colnames(train_X) <- predictor_names
  colnames(test_X) <- predictor_names
  
  data <- data.frame(train_X)
  data$group <- as.factor(group_train)
  data$y <- matrix(train_y, ncol = 1)
  
  data_test <- data.frame(test_X)
  data_test$group <- factor(group_test, levels = levels(data$group))
  # Check if any NAs were created
  print(paste("NAs were created: ", sum(is.na(data_test$group))))
  
  # Remove rows with NA groups from test data
  complete_rows <- complete.cases(data_test$group)
  if (sum(!complete_rows) > 0) {
    print(paste("Removing", sum(!complete_rows), "rows with NA groups"))
    data_test <- data_test[complete_rows, , drop = FALSE]
    
    # Handle group_test based on its structure
    if (is.null(dim(group_test))) {
      # group_test is a vector
      group_test <- group_test[complete_rows]
    } else {
      # group_test is a matrix/data.frame
      group_test <- group_test[complete_rows, , drop = FALSE]
    }
  }
  
  # Verify NAs are gone
  print(paste("NAs remaining: ", sum(is.na(data_test$group))))
  
  formula_fixed <- as.formula(paste("y ~ -1 +", paste(predictor_names, collapse = " + ")))
  
  # Time fitting and prediction together
  timing <- system.time({
    fit.lqmm <- lqmm(
      fixed = formula_fixed,
      random = ~ 1,
      group = group,
      data = data,
      tau = target_quantile,
      nK = 30,
      type = "normal",
      control = list(verbose = FALSE, LP_tol_ll = 1e-6, LP_max_iter = 1000)
   
    #fit.boot <- boot(fit.lqmm, R = 50, startQR = TRUE)
       )
    print("fitting succesfull")
    
    X_test <- as.matrix(data_test[, predictor_names])
    fixed_pred <- X_test %*% fit.lqmm$theta_x
    
    re_train <- ranef(fit.lqmm)
    train_groups <- rownames(re_train)
    
    re_test <- rep(0, length(group_test))
    names(re_test) <- as.character(group_test)
    matching_indices <- as.character(group_test) %in% train_groups
    re_test[matching_indices] <- re_train[as.character(group_test[matching_indices]), 1]
    
    pred <- drop(fixed_pred + re_test)
    
    # bootstrap
    # re_b <- extractBoot(fit.boot, "random")
    
  })
  
  scale_param <- fit.lqmm$scale
  varcorr <- VarCorr(fit.lqmm)
  # Pack into a list
  # Unpack into top-level list
  hyper_params <- c(as.list(varcorr), list(noise_variance = scale_param))
  
  return(list(
    predictions = pred,
    random_effects = re_test,
    fixed_effects = drop(fixed_pred),
    fit_time = timing["elapsed"],
    hyper_params = hyper_params
  ))
}

#### BRMS and BAYESQR WITH BETTER GROUPS HANDLING ###

model_brms_quantile_v2 <- function(train_X, train_y, group_train,
                                test_X, group_test,
                                target_quantile = 0.5) {
  
  n_predictors <- ncol(train_X) - 1
  
  if (n_predictors > 0) {
    predictor_names <- c("Intercept", paste0("X", 1:n_predictors))
  } else {
    predictor_names <- "Intercept"
  }
  colnames(train_X) <- predictor_names
  data_train <- data.frame(train_X)
  data_train$y <- train_y
  
  # FIX: Handle arrays properly by converting to vectors first
  if (is.array(group_train) && length(dim(group_train)) == 1) {
    # Convert 1D array to vector
    group_train <- as.vector(group_train)
  }
  if (is.array(group_test) && length(dim(group_test)) == 1) {
    # Convert 1D array to vector  
    group_test <- as.vector(group_test)
  }
  
  # Now handle as before, but with proper vector handling
  if (is.null(dim(group_train))) {
    # Single grouping variable as vector
    group_train_df <- data.frame(group1 = group_train)
    group_test_df <- data.frame(group1 = group_test)
  } else {
    # Multiple grouping variables as data frame
    group_train_df <- as.data.frame(group_train)
    group_test_df <- as.data.frame(group_test)
    # Ensure same column names
    colnames(group_test_df) <- colnames(group_train_df)
  }
  
  group_vars <- names(group_train_df)
  
  # Add grouping variables to training data
  for (v in group_vars) {
    data_train[[v]] <- factor(group_train_df[[v]])
  }
  
  # Prepare test data
  colnames(test_X) <- predictor_names
  data_test <- data.frame(test_X)
  
  # Add grouping variables to test data with careful handling
  for (v in group_vars) {
    train_levels <- levels(data_train[[v]])
    test_values <- group_test_df[[v]]
    
    # Check for any issues with test values
    if (length(test_values) != nrow(data_test)) {
      stop(paste("Length mismatch: test_values has", length(test_values), 
                 "elements but test data has", nrow(data_test), "rows"))
    }
    
    # Check for NA/NaN values
    if (any(is.na(test_values)) || any(is.nan(test_values))) {
      warning(paste("Found NA/NaN values in test grouping variable:", v))
      # You might want to handle this by removing those rows or imputing
    }
    
    # Check for unseen levels
    test_values_clean <- test_values[!is.na(test_values) & !is.nan(test_values)]
    unseen_levels <- setdiff(unique(test_values_clean), train_levels)
    if (length(unseen_levels) > 0) {
      cat("Unseen levels in test data:", paste(head(unseen_levels, 5), collapse = ", "), "\n")
    }
    
    # Create factor with training levels
    data_test[[v]] <- factor(test_values, levels = train_levels)
    
    # Report NA creation
    na_count <- sum(is.na(data_test[[v]]))
    if (na_count > 0) {
      cat("Variable", v, "created", na_count, "NAs out of", length(test_values), "values\n")
    }
  }
  
  # Check final data integrity
  if (any(sapply(data_train, function(x) any(is.na(x) | is.nan(x))))) {
    stop("Training data contains NA/NaN values")
  }
  
  # Build formula
  random_effects <- paste0("(1 | ", group_vars, ")", collapse = " + ")
  formula_text <- paste("y ~ -1 +", paste(predictor_names, collapse = " + "), "+", random_effects)
  formula_fixed <- bf(as.formula(formula_text), quantile = target_quantile)
  
  # Fit model and predict with timeout
  fit_time_start <- Sys.time()
  timeout_seconds <- 3600 #00  # 1 hour
  
  # Try to fit with timeout
  fit_result <- tryCatch({
    # Use R.utils::withTimeout or base R with alarm
    R.utils::withTimeout({
      fit <- brm(
        formula = formula_fixed,
        data = data_train,
        family = asym_laplace(),
        chains = 2, iter = 2000, refresh = 0,
        control = list(adapt_delta = 0.95),
        seed = 42
      )
      
      # Handle prediction
      has_na_groups <- any(sapply(data_test[group_vars], function(x) any(is.na(x))))
      
      if (has_na_groups) {
        warning("Test data has NA values in grouping variables. Using population-level predictions.")
        pred <- fitted(fit, newdata = data_test, re_formula = NA)
      } else {
        pred <- fitted(fit, newdata = data_test, re_formula = NULL, allow_new_levels = TRUE)
      }
      
      list(fit = fit, pred = pred, success = TRUE)
    }, timeout = timeout_seconds, onTimeout = "error")
    
  }, TimeoutException = function(e) {
    warning(paste("brms fitting timed out after", timeout_seconds, "seconds"))
    list(success = FALSE)
  }, error = function(e) {
    warning(paste("brms fitting failed:", e$message))
    list(success = FALSE)
  })
  
  fit_time <- as.numeric(difftime(Sys.time(), fit_time_start, units = "secs"))
  
  # Return NAs if failed
  if (!fit_result$success) {
    n_test <- nrow(data_test)
    return(list(
      predictions = rep(NA_real_, n_test),
      se = rep(NA_real_, n_test),
      fit_time = fit_time,
      hyper_params = list(noise_variance = NA_real_),
      model = NULL,
      status = "timeout_or_error"
    ))
  }
  
  # Extract results if successful
  fit <- fit_result$fit
  pred <- fit_result$pred
  
  # Extract variance components
  vc <- VarCorr(fit)
  cov_pars <- lapply(group_vars, function(g) vc[[g]]$sd[, "Estimate"]^2)
  names(cov_pars) <- group_vars
  noise_variance <- vc$residual__$sd[, "Estimate"]^2
  hyper_params <- c(cov_pars, list(noise_variance = noise_variance))
  
  return(list(
    predictions = pred[, "Estimate"],
    se = pred[, "Est.Error"],
    fit_time = fit_time,
    hyper_params = hyper_params,
    model = fit,
    status = "success"
  ))
  
}
  

model_bayesqr_v2 <- function(train_X, train_y, group_train,
                          test_X, group_test,
                          target_quantile = 0.5,
                          ndraw = 5000, keep = 1) {
  
  # --- Prepare predictor matrix ---
  n_predictors <- ncol(train_X) - 1  # Number of non-intercept predictors
  
  if (n_predictors > 0) {
    predictor_names <- c("intercept", paste0("X", 1:n_predictors))
  } else {
    predictor_names <- "intercept"
  }
  colnames(train_X) <- predictor_names
  colnames(test_X) <- predictor_names
  data_train <- as.data.frame(train_X)
  data_test <- as.data.frame(test_X)
  
  # --- FIX: Handle arrays properly by converting to vectors first ---
  if (is.array(group_train) && length(dim(group_train)) == 1) {
    # Convert 1D array to vector
    group_train <- as.vector(group_train)
  }
  if (is.array(group_test) && length(dim(group_test)) == 1) {
    # Convert 1D array to vector  
    group_test <- as.vector(group_test)
  }
  
  # --- Grouping ---
  if (is.null(dim(group_train))) {
    group_train <- data.frame(group1 = factor(group_train))
    group_test <- data.frame(group1 = factor(group_test, levels = levels(group_train$group1)))
  } else {
    group_train <- as.data.frame(group_train)
    group_test <- as.data.frame(group_test)
    group_train[] <- lapply(group_train, factor)
    for (v in names(group_train)) {
      group_test[[v]] <- factor(group_test[[v]], levels = levels(group_train[[v]]))
    }
  }
  
  # Remove rows with NA groups from test data
  complete_rows <- complete.cases(group_test)
  data_test <- data_test[complete_rows, , drop = FALSE]
  group_test <- group_test[complete_rows, , drop = FALSE]
  
  group_vars <- names(group_train)
  
  # --- Add group dummies to design matrix ---
  group_dummies_train <- model.matrix(~ ., data = group_train)
  group_dummies_test <- model.matrix(~ ., data = group_test)
  
  # Remove the intercept column from group dummies to avoid collinearity
  group_dummies_train <- group_dummies_train[, -1, drop = FALSE]
  group_dummies_test <- group_dummies_test[, -1, drop = FALSE]
  

  # Single QR decomposition
  combined_matrix <- cbind(as.matrix(data_train), group_dummies_train)
  #qr_decomp <- qr(combined_matrix)
  #print(qr_decomp$rank)
  #if (qr_decomp$rank < ncol(combined_matrix)) {
   # print("Removing linearly dependent columns")
  #  keep_cols <- qr_decomp$pivot[1:qr_decomp$rank]
  #  X_train <- combined_matrix[, keep_cols, drop = FALSE]
    # Apply same column selection to test data
  #  X_test <- cbind(as.matrix(data_test), group_dummies_test)[, keep_cols, drop = FALSE]
  #} else {
  X_train <- as.data.frame(combined_matrix)
  X_test <- cbind(as.matrix(data_test), group_dummies_test)
  #}
  
  # --- Fit BayesQR model ---
  # After creating X_train and X_test
  print("Before fitting - dimensions:")
  print(paste("X_train:", paste(dim(X_train), collapse="x")))
  print(paste("X_test:", paste(dim(X_test), collapse="x")))
  
  # Prepare data for formula interface
  X_train$y <- train_y
  
  # Create formula
  predictor_vars <- setdiff(names(X_train), "y")
  formula_str <- paste("y ~", paste(predictor_vars, collapse = " + "), "- 1")  # -1 to remove default intercept since we have our own
  ndraw = 2000
  
  fit_time <- system.time({
      
      fit <- bayesQR(
        formula = as.formula(formula_str),
        data = X_train,
        normal.approx = TRUE, 
        quantile = target_quantile,
        ndraw = ndraw
      )
      print("bayesQR completed successfully!")
      # Get summary with burnin to access betadraw
      burnin <- ndraw %/% 2
      print(paste("burnin value:", burnin))
      print("About to call summary...")
      tryCatch({
        fit_summary <- summary(fit, burnin = burnin)
        print("2 - summary completed")
      }, error = function(e) {
        print(paste("summary() failed:", e$message))
        
        # Debug the fit object
        print("Checking fit object structure:")
        print(names(fit))
        
        return(list(error = "summary failed"))
      })
      beta_draws <- fit_summary[[1]]$betadraw
  
      # Posterior mean prediction
      beta_post_mean <- rowMeans(beta_draws)  # Note: rowMeans because betadraw is parameters x draws
      X_test_matrix <- as.matrix(X_test)
      
      preds <- as.numeric(X_test_matrix %*% beta_post_mean)
     
      # Posterior std deviation of predictions
      preds_samples <- t(beta_draws) %*% t(X_test_matrix)  # transpose beta_draws to get draws x parameters
      
      preds_se <- apply(preds_samples, 2, sd)
  })[["elapsed"]]
  
  return(list(
    predictions = preds,
    se = preds_se,
    fit_time = fit_time,
    model = fit
  ))
}



#### UTILS FOR VECCHIA ####
# 2) Vecchia adjacency function (works with 1D and 2D inputs)
vecchia_adj <- function(X, num_neighbors) {
  X <- as.matrix(X)
  N <- nrow(X)
  adj <- matrix(0L, N, N)
  
  for (i in 2:N) {
    previous <- 1:(i-1)
    diffs <- sweep(X[previous, , drop=FALSE], 2, X[i, ], "-")
    dists <- rowSums(diffs^2)
    neighbors <- previous[order(dists)[seq_len(min(num_neighbors, length(previous)))]]
    adj[neighbors, i] <- 1L
  }
  adj
}

# 3) Convert adjacency matrix to lattice predecessors matrix
lattice_predecessors <- function(adj, num_neighbors) {
  N <- ncol(adj)
  lattice <- matrix(-1L, nrow = N, ncol = num_neighbors + 1)
  lattice[, num_neighbors + 1] <- 1:N
  for (i in seq_len(N)) {
    preds <- which(adj[, i] == 1)
    if (length(preds) < num_neighbors) {
      preds <- c(preds, rep(-1L, num_neighbors - length(preds)))
    }
    lattice[i, 1:num_neighbors] <- preds
  }
  lattice
}

# 4) Convert lattice predecessors to edge index for Stan (1-based)
predecessors_to_edge_index <- function(predecessors) {
  N <- nrow(predecessors)
  num_neighbors <- ncol(predecessors) - 1
  edges_list <- vector("list", N)
  
  for (i in 1:N) {
    node <- predecessors[i, num_neighbors + 1]
    parents <- predecessors[i, 1:num_neighbors]
    parents <- parents[parents >= 0]  # remove padding
    if (length(parents) == 0) {
      edges_list[[i]] <- NULL
    } else {
      edges_list[[i]] <- cbind(parents, rep(node, length(parents)))
    }
  }
  edges <- do.call(rbind, edges_list)
  edges
}

#### ACTUAL MODEL ####

model_vecchia_gp <- function(train_X, train_y, test_X,
                                      target_quantile = 0.5,
                                      stan_model_path = "R/stan/asym_laplace_matern32_noncentered_sparse.stan",
                                      m = 5)
                             {
  
  X <- rbind(train_X, test_X)
  N <- nrow(X)
  D <- ncol(X)
  
  adj <- vecchia_adj(X, num_neighbors = m)
  lattice <- lattice_predecessors(adj, m)
  edge_index <- predecessors_to_edge_index(lattice)
  
  
  n_train = nrow(train_X)
  n_test = nrow(test_X)
  
  is_observed <- rep(0, N)
  is_observed[1:n_train] <- 1
  
  y_test <- rep(0, n_test)
  
  y_masked <- c(train_y, y_test)
  # y_masked[is_observed == 0] <- 0  # dummy value, won't be used in model
  
  
  data_list <- list(
    N = N,
    D = D,
    x_mat = X,
    y = y_masked,
    is_observed = is_observed,
    tau = target_quantile,
    epsilon = 1e-6,      # jitter
    num_edges = nrow(edge_index),
    edge_index = t(edge_index)
    
  )
  
  # 4. Compile and fit the Stan model --------------------------------------------
  
  mod <- cmdstan_model(stan_model_path, 
                       include_paths = gptools_include_path())
  fit_time <- system.time({
    fit <- mod$sample(
      data = data_list,
      chains = 2,
      iter_warmup = 200,
      iter_sampling = 200
    )})[["elapsed"]]
  print("here 1")
  f_samples <- fit$draws("f")
  print("here 2")
  f_samples_test <- f_samples[, , (n_train + 1):N]
  print("here 3")
  f_mean <- apply(f_samples, 3, mean)
  f_mean_test <- f_mean[(n_train + 1):N]
  print("here 4")
  
  
  return(list(
    predictions = f_mean_test,
    samples = f_samples_test,
    fit_time = fit_time))
}

## GP via STAN/MCMC ##
fit_gp_stan <- function(train_X, train_y, test_X, target_quantile, 
                        file_path = "R/stan/asym_laplace_matern32_noncentered.stan",
                        iter = 2000, warmup = 1000, chains = 4, seed = 42) {
  
  N <- length(train_y)
  data_list <- list(
    N = length(train_X),
    N_test = length(test_X),
    D = dim(train_X)[2],
    x = train_X,
    y = train_y,
    x_test = test_X,
    tau = target_quantile
  )
  stan_model <- rstan::stan_model(file = file_path)
  
  fit_time <- system.time({
    fit <- rstan::sampling(
      stan_model, 
      data = data_list, 
      iter = iter, 
      warmup = warmup, 
      chains = chains, 
      seed = seed,
      control = list(max_treedepth = 15)
    )
  })
  
  f_test_samples <- extract(fit)$f_test_mean  # matrix [num_draws x N_test]
  
  return(list(
    test_posterior_samples = f_test_samples,
    fit_time = fit_time))
  
}

### ----------------- EVALUATION METRICS ---------------- ###

quantile_score <- function(y, preds, quantile) {
  # Compute the quantile score (i.e. pin-ball loss).
  score <- numeric(length(y))
  index <- (preds - y) >= 0
  score[index] <- (preds - y)[index] * (1 - quantile)
  score[!index] <- -(preds - y)[!index] * quantile
  qs <- mean(score)
  
  return(qs)
}

interval_score <- function(y, pred_low, pred_up, alpha) {
  # Compute the interval score.
  
  dispersion <- mean(pred_up - pred_low)
  over_prediction <- 2 * mean((y - pred_up)[(y - pred_up) >= 0]) / alpha
  under_prediction <- 2 * mean((pred_low - y)[(pred_low - y) >= 0]) / alpha
  mis <- sum(c(dispersion, over_prediction, under_prediction), na.rm = TRUE)
  
  return(mis)
}

coverage_and_width <- function(y, pred_low, pred_up) {
  # Compute coverage and width of intervals formed by [pred_low, pred_up].
  
  coverage <- mean((y >= pred_low) & (y <= pred_up))
  width <- mean(pred_up - pred_low)
  
  return(c(coverage, width))
}



###### MIXED MODELS SIMULATION ######














