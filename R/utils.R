library(qgam)
library(cmdstanr)
library(gptoolsStan)
library(brms)


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

load_X_y_preprocessed <- function(dataset_name, dir) {
  path <- file.path(dir, paste0(dataset_name, ".txt"))
  if (!file.exists(path)) {
    stop(sprintf("Dataset '%s' not found at %s", dataset_name, path))
  }
  
  if (dataset_name %in% c("protein", "elevators")) {
    df <- read.table(path, sep = " ", header = FALSE)
  }
  else {
    df <- read.table(path, sep = " ", header = TRUE)
  }
  
  
  
  # Features & Response
  y <- df[[1]]
  X <- df[, -1]
  
  return(list(X = X, y = y))
}


load_cv_splits <- function(dataset_name, dir = "data/real_data", n_splits = 5) {
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




### ----------------- MODEL FITTING --------------------- ###


### LQMM (Geraci) ###
model_lqmm <- function(train_X, train_y, group_train, 
                       test_X, group_test, target_quantile = 0.5) {
  
  
  predictor_names <- c("intercept", paste0("X", 1:(ncol(train_X) - 1)))
  colnames(train_X) <- predictor_names
  colnames(test_X) <- predictor_names
  
  data <- data.frame(train_X)
  data$group <- as.factor(group_train)
  data$y <- matrix(train_y, ncol = 1)
  
  data_test <- data.frame(test_X)
  data_test$group <- factor(group_test, levels = levels(data$group))
  
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
      control = list(verbose = TRUE, LP_tol_ll = 1e-6, LP_max_iter = 1000)
    )
    
    X_test <- as.matrix(data_test[, predictor_names])
    fixed_pred <- X_test %*% fit.lqmm$theta_x
    
    re_train <- ranef(fit.lqmm)
    train_groups <- rownames(re_train)
    
    re_test <- rep(0, length(group_test))
    names(re_test) <- as.character(group_test)
    matching_indices <- as.character(group_test) %in% train_groups
    re_test[matching_indices] <- re_train[as.character(group_test[matching_indices]), 1]
    
    pred <- drop(fixed_pred + re_test)
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

### BRMS ###
model_brms_quantile <- function(train_X, train_y, group_train,
                                test_X, group_test,
                                target_quantile = 0.5) {
  
  # Prepare training data
  predictor_names <- c("intercept", paste0("X", 1:(ncol(train_X) - 1)))
  colnames(train_X) <- predictor_names
  data_train <- data.frame(train_X)
  data_train$y <- train_y
  data_train$group <- factor(group_train)
  
  # Prepare test data
  colnames(test_X) <- predictor_names
  data_test <- data.frame(test_X)
  data_test$group <- factor(group_test, levels = levels(data_train$group))  # align levels
  
  # Build formula: no intercept if it's already in X
  formula_fixed <- bf(
    as.formula(
      paste("y ~ -1 +", paste(predictor_names, collapse = " + "), "+ (1 | group)")
    ),
    quantile = target_quantile
  )
  
  # Fit model with timing
  fit_time <- system.time({
    fit <- brm(
      formula = formula_fixed,
      data = data_train,
      family = asym_laplace(),
      chains = 2, iter = 2000, refresh = 0,
      control = list(adapt_delta = 0.95),
      seed = 42
    )
    
    # Predict on test set, including random effects (level 1)
    pred <- fitted(fit, newdata = data_test, re_formula = NULL)
  })[["elapsed"]]
  
  # Extract hyperparameters
  cov_pars <- VarCorr(fit)$group$sd[, "Estimate"] # Random effect SD
  noise_variance <- VarCorr(fit)$residual__$sd[, "Estimate"]    # Residual scale
  
  # Unpack into top-level list
  hyper_params <- c(as.list(cov_pars), list(noise_variance = noise_variance))
  
  return(list(
    predictions = pred[, "Estimate"],
    se = pred[, "Est.Error"],
    fit_time = fit_time,
    hyper_params = hyper_params,
    model = fit  # optional: for future inspection
  ))
}



## QGAM (Fasiolo) ##

model_qgam <- function(train_X, train_y, test_X, target_quantile = 0.5, smooth_term = 20) {
  # Function to fit the QGAM model and predict on the test set

  # Dynamically create the training data frame (with y as the response variable)
  train_data <- data.frame(y = train_y)
  # Create a data frame for the test data
  test_data <- as.data.frame(test_X)  # Directly convert test_X into a data frame
  # Rename the columns in the test data
  colnames(test_data) <- paste0("X", 1:ncol(test_data))
  
  # Add each column of X as separate predictors in the data frame
  for (i in 1:ncol(train_X)) {
    train_data[[paste0("X", i)]] <- train_X[, i]
  }
  
  # Dynamically create the formula for qgam model
  formula_parts <- sapply(1:ncol(train_X), function(i) {
    paste0("s(X", i, ", k = smooth_term, bs = 'ad')")
  })
  formula <- as.formula(paste("y ~", paste(formula_parts, collapse = " + ")))
  # print(paste0("formula qgam: ", formula))
  
  # Fit the QGAM model
  fit_time <- system.time({
    fit <- qgam(formula, data = train_data, qu = target_quantile)
    # Make predictions on the test set
    pred <- predict(fit, newdata = test_data, se = TRUE)
  })[["elapsed"]]
  
  
  # Return predictions along with standard errors
  return(list(
    predictions = pred$fit,
    se = pred$se.fit,
    fit_time = fit_time))
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














