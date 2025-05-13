library(qgam)


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


# Load configuration from a YAML file
load_config <- function(config_path) {
  config <- yaml::read_yaml(config_path)
  return(config)
}

# load data and train and test split
load_data <- function(likelihood, sample_size, input_dim, replicate, train_split, file_path = NULL) {
  folder_name <- paste0(sample_size, "_", input_dim)
  file_name <- paste0("data_replicate_", replicate, ".npz")
  
  if (is.null(file_path)) {
    file_path <- file.path("data", "simulated_data", folder_name, likelihood, file_name)
  }
  
  np <- reticulate::import("numpy")
  data <- np$load(file_path)
  
  X <- data[["X"]]
  y <- data[["y"]]
  f <- data[["f"]]
  
  n <- nrow(X)
  n_train <- floor(train_split * n)
  X_train <- X[1:n_train, , drop = FALSE]
  y_train <- y[1:n_train]
  X_test <- X[(n_train + 1):n, , drop = FALSE]
  y_test <- y[(n_train + 1):n]
  
  list(f = f, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
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
  
  df <- read.table(path, sep = " ", header = TRUE)
  
  # Features & Response
  y <- df[[1]]
  X <- df[, -1]
  
  return(list(X = X, y = y))
}


load_cv_splits <- function(dataset_name, dir = "data/real_data_splits", n_splits = 5) {
  path <- file.path(dir, paste0(dataset_name, "_cv", n_splits, "_splits.json"))
  
  if (!file.exists(path)) {
    stop(sprintf("Split file for dataset '%s' not found at %s", dataset_name, path))
  }
  
  splits <- jsonlite::fromJSON(path)
  
  return(splits)
}


# Recover true quantile
obtain_quantile <- function(f, noise, pars, target_quantile, g = NULL) {
  n <- length(f)
  if (!is.null(g)) {
    stopifnot(length(g) == n)
  }
  
  if (noise == "gaussian") {
    scale <- if (!is.null(g)) g else pars[["gaussian"]][["scale"]]
    delta <- qnorm(target_quantile) * scale
    
  } else if (noise == "ald") {
    q <- pars[["ald"]][["q"]]
    scale <- if (!is.null(g)) g else pars[["ald"]][["scale"]]
    delta <- quantile_func_asym_laplace(target_quantile, q, 1) * scale
    
  } else if (noise == "t") {
    df <- pars[["t"]][["df"]]
    scale <- if (!is.null(g)) g else pars[["t"]][["scale"]]
    delta <- qt(target_quantile, df = df) * scale
    
  } else if (noise == "chi") {
    df <- pars[["chi"]][["df"]]
    scale <- if (!is.null(g)) g else pars[["chi"]][["scale"]]
    delta <- qchisq(target_quantile, df = df) * scale
    
  } else {
    stop(paste("Unsupported noise model:", noise))
  }
  
  return(f + delta)
}



### ----------------- MODEL FITTING --------------------- ###

## QGAM (Fasiolo) ##

model_qgam <- function(train_X, train_y, test_X, target_quantile = 0.5, smooth_term = 20) {
  # Function to fit the QGAM model and predict on the test set

  # Dynamically create the training data frame (with y as the response variable)
  train_data <- data.frame(y = train_y)
  
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
  })[["elapsed"]]
  
  # Create a data frame for the test data
  test_data <- as.data.frame(test_X)  # Directly convert test_X into a data frame
  
  # Rename the columns in the test data
  colnames(test_data) <- paste0("X", 1:ncol(test_data))
  
  # Make predictions on the test set
  pred <- predict(fit, newdata = test_data, se = TRUE)
  
  # Return predictions along with standard errors
  return(list(
    predictions = pred$fit,
    se = pred$se.fit,
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


















