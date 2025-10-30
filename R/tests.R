### check functions/ tests ###

source("R/utils.R")

## load data from sim


### fit bayesqr
randeff <- "One_random_effect"
likelihood <- "gaussian"
n_groups <- 10
group_size <- 100
replicate <- 1
is_heteroscedastic <- FALSE

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

# For one grouped random effect
expected_num_cov_pars <- 1
expected_cov_par_names <- c("Group_1")  # Define expected names

# Define a "safe empty" hyper_params list to use on failures
empty_hyper_params <- c(
  setNames(rep(NA_real_, expected_num_cov_pars), expected_cov_par_names),
  list(noise_variance = NA_real_)
)


# --- Prepare predictor matrix ---
n_predictors <- ncol(train_X) - 1  # Number of non-intercept predictors

n_predictors

if (n_predictors > 0) {
  predictor_names <- c("intercept", paste0("X", 1:n_predictors))
} else {
  predictor_names <- "intercept"
}
colnames(train_X) <- predictor_names
colnames(test_X) <- predictor_names
data_train <- as.data.frame(train_X)
data_test <- as.data.frame(test_X)

data_train
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

target_quantile <- 0.5
fit <- bayesQR(
  formula = as.formula(formula_str),
  data = X_train,
  normal.approx = TRUE, 
  quantile = target_quantile,
  ndraw = ndraw
)

class(X_train)





fit_time <- system.time({
  
  fit <- bayesQR(
    formula = as.formula(formula_str),
    data = X_train,
    normal.approx = FALSE, 
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
  print("2")
  # Posterior mean prediction
  beta_post_mean <- rowMeans(beta_draws)  # Note: rowMeans because betadraw is parameters x draws
  X_test_matrix <- as.matrix(X_test)
  print("3")
  preds <- as.numeric(X_test_matrix %*% beta_post_mean)
  print("4")
  # Posterior std deviation of predictions
  preds_samples <- t(beta_draws) %*% t(X_test_matrix)  # transpose beta_draws to get draws x parameters
  print("5")
  preds_se <- apply(preds_samples, 2, sd)
})[["elapsed"]]

beta_draws

fit_summary[2]

fit_summary[[1]]$sigmadraw[1]



tryCatch({
  # For grouped random effects, look for variance parameters
  # bayesQR typically stores these in the fit object or summary
  
  # Method 1: Check if sigma draws are available
  if ("sigmadraw" %in% names(fit_summary[[1]])) {
    sigma_draws <- fit_summary[[1]]$sigmadraw
    noise_variance <- mean(sigma_draws^2)  # Convert to variance
  } else {
    # Method 2: Extract from model coefficients if available
    # Look for variance parameters in the coefficient names
    coef_names <- rownames(beta_draws)
    var_indices <- grep("sigma|var|tau", coef_names, ignore.case = TRUE)
    
    if (length(var_indices) > 0) {
      # Extract variance components from the draws
      var_draws <- beta_draws[var_indices, , drop = FALSE]
      var_estimates <- rowMeans(var_draws^2)  # Convert to variance if needed
      names(var_estimates) <- coef_names[var_indices]
      
      # Separate group variances from noise variance
      group_vars <- var_estimates[!grepl("residual|sigma", names(var_estimates))]
      noise_variance <- var_estimates[grepl("residual|sigma", names(var_estimates))]
      
      if (length(noise_variance) == 0) {
        noise_variance <- NA  # Set to NA if not found
      }
      
      hyper_params <- list(
        group_variances = group_vars,
        noise_variance = as.numeric(noise_variance)
      )
    } else {
      # bayesQR might not have explicit variance components
      # In this case, return NA or estimate from residuals
      hyper_params <- list(
        group_variances = NA,
        noise_variance = NA
      )
    }
  }
}, error = function(e) {
  print(paste("Variance extraction failed:", e$message))
  hyper_params <- list(
    group_variances = NA,
    noise_variance = NA
  )
})

hyper_params



res <- model_bayesqr_v2(train_X = train_X, train_y = train_y,
                        group_train = group_train,
                        test_X = test_X, group_test = group_test,
                        target_quantile = target_quantile)

latent_pred <- res$predictions
hyper_params <- res$hyper_params
fit_time <- res$fit_time
































# load data
df_name = "cars_crossed"
DIR = "data/real_data_mm/"
data <- load_X_y_preprocessed(dataset_name = df_name, dir = DIR)
X <- data$X
group_data <- data$group_data
y <- data$Y
n_splits <- 2
folds <- load_cv_splits(dataset_name = df_name, dir = DIR,
                        n_splits = n_splits)
train_idx <- unlist(fold$train_idx)[1:5000]
test_idx <- unlist(fold$test_idx)[1:5000]

train_X <- X[train_idx, , drop = FALSE]
test_X  <- X[test_idx, , drop = FALSE]
group_train <- group_data[train_idx, , drop = FALSE]
group_test <- group_data[test_idx, ,drop = FALSE]
# scale y
y <- (y - mean(y)) / sd(y)

train_y <- y[train_idx]
test_y  <- y[test_idx]
# try model.matrix
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
qr_decomp <- qr(combined_matrix)
print(qr_decomp$rank)
if (qr_decomp$rank < ncol(combined_matrix)) {
  print("Removing linearly dependent columns")
  keep_cols <- qr_decomp$pivot[1:qr_decomp$rank]
  X_train <- combined_matrix[, keep_cols, drop = FALSE]
  # Apply same column selection to test data
  X_test <- cbind(as.matrix(data_test), group_dummies_test)[, keep_cols, drop = FALSE]
} else {
  X_train <- combined_matrix
  X_test <- cbind(as.matrix(data_test), group_dummies_test)
}

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
ndraw = 500


















### LOAD CONFIGS ###
config_path <- "configs/config_data_generation.yaml"
configs <- load_config(config_path)
class(configs) # list of lists
configs[1] # simulation
configs[2] # gp parameters
configs[3] # output

configs$simulation$replicates

configs_sim <- load_config("configs/config_run_simulation.yaml")
train_split <- configs_sim$train_split


### LOAD DATA ###
likelihood <- "gaussian"
sample_size <- 300
input_dim <- 2
replicate <- 2

data <- load_data(likelihood, sample_size, input_dim, replicate, train_split) 
class(data)
f <- data$f
n <- length(f)
data$X_train

#####

pars <- configs$simulation$pars
target_quantile <- 0.9
latent_quantiles <- obtain_quantile(f, likelihood, pars, target_quantile)
latent_quantiles
train_index <- floor(n * train_split)
train_latent_quantiles <- latent_quantiles[1:train_index]
test_latent_quantiles <- latent_quantiles[(train_index + 1): n]

plot(data$X_train, train_latent_quantiles)
points(x = data$X_train, y = data$y_train, col = "green")
# noise --> looks homoscedastic and heavy tailed --> ALD!
plot(data$X_train, train_latent_quantiles -data$y_train)

### DATA PREP ###
X_train <- data$X_train
y_train <- data$y_train
X_test <- data$X_test
y_test <- data$y_test


# fit QGAM 
res_qgam <- model_qgam(train_X = X_train, train_y = y_train, test_X = X_test,
           quantile = target_quantile, smooth_term = 20)
preds <- res_qgam$predictions
pred_std <- res_qgam$se
fit_time <- res_qgam$fit_time
fit_time

# Construct PIs
alpha <- configs_sim$alpha
t <- qnorm(1 - (1-alpha)/2)
t

pred_low <- preds - t * pred_std
pred_up <- preds + t * pred_std


# Compute metrics 
qs <- quantile_score(y_test, preds, target_quantile)
qs
# on true latent quantile
is <- interval_score(test_latent_quantiles, pred_low, pred_up, alpha)
is
cov_wid <- coverage_and_width(test_latent_quantiles, pred_low, pred_up)
cov <- cov_wid[1]
cov
wid <- cov_wid[2]
wid

# Sort X_test and preds accordingly
ord <- order(X_test)
X_test_sorted <- X_test[ord]
preds_sorted <- preds[ord]
# Sort interval bounds accordingly
preds_lower_sorted <- pred_low[ord]
preds_upper_sorted <- pred_up[ord]

# Plot as before
plot(X_test_sorted, preds_sorted, 
     col = "black", type = "l", lwd = 2,
     xlab = "X", ylab = "Quantile value", 
     main = paste0("Predicted Quantile (q = ", target_quantile, ")"),
     ylim = range(c(preds_lower_sorted, preds_upper_sorted,
                    train_latent_quantiles, test_latent_quantiles)))

# Add shaded interval
polygon(c(X_test_sorted, rev(X_test_sorted)),
        c(pred_lower_sorted, rev(pred_upper_sorted)),
        col = adjustcolor("grey", alpha.f = 0.4), border = NA)

# Add points
points(X_train, train_latent_quantiles, 
       col = "purple", pch = 16, cex = 0.7)
points(X_train, y_train, 
       col = "green", pch = 16, cex = 0.7)
points(X_test, test_latent_quantiles, 
       col = "red", pch = 17, cex = 0.7)

# Legend
legend("topright", 
       legend = c("Predicted (test)", "Latent (train)", "Latent (test)", "Prediction interval"),
       col = c("black", "purple", "red", "grey"),
       lwd = c(2, NA, NA, NA),
       pch = c(NA, 16, 17, 15),
       pt.cex = c(NA, 0.7, 0.7, 1),
       bty = "n")




## FIT STAN/MCMC MODEL ##
fit_gp_stan(X_train, y_train, X_test, target_quantile, 
                        file_path = "R/stan/asym_laplace_matern32_noncentered.stan",
                        iter = 2000, warmup = 1000, chains = 4, seed = 42) 


df_name <- "house"
DIR <- "data/real_data/"
data <- load_X_y_preprocessed(dataset_name = df_name, dir = DIR)
X <- data$X
y <- data$y
y

folds <- load_cv_splits(dataset_name = df_name, dir = DIR,
                        n_splits = n_splits)


