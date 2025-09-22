### check functions/ tests ###

source("R/utils.R")


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


