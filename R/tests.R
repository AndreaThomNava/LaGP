### check functions/ tests ###

source("R/utils.R")

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






