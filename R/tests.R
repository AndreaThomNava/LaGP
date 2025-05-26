### check functions/ tests ###

source("R/utils.R")

### LOAD CONFIGS ###
config_path <- "configs/config_test.yaml"
configs <- load_config(config_path)
class(configs) # list of lists
configs[1] # simulation
configs[2] # gp parameters
configs[3] # output

configs$replicate


### LOAD DATA ###
likelihood <- "gaussian"
n_groups <- 200
group_size <- 10
replicate <- 2
randeff <- configs$randeff
# Load the dataset
data <- load_data(
  randeff = randeff,
  likelihood = likelihood,
  n_groups = n_groups,
  group_size = group_size,
  replicate = replicate
)
names(data)

train_X <- data$X_train
train_y <- data$y_train
test_X <- data$X_test
test_y <- data$y_test
group_train <- data$group_train
group_test <- data$group_test
eps_test <- data$eps_test 
class(data)


#####

# Compute true latent quantile
is_heteroscedastic <- grepl("heteroscedastic", likelihood)
g <- if (is_heteroscedastic) data$g else NULL
target_quantile <- configs$target_quantile
test_true_latent_quantile <- obtain_quantile(
  eps = eps_test,
  noise = likelihood,
  pars = configs$data_generation$pars,
  target_quantile = target_quantile,
  g = g
)
test_true_latent_quantile

# test LQMM and BRMS
library("lqmm")


# Test example
set.seed(123)

M <- 50
n <- 10
test <- data.frame(x = runif(n*M,0,1), group = rep(1:M,each=n))
test$y <- 10*test$x + rep(rnorm(M, 0, 2), each = n) + rchisq(n*M, 3)

# Assign column names if train_X is a matrix

predictor_names <- c("intercept", paste0("X", 1:(ncol(train_X)-1)))
colnames(train_X) <- predictor_names
colnames(test_X) <- predictor_names
# Combine all into a data frame
data <- data.frame(train_X)
data$group <- as.factor(group_train)
data$y <- matrix(train_y, ncol = 1)

data_test <- data.frame(test_X)
data_test$group <- factor(group_test, levels = levels(data$group))  # important!
# data_test$y <- matrix(test_y, ncol = 1)

# Construct formula manually
formula_fixed <- as.formula(paste("y ~ -1 +", paste(predictor_names, collapse = " + ")))

# Fit the model
fit.lqmm <- lqmm(
  fixed = formula_fixed,
  random = ~ 1,
  group = group,
  data = data,
  tau = 0.8,
  nK = 11,
  type = "normal"
)

fit.lqmm

# Extract estimates
VarCorr(fit.lqmm)
coef(fit.lqmm)

re <- ranef(fit.lqmm)
re
str(fit.lqmm)

fit.lqmm$scale
VarCorr(fit.lqmm)

# 1. Fixed effects prediction
X_test <- as.matrix(data_test[, predictor_names])
fixed_pred <- X_test %*% fit.lqmm$theta_x  # (n_test x 1) matrix

# Get group labels from training
re_train <- ranef(fit.lqmm)
train_groups <- rownames(re_train)
re_test <- rep(0, length(group_test))
names(re_test) <- as.character(group_test)
matching_indices <- as.character(group_test) %in% train_groups
re_test[matching_indices] <- re_train[as.character(group_test[matching_indices]),1]

# 5. Final prediction = fixed + random
pred <- drop(fixed_pred + re_test)
class(pred)

rownames(eps_test) <- group_test
rownames(eps_test)
eps_test

re_test

# in the comparison can remove where re_test is zero
mean((eps_test[re_test != 0] - re_test[re_test!= 0])**2)


res <- model_brms_quantile(train_X = train_X, train_y = train_y,
           group_train = group_train, test_X = test_X, group_test = group_test,
           target_quantile = target_quantile)


fit.test <- lqmm(
  fixed = y ~ X2,
  random = ~ 1,
  group = group,
  data = data,
  tau = 0.5,
  nK = 5,      # fewer quadrature nodes
  type = "normal"
)

ranef(fit.test)  # should now work

## Orthodont data
data(Orthodont)

# Random intercept model
fitOi.lqmm <- lqmm(distance ~ age, random = ~ 1, group = Subject,
                   tau = c(0.1,0.5,0.9), data = Orthodont)
coef(fitOi.lqmm)
ranef(fitOi.lqmm)


# Random slope model
fitOs.lqmm <- lqmm(distance ~ age, random = ~ age, group = Subject,
                   tau = c(0.1,0.5,0.9), cov = "pdDiag", data = Orthodont)



ranef(fitOs.lqmm)





# fit QGAM 
res_qgam <- model_qgam(train_X = train_X, train_y = train_y, test_X = test_X,
                       target_quantile = target_quantile, smooth_term = 5)
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






