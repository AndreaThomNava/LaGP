library(MASS)
library(Matrix)
library(cmdstanr)
library(ggplot2)
library(gptoolsStan)

# 1) Simulate GP Poisson data with Matern kernel (nu=1.5)
simulate_gp_poisson <- function(X, sigma=1, length_scale=10, mu=0) {
  X <- as.matrix(X)
  N <- nrow(X)
  
  dists <- as.matrix(dist(X))^2
  dists_sqrt <- sqrt(dists)
  
  K <- sigma^2 * (1 + sqrt(3) * dists_sqrt / length_scale) * exp(-sqrt(3) * dists_sqrt / length_scale)
  diag(K) <- diag(K) + 1e-6  # nugget for stability
  
  eta <- as.numeric(MASS::mvrnorm(1, mu = rep(mu, N), Sigma = K))
  rate <- exp(eta)
  y <- rpois(N, lambda = rate)
  
  list(x = X, y = y, eta = eta, rate = rate, sigma=sigma, length_scale=length_scale)
}

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

# 5) Main workflow

set.seed(42)
N <- 64
X <- matrix(seq_len(N), ncol=1)
num_neighbors <- 20

# Simulate data
sample <- simulate_gp_poisson(X, sigma=1, length_scale=10, mu=0)

# Build Vecchia adjacency and lattice
adj <- vecchia_adj(sample$x, num_neighbors)
lattice <- lattice_predecessors(adj, num_neighbors)
edge_index <- predecessors_to_edge_index(lattice)

# Prepare data list for Stan
stan_data <- list(
  epsilon = 1e-6,      # small nugget, adjust as needed
  n = N,
  y = as.integer(sample$y),
  X = sample$x,
  num_edges = nrow(edge_index),
  edge_index = t(edge_index) # transpose to 2 x num_edges for Stan
)

# Compile the Stan model (adjust path accordingly)
model <- cmdstan_model(
  stan_file = "stan/poisson_noncentered.stan",
  include_paths = gptools_include_path()
)

# Sample from the model
fit <- model$sample(data = stan_data, chains = 1, iter_warmup = 100, iter_sampling = 100)

# Extract posterior samples of latent eta (log-rate)
eta_post <- fit$draws("eta", format = "matrix")

# Plot true vs inferred rates
library(ggplot2)
eta_mean <- apply(eta_post, 2, mean)
rate_est <- exp(eta_mean)

df <- data.frame(
  x = sample$x[,1],
  y_obs = sample$y,
  rate_true = sample$rate,
  rate_est = rate_est
)

ggplot(df) +
  geom_point(aes(x = x, y = y_obs), color = "black", alpha=0.6, size=1.5) +
  geom_line(aes(x = x, y = rate_true), color = "blue", size = 1, linetype = "dashed") +
  geom_line(aes(x = x, y = rate_est), color = "red", size = 1) +
  labs(title = "GP Poisson Model: True vs Inferred Rate",
       y = "Counts / Rate",
       x = "X") +
  theme_minimal() +
  theme(legend.position = "none")



library(cmdstanr)
library(gptoolsStan)
source("utils.R")

# 1. Simulate data -------------------------------------------------------------

set.seed(124)
N <- 500
x <- matrix(seq(0, 10, length.out = N), ncol = 1)
tau <- 0.8  # median quantile

# True function (GP sample, for example)
true_f <- sin(x[,1])  
sigma_obs <- 0.3

error <- ralaplace(N, tau, sigma_obs)
y <- true_f + error

# 2. Build Vecchia graph -------------------------------------------------------

num_neighbors <- 5
adj <- vecchia_adj(x, num_neighbors = num_neighbors)
lattice <- lattice_predecessors(adj, num_neighbors)
edge_index <- predecessors_to_edge_index(lattice)

# 3. Prepare data list for Stan ------------------------------------------------

# Random split: 75% train, 25% test
train_frac <- 0.75
train_size <- floor(train_frac * N)

indices <- sample(1:N)  # randomly shuffle indices
train_idx <- sort(indices[1:train_size])  # optional: sort for plotting
test_idx  <- sort(indices[(train_size + 1):N])

is_observed <- rep(0, N)
is_observed[train_idx] <- 1

y_masked <- y
y_masked[is_observed == 0] <- 0  # dummy value, won't be used in model


data_list <- list(
  N = N,
  D = ncol(x),
  x_mat = x,
  y = y_masked,
  is_observed = is_observed,
  tau = tau,
  epsilon = 1e-6,      # jitter
  num_edges = nrow(edge_index),
  edge_index = t(edge_index)
  
)

# 4. Compile and fit the Stan model --------------------------------------------

mod <- cmdstan_model("stan/asym_laplace_matern32_noncentered_sparse.stan", 
                     include_paths = gptools_include_path())

fit <- mod$sample(
  data = data_list,
  chains = 2,
  iter_warmup = 200,
  iter_sampling = 200
)

print(fit$summary("f"))

f_samples <- fit$draws("f")
f_mean <- apply(f_samples, 3, mean)
f_mean

N <- length(f_mean)
df <- data.frame(
  x = 1:N,  # or your original x locations
  f_true = true_f,  # your simulated latent f
  f_post_mean = f_mean
)


# Example: create separate data frames for train and test points
df_train <- data.frame(
  x = train_idx,
  y = y_masked[train_idx],       # observed y for training points
  f_true = true_f[train_idx],
  f_post_mean = f_mean[train_idx],
  data_type = "Train"
)

df_test <- data.frame(
  x = test_idx,
  y = y[test_idx],                        # no observed y for test
  f_true = true_f[test_idx],
  f_post_mean = f_mean[test_idx],
  data_type = "Test"
)

# Combine for plotting
df_all <- rbind(df_train, df_test)

ggplot(df_all, aes(x = x)) +
  # Plot observed y only for train
  geom_point(data = subset(df_all, data_type == "Train"), aes(y = y), color = "black", size = 1) +
  
  geom_point(data = subset(df_all, data_type == "Test"), aes(y = y), color = "green", size = 1) +
  
  # True latent for all points
  geom_line(aes(y = f_true, group = 1), color = "blue", size = 1, linetype = "dashed") +
  
  # Posterior mean for all points
  geom_line(aes(y = f_post_mean, group = 1), color = "red", size = 1) +
  
  labs(y = "Latent function f", title = "Train vs Test: True vs Posterior Mean") +
  theme_minimal()

