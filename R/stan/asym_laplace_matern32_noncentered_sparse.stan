functions {
  real asym_laplace_lpdf(real y, real mu, real tau, real sigma) {
    real z = (y - mu) / sigma;
    return log(tau * (1 - tau)) - log(sigma) -
           (z >= 0 ? tau * z : (tau - 1) * z);
  }

  #include gptools/util.stan
  #include gptools/graph.stan
}

data {
  real<lower=0> epsilon;       // jitter for numerical stability

  // Observed data
  int<lower=1> N;
  int<lower=1> D;
  matrix[N, D] x_mat;
  vector[N] y;
  array[N] int<lower=0, upper=1> is_observed;
  real<lower=0, upper=1> tau;

  // Graph structure for sparse GP
  int<lower=1> num_edges;
  array[2, num_edges] int edge_index;
}

transformed data {
  array[N] int degrees = out_degrees(N, edge_index);
  
  array[N] vector[D] x;
  for (i in 1:N) {
    x[i] = to_vector(row(x_mat, i));
  }
}

parameters {
  
   // GP hyperparameters (now inferred)
  real<lower=0> sigma;
  real<lower=0> lengthscale;
  
  vector[N] z;           // non-centered GP latent
  real<lower=0> obs_sigma;  // scale for asym Laplace likelihood
}

transformed parameters {
  vector[N] f;
  f = gp_inv_graph_matern32_cov(
    z, rep_vector(0, N), x, sigma, lengthscale,
    edge_index, degrees, epsilon
  );
  
}

model {
  // Priors
  z ~ normal(0, 1);
  obs_sigma ~ lognormal(0, 1);     // scale of ALD
  sigma ~ lognormal(0, 1);         // marginal std dev of GP
  lengthscale ~ lognormal(0, 1);   // smoothness scale of GP

  // Likelihood
  for (n in 1:N) {
    if (is_observed[n])
      target += asym_laplace_lpdf(y[n] | f[n], tau, obs_sigma);
  }
}
