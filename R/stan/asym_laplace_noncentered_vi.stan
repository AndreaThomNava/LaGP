functions {
  real asym_laplace_lpdf(real y, real mu, real tau, real sigma) {
    real z = (y - mu) / sigma;
    return log(tau * (1 - tau)) - log(sigma) - 
           (z >= 0 ? tau * z : (tau - 1) * z);
  }
}

data {
  int<lower=1> N;                 // number of observations
  vector[N] x;                    // input locations
  vector[N] y;                    // target values
  real<lower=0, upper=1> tau;     // quantile level
}

transformed data {
  matrix[N, N] D;
  for (i in 1:N)
    for (j in 1:N)
      D[i, j] = square(x[i] - x[j]);  // squared distance matrix
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> variance;
  real<lower=0> sigma;
  vector[N] z; // standard normal for non-centered GP
}

transformed parameters {
  matrix[N, N] K = variance * exp(-0.5 * D / square(lengthscale));
  matrix[N, N] L_K = cholesky_decompose(K + diag_matrix(rep_vector(1e-6, N)));
  vector[N] f = L_K * z;
}

model {
  // Non-centered: standard normal prior on z
  z ~ normal(0, 1);

  // Optional weak priors (feel free to remove or adjust)
  lengthscale ~ lognormal(0, 1); 
  variance ~ lognormal(0, 1);
  sigma ~ lognormal(0, 1);

  for (n in 1:N)
    target += asym_laplace_lpdf(y[n] | f[n], tau, sigma);
}
