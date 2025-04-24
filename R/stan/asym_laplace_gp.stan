functions {
  real asym_laplace_lpdf(real y, real mu, real tau, real sigma) {
    real z = (y - mu) / sigma;
    real log_scale = log(sigma);
    return log(tau * (1 - tau)) - log_scale -
           (z >= 0 ? tau * z : (tau - 1) * z);
  }
}

data {
  int<lower=1> N;                 // number of observations
  vector[N] x;                    // input locations (1D for simplicity)
  vector[N] y;                    // target values
  real<lower=0, upper=1> tau;     // quantile level
}

transformed data {
  matrix[N, N] D;
  for (i in 1:N)
    for (j in 1:N)
      D[i, j] = square(x[i] - x[j]);  // squared distance
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> variance;
  real<lower=0> sigma;           // scale of asymmetric Laplace
  vector[N] f;                   // latent GP function
}

model {
  // GP prior
  matrix[N, N] K = variance * exp(-0.5 * D / square(lengthscale));
  K += diag_matrix(rep_vector(1e-6, N)); // jitter for stability
  f ~ multi_normal(rep_vector(0, N), K);

  // Likelihood
  for (n in 1:N)
    target += asym_laplace_lpdf(y[n] | f[n], tau, sigma);
}
