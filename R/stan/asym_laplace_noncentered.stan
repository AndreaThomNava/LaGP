functions {
  real asym_laplace_lpdf(real y, real mu, real tau, real sigma) {
    real z = (y - mu) / sigma;
    return log(tau * (1 - tau)) - log(sigma) -
           (z >= 0 ? tau * z : (tau - 1) * z);
  }
}

data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
  real<lower=0, upper=1> tau;
}

transformed data {
  matrix[N, N] D;
  for (i in 1:N)
    for (j in 1:N)
      D[i, j] = square(x[i] - x[j]);
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> variance;
  real<lower=0> sigma;
  vector[N] z;  // non-centered latent
}

transformed parameters {
  matrix[N, N] K = variance * exp(-0.5 * D / square(lengthscale));
  matrix[N, N] L_K = cholesky_decompose(K + diag_matrix(rep_vector(1e-6, N)));
  vector[N] f = L_K * z;
}

model {
  z ~ normal(0, 1);

  lengthscale ~ lognormal(0, 1);
  variance ~ lognormal(0, 1);
  sigma ~ lognormal(0, 1);

  for (n in 1:N)
    target += asym_laplace_lpdf(y[n] | f[n], tau, sigma);
}
