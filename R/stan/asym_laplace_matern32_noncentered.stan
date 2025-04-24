functions {
  real asym_laplace_lpdf(real y, real mu, real tau, real sigma) {
    real z = (y - mu) / sigma;
    return log(tau * (1 - tau)) - log(sigma) - 
           (z >= 0 ? tau * z : (tau - 1) * z);
  }

 matrix cov_matern32(matrix x1, matrix x2, real lengthscale, real variance) {
    int N1 = rows(x1);  // Number of rows in x1
    int N2 = rows(x2);  // Number of rows in x2
    int D = cols(x1);   // Number of columns in x1 (features)
    matrix[N1, N2] K;

    for (i in 1:N1) {
        for (j in 1:N2) {
            real d = 0;
            // Compute the squared Euclidean distance between rows i and j of x1 and x2
            for (k in 1:D) {
                d += square(x1[i, k] - x2[j, k]);  // Compute the squared difference for each feature
            }
            real sqrt3_d = sqrt(3) * sqrt(d) / lengthscale;
            K[i, j] = variance * (1 + sqrt3_d) * exp(-sqrt3_d);
        }
    }

    return K;
}
}

data {
  int<lower=1> N;               // Training size
  int<lower=1> N_test;          // Test size
  int<lower=1> D;
  matrix[N, D] x;
  matrix[N_test, D] x_test;
  vector[N] y;
  real<lower=0, upper=1> tau;
}

parameters {
  real<lower=0> lengthscale;
  real<lower=0> variance;
  real<lower=0> sigma;
  vector[N] z;  // non-centered latent
}

transformed parameters {
  matrix[N, N] K_train = cov_matern32(x, x, lengthscale, variance);
  matrix[N, N] L_K_train = cholesky_decompose(K_train + diag_matrix(rep_vector(1e-6, N)));
  vector[N] f = L_K_train * z;
}

model {
  // Priors
  z ~ normal(0, 1);
  lengthscale ~ lognormal(0, 1);
  variance ~ lognormal(0, 1);
  sigma ~ lognormal(0, 1);

  // Likelihood
  for (n in 1:N)
    target += asym_laplace_lpdf(y[n] | f[n], tau, sigma);
}

/*
generated quantities {
  // This stores the posterior latent values at the test locations (f_test)
  vector[N_test] f_test_mean;  // Posterior latent at the test locations

  {
    matrix[N, N] K_pred_train = cov_matern32(x, x, lengthscale, variance);
    matrix[N_test, N] K_pred_star = cov_matern32(x_test, x, lengthscale, variance);
    
    // Add jitter for numerical stability
    matrix[N, N] K_eps = K_pred_train + diag_matrix(rep_vector(1e-6, N));

    // Compute the mean latent values at the test locations using the posterior latent values
    f_test_mean = K_pred_star * mdivide_left_spd(K_eps, f);
  }
}

*/