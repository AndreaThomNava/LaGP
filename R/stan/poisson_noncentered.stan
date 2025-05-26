functions {
    #include gptools/util.stan
    #include gptools/graph.stan
}

data {
    real<lower=0> sigma, length_scale, epsilon;
    int n;
    array[n] int y;
    array[n] vector[1] X;
    int num_edges;
    array[2, num_edges] int edge_index;
}

transformed data {
    array[n] int degrees = out_degrees(n, edge_index);
}

parameters {
    vector[n] z;  // standard normal latent variables
}

transformed parameters {
    vector[n] eta;
    eta = gp_inv_graph_matern32_cov(z, zeros_vector(n), X, sigma, length_scale, edge_index, degrees, epsilon);
}

model {
    z ~ normal(0, 1);  // standard normal prior on z
    y ~ poisson_log(eta);
}

