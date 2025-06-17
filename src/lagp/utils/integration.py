import pandas as pd
import numpy as np
from lagp.utils.generate_data import quantile_func_asym_laplace, pdf_asym_laplace, generate_input_grid, simulate_latentGP, simulate_response
from scipy import stats
from scipy.integrate import quad, trapezoid
import gpboost as gpb
from scipy.special import roots_hermite
from scipy.special import logsumexp
import gpytorch as gpy
import pymc as pm
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def simulate_grouped_response(group_size, num_groups, re_mean, re_std, quantile, scale, misspecified=False):
    N = group_size * num_groups
    group_effects = np.random.normal(loc=re_mean, scale=re_std, size=num_groups)
    random_effects = np.repeat(group_effects, repeats=group_size)
    
    if not misspecified:
        error = quantile_func_asym_laplace(
            x=np.random.uniform(0, 1, N), q=quantile, scale=scale
        )
        Y = random_effects + error
    else:
        Y = random_effects + np.random.normal(0, 1, size=N)
    
    return Y, group_effects


def compute_naive_log_marglik(Y, group_size, num_groups, quantile, scale, re_std):
    res = 0
    for group in range(num_groups):
        yc = Y[group_size*group : group_size*(group+1)]
        def pjoint(b):
            b = [b]
            return np.exp(
                pdf_asym_laplace(y=yc, b=b, q=quantile, scale=scale, log=True) +
                stats.norm.logpdf(b, scale=np.sqrt(re_std))
            )
        integral, _ = quad(pjoint, -5, 5, epsrel=1e-8)
        res += np.log(integral)
    return -res



def gpboost_nll_min_dec(Y, laplace_approx, min_dec, num_groups, group_size, quantile, scale, estimate_scale = True):
    # Step 4: Fit GPBoost model with specified Laplace approximation
    gpq = gpb.GPModel(group_data= np.repeat(range(num_groups), repeats = group_size),
                likelihood=laplace_approx, 
                likelihood_additional_param = quantile,
                cover_tree_radius = min_dec)
    # need to estimate extra params
    params = {"estimate_aux_pars": estimate_scale, "init_aux_pars" : np.array([scale]), "trace":False}
    # do i need to fit it?
    gpq.fit(X = np.ones(group_size*num_groups), y=Y, params = params) 
    # obtain neg ll via laplace approximation
    gpb_nll = gpq.neg_log_likelihood(cov_pars=[1.], y = Y) 
    gpb_nll_current = gpq.get_current_neg_log_likelihood() 

    if estimate_scale:
        # get the scale parameter
        scale_est = gpq.get_aux_pars()["scale"]["Param."] #["Param. "]
    else:
        scale_est = np.array([scale])    
    
    return gpq, gpb_nll, gpb_nll_current, scale_est


def compute_aghq(Y, group_size, num_groups, predicted_re, quantile, scale, re_std, K=100, use_pred_var=True):
    
    nodes, weights = roots_hermite(K)
    # why?
    adaptive_weights = weights * np.exp(nodes**2)
    
    def g(y, eval_points):
        return pdf_asym_laplace(y=y, b=eval_points, q=quantile, scale = scale, log=True) + \
               stats.norm.logpdf(eval_points, scale=np.sqrt(re_std))

    res_aghq = 0
    for group in range(num_groups):
        yc = Y[group_size*group : group_size*(group+1)]
        mode = predicted_re.iloc[group_size*group, 0]
        if use_pred_var:
            sigma = np.sqrt(predicted_re.iloc[0, 1])
        else: 
            sigma = 0.2
        eval_points = mode + np.sqrt(2) * sigma * nodes
        x_i = np.log(adaptive_weights) + g(y=yc, eval_points=eval_points)
        logsum = logsumexp(x_i)
        res_aghq += np.log(sigma * np.sqrt(2)) + logsum
    return -res_aghq


def run_simulation(
    laplace_approximation,
    group_size,
    num_groups,
    re_mean,
    re_std,
    scale,
    quantile,
    min_decreases,
    K,
    B,
    misspecified,
    estimate_scale=True,
    use_pred_var = True
):
    gps = {str(min_dec): [] for min_dec in min_decreases}
    naives = {str(min_dec): [] for min_dec in min_decreases}
    aghqs = {str(min_dec): [] for min_dec in min_decreases}

    for b in range(B):
        Y, group_effects = simulate_grouped_response(
            group_size=group_size,
            num_groups=num_groups,
            re_mean=re_mean,
            re_std=re_std,
            scale=scale,
            quantile=quantile,
            misspecified=misspecified
        )

        naive = compute_naive_log_marglik(
            Y=Y,
            group_size = group_size,
            num_groups= num_groups,
            quantile=quantile,
            scale=scale,
            re_std=re_std,
        )

        for min_dec in min_decreases:
            gpq, gpboost_ll, gpboost_ll_current, used_scale = gpboost_nll_min_dec(
                Y=Y,
                laplace_approx=laplace_approximation,
                min_dec=min_dec, 
                num_groups= num_groups,
                group_size = group_size,
                quantile=quantile,
                scale=scale,
                estimate_scale=estimate_scale,
            
            )

            predicted_re = gpq.predict_training_data_random_effects(predict_var=True)

            aghq = compute_aghq(
                Y=Y,
                group_size = group_size,
                num_groups= num_groups,
                predicted_re=predicted_re, 
                quantile=quantile,
                scale=used_scale, # IMPORTANT: use the scale estimated by GPBoost
                re_std=re_std,
                K = K,
                use_pred_var=use_pred_var
                
            )

            gps[str(min_dec)].append(gpboost_ll)
            naives[str(min_dec)].append(naive)
            aghqs[str(min_dec)].append(aghq)

    return gps, naives, aghqs






#### GP MARGINAL LOG-LIKELIHOOD  #####

def gpboost_nll_gp(y, X, laplace_approx, min_dec, scale, quantile):
    # Step 4: Fit GPBoost model with specified Laplace approximation
    likelihood = laplace_approx
    gpq = gpb.GPModel(gp_coords  = X,
                    cov_function= "matern", # "matern",
                    cov_fct_shape= 1.5,
                likelihood=likelihood, 
                likelihood_additional_param = quantile,
                cover_tree_radius = min_dec)
    # need to estimate extra params
    params = {"estimate_aux_pars": False, "init_aux_pars" : np.array([scale]), "trace":False}
    # with intercept
    gpq.fit(X = np.ones(len(y)), y=y, params = params) 
    # obtain neg ll via laplace approximation
    gpb_nll = gpq.get_current_neg_log_likelihood() 
    
    return gpq, gpb_nll
    


def run_thermo_ais(config):
    """
    Run a simulation of thermodynamic integration and annealed importance sampling, comapre it to GPBoost.
    
    """
    d = config["dimension"]
    N = config["sample_size"]
    X = generate_input_grid(
                d=d, sample_size=N, lims=config["grid_lims"]
            )  
    quantile = config["quantile"]
    
    nu = config["kernel"]["nu"]
    lengthscale = config["kernel"]["lengthscale"]
    signal_variance = config["kernel"]["signal_variance"]
    base_kernel = gpy.kernels.MaternKernel(nu=nu)
    base_kernel.lengthscale = lengthscale
    kernel = gpy.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = signal_variance

    f = simulate_latentGP(X, kernel, n_samples=1)
    f = f.reshape(N)

    y_obs = simulate_response(f, noise=config["likelihood"], pars = config["pars"])

    # Define temperature schedule (β values)
    num_temps = 20 # 20
    betas = np.linspace(0, 1, num_temps)
    betas[0] = 0.01  # Avoid exact zero

    # Store log-likelihoods for each β
    log_likelihoods = []
    AIS = []

    # Define GP Model with Asymmetric Laplace Likelihood
    with pm.Model() as model:
        # Thermodynamic integration parameter (β)
        beta = pm.Data("beta", 1.0)  # Will be updated dynamically

        # GP Kernel
        cov = signal_variance * pm.gp.cov.Matern32(input_dim = d, ls = lengthscale)
        gp = pm.gp.Latent(cov_func=cov)

        # GP Prior
        f = gp.prior("f", X=X)

        # Likelihood: Asymmetric Laplace with β-scaling
        # IF WELL-SPECIFIED !
        sigma = config["scale_laplace"] 
        y = pm.AsymmetricLaplace("y", q=quantile, mu=f, b = beta / (sigma), observed=y_obs) # 1 / (sigma * pm.math.sqrt(beta))

        # Run thermodynamic integration over different β values
        for i, beta_value in enumerate(betas):
            
            pm.set_data({"beta": beta_value})  # Update β in the model
            
            # Sample posterior
            trace = pm.sample(1000, tune=500, step = pm.Metropolis())

            # Extract posterior samples
            # take only first chain
            f_samples = trace.posterior["f"].values[0,:,:]  # Shape (num_samples, n_samples)
    

            # Compute log-likelihood manually for Thermo Integration
            log_lik_samples = np.array([pdf_asym_laplace(y = y_obs, b = f, q = quantile,
                                                        scale = sigma, log = True) for f in f_samples])
            
            # For Annealed Importance Sampling
            # why sqrt?
            if i > 1:
                delta_beta_value = beta_value - betas[i-1]
                ais = np.array([pdf_asym_laplace(y = y_obs, b = f, q = quantile,
                                                        scale = sigma, log = False) for f in f_samples])
                ais = ais**(delta_beta_value)
                AIS.append(ais.mean())
            # Average over posterior draws
            log_likelihoods.append(log_lik_samples.mean())

    # Convert to NumPy array
    nlog_likelihoods = - np.array(log_likelihoods)

    # Compute log-marginal likelihood using trapezoidal integration
    nlog_marginal_likelihood = trapezoid(nlog_likelihoods, betas)

    # compute log-marginal likelihood using AIS
    # product of elements in AIS
    AIS = np.array(AIS)
    nlog_ais = - np.log(AIS.prod())


    # GPBOOST
    laplace_approxs = ["asymmetric_laplace",
                  "asymmetric_laplace_constant_curvature_fisher_mode_finding",
                  "asymmetric_laplace_lls_laplace_fisher_mode_finding"] 
    names = ["Fisher", "CC", "LLS"]
    min_dec = config["gpboost"]["min_decrease"]
    gp_results = {f"{name}": [] for name in names}
    for approx, name in zip(laplace_approxs, names):
        # Step 4: Fit GPBoost model with specified Laplace apddddion
        gpq, gp_nll = gpboost_nll_gp(y_obs, X, approx, min_dec = min_dec, scale = sigma, quantile = quantile)
        gp_results[f"{name}"].append(gp_nll)

    return nlog_marginal_likelihood, nlog_ais, gp_results

### CARL ####

class RatioEstimator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()  # Outputs probability of joint sample
        )

    def forward(self, x):
        return self.model(x)
    


def create_carl_dataset(config, train: bool = True):
    if train:
        num_samples = config["num_samples"]
    else:
        num_samples = 1 # make test config
    d = config["dimension"]
    N = config["sample_size"]
    X = generate_input_grid(
                d=d, sample_size=N, lims=config["grid_lims"]
            )  
    quantile = config["quantile"]
    
    
    # kernel params
    nu = config["kernel"]["nu"]
    lengthscale = config["kernel"]["lengthscale"]
    signal_variance = config["kernel"]["signal_variance"]
    base_kernel = gpy.kernels.MaternKernel(nu=nu)
    base_kernel.lengthscale = lengthscale
    kernel = gpy.kernels.ScaleKernel(base_kernel)
    kernel.outputscale = signal_variance
    
    
    y_obs_list = []
    f_list = []
    for b in range(num_samples):
        f = simulate_latentGP(X, kernel, n_samples=1)
        f = f.reshape(N)
        f_list.append(f)
        y_obs_list.append(simulate_response(f, noise=config["likelihood"], pars = config["pars"]))

    return np.array(y_obs_list), np.array(f_list)


def run_carl(config):
    """
    
    """
    num_samples = config["num_samples"]
    quantile = config["quantile"]
    sigma = config["pars"]["ald"]["scale"]
    data_samples, gp_realizations = create_carl_dataset(config=config, train = True)
    # Shuffle for negative class   
    gp_marginal = np.random.permutation(gp_realizations)  

    # Labels: 1 for joint samples, 0 for marginal
    X_joint = np.hstack([gp_realizations, data_samples])
    X_marginal = np.hstack([gp_marginal, data_samples])
    y_joint = np.ones((num_samples, 1))
    y_marginal = np.zeros((num_samples, 1))

    # Concatenate and shuffle
    X = np.vstack([X_joint, X_marginal])
    y = np.vstack([y_joint, y_marginal])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train the classifier
    def train_classifier(model, X_train, y_train, epochs=50, batch_size=64, lr=0.001):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y.squeeze())
                loss.backward()
                optimizer.step()

        return model
        
    model = RatioEstimator(input_dim=X.shape[1])
    trained_model = train_classifier(model, X_train, y_train)

    def estimate_evidence(trained_model, num_points = 100):
   
        gp_samples, data_samples = create_carl_dataset(config, train = False)

        X_mc = np.hstack([gp_samples, data_samples])
        X_mc_tensor = torch.tensor(X_mc, dtype=torch.float32)

        with torch.no_grad():
            ratio_predictions = trained_model(X_mc_tensor).numpy()
        ratio_predictions = ratio_predictions/(1- ratio_predictions)

        # get likelihood of samples
        eps = 0 #0.01
        # use asymmetric laplace likelihood
        gp_sample = gp_samples[0]
        data_sample = data_samples[0]
        likelihood = [pdf_asym_laplace(y=data_sample[idx], b = [sample], q=quantile, scale = sigma, log=True) for idx, sample in enumerate(gp_sample)]
        log_likelihood = (np.array(likelihood).sum())
        evidence_estimate = np.exp(log_likelihood)/ratio_predictions

        return evidence_estimate, ratio_predictions, log_likelihood, data_samples

    estimated_evidence, ratio_predictions, log_likelihood, data_samples = estimate_evidence(trained_model, num_points = num_samples)
    
    
    return estimated_evidence, ratio_predictions, log_likelihood, data_samples