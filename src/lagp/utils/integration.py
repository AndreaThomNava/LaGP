import pandas as pd
import numpy as np
from lagp.utils.generate_data import quantile_func_asym_laplace, pdf_asym_laplace, simulate_response
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



def gpboost_nll_min_dec(Y, laplace_approx, min_dec, num_groups, group_size, quantile, scale):
    # Step 4: Fit GPBoost model with specified Laplace approximation
    gpq = gpb.GPModel(group_data= np.repeat(range(num_groups), repeats = group_size),
                likelihood=laplace_approx, 
                likelihood_additional_param = quantile,
                cover_tree_radius = min_dec)
    # need to estimate extra params
    params = {"estimate_aux_pars": True, "init_aux_pars" : np.array([scale]), "trace":False}
    # do i need to fit it?
    gpq.fit(X = np.ones(group_size*num_groups), y=Y, params = params) 
    # obtain neg ll via laplace approximation
    gpb_nll = gpq.neg_log_likelihood(cov_pars=[1.], y = Y) 
    gpb_nll_current = gpq.get_current_neg_log_likelihood() 
    
    return gpq, gpb_nll, gpb_nll_current


def compute_aghq(Y, group_size, num_groups, predicted_re, quantile, scale, re_std, K=100):
    
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
        sigma = np.sqrt(predicted_re.iloc[0, 1])
        # try fixed for a sec
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
    misspecified
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
            gpq, gpboost_ll, gpboost_ll_current = gpboost_nll_min_dec(
                Y=Y,
                laplace_approx=laplace_approximation,
                min_dec=min_dec, 
                num_groups= num_groups,
                group_size = group_size,
                quantile=quantile,
                scale=scale,
            
            )

            predicted_re = gpq.predict_training_data_random_effects(predict_var=True)

            aghq = compute_aghq(
                Y=Y,
                group_size = group_size,
                num_groups= num_groups,
                predicted_re=predicted_re, 
                quantile=quantile,
                scale=scale,
                re_std=re_std,
                K = K
                
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
    