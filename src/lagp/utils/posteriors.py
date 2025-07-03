
from lagp.utils.generate_data import pdf_asym_laplace
from lagp.utils.integration import compute_aghq

from scipy.stats import norm
import numpy as np
import gpboost as gpb
import matplotlib.pyplot as plt



def un_norm_log_posterior(y, b, q, scale, re_std):
    
    p = pdf_asym_laplace(y=y, b=[b], q=q, scale=scale, log=True) + norm.logpdf(x=b, loc = 0, scale = re_std)

    return p

def laplace_log_posterior(b, mean, var):

    p = norm.logpdf(b, loc = mean, scale = np.sqrt(var))

    return p

def norm_log_posterior(y, b, quantile, scale, re_std, log_normalising_constant):

    # compute normalising constant via adaptive Guass-Hermite Quadrature
    p = un_norm_log_posterior(y, b, quantile, scale, re_std) - log_normalising_constant

    return p


def laplace_approx(Y, approx, min_dec, quantile, num_groups, group_size):

    # fit model and extract mode and pred var
    gpq = gpb.GPModel(group_data= np.repeat(range(num_groups), repeats = group_size) ,
                    likelihood=approx, 
                    likelihood_additional_param = quantile,
                    cover_tree_radius = min_dec)
    # need to estimate extra params
    params = {"estimate_aux_pars": True, "init_aux_pars" : np.array([1]), "trace":False}
    gpq.fit(y=Y, params = params) 
    nll = gpq.get_current_neg_log_likelihood()
    re_estimate = gpq.predict_training_data_random_effects(predict_var=True)
    mode = re_estimate.iloc[0,0]
    pred_var = re_estimate.iloc[0,1]

    estimated_scale = gpq.get_aux_pars()["scale"]["Param."]

    return re_estimate, mode, pred_var, nll, estimated_scale

def laplace_approx_fixed_scale(Y, approx, min_dec, quantile, num_groups, group_size, scale):

    # fit model and extract mode and pred var
    gpq = gpb.GPModel(group_data= np.repeat(range(num_groups), repeats = group_size) ,
                    likelihood=approx, 
                    likelihood_additional_param = quantile,
                    cover_tree_radius = min_dec)
    # need to estimate extra params
    params = {"estimate_aux_pars": False, "init_aux_pars" : np.array([scale]), "trace":False}
    gpq.fit(y=Y, params = params) 
    nll = gpq.get_current_neg_log_likelihood()
    re_estimate = gpq.predict_training_data_random_effects(predict_var=True)
    mode = re_estimate.iloc[0,0]
    pred_var = re_estimate.iloc[0,1]

    estimated_scale = gpq.get_aux_pars()["scale"]["Param."]

    return re_estimate, mode, pred_var, nll, estimated_scale

def generalised_laplace_approx(Y, approx, min_dec, quantile, num_groups, group_size, lr):

    # fit model and extract mode and pred var
    gpq = gpb.GPModel(group_data= np.repeat(range(num_groups), repeats = group_size) ,
                    likelihood=approx, 
                    likelihood_additional_param = quantile,
                    cover_tree_radius = min_dec,
                    likelihood_learning_rate=lr)
    # need to estimate extra params
    params = {"estimate_aux_pars": True, "init_aux_pars" : np.array([1]), "trace":False}
    gpq.fit(y=Y, params = params) 
    print(gpq.likelihood_learning_rate)
    nll = gpq.get_current_neg_log_likelihood()
    re_estimate = gpq.predict_training_data_random_effects(predict_var=True)
    mode = re_estimate.iloc[0,0]
    pred_var = re_estimate.iloc[0,1]

    estimated_scale = gpq.get_aux_pars()["scale"]["Param."]

    return re_estimate, mode, pred_var, nll, estimated_scale


def plot_posteriors(bs, Y, quantile, scale, re_std, laplace_mean, laplace_var, log_normalising_constant):

    posterior_laplace =  [np.exp(laplace_log_posterior(b = b, mean = laplace_mean, var = laplace_var)) for b in bs]
    posterior_numerical = [np.exp(norm_log_posterior(y=Y, b=b, quantile=quantile, scale = scale, re_std = re_std, log_normalising_constant = log_normalising_constant)) for b in bs]

    plt.plot(bs, posterior_numerical, label = "True")
    plt.plot(bs, posterior_laplace, label = "LA")
    plt.xlabel("b")
    plt.legend(labelcolor = "black")
    delta = np.sqrt(laplace_var)*5
    plt.xlim(laplace_mean - delta, laplace_mean + delta)
    plt.title(f"Posterior of random effect (sample size: {len(Y)})", color = "black")
    plt.show()



