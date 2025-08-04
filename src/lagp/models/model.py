import gpboost as gpb
import gpytorch as gpy
import numpy as np
import pandas as pd
import torch
from torch import tensor
from gpytorch.kernels import MaternKernel, ScaleKernel
import time
from gpytorch.distributions import Distribution
import viva
from viva import VIVACpp as VIVA, my_train_cpp as my_train
from lagp.utils.gpytorch_utils import (AsymmetricLaplaceLikelihood,
                                       GPRegressionModel, AsymmetricLaplaceLikelihood_DKLGP)
import lightgbm as lgb

def model_boosting_l1(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    """
    Fit a LightGBM model with L1 loss (quantile=0.5) and predict on the test set.

    Input:
        - train_X: Feature matrix for training data
        - train_y: Target values for training data
        - test_X: Feature matrix for test data
        - test_y: Target values for test data

    Output:
        - pred: Predicted values for test data
    """

    start_time = time.time()

    # Setup dataset
    dtrain = lgb.Dataset(train_X, label=train_y)

    # Define model parameters
    params = {
        "objective": "quantile",
        "alpha": quantile,
        "verbosity": -1,
    }

    # Fit model
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=100,
    )

    # Predict
    pred = model.predict(test_X)
    pred_train = model.predict(train_X)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # "Hyperparameters" for consistency — mimic GPBoost
    hyper_params = {
        "lengthscale": None,
        "signal_variance": None,
        "noise_variance": None,
    }

    return pred, pred_train, elapsed_time, hyper_params


def model_gpboost(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    approx: str,
    delta_logl: float,
    n_vecchia: int = 1000,
) -> dict:
    """
    Fit the GPBoost model and predict on the test set.

    Input:
        - train_X: Feature matrix for training data
        - train_y: Target values for training data
        - test_X: Feature matrix for test data
        - test_y: Target values for test data

    Output:
        - pred: Predicted values for test data
    """

    N = len(train_X)
    vecchia = N > n_vecchia
    
    start_time = time.time()
    gpq = gpb.GPModel(
        gp_coords=train_X,
        cov_function="matern_ard",
        cov_fct_shape=1.5,
        gp_approx="vecchia" if vecchia else "none",
        num_neighbors=30,
        matrix_inversion_method= "iterative" if (vecchia and approx != "gaussian") else "cholesky",
        likelihood=approx,
        likelihood_additional_param=quantile if approx != "gaussian" else 1.,
        cover_tree_radius=delta_logl,
        num_parallel_threads=2,
    )
    
    params = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([np.std(train_y)]),
        "trace": True,
    }
    gpq.set_optim_params({
        "cg_preconditioner_type": "vadu",
       # "delta_rel_conv": 1e-8,
        "cg_max_num_it": 1500,
        "cg_max_num_it_tridiag" : 1500,
    }
    )
    # fit
    if approx != "gaussian":
        gpq.fit(X=np.ones(N), y=train_y, params=params)
    else:
        gpq.fit(X=np.ones(N), y=train_y)

    # predict
    pred = gpq.predict(
        X_pred=np.ones(len(test_X)),
        gp_coords_pred=test_X,
        predict_response=False,  # get the latent data
        predict_var=True,
    )

    # in-sample predictions
    pred_train =    gpq.predict(
        X_pred=np.ones(len(train_X)),
        gp_coords_pred=train_X,
        predict_response=False,  # get the latent data
        predict_var=True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # extract hyper-parames
    cov_pars = gpq.get_cov_pars()
    if approx == "gaussian":
        print(cov_pars)

    range_keys = [k for k in cov_pars.columns if k.startswith("GP_range")]
    estimated_ranges = [cov_pars[k].iloc[0] for k in range_keys]
    # Aggregate — mean range
    length_scale = np.mean(estimated_ranges)
    #length_scale = cov_pars["GP_range"].iloc[0]

    output_variance = cov_pars["GP_var"].iloc[0]
    noise_variance = gpq.get_aux_pars()["scale"] if approx != "gaussian" else cov_pars["Error_term"]

    hyper_params = {"lengthscale": length_scale,
                    "signal_variance":output_variance,
                    "noise_variance": noise_variance,
                    }
    

    return pred, pred_train, elapsed_time, hyper_params


def model_gpytorch(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    lr: float,
    inducing_threshold: int,
    inducing_points: int,
) -> gpy.distributions.MultivariateNormal:
    """
    Fit the GPyTorch model and predict on the test set.

    Input:
        - train_X: Feature matrix for training data
        - train_y: Target values for training data
        - test_X: Feature matrix for test data
        - test_y: Target values for test data
        - epochs
        - lr
        - inducing_threshold
        - inducing_points

    Output:
        - predictions: quantile predictions on test data
    """

    torch.set_num_threads(2) # same as gpboost
    if isinstance(train_X, pd.DataFrame):
        train_X = train_X.values
    if isinstance(train_y, (pd.Series, pd.DataFrame)):
        train_y = train_y.values.ravel()  # Ensure 1D array if needed

    if isinstance(test_X, pd.DataFrame):
        test_X = test_X.values
    if isinstance(test_y, (pd.Series, pd.DataFrame)):
        test_y = test_y.values.ravel()

    # Assuming train_X, train_y, test_X, test_y are numpy arrays
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    likelihood = AsymmetricLaplaceLikelihood(quantile=quantile)
    model = GPRegressionModel(train_X_tensor, inducing_threshold, inducing_points)  # Instantiate the GPyTorch model

    # ---- Training ----
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    mll = gpy.mlls.VariationalELBO(likelihood, model, num_data=train_y_tensor.size(0))

    num_epochs = epochs
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_X_tensor)
        loss = -mll(output, train_y_tensor)
        loss.backward()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        optimizer.step()
    # ---- Evaluation ----
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = model(test_X_tensor)
        # predictions = pred.mean  # Mode or median of asymmetric Laplace
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Access the estimated hyperparameters: detach grads and make numpy
    lengthscale = np.float64(np.mean(model.covar_module.base_kernel.lengthscale.detach().numpy()))
    output_variance = np.float64(model.covar_module.outputscale.detach().numpy().item())
    noise_variance = np.float64(likelihood.scale.detach().numpy().item())

    hyper_params = {"lengthscale": lengthscale,
                    "signal_variance":output_variance,
                    "noise_variance": noise_variance,
                    }

    return pred, elapsed_time, hyper_params


def model_viva_gp(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    rho: float = 1.5,
    lengthscale_init: float = 0.25,
    outputscale_init: float = 0.25,
    nu: float = 1.5,
    epochs: int = 200,
    use_ic0: bool = True,
    classify: bool = False
):
    """
    Fit a VIVA (Vecchia approximation) GP model with asymmetric Laplace likelihood.

    Args:
        quantile: Target quantile.
        train_X, train_y: Training data.
        test_X, test_y: Test data.
        rho: Expansion factor for neighbor search.
        lengthscale_init, outputscale_init: Kernel init hyperparams.
        nu: Smoothness for Matern kernel.
        epochs: Number of training iterations.
        use_ic0: Whether to use IC0 initialization.
        classify: Flag for classification (False for regression).
    
    Returns:
        mu_post: Predictive mean at test points.
        sd_post: Predictive std deviation at test points.
        elapsed_time: Training time in seconds.
    """

    # Concatenate train/test for VIVA format
    X_full = np.vstack((train_X, test_X))
    y_full = np.concatenate((train_y, np.zeros(len(test_y))))  # dummy y for test

    X_tensor = torch.tensor(X_full, dtype=torch.float32)
    y_tensor = torch.tensor(y_full, dtype=torch.float32)

    # Define likelihood
    likelihood = AsymmetricLaplaceLikelihood_DKLGP(quantile=quantile)

    # Define kernel
    d = X_full.shape[1]
    K = ScaleKernel(MaternKernel(ard_num_dims=d, nu=nu))
    K.base_kernel.lengthscale = lengthscale_init
    K.outputscale = outputscale_init

    # Instantiate VIVA model
    n_test = test_X.shape[0]

    start_time = time.time()
    model = VIVA(
        X_tensor,
        y_tensor,
        K,
        likelihood=likelihood,
        rho=rho,
        n_test=n_test,
        classify=classify,
        use_ic0=use_ic0,
    )

    # Training
    my_train(model, n_Epoch=epochs)
    

    # Prediction
    model.eval()
    mu_post, var_post = model.predict()
    elapsed_time = time.time() - start_time
    mu = mu_post.detach().numpy()
    sd = var_post.sqrt().detach().numpy()

    signal_variance = np.float64((K.outputscale).detach().numpy().item())
    lengthscale = np.float64(np.mean(K.base_kernel.lengthscale.detach().numpy()))
    noise_variance = np.float64(likelihood.noise.detach().numpy().item())

    # extract hyper
    hyper_params = {"lengthscale": lengthscale,
                    "signal_variance":signal_variance,
                    "noise_variance": noise_variance,
                    }

    return mu[-n_test:], sd[-n_test:], elapsed_time, hyper_params



