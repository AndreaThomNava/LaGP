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


def model_gpboost(
    quantile: float,
    train_X: np.ndarray,
    group_train: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    group_test: np.ndarray,
    test_y: np.ndarray,
    approx: str,
    delta_logl: float,
    estimate_hyper: bool = True,
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

    start_time = time.time()
    # determine if crossed random effects
    n_re_groups = group_train.shape[1] if group_train.ndim > 1 else 1
    crossed = n_re_groups > 1

    gpq = gpb.GPModel(
        group_data=group_train,
        likelihood=approx,
        likelihood_additional_param=quantile if approx != "gaussian" else 1.,
        matrix_inversion_method = "iterative" if crossed else "cholesky",
        cover_tree_radius=delta_logl,
        num_parallel_threads=2,
        )
    
    if estimate_hyper:
        params = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
       #"init_cov_pars": np.array([1,1]),
        "trace": True,
        }
    else:
        params = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
        "init_cov_pars": np.array([1]), # set to true params
        "estimate_cov_par_index": np.array([0]),
        "trace": True,
        }

    # fit
    if approx != "gaussian":
        gpq.fit(X=train_X, y=train_y, params=params)
    else:
        gpq.fit(X=train_X, y=train_y)


    # predict (contains also fixed effects predictors)
    pred = gpq.predict(
        X_pred=test_X,
        group_data_pred=group_test,
        predict_response=False,  # get the latent data
        predict_var=True,
        )

    # predict latent random effect
    re_estimate = gpq.predict_training_data_random_effects(predict_var=True)
  
    end_time = time.time()
    elapsed_time = end_time - start_time

    # extract hyper-parames
    cov_pars = gpq.get_cov_pars().to_dict()
    if approx == "gaussian":
        print(cov_pars)
    flat_cov_pars = {group: next(iter(val.values())) for group, val in cov_pars.items() if "Group_" in group}

    noise_variance = np.float64(gpq.get_aux_pars()["scale"].iloc[0]) if approx != "gaussian" else cov_pars["Error_term"]["Param."]
    print(noise_variance)
    hyper_params = {**flat_cov_pars,
                    "noise_variance": noise_variance,
                    }
    

    return pred, re_estimate, elapsed_time, hyper_params


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
    lengthscale = np.float64(model.covar_module.base_kernel.lengthscale.detach().numpy().item())
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
    lengthscale = np.float64((K.base_kernel.lengthscale[0][0]).detach().numpy().item())
    noise_variance = np.float64(likelihood.noise.detach().numpy().item())

    # extract hyper
    hyper_params = {"lengthscale": lengthscale,
                    "signal_variance":signal_variance,
                    "noise_variance": noise_variance,
                    }

    return mu[-n_test:], sd[-n_test:], elapsed_time, hyper_params
