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
        "init_aux_pars": np.array([np.std(train_y)]),
       #"init_cov_pars": np.array([1,1]),
        "trace": False,
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



def model_gpboost_twostage(
    quantile: float,
    train_X: np.ndarray,
    group_train: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    group_test: np.ndarray,
    test_y: np.ndarray,
    approx: str,
    delta_logl: float = 1.0,
    target_coverage: float = 0.9,
    margin: float = 0.02,
    lr_init: float = 1.0,
    lr_step: float = 0.05,
    max_iter: int = 20,
) -> dict:
    """
    Two-stage approach to learn likelihood_learning_rate for grouped random effects.
    
    Stage 1: Split train in two (by groups), fit independent models
    Stage 2: Find optimal learning rate for target coverage
    Final: Refit on full data with learned learning rate
    """
    
    start_time = time.time()
    
    # Determine if crossed random effects
    n_re_groups = group_train.shape[1] if group_train.ndim > 1 else 1
    crossed = n_re_groups > 1
    
    
   # Random split of observations (groups appear in both)
    n_train = len(train_X)
    split_idx = n_train // 2
    indices = np.random.permutation(n_train)
    train1_idx, train2_idx = indices[:split_idx], indices[split_idx:]
    print("1")
    X_train1, X_train2 = train_X[train1_idx], train_X[train2_idx]
    print("2")
    y_train1, y_train2 = train_y[train1_idx], train_y[train2_idx]
    group_train1 = group_train[train1_idx] if n_re_groups == 1 else group_train[train1_idx, :]
    group_train2 = group_train[train2_idx] if n_re_groups== 1 else group_train[train2_idx, :]
    N1, N2 = len(X_train1), len(X_train2)
    print(f"Split: {N1} train1, {N2} train2")
    
    # ===== Stage 1: Fit two independent models =====
    print("Stage 1: Fitting model 1...")
    gpq1 = gpb.GPModel(
        group_data=group_train1,
        likelihood=approx,
        likelihood_additional_param=quantile if approx != "gaussian" else 1.,
        matrix_inversion_method="iterative" if crossed else "cholesky",
        cover_tree_radius=delta_logl,
        num_parallel_threads=1,
        likelihood_learning_rate=1.0,
    )
    
    params1 = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
        "trace": False,
    }
    
    if approx != "gaussian":
        gpq1.fit(X=X_train1, y=y_train1, params=params1)
    else:
        gpq1.fit(X=X_train1, y=y_train1)
    
    # Save hyperparameters
    cov_pars1 = gpq1.get_cov_pars(format_pandas=False)
    aux_pars1 = gpq1.get_aux_pars(format_pandas=False)
    coeff_pars1 = gpq1.get_coef(format_pandas=False)
    
    print("Stage 1: Fitting model 2...")
    gpq2 = gpb.GPModel(
        group_data=group_train2,
        likelihood=approx,
        likelihood_additional_param=quantile if approx != "gaussian" else 1.,
        matrix_inversion_method="iterative" if crossed else "cholesky",
        cover_tree_radius=delta_logl,
        num_parallel_threads=1,
        likelihood_learning_rate=1.0,
    )
    
    params2 = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
        "trace": False,
    }
    
    if approx != "gaussian":
        gpq2.fit(X=X_train2, y=y_train2, params=params2)
    else:
        gpq2.fit(X=X_train2, y=y_train2)
    
    # Get pseudo-truth from model 2
    pred2_on_train2 = gpq2.predict(
        X_pred=X_train2,
        group_data_pred=group_train2,
        predict_response=False,
        predict_var=False,
    )
    pseudo_truth = pred2_on_train2["mu"]
    
    # ===== Stage 2: Learn optimal learning rate =====
    print("\nStage 2: Learning likelihood_learning_rate...")
    
    lr_history = []
    coverage_history = []
    
    # Get critical value for intervals
    from scipy.stats import norm
    normv = norm()
    alpha = 1 - target_coverage
    t = normv.ppf(1 - alpha / 2)
    
    lr = lr_init
    for iteration in range(max_iter):
        # Initialize model with current learning rate
        gpq1_lr = gpb.GPModel(
            group_data=group_train1,
            likelihood=approx,
            likelihood_additional_param=quantile if approx != "gaussian" else 1.,
            matrix_inversion_method="iterative" if crossed else "cholesky",
            cover_tree_radius=delta_logl,
            num_parallel_threads=1,
            likelihood_learning_rate=lr,
        )
        
        # Fit with fixed hyperparameters
        params_fixed = {
            "estimate_aux_pars": False,
            "init_aux_pars": aux_pars1,
            "trace": False,
            "init_cov_pars": cov_pars1,
            "estimate_cov_par_index": [0] * len(cov_pars1),
            "init_coef": coeff_pars1,
        }
        
        gpq1_lr.fit(X=X_train1, y=y_train1, params=params_fixed)
        
        # Predict on train2
        pred1_on_train2 = gpq1_lr.predict(
            X_pred=X_train2,
            group_data_pred=group_train2,
            predict_response=False,
            predict_var=True,
        )
        
        # remove model after prediction
        del gpq1_lr
        mu1 = pred1_on_train2["mu"]
        std1 = np.sqrt(pred1_on_train2["var"])
        
        # Compute intervals
        lower = mu1 - t * std1
        upper = mu1 + t * std1
        
        # Check coverage
        coverage = np.mean((pseudo_truth >= lower) & (pseudo_truth <= upper))
        
        lr_history.append(lr)
        coverage_history.append(coverage)
        
        print(f"  Iter {iteration+1}: lr={lr:.3f}, coverage={coverage:.3f}, target={target_coverage:.3f}")
        
        # Check convergence
        if abs(coverage - target_coverage) <= margin:
            print(f"  Converged!")
            break
        
        # Update learning rate
        if coverage < target_coverage:
            lr -= lr_step  # Need wider intervals
        else:
            lr += lr_step  # Can use narrower intervals
        
        lr = max(0.1, lr)
    
    # Select best learning rate
    best_lr_idx = np.argmin(np.abs(np.array(coverage_history) - target_coverage))
    best_lr = lr_history[best_lr_idx]
    print(f"\nLearned likelihood_learning_rate: {best_lr:.3f}")
    
    # ===== Final: Refit on full data ??? OR PART 1 ??? =====
    print("\nFinal: Refitting on full training data...")
    gpq_final = gpb.GPModel(
        group_data=group_train1,
        likelihood=approx,
        likelihood_additional_param=quantile if approx != "gaussian" else 1.,
        matrix_inversion_method="iterative" if crossed else "cholesky",
        cover_tree_radius=delta_logl,
        num_parallel_threads=1,
        likelihood_learning_rate=best_lr,
    )

    params_final = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
        "trace": True,
    }
    
    if approx != "gaussian":
        gpq_final.fit(X=X_train1, y=y_train1, params=params_final)
    else:
        gpq_final.fit(X=X_train1, y=y_train1)
    
    # Predict on test
    pred = gpq_final.predict(
        X_pred=test_X,
        group_data_pred=group_test,
        predict_response=False,
        predict_var=True,
    )
    
    # Predict random effects
    re_estimate = gpq_final.predict_training_data_random_effects(predict_var=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Extract hyperparameters
    cov_pars = gpq_final.get_cov_pars().to_dict()
    flat_cov_pars = {group: next(iter(val.values())) for group, val in cov_pars.items() if "Group_" in group}
    noise_variance = np.float64(gpq_final.get_aux_pars()["scale"].iloc[0]) if approx != "gaussian" else cov_pars["Error_term"]["Param."]
    
    hyper_params = {
        **flat_cov_pars,
        "noise_variance": noise_variance,
      #  "learned_lr": best_lr,
     #   "lr_history": lr_history,
     #   "coverage_history": coverage_history,
      #  "validation_coverage": coverage_history[best_lr_idx] if coverage_history else None,
    }
    
    return pred, re_estimate, elapsed_time, hyper_params, group_train1


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
