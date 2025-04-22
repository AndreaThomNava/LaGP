import gpboost as gpb
import gpytorch as gpy
import numpy as np
import torch
from gpytorch.distributions import Distribution
from lagp.utils.gpytorch_utils import AsymmetricLaplaceLikelihood, GPRegressionModel


def model_gpboost(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    approx: str,
    delta_logl: float,
    vecchia: bool = False,
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
    gpq = gpb.GPModel(
        gp_coords=train_X,
        cov_function="matern",
        cov_fct_shape=1.5,
        gp_approx="vecchia" if vecchia else "none",
        num_neighbors=30,
        matrix_inversion_method="iterative" if vecchia else "cholesky",
        likelihood=approx,
        likelihood_additional_param=quantile,
        cover_tree_radius=delta_logl,
    )
    params = {
        "estimate_aux_pars": True,
        "init_aux_pars": np.array([0.1]),
        "trace": False,
    }
    # fit
    gpq.fit(X=np.ones(N), y=train_y, params=params)
    # predict
    pred = gpq.predict(
        X_pred=np.ones(len(test_X)),
        gp_coords_pred=test_X,
        predict_response=False,  # get the latent data
        predict_var=True,
    )

    return pred


def model_gpytorch(
    quantile: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    epochs: int,
    lr: float,
) -> gpy.distributions.MultivariateNormal:
    """
    Fit the GPyTorch model and predict on the test set.

    Input:
        - train_X: Feature matrix for training data
        - train_y: Target values for training data
        - test_X: Feature matrix for test data
        - test_y: Target values for test data
        -
    Output:
        - predictions: quantile predictions on test data
    """

    # Assuming train_X, train_y, test_X, test_y are numpy arrays
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    likelihood = AsymmetricLaplaceLikelihood(quantile=quantile)
    model = GPRegressionModel(train_X_tensor)  # Instantiate the GPyTorch model

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
        #predictions = pred.mean  # Mode or median of asymmetric Laplace

    return pred
