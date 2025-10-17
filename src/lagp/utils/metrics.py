from typing import Tuple
import pandas as pd
import numpy as np


def quantile_score(y: np.ndarray, preds: np.ndarray, quantile: float) -> float:
    """
    Compute the quantile score (i.e. pin-ball loss).

    Input:
        - y: (np.ndarray) observed responses.
        - pred: (np.ndarray) predicted quantiles.
        - quantile: (float) target quantile.

    Output:
        - qs: (float) quantile score.

    """

    score = np.zeros(len(y))
    index = (preds - y) >= 0
    score[index] = (preds - y)[index] * (1 - quantile)
    score[~index] = -(preds - y)[~index] * quantile
    qs = score.mean()

    return qs


def interval_score(y, pred_low, pred_up, alpha):
    """
    Compute the interval score.

    Input:
        - y: (np.ndarray) observed responses.
        - pred_low: (np.ndarray) predicted lower quantiles.
        - pred_up: (np.ndarray) predicted upper quantiles.
        - alpha: (float) confidence level of prediction interval.

    Output:
        - mis: (float) mean interval score.
    """

    dispersion = (pred_up - pred_low).mean()
    over_prediction = 2 * ((y - pred_up)[(y - pred_up) >= 0]).mean() / alpha
    under_prediction = 2 * ((pred_low - y)[(pred_low - y) >= 0]).mean() / alpha
    mis = np.nansum([dispersion, over_prediction, under_prediction])

    return mis


def coverage_and_width(
    y: np.ndarray, pred_low: np.ndarray, pred_up: np.ndarray
) -> Tuple[float, float]:
    """
    Compute coverage and width of intervals formed by [pred_low, pred_up].

    Input:
        - y: (np.ndarray)
        - pred_low: (np.ndarray)
        - pred_up: (np.ndarray)

    Output:
        - coverage: (float)
        - width: (float)
    """

    coverage = ((y >= pred_low) & (y <= pred_up)).mean()
    width = (pred_up - pred_low).mean()

    return coverage, width


def compute_bias_mse(true_param: np.ndarray, estimates: np.ndarray):
    """
    Compute bias and MSE of estimated parameters.
    
    Parameters
    ----------
    true_param : np.ndarray of shape (d,)
        The true values of the parameters.
    estimates : np.ndarray of shape (n_rep, d)
        Estimated parameters over multiple replicates.
    
    Returns
    -------
    dict
        Dictionary with 'bias' and 'mse', each of shape (d,)
    """
    assert estimates.ndim == 2, "estimates should be (n_rep, d)"
    assert true_param.shape[0] == estimates.shape[1], "Dimension mismatch"

    bias = np.mean(estimates, axis=0) - true_param
    mse = np.mean((estimates - true_param) ** 2, axis=0)
    
    return {"bias": bias, "mse": mse}


def align_re(group_train, group_test, re, test_true_latent_quantile, pred_std, true_var, group_size):

    group_train_df = pd.DataFrame({"group_id":group_train})
    group_test_df = pd.DataFrame({"group_id":group_test})

    predicted_re = pd.DataFrame({"pred_re": re["Group_1"]})
    predicted_var = pd.DataFrame({"true_var": true_var})
    
    true_var = pd.DataFrame({"pred_std": pred_std})


    train_re = pd.concat([group_train_df, predicted_re], axis = 1)
    test_re = pd.concat([group_test_df, pd.DataFrame({"true_re": test_true_latent_quantile})],axis = 1)
    test_re = pd.concat([test_re, predicted_var],axis = 1)

    
    test_re = pd.concat([test_re, true_var],axis = 1)

    # align by merging on id
    merged = test_re.merge(train_re, on = "group_id", how= "inner").drop_duplicates(subset="group_id")
    
    # Divide true_var by group size
    merged["true_var"] = merged["true_var"] / (group_size - merged.groupby("group_id")["group_id"].transform("size"))

    return merged["true_re"], merged["pred_re"], merged["pred_std"], merged["true_var"]



def compute_rmse(f_true, f_pred):
    """
    Returns the Root Mean Squared Error (RMSE) for the predicted quantiles.
    """

    rmse = np.sqrt(np.mean((f_true - f_pred)**2))

    return rmse
