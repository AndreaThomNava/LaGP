import numpy as np
from typing import Tuple

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
    score[~index] = - (preds - y)[~index] * quantile
    qs = score.mean()
    
    return qs

def interval_score(y, pred_low, pred_up, quantile):
    """
    Compute the interval score.

    Input:
        - y: (np.ndarray) observed responses.
        - pred_low: (np.ndarray) predicted lower quantiles.
        - pred_up: (np.ndarray) predicted upper quantiles.
        - quantile: (float) target quantile.
    
    Output:
        - mis: (float) mean interval score.
    """

    dispersion = (pred_up - pred_low).mean()
    over_prediction = 2 * ((y - pred_up)[(y - pred_up) >= 0]).mean() / quantile
    under_prediction = 2 * ((pred_low - y)[(pred_low - y) >= 0]).mean() / quantile
    mis = np.nansum([dispersion, over_prediction, under_prediction])
    
    return mis


def coverage_and_width(y: np.ndarray, pred_low: np.ndarray, pred_up: np.ndarray) -> Tuple[float, float]:
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

