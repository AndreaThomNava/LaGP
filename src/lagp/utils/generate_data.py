import numpy as np
import pandas as pd


def pdf_asym_laplace(y: np.ndarray, b: np.ndarray, q: float, scale: float, log=True) -> np.ndarray:
    """
    Computes the PDF of the asymmetric Laplace distribution.
    
    Input:
        - y (np.ndarray): Data points.
        - b (np.ndarray): Quantile values.
        - q (float): Asymmetry parameter (0 < alpha < 1).
        - scale (float): Scale parameter (> 0).
        - log (bool): If True, returns log-PDF. If False, returns PDF.
    
    Output:
        - ret: (np.ndarray) PDF or log-PDF values.
    """
    y = np.asarray(y)
    b = np.asarray(b)
    
    ret = np.empty_like(b, dtype=np.float64)
    
    for i in range(len(b)):
        llis = np.empty_like(y, dtype=np.float64)
        ind = y <= b[i]
        
        llis[ind] = np.log(q * (1 - q) / scale) + (1 - q) * (y[ind] - b[i]) / scale
        llis[~ind] = np.log(q * (1 - q) / scale) - q * (y[~ind] - b[i]) / scale
        
        if log:
            ret[i] = np.sum(llis)
        else:
            ret[i] = np.exp(np.sum(llis))
    
    return ret

def quantile_func_asym_laplace(x: float, q: float, scale: float) -> np.ndarray:
    """
    Quantile function for the asymmetric laplace likelihood.

    Input:
        - x: (float) values from 0 to 1, ie the desired quantile.
        - q: (float) quantile/asymmetry parameter of the asymmetric laplace distribution. 
        - scale: (float) Scale parameter (> 0).
    
    Output:
        - rvs: (np.ndarray) quantiles of the asymmetric laplace distribution.
    """
    rvs = np.zeros(len(x))
    ind = (x <= q) 
    rvs[ind] = np.log(x[ind]/q) * scale / (1-q) 
    ind = ind == False
    rvs[ind] = - np.log((1-x[ind]) / (1-q)) * scale / q

    return rvs


def generate_input_grid():
    """
    Function to generate the input of the latent GP.

    Input:
        - d: (float) input dimension.
        - sample_size: (float) number of data points/sample size.

    Output:
        - grid: (np.ndarray) (nxd) grid
    """

    return "in progress"


def simulate_latentGP():
    """
    Function to simulate the latent Gaussian Process f.

    Input:
        - X: (np.ndarray) (nxd) input grid.
        - kernel:
        - n_samples: (float) number of deired GP samples. 

    Output:
        - f: (np.ndarray) (n x m_samples) GP samples. 
    """


    return "in progress"

def simulate_response():
    """
    Simulate the noise and add it to the latent GP.
    
    """



    return "in progress"