import os

import gpytorch as gpy
import numpy as np
import pandas as pd
import torch
import json
from scipy.stats import chi2, norm, t
from sklearn.model_selection import KFold



def pdf_asym_laplace(
    y: np.ndarray, b: np.ndarray, q: float, scale: float, log=True
) -> np.ndarray:
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


def quantile_func_asym_laplace(x, q: float, scale: float) -> np.ndarray:
    """
    Quantile function for the asymmetric laplace likelihood.

    Input:
        - x: (float or np.ndarray) values from 0 to 1, ie the desired quantile.
        - q: (float) quantile/asymmetry parameter of the asymmetric laplace distribution.
        - scale: (float) Scale parameter (> 0).

    Output:
        - rvs: (np.ndarray) quantiles of the asymmetric laplace distribution.
    """
    x = np.asarray(x, dtype=np.float64)
    rvs = np.zeros_like(x)
    ind = x <= q
    rvs[ind] = np.log(x[ind] / q) * scale / (1 - q)
    ind = ind == False
    rvs[ind] = -np.log((1 - x[ind]) / (1 - q)) * scale / q

    return rvs


def generate_input_grid(d: float, sample_size: float, lims: list) -> np.ndarray:
    """
    Function to generate the input of the latent GP.

    Input:
        - d: (float) input dimension.
        - sample_size: (float) number of data points/sample size.

    Output:
        - grid: (np.ndarray) (nxd) grid
    """
    lower_lim = lims[0]
    upper_lim = lims[1]
    grid = np.random.uniform(lower_lim, upper_lim, size=(sample_size, d))

    return grid


def simulate_latentGP(
    X: np.ndarray, kernel: gpy.kernels.Kernel, n_samples: int
) -> np.ndarray:
    """
    Function to simulate the latent Gaussian Process f.

    Input:
        - X: (np.ndarray) (nxd) input grid.
        - kernel:
        - n_samples: (float) number of deired GP samples.

    Output:
        - f: (np.ndarray) (n x m_samples) GP samples.
    """

    N = X.shape[0]
    X_torch = torch.tensor(X, dtype=torch.float)
    # Evaluate the kernel
    covar = kernel(X_torch)
    mean = torch.zeros(N)  # Zero mean GP

    mvn = gpy.distributions.MultivariateNormal(mean, covar)
    f = mvn.sample(sample_shape=torch.Size([n_samples]))  # (n_samples x N)

    return f.numpy().T


def simulate_response(f: np.ndarray, noise: str, pars: dict) -> np.ndarray:
    """
    Simulate the noise and add it to the latent GP.

    Input:
        - f: (np.ndarray) latent GP.
        - noise: (str) noise model.
        - pars: (dict) dictionary containing the simulation parameters.

    Output:
        - y: (np.ndarray) reponse = f (latent GP) + error (from noise model)

    """
    n = len(f)

    if noise == "gaussian":
        sigma = pars["gaussian"]["scale"]
        # normv = norm()
        error = np.random.normal(0, sigma, n)

    elif noise == "ald":
        q = pars["ald"]["q"]
        scale = pars["ald"]["q"]
        error = quantile_func_asym_laplace(
            x=np.random.uniform(0, 1, n), q=q, scale=scale
        )

    elif noise == "t":
        sigma = pars["t"]["scale"]
        df = pars["t"]["df"]
        error = np.random.standard_t(df=df, size=n) * sigma

    elif noise == "chi":
        sigma = pars["chi"]["scale"]
        df = pars["chi"]["df"]
        error = np.random.chisquare(df=df, size=n) * sigma

    y = f + error

    return y


def obtain_quantile(
    f: np.ndarray, noise: str, pars: dict, target_quantile: float
) -> np.ndarray:
    """
    Recover true quantile.

    Input:
        - f: (np.ndarray) latent GP.
        - noise: (str) noise model.
        - pars: (dict) dictionary containing the simulation parameters.
        - target_quantile: (float) the quantile level of interest.

    Output:
        - true latent quantile: (np.ndarray)

    """
    n = len(f)

    if noise == "gaussian":
        sigma = pars["gaussian"]["scale"]
        normv = norm()
        delta = normv.ppf(target_quantile) * sigma

    elif noise == "ald":
        q = pars["ald"]["q"]
        scale = pars["ald"]["scale"]
        delta = quantile_func_asym_laplace(np.array(target_quantile), q, scale)

    elif noise == "t":
        sigma = pars["t"]["scale"]
        df = pars["t"]["df"]
        tv = t(df=df)
        delta = tv.ppf(target_quantile) * sigma

    elif noise == "chi":
        sigma = pars["chi"]["scale"]
        df = pars["chi"]["df"]
        chi = chi2(df=df)
        delta = chi.ppf(target_quantile) * sigma

    quantile = f + delta

    return quantile


def train_test_split(X: np.ndarray, y: np.ndarray, train_split: float) -> np.ndarray:
    """
    Deterministic train and test split.

    Input:
        - X: (np.ndarray) (n,d)
        - y: (np.ndarray) (n,0)
        - train_split: (float) percentage of data for training

    Output:
        - X_train, y_train, X_test, y_test: (np.ndarray)
    """

    N = len(y)
    train_size = int(train_split * N)
    X_train = X[:train_size, :]
    y_train = y[:train_size]
    X_test = X[train_size:, :]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test


##### LOAD DATA ######


def load_data(likelihood, sample_size, input_dim, replicate, train_split, file_path = None):
    """
    Load data from a .npz file containing X, y, and optionally f.
    Assumes the function is called from the root directory, so that paths work as expected.

    Input:


    Output:

    """
    folder_name = f"{sample_size}_{input_dim}"
    file_name = f"data_replicate_{replicate}.npz"
    if file_path is None:
        file_path = os.path.join(
            "data", "simulated_data", folder_name, likelihood, file_name
        )
    # print(f"path: {file_path}")
    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
    f = data["f"]

    X_train, y_train, X_test, y_test = train_test_split(
        X=X, y=y, train_split=train_split
    )

    return f, X_train, y_train, X_test, y_test


### FOR REAL DATASETS ###

def load_X_y(dataset_name, dir):
    path = os.path.join(dir, f"{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {path}")

    df = pd.read_csv(path)

    if dataset_name == "bike":
        # Example: predict count from weather and time features
        y = df["cnt"]
        X = df.drop(columns=["cnt", "casual", "registered", "dteday"])
    elif dataset_name == "house":
        y = df["median_house_value"]
        X = df.drop(columns=["median_house_value"])
    elif dataset_name == "power":
        y = df["Global_active_power"]
        X = df.drop(columns=["Global_active_power"])
    elif dataset_name == "protein":
        y = df["target"] if "target" in df else df.iloc[:, -1]
        X = df.drop(columns=[y.name])
    elif dataset_name == "elevators":
        y = df["failure"] if "failure" in df else df.iloc[:, -1]
        X = df.drop(columns=[y.name])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return X, y


def save_cv_splits(dataset_name, n_splits=5, dir="data/real_data", seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    os.makedirs(dir, exist_ok=True)

    X, y = load_X_y(dataset_name=dataset_name, dir = dir)

    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        splits.append({
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        })

    out_path = os.path.join(dir, f"{dataset_name}_cv{n_splits}_splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f)
    
    print(f"Saved {n_splits}-fold CV splits for '{dataset_name}' to {out_path}")


def load_cv_splits(dataset_name, dir="data/real_data_splits", n_splits=5):
    path = os.path.join(dir, f"{dataset_name}_cv{n_splits}_splits.json")
    with open(path, "r") as f:
        splits = json.load(f)
    
    return splits




















############# NOT IN USE ##################
# load real data
def load_real_data(name, data_dir="data/real_data", train_split=0.75):
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{name}' not found at {path}")

    df = pd.read_csv(path)

    if name == "bike":
        # Example: predict count from weather and time features
        y = df["cnt"]
        X = df.drop(columns=["cnt", "casual", "registered", "dteday"])
    elif name == "house":
        y = df["median_house_value"]
        X = df.drop(columns=["median_house_value"])
    elif name == "power":
        y = df["Global_active_power"]
        X = df.drop(columns=["Global_active_power"])
    elif name == "protein":
        y = df["target"] if "target" in df else df.iloc[:, -1]
        X = df.drop(columns=[y.name])
    elif name == "elevators":
        y = df["failure"] if "failure" in df else df.iloc[:, -1]
        X = df.drop(columns=[y.name])
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # make it a np.ndarray
    X = np.array(X)
    y = np.array(y)
    X_train, y_train, X_test, y_test  = train_test_split(
        X, y, train_split=train_split)
    

    return X_train, y_train, X_test, y_test




def simulate_latentGP_old(
    X: np.ndarray, kernel: gpy.kernels.Kernel, n_samples: int
) -> np.ndarray:
    """
    Function to simulate the latent Gaussian Process f.

    Input:
        - X: (np.ndarray) (nxd) input grid.
        - kernel:
        - n_samples: (float) number of deired GP samples.

    Output:
        - f: (np.ndarray) (n x m_samples) GP samples.
    """

    class ExactGPModel(gpy.models.ExactGP):
        def __init__(self, train_x, input_kernel):
            dummy_y = torch.zeros(train_x.shape[0])
            likelihood = gpy.likelihoods.GaussianLikelihood()
            super().__init__(train_x, dummy_y, likelihood)
            self.mean_module = gpy.means.ConstantMean()
            self.covar_module = input_kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    # Convert X to torch tensor
    X_torch = torch.tensor(X, dtype=torch.float)
    model = ExactGPModel(X_torch, kernel)

    # Switch to Evaluation Mode
    model.eval()
    with torch.no_grad():
        mvn = model(X_torch)
        f = mvn.sample(sample_shape=torch.Size([n_samples]))

    # Evaluate the kernel
    covar = kernel(X_torch)
    mean = torch.zeros(X.shape[0])  # Zero mean GP

    mvn = gpy.distributions.MultivariateNormal(mean, covar)
    f = mvn.sample(sample_shape=torch.Size([n_samples]))  # (n_samples x N)

    return f.numpy().T, mvn
