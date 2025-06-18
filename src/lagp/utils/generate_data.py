import os

import gpytorch as gpy
import numpy as np
import pandas as pd
import torch
import json
from scipy.stats import chi2, norm, t, bernoulli, poisson, gamma
from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit


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


def make_latents(n_groups, group_size, pars: dict):

    N = n_groups * group_size
    signal_variance = pars.get("signal_variance", 1)

    signal_variance_2 = pars.get("signal_variance_2", 1)
    n_groups_2 = pars.get("n_groups_2", 10)

    randef = pars.get("randeff", "One_random_effect")


    sigma2 = pars.get("sigma2", 0.5**2)
    sigma2_1 = pars.get("sigma2_1", 0.5**2)
    sigma2_2 = pars.get("sigma2_2", 0.5**2)
    sigma2_3 = pars.get("sigma2_3", 0.5**2)
    #randef = pars.get("randef", "One_random_effect")
    has_F = pars.get("has_F", False)
    factor_m2 = pars.get("factor_m2", 1)
    num_covariates = pars.get("num_covariates", 5)

    
    # Grouping var 1
    group = np.repeat(np.arange(n_groups), repeats = group_size)
    
    # Random effect 1
    b1 = np.random.normal(0, np.sqrt(signal_variance), size=n_groups)
    
    if randef == "One_random_effect":
        eps = b1[group]
        group_data = group
    
    elif randef == "Two_completely_crossed_random_effects":
        # assert group_size >= n_groups_2, (
        # "To create a completely crossed design, group_size must be >= n_groups_2 "
        # "so each level of the second random effect is observed within each level of the first."
        # )
        # assert N % n_groups_2 == 0, (
        # "N must be divisible by n_groups_2 to tile group2 evenly across all observations."
        # )
        group2 = np.tile(np.arange(n_groups_2), N//n_groups_2)
        b2 = np.random.normal(0, np.sqrt(signal_variance_2), size=n_groups_2)
        eps = b1[group] + b2[group2]
        group_data = np.column_stack([group, group2])
    
    
    elif randef == "Two_randomly_crossed_random_effects":
        m2 = int(factor_m2 * group_size) if factor_m2 != 1 else group_size
        group2 = np.repeat(np.arange(m2), N // m2)
        np.random.shuffle(group2)
        b2 = np.random.normal(0, np.sqrt(signal_variance_2), size=m2)
        eps = b1[group] + b2[group2]
        group_data = np.column_stack([group, group2])


        ### ABOVE HERE CORRECT ###
    
    elif randef == "Two_nested_random_effects":
        m_nested = group_size * 2
        group2 = np.repeat(np.arange(m_nested), N // m_nested)
        b2 = np.random.normal(0, np.sqrt(sigma2_2), size=m_nested)
        eps = b1[group] + b2[group2]
        group_data = np.column_stack([group, group2])
    
    elif randef == "Three_randomly_crossed_random_effects":
        group2 = group.copy()
        group3 = group.copy()
        np.random.shuffle(group2)
        np.random.shuffle(group3)
        b2 = np.random.normal(0, np.sqrt(sigma2_2), size=group_size)
        b3 = np.random.normal(0, np.sqrt(sigma2_3), size=group_size)
        eps = b1[group] + b2[group2] + b3[group3]
        group_data = np.column_stack([group, group2, group3])
    
    else:
        raise ValueError(f"Unknown randef: {randef}")
    
    # Fixed effects
    if has_F:
        beta = np.concatenate(([0], np.ones(num_covariates)))
        X = np.random.normal(0, 1, size=(N, num_covariates))
        X = np.column_stack([np.ones(N), X])
        f = X @ beta
        delta_var_f = np.sqrt(sigma2 / np.var(f))
        X[:, 1:] *= delta_var_f
        fe = X @ beta
    else:
        X = np.zeros((N, 1))
        X[:, 0] = 1
        fe = np.zeros(N)
    
    # Simulate response y
    return X, fe, eps, group_data

def simulate_response(pars: dict, X, fe, eps, group_data, likelihood, g = None):

    N = len(X)
    # Simulate response y
    eta = fe + eps
    pars = pars["pars"]
    if likelihood == "ald":
        q = pars["ald"]["q"]
        scale = g if g is not None else pars["ald"]["scale"]
        u = np.random.uniform(0, 1, N)
        error = quantile_func_asym_laplace(x=u, q=q, scale=1.0) * scale
        y = eta + error
    elif likelihood == "gaussian":
        scale = pars["gaussian"]["scale"]
        error = np.random.normal(0, scale, size=N)
        y = eta + error
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")
    
    return {"y": y, "X": X, "group_data": group_data, "eps": eps, "fe": fe}

def obtain_quantile(
    eps: np.ndarray, noise: str, pars: dict, target_quantile: float, g: np.ndarray = None) -> np.ndarray:
    """
    Recover true quantile.

    Input:
        - f: (np.ndarray) latent GP.
        - noise: (str) noise model.
        - pars: (dict) dictionary containing the simulation parameters.
        - target_quantile: (float) the quantile level of interest.
        - g: (np.ndarray) GP that determines the scale of the noise.

    Output:
        - true latent quantile: (np.ndarray)

    """

    n = len(eps)
    if g is not None:
        assert len(g) == n, "g must have the same length as f."

    if noise == "gaussian":
        scale = g if g is not None else pars["gaussian"]["scale"]
        delta = norm.ppf(target_quantile) * scale

    elif noise == "ald":
        q = pars["ald"]["q"]
        scale = g if g is not None else pars["ald"]["scale"]
        # Quantile function of ALD with scale=1, then multiply
        delta = quantile_func_asym_laplace(np.array(target_quantile), q, 1) * scale

    elif noise == "t":
        df = pars["t"]["df"]
        scale = g if g is not None else pars["t"]["scale"]
        delta = t(df=df).ppf(target_quantile) * scale

    elif noise == "chi":
        df = pars["chi"]["df"]
        scale = g if g is not None else pars["chi"]["scale"]
        delta = chi2(df=df).ppf(target_quantile) * scale

    else:
        raise ValueError(f"Unsupported noise model: {noise}")

    return eps + delta
    

def compute_dict_pars(signal_variance, snr, quantile):
    """
    Assumes GP signal variance to be signal_variance. Hence target variance is signal_variance/snr.
    If snr = 10 and signal_variance = 1, then target variance = 0.1.
    """
    target_variance = signal_variance/snr
    # guassian
    scale_gaussian = np.sqrt(target_variance)

    # ald
    scale_ald = np.sqrt(target_variance * (quantile**2 * (1-quantile)**2/(1-2*quantile + 2*quantile**2)))

    # student t with 3 dfs
    scale_t = np.sqrt(0.5 * target_variance)

    # chi ?

    pars = {
    "gaussian": {"scale": scale_gaussian}, 
    "ald": {"q": quantile, "scale": scale_ald}, 
    "t": {"df": 3, "scale": scale_t},
    "chi": {"df": 1, "scale": 1} # fixed to 1 for now
    }

    return pars


def train_test_split_mixed_data(
    X, y, group_data, fe, eps, n_groups, group_size, likelihood,
    randeff, replicate=0, test_size=0.2, extrapolate=False,
    output_folder="simulated_data_mm"
):
    """
    Perform train-test split (i.i.d. or group-aware) and save to .npz file.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Targets.
        group_data (np.ndarray): Group assignments.
        sample_size (int): Total number of samples.
        input_dim (int): Input dimensionality.
        likelihood (str): Name of the likelihood/noise type.
        replicate (int): Replication ID.
        test_size (float): Fraction of data to allocate to test set.
        random_state (int): Seed for reproducibility.
        extrapolate (bool): If True, test set will contain unseen groups.
        f (np.ndarray or None): Latent function, optional.
        output_folder (str): Base folder to save the data.
    """

    if not extrapolate:
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=1 + replicate # each replicate gets a different split
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        group_train, group_test = group_data[train_idx], group_data[test_idx]
        fe_train, fe_test = fe[train_idx], fe[test_idx]
        eps_train, eps_test = eps[train_idx], eps[test_idx]
    
    else:
        if group_data.ndim == 1:
            groups = group_data
        else:
            groups = group_data[:, 0]  # Can extend to tuple(group_data.T) for composite groups

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=12)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        group_train, group_test = group_data[train_idx], group_data[test_idx]
        fe_train, fe_test = fe[train_idx], fe[test_idx]
        eps_train, eps_test = eps[train_idx], eps[test_idx]

    # Define file path
    output_folder_randeff = os.path.join(output_folder, randeff)
    os.makedirs(output_folder_randeff, exist_ok=True)
    folder = os.path.join(output_folder_randeff, f"{n_groups}_{group_size}", likelihood)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"data_replicate_{replicate}.npz")

    # Save data
    np.savez_compressed(
        file_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        group_train=group_train,
        group_test=group_test,
        fe_train = fe_train,
        fe_test = fe_test,
        eps_train = eps_train,
        eps_test = eps_test  
    )

    print(f"Saved to: {file_path}")


def load_data(
    randeff: str, 
    likelihood: str,
    n_groups: int,
    group_size: int,
    replicate: int,
    file_path: str = None,
):
    """
    Load pre-split training and testing data from a .npz file.

    Parameters:
        likelihood (str): Type of noise model used.
        sample_size (int): Number of total samples.
        input_dim (int): Input dimensionality.
        replicate (int): Replication number.
        file_path (str, optional): If provided, loads directly from this path.

    Returns:
        dict: Contains f, X_train, y_train, X_test, y_test, group_train, group_test
    """
    folder_name = os.path.join(randeff, f"{n_groups}_{group_size}")
    file_name = f"data_replicate_{replicate}.npz"
    if file_path is None:
        file_path = os.path.join(
            "data", "simulated_data_mm", folder_name, likelihood, file_name
        )

    data = np.load(file_path)
    
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "group_train": data["group_train"] if "group_train" in data else None,
        "group_test": data["group_test"] if "group_test" in data else None,
        "fe_train": data["fe_train"] if "fe_train" in data else None,
        "fe_test": data["fe_test"] if "fe_test" in data else None,
        "eps_train": data["eps_train"] if "eps_train" in data else None,
        "eps_test": data["eps_test"] if "eps_test" in data else None,
    }




##################### 
def load_scale_gp(likelihood, sample_size, input_dim, replicate, file_path=None):
    """
    Load the scale GP (g) from a .npz file. Assumes g exists in the file for heteroscedastic models.

    Input:
        - likelihood (str): Name of the likelihood.
        - sample_size (int): Number of samples.
        - input_dim (int): Input dimensionality.
        - replicate (int): Replicate index (starting from 1).
        - file_path (str, optional): Override the default path.

    Output:
        - g (np.ndarray): Scale GP.
    """
    folder_name = f"{sample_size}_{input_dim}"
    file_name = f"data_replicate_{replicate}.npz"
    if file_path is None:
        file_path = os.path.join("data", "simulated_data", folder_name, likelihood, file_name)

    data = np.load(file_path)
    if "g" not in data:
        raise ValueError(f"No scale GP 'g' found in file: {file_path}")

    return data["g"]

########################  FOR REAL DATASETS ##################

def prepare_real_data(dataset_name, dir="data/real_data_mm", subsample: bool = False):

    my_data = pd.read_csv(os.path.join(dir, f"{dataset_name}.csv"))

    if subsample: 
        # Subsample to 1000 rows for faster processing
        my_data = my_data.sample(n=1000, random_state=42)
        
           
    if dataset_name == "cars":
        # Categorical variables
        cat_vars = ["model_id"] #, "location_id"]
        group_data_df = my_data[cat_vars]

        # Convert categorical variables to numeric factor-like codes
        group_data = group_data_df.apply(lambda col: col.astype("category").cat.codes)

        # If you need it as a NumPy array (like the R matrix)
        group_data_np = group_data.to_numpy()

        # Drop target and specified dummy columns
        excluded_cols = [
            "price", "model_id", "location_id",
            "manufacturerbmw", "conditionexcellent", "fueldiesel", "title_statusclean",
            "transmissionautomatic", "drive4wd", "sizecompact", "typebus", "paint_colorblack"
        ]
        feat_cols = [col for col in my_data.columns if col not in excluded_cols]
        X = my_data[feat_cols].to_numpy()
        # Add intercept term
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack([intercept, X])        
        # Response
        Y = np.log(my_data["price"].to_numpy())


    np.savez_compressed(
        os.path.join(dir, f"{dataset_name}_preprocessed.npz"),
        X=X,
        group_data=group_data_np,
        Y=Y
    )

    
def load_X_y_preprocessed(dataset_name, dir):
    
    path = os.path.join(dir, f"{dataset_name}_preprocessed.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {path}")
    
    data = np.load(path)
    X = data["X"]   
    group_data = data["group_data"]
    y = data["Y"]

    return X, group_data, y


def save_cv_splits_preprocessed(dataset_name, n_splits=5, dir="data/real_data_mm", seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    os.makedirs(dir, exist_ok=True)

    X, group_data, y = load_X_y_preprocessed(dataset_name=dataset_name, dir = dir)

    splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        splits.append({
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "train_groups": group_data[train_idx].tolist(),
            "test_groups": group_data[test_idx].tolist(),
        })

    out_path = os.path.join(dir, f"{dataset_name}_cv{n_splits}_splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f)
    
    print(f"Saved {n_splits}-fold CV splits for '{dataset_name}' to {out_path}")


def load_cv_splits(dataset_name, dir="data/real_data_mm", n_splits=5):
    path = os.path.join(dir, f"{dataset_name}_cv{n_splits}_splits.json")
    with open(path, "r") as f:
        splits = json.load(f)
    
    return splits



##### FOR COHERENT SIGNAL TO NOISE RATIO DATA GENERATION #########

def compute_dict_pars(signal_variance, snr, quantile):
    """
    Assumes GP signal variance to be signal_variance. Hence target variance is signal_variance/snr.
    If snr = 10 and signal_variance = 1, then target variance = 0.1.
    """
    target_variance = signal_variance/snr
    # guassian
    scale_gaussian = np.sqrt(target_variance)

    # ald
    scale_ald = np.sqrt(target_variance * (quantile**2 * (1-quantile)**2/(1-2*quantile + 2*quantile**2)))

    # student t with 3 dfs
    scale_t = np.sqrt(0.5 * target_variance)

    # chi ?

    pars = {
    "gaussian": {"scale": scale_gaussian}, 
    "ald": {"q": quantile, "scale": scale_ald}, 
    "t": {"df": 3, "scale": scale_t},
    "chi": {"df": 1, "scale": 1} # fixed to 1 for now
    }

    return pars


def compute_u_scale_gp(signal_variance, pars):
    """
    Extract scale and compute mu, given also signal_variance to get correct mean for scale gp.
    """
    # loop over likelihoods
    def get_mu(scale):
        mu = np.log(scale)-signal_variance/2
        return mu
    
    mu_scale_gp = {l: get_mu(dic["scale"]) for l, dic in pars.items()}

    return mu_scale_gp
















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
