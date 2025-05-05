import torch
import gpytorch as gpy
from gpytorch.distributions import Distribution


class AsymmetricLaplaceOutput(Distribution):
    def __init__(self, function_samples, quantile, scale):
        self.function_samples = function_samples
        self.quantile = quantile
        self.scale = scale
        super().__init__()

    def log_prob(self, observations):
        residuals = observations - self.function_samples
        scale = torch.abs(self.scale) + 1e-6
        tau = self.quantile
        log_const = torch.log(tau * (1 - tau) / scale)
        rho = torch.where(residuals >= 0, tau * residuals, (tau - 1) * residuals)
        return log_const - rho / scale

    @property
    def mean(self):
        return self.function_samples
    
class AsymmetricLaplaceLikelihood(gpy.likelihoods.Likelihood):
    def __init__(self, quantile=0.5):
        super().__init__()
        assert 0 < quantile < 1, "Quantile must be in (0,1)"
        self.quantile = quantile
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, function_samples, **kwargs):
        return AsymmetricLaplaceOutput(function_samples, self.quantile, self.scale)

    def log_prob(self, observations, function_samples):
        # This may now be unused, but fine to keep
        residuals = observations - function_samples
        scale = torch.abs(self.scale) + 1e-6
        tau = self.quantile
        log_const = torch.log(tau * (1 - tau) / scale)
        rho = torch.where(residuals >= 0, tau * residuals, (tau - 1) * residuals)
        return log_const - rho / scale
    

class GPRegressionModel(gpy.models.ApproximateGP):
    def __init__(self, train_x, inducing_threshold: int, num_inducing: int = 500):
        # Use full Cholesky variational distribution (full rank, no inducing point approximation)
        variational_distribution = gpy.variational.CholeskyVariationalDistribution(num_inducing if train_x.size(0) > inducing_threshold
                                                                                else train_x.size(0))
        
        # Ensure the inducing points are set as parameters
        num_features = train_x.size(1) if len(train_x.size()) > 1 else 1 
        inducing_points = torch.nn.Parameter(torch.randn(num_inducing, num_features))
        
        variational_strategy = gpy.variational.VariationalStrategy(
            self,
            inducing_points = inducing_points if train_x.size(0) > inducing_threshold else train_x,  # make it [N, D]
            variational_distribution=variational_distribution,
            learn_inducing_locations= True if train_x.size(0) > inducing_threshold else False
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.MaternKernel(nu = 1.5)
        )
        self.covar_module.jitter = 1e-4 # default is 1e-6

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean, cov)