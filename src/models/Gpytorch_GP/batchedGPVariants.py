import gpytorch
import math
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import torch.nn.functional as F
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.constraints import Interval

"""NOTATIONS
   D: input dimensions
   N: number of training data points
   T: number of testing data points
   I: number of inducing points
   B: number of GPs in the batch 
   b: batch size for hyper-parameter stochastic optimization
"""

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, input_dims, learn_inducing, batch):
        """Initialize the standardGP model with approximate (variational) inference strategy

        Args:
            inducing_points (torch Tensor): The intial inducing points guess [I X D]
            input_dims (int): The number of input dimensions [D]
            learn_inducing (bool): Learn best inducing points or not
            batch (int): The number of GPs to train independently in the batch [B]
        """
        outputscale_constraint = Interval(0.05, 20.0)
        noise_constraint = Interval(1e-8, 0.2)
        lengthscale_constraint = Interval(0.005, 10.0)  

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([batch]))
        self.base_kernel = gpytorch.kernels.MaternKernel(lengthscale_constraint=lengthscale_constraint,nu=2.5,ard_num_dims=input_dims,batch_shape=torch.Size([batch]))
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel,batch_shape=torch.Size([batch]),outputscale_constraint=outputscale_constraint) #
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([batch]), noise_constraint=noise_constraint)
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dims, batch_shape=torch.Size([batch])), batch_shape=torch.Size([batch]))
        #self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    def forward(self, x):
        """Forward evaluation of the GP model

        Args:
            x (torch Tensor): The input values [B X N X D or B X b X D]

        Returns:
            gpytorch MultivariateNormal object: The GP prediction as a MultivariateNormal object
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    def predict(self, test_loader, n_samples=None):
        """Predict GP means and variances as a batch

        Args:
            test_loader (torch DataLoader object): The testing dataset as a DataLoader object

        Returns:
            torch Tensor: The predictive means [B X T]
            torch Tensor: The predictive variances [B X T]
        """
        with torch.no_grad():
            mus = []
            variances = []
            samples = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                if n_samples is None:
                    mus.append(preds.mean)
                    variances.append(preds.variance)
                else:
                    samples.append(preds.sample(torch.Size((n_samples,))))
            
            if len(samples) <= 0:
                return (torch.cat(mus, dim=0), torch.cat(variances, dim=0))
            else:
                return torch.cat(samples, dim=2)


