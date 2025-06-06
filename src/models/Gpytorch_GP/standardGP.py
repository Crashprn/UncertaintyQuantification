import sys
import os

sys.path.append(os.path.abspath('.'))

import torch
import gpytorch
from src.models.Gpytorch_GP.batchedGPVariants import GPModel
from torch.optim import Adam
import math
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.constraints import Interval

"""NOTATIONS
   D: input dimensions
   N: number of training data points
   T: number of testing data points
   I: number of inducing points
   B: number of GPs in the batch 
   b: batch size for hyper-parameter stochastic optimization
"""

class standardGP():
    def __init__(self, num_inducing, 
                 initial_inducing_pts, 
                 learn_inducing, 
                 num_dim, num_GPs, 
                 train_inp, train_out,
                 device):
        """Initialize the standard GP model trained with variational inference inducing points method

        Args:
            num_inducing (int): The number of inducing point locations
            initial_inducing_pts (torch Tensor): The intial inducing points guess [I X D]
            learn_inducing (bool): Learn best inducing points or not
            num_dim (int): The number of input dimensions [D]
            num_GPs (int): The number of GP trained independently in the batch
            train_x (torch Tensor): The input training data [B X N X D]
            train_y (torch Tensor): The output training data [B X N 1]
        """
        self.num_inducing = num_inducing
        self.initial_inducing_pts = initial_inducing_pts
        self.num_dim = num_dim
        self.num_GPs = num_GPs
        self.learn_inducing = learn_inducing
        self.train_x = train_inp
        self.train_y = train_out
        self.device = device
    def train(self, epochs, learning_rate, model_params=None, train_noise=False):
        """Train the GP model

        Args:
            epochs (int): The number of training epochs
            learning_rate (real): The learning rate

        Returns:
            model: The trained GP model as a gpytorch object    
        """
        model = GPModel(inducing_points=self.initial_inducing_pts, input_dims=self.num_dim, learn_inducing=self.learn_inducing, batch=self.num_GPs).to(self.device, dtype=torch.float64)
        if model_params is not None:
            model.load_state_dict(model_params)
        if not train_noise:
            model.likelihood.noise_covar.raw_noise.requires_grad = False
        model.train()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=self.train_x.shape[1])
        for i in range(epochs):
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y).sum()
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   ' % (
                i + 1, epochs, loss.detach().item(),
            ))
            optimizer.step()
        model.eval()
        return model

    def predict(self, test_x, model, batch_size=1, n_samples=None):
        """Predict using the GP model

        Args:
            test_x (torch tensor): The testing inputs [T X D]

        Returns:
            means: The GP predictive means in a batch [B X T]
            variances: The GP predictive variances in a batch [B X T]
        """
        test_y = torch.ones(test_x.size()).to(test_x.device) # this is dummy
        with torch.no_grad():
            test_dataset = TensorDataset(test_x, test_y)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            out = model.predict(test_loader, n_samples)

        if n_samples is None:
            return out[0], out[1]
        else:
            return out 
