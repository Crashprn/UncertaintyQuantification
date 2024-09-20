from torch import nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

class RBFKernel(nn.Module):
    def __init__(self, length_scale=1.0, amplitude=1.0, device=None):
        super(RBFKernel, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.length_scale = nn.Parameter(torch.tensor(length_scale))
        self.lower_bound = torch.tensor([1e-6], device=self.device)
        self.upper_bound = torch.tensor([1e6], device=self.device)

    def get_params(self):
        return {'amplitude': self.amplitude.cpu().detach().item(), 'length_scale': self.length_scale.cpu().detach().item()} 

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(0)
        amplitude = self.amplitude.clamp(self.lower_bound, self.upper_bound)
        length_scale = self.length_scale.clamp(self.lower_bound, self.upper_bound)

        dists = (x1 - x2).pow(2).sum(2)

        return amplitude ** 2 * torch.exp(-0.5 * dists / length_scale)


        #return amplitude ** 2 * torch.exp(-0.5 * (torch.linalg.norm(x1-x2, dim=2)) / length_scale)


class RBFKernelIndependent(nn.Module):
    def __init__(self, length_scale=[1.0, 1.0], amplitude=1.0, device=None):
        super(RBFKernelIndependent, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.length_scale = nn.Parameter(torch.tensor(length_scale))
        self.lower_bound = torch.tensor([1e-6], device=self.device)
        self.upper_bound = torch.tensor([1e6], device=self.device)
    
    def get_params(self):
        return {'amplitude': self.amplitude.cpu().detach().item(), 'length_scale': self.length_scale.cpu().detach().tolist()}

    def forward(self, x1, x2):
        amplitude = self.amplitude.clamp(self.lower_bound, self.upper_bound)
        length_scale = self.length_scale.clamp(self.lower_bound, self.upper_bound)

        return amplitude ** 2 * torch.exp(-0.5 * (x1 - x2) @ torch.diag(1 / length_scale ** 2) @ (x1 - x2).T)

class SinusoidalKernel(nn.Module):
    def __init__(self, amplitude=1.0, length_scale=1.0, period=1.0, device=None):
        super(SinusoidalKernel, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.length_scale = nn.Parameter(torch.tensor(length_scale))
        self.period = nn.Parameter(torch.tensor(period))
        self.lower_bound = torch.tensor([1e-6], device=self.device)
        self.upper_bound = torch.tensor([1e6], device=self.device)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(0)
        amplitude = self.amplitude.clamp(self.lower_bound, self.upper_bound)
        length_scale = self.length_scale.clamp(self.lower_bound, self.upper_bound)
        period = self.period.clamp(self.lower_bound, self.upper_bound)

        return amplitude ** 2 * torch.exp(-2 * torch.sin(torch.pi *torch.linalg.norm(x1 - x2, dim=2)*period)**2 / length_scale)

class GaussianProcessRegressor:
    def __init__(self, kernel, noise=0.0, max_iter=1000, lr=1e-3, batch_size=32, tol=1e-2, device=None, delta=1e-10, verbose=True):
        self.kernel = kernel
        self.noise = noise
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.delta = delta
        self.batch_size = batch_size
        self.verbose = verbose
    
    def fit(self, X, y):
        self.L = None
        self.X_train = X
        self.y_train = y
        self.losses = []
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)


        optimizer = torch.optim.Adam(self.kernel.parameters(), lr=self.lr)

        for i in range(self.max_iter):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                K = self.kernel(X_batch, X_batch) + (self.noise + self.delta) * torch.eye(X_batch.shape[0], device=self.device)
                self.L = torch.linalg.cholesky(K)

                loss = self.neg_log_likelihood(self.L, y_batch)
                if i > 0 and abs(self.losses[-1] - loss.item()) < self.tol:
                    break
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
            if i % 50 == 0 and self.verbose:
                print(f'Iteration {i:6d}, Loss: {loss.item():10.2f}')
        
        # Compute final Cholesky decomposition of the kernel matrix and save the inverse
        K = self.kernel(X, X) + (self.noise + self.delta) * torch.eye(X.shape[0], device=self.device)
        self.L = torch.linalg.cholesky(K)
        self.K_inv = torch.cholesky_inverse(self.L)

        return self
    
    def neg_log_likelihood(self, L, y):
        mvnormal = MultivariateNormal(torch.zeros(y.shape[0], device=self.device, dtype=y.dtype), scale_tril=L)

        return -mvnormal.log_prob(y.squeeze()).sum()
    
    def predict(self, X, num_samples=None, return_std=False):
        with torch.no_grad():
            K_test = self.kernel(X, self.X_train)
            mean = K_test @ self.K_inv @ self.y_train

            if return_std or num_samples is not None:
                cov = self.kernel(X, X) - K_test @ self.K_inv @ K_test.T
                cov = cov + self.delta * torch.eye(X.shape[0], device=self.device)

            if num_samples is None:
                if return_std:
                    return mean, cov.diag().sqrt()
                else:
                    return mean
            else:
                mv_normal = MultivariateNormal(mean.squeeze(), covariance_matrix=cov)
                if return_std:
                    return mv_normal.sample((num_samples,)), cov.diag().sqrt()
                else:
                    return mv_normal.sample((num_samples,))
