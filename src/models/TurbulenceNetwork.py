import torch
import torch.nn as nn
import typing as t

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


'''
Feedforward neural network for turbulence modeling.
'''
class TurbulenceNetwork(nn.Module):
    def __init__(
            self,
            input_dim: int, output_dim: int,
            num_layers: int, h_dim:int,
            dropout:list[float]=None,
            out_noise:str = 'none',
            out_noise_scale:float = 0.1
        ) -> None:
        super(TurbulenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.h_dim = h_dim
        self.out_noise = out_noise
        self.heteroscedastic = True if out_noise == "heteroscedastic" else False
        self.homoscedastic = True if out_noise == "homoscedastic" else False

        # Set the dropout rates for each layer
        if dropout is None:
            self.dropout = [0.0]*num_layers
        else:
            self.dropout = dropout

        # If heteroscedastic noise is used, double output dimension
        if self.heteroscedastic:
            self.output_dim = 2 * output_dim

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout[i]))
            elif i == num_layers - 1:
                layers.append(nn.Linear(h_dim, self.output_dim))
            else:
                layers.append(nn.Linear(h_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout[i]))
        
        if self.homoscedastic:
            self.sigma = nn.Parameter(torch.ones((1,output_dim))*out_noise_scale, requires_grad=True)

        self.layers = nn.Sequential(*layers)
                
    '''
    Turn on dropout layers prediction time.
    '''
    def dropout_on(self) -> None:
        for m in self.layers.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x: torch.Tensor, noise=True) -> torch.Tensor:
        out = self.layers(x)

        if self.heteroscedastic and noise:
            mu = out[:, :self.output_dim//2]
            sigma = out[:, self.output_dim//2:]
            sigma = torch.exp(sigma)
            return mu, sigma
        elif self.heteroscedastic and not noise:
            return out[:, :self.output_dim//2]
        elif self.homoscedastic and noise:
            return out, self.sigma**2
        else:
            return out

'''
Bayesian feedforward neural network for turbulence modeling with Gaussian priors.
'''
class TurbulenceNetworkBayesian(PyroModule):
    def __init__(
            self,
            input_dim: int, 
            output_dim: int, 
            h_nodes: int, 
            num_layers: int,
            device,
            data_size: int,
            layer_prior=(0,1), 
            output_prior_conc_rate=(3.0, 1.0),
            activation=nn.ReLU,
            train_noise=False
        ) -> None:
        super().__init__()
        self.input_dim = torch.tensor(input_dim , device=device)
        self.output_dim = torch.tensor(output_dim, device=device)
        self.h_nodes = torch.tensor(h_nodes, device=device)
        self.device = device
        self.output_prior_conc_rate = output_prior_conc_rate
        self.data_size = data_size
        self.activation = activation
        self.train_noise = train_noise

        layers = []

        if num_layers <= 1:
            layer = PyroModule[nn.Linear](self.input_dim, self.output_dim)
            layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.output_dim)).expand([self.output_dim, self.input_dim]).to_event(2))
            layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.output_dim]).to_event(1))
            layers.append(layer)
        else:
            for i in range(num_layers):
                if i == 0:
                    layer = PyroModule[nn.Linear](self.input_dim, self.h_nodes)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.h_nodes)).expand([self.h_nodes, self.input_dim]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.h_nodes]).to_event(1))
                    layers.append(layer)
                    if activation is not None:
                        layers.append(activation())
                elif i == num_layers - 1:
                    layer = PyroModule[nn.Linear](self.h_nodes, self.output_dim)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.output_dim)).expand([self.output_dim, self.h_nodes]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.output_dim]).to_event(1))
                    layers.append(layer)
                else:
                    layer = PyroModule[nn.Linear](self.h_nodes, self.h_nodes)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.h_nodes)).expand([self.h_nodes, self.h_nodes]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.h_nodes]).to_event(1))
                    layers.append(layer)
                    if activation is not None:
                        layers.append(activation())

        self.layers = PyroModule[nn.ModuleList](layers)

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:

        for i in range(len(self.layers)):
            x = self.layers[i](x)

        mu = x

        if self.train_noise:
            sigma = pyro.sample("sigma", dist.Gamma(self.output_prior_conc_rate[0], self.output_prior_conc_rate[1]).expand([self.output_dim]).to_event(1))
            sigma = torch.diag(sigma**2)
        else:
            sigma = torch.eye(self.output_dim, device=self.device)

        with pyro.plate("data", size=self.data_size, subsample_size=x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=sigma), obs=y)

        return mu
