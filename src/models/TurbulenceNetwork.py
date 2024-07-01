import torch
import torch.nn as nn
import typing as t

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class TurbulenceNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, h_dim:int, dropout:list[float]=None) -> None:
        super(TurbulenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.h_dim = h_dim
        if dropout is None:
            self.dropout = [0.0]*num_layers
        else:
            self.dropout = dropout
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout[i]))
            elif i == num_layers - 1:
                layers.append(nn.Linear(h_dim, output_dim))
            else:
                layers.append(nn.Linear(h_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout[i]))


        self.layers = nn.Sequential(*layers)
                
    
    def dropout_on(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.layers(x)
        return out1


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
            output_prior=(0,5),
            activation=nn.ReLU
        ) -> None:
        super().__init__()
        self.input_dim = torch.tensor(input_dim , device=device)
        self.output_dim = torch.tensor(output_dim, device=device)
        self.h_nodes = torch.tensor(h_nodes, device=device)
        self.device = device
        self.output_prior = output_prior
        self.data_size = data_size
        self.activation = activation

        self.layers = PyroModule[nn.ModuleList]()

        if num_layers <= 1:
            layer = PyroModule[nn.Linear](self.input_dim, self.output_dim)
            layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.output_dim)).expand([self.output_dim, self.input_dim]).to_event(2))
            layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.output_dim]).to_event(1))
            self.layers.append(layer)
        else:
            for i in range(num_layers):
                if i == 0:
                    layer = PyroModule[nn.Linear](self.input_dim, self.h_nodes)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.h_nodes)).expand([self.h_nodes, self.input_dim]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.h_nodes]).to_event(1))
                    self.layers.append(layer)
                    if activation is not None:
                        self.layers.append(activation())
                elif i == num_layers - 1:
                    layer = PyroModule[nn.Linear](self.h_nodes, self.output_dim)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.output_dim)).expand([self.output_dim, self.h_nodes]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.output_dim]).to_event(1))
                    self.layers.append(layer)
                else:
                    layer = PyroModule[nn.Linear](self.h_nodes, self.h_nodes)
                    layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.h_nodes)).expand([self.h_nodes, self.h_nodes]).to_event(2))
                    layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.h_nodes]).to_event(1))
                    self.layers.append(layer)
                    if activation is not None:
                        self.layers.append(activation())
        
        self.sigma = torch.diag(self.output_prior)
        

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:

        for i in range(len(self.layers)):
            x = self.layers[i](x)

        mu = x

        #sigma = pyro.sample('sigma', dist.Uniform(self.output_prior[0], self.output_prior[1]).expand([self.output_dim]).to_event(1))
        #sigma = torch.diag(sigma)


        with pyro.plate("data", size=self.data_size, subsample_size=x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=self.sigma), obs=y)
        return mu
