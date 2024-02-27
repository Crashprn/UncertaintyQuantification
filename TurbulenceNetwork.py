import torch
import torch.nn as nn
import typing as t

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class TurbulenceNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, nodes: t.List[int], dropout=0.0) -> None:
        super(TurbulenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.nodes = nodes
        self.dropout = dropout
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, nodes[i], dtype=torch.float64))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

            elif i == num_layers - 1:
                layers.append(nn.Linear(nodes[i-1], output_dim, dtype=torch.float64))
                #layers.append(nn.Tanh())

            else:
                layers.append(nn.Linear(nodes[i-1], nodes[i], dtype=torch.float64))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

                
        self.model = nn.Sequential(*layers)
    
    def dropout_on(self, p):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.model(x)
        return out1


class TurbulenceNetworkBayesian(PyroModule):
    def __init__(
            self,
            input_dim: int, 
            output_dim: int, 
            h_nodes: t.List[int], 
            layers: int,
            device,
            data_size: int,
            layer_prior=(0,1), 
            output_prior=5
        ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.output_prior = output_prior
        self.data_size = data_size

        self.layer_sizes = torch.tensor([input_dim] + [h_nodes]*layers + [output_dim])

        layer_list = [PyroModule[nn.Linear](self.layer_sizes[i-1], self.layer_sizes[i]) for i in range(1, len(self.layer_sizes))]
        self.layers = PyroModule[nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.layer_sizes[layer_idx])).expand([self.layer_sizes[layer_idx+1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(*layer_prior).expand([self.layer_sizes[layer_idx+1]]).to_event(1))

        self.activation = nn.ReLU()

#        self.sigma = pyro.param('sigma', torch.eye(self.output_dim, device=self.device)*self.output_prior, constraint=dist.constraints.positive)

    
    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:

        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        mu = self.layers[-1](x)

        sigma = pyro.param('sigma', torch.eye(self.output_dim, device=self.device)*self.output_prior, constraint=dist.constraints.positive)

        with pyro.plate("data", size=self.data_size, subsample_size=x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=sigma), obs=y)
        return mu


        
