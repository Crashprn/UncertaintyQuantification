import torch
import torch.nn as nn
import typing as t

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class TurbulenceNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, h_dim:int, dropout=0.0) -> None:
        super(TurbulenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.h_dim = h_dim
        self.dropout = dropout
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, h_dim, dtype=torch.float64))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

            elif i == num_layers - 1:
                layers.append(nn.Linear(h_dim, output_dim, dtype=torch.float64))
                #layers.append(nn.Tanh())

            else:
                layers.append(nn.Linear(h_dim, h_dim, dtype=torch.float64))
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
            residual_blocks: int,
            layer_per_residual: int,
            device,
            data_size: int,
            layer_prior=(0,1), 
            output_prior=5,
            activation=nn.ReLU
        ) -> None:
        super().__init__()
        self.input_dim = torch.tensor(input_dim , device=device)
        self.output_dim = torch.tensor(output_dim, device=device)
        self.h_nodes = torch.tensor(h_nodes, device=device)
        self.device = device
        self.output_prior = output_prior
        self.data_size = data_size

        # Creating stem for the network

        self.stem = PyroModule[nn.Linear](self.input_dim, self.h_nodes)
        self.stem.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.h_nodes)).expand([self.h_nodes, self.input_dim]).to_event(2))
        self.stem.bias = PyroSample(dist.Normal(*layer_prior).expand([self.h_nodes]).to_event(1))

        # Defining Residual Layers
        layer_list = [BayesianLinearResidualBlock(self.h_nodes, self.h_nodes, layer_per_residual, layer_prior, activation) for i in range(residual_blocks)]  
        self.layers = PyroModule[nn.ModuleList](layer_list)

        # Creating output layer
        self.output = PyroModule[nn.Linear](self.h_nodes, self.output_dim)
        self.output.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.output_dim)).expand([self.output_dim, self.h_nodes]).to_event(2))
        self.output.bias = PyroSample(dist.Normal(*layer_prior).expand([self.output_dim]).to_event(1))


    
    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:

        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
        mu = self.output(x)

        sigma = pyro.param('sigma', torch.eye(self.output_dim, device=self.device)*self.output_prior, constraint=dist.constraints.positive)

        with pyro.plate("data", size=self.data_size, subsample_size=x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, covariance_matrix=sigma), obs=y)
        return mu



class BayesianLinearResidual(PyroModule):
    def __init__(self, input_dim: int, output_dim: int, layer_prior=(0,1), nonlinearity=nn.ReLU, residual = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual

        self.linear = PyroModule[nn.Linear](input_dim, output_dim)

        self.linear.weight = PyroSample(dist.Normal(layer_prior[0], layer_prior[1]*torch.sqrt(2/self.input_dim)).expand([output_dim, input_dim]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(*layer_prior).expand([output_dim]).to_event(1))

        self.activation = nonlinearity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))

        return x + out if self.residual else out


class BayesianLinearResidualBlock(PyroModule):
    def __init__(self, input_dim: int, output_dim: int, layers:int, layer_prior=(0,1), nonlinearity=nn.ReLU) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


        layers = [BayesianLinearResidual(input_dim, output_dim, layer_prior, nonlinearity, residual=layers>1) for _ in range(layers)]
        self.layers = PyroModule[nn.ModuleList](layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for layer in self.layers:
            out = layer(out)

        return x + out