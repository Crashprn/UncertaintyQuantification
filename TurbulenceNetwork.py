import torch
import torch.nn as nn
import typing as t

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, Sequential

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
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, nodes: t.List[int], layer_prior=(0,1)) -> None:
        super(TurbulenceNetworkBayesian, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.nodes = nodes

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(PyroSample(dist.Normal(*layer_prior).expand([nodes[i], input_dim]).to_event(2)))
                layers.append(nn.ReLU())

            elif i == num_layers - 1:
                layers.append(PyroSample(dist.Normal(*layer_prior)).expand([output_dim, nodes[i]]).to_event(2)))

            else:
                layers.append(PyroSample(dist.Normal(*layer_prior).expand([nodes[i], nodes[i-1]]).to_event(2)))
                layers.append(nn.ReLU())
        
        self.model = Sequential(*layers)
