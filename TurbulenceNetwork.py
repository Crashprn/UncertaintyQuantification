import torch
import torch.nn as nn
import typing as t

class TurbulenceNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, nodes: t.List[int]) -> None:
        super(TurbulenceNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = num_layers
        self.nodes = nodes
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, nodes[i], dtype=torch.float64))
                layers.append(nn.ReLU())

            elif i == num_layers - 1:
                layers.append(nn.Linear(nodes[i-1], output_dim, dtype=torch.float64))
                #layers.append(nn.Tanh())

            else:
                layers.append(nn.Linear(nodes[i-1], nodes[i], dtype=torch.float64))
                layers.append(nn.ReLU())

                
        self.model1 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.model1(x)
        return out1
        