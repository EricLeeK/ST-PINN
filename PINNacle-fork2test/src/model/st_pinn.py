# src/model/st_pinn.py

import torch
import torch.nn as nn
import deepxde as dde

# --------------------------------------------------------------------------
# 确保这个类的名字是 SeparatedNetPolynomial，一个字母都不能错
# --------------------------------------------------------------------------
class SeparatedNetPolynomial(dde.nn.pytorch.NN):
    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        spatial_layers=[64, 64, 64],
        poly_degree=20
    ):
        super().__init__()
        self.poly_degree = poly_degree
        
        spatial_input_dim = 1
        spatial_output_dim = poly_degree
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        
        spatial_features = self.spatial_net(x)
        temporal_features = torch.cat([t.pow(i) for i in range(self.poly_degree)], dim=1)
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output