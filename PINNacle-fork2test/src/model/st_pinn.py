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


class SeparatedNetFourier(dde.nn.pytorch.NN):
    """
    Separated neural network with Fourier time basis.
    Solution form: u(x,t) = Σ C_i(x) * [sin(ω_i*t), cos(ω_i*t)]
    """
    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        spatial_layers=[64, 64, 64],
        num_frequencies=10,
        freq_type="linear",
        freq_scale=1.0
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        spatial_input_dim = 1
        spatial_output_dim = 2 * num_frequencies  # sin and cos for each frequency
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)
        
        # Generate frequencies
        if freq_type == "linear":
            self.freqs = freq_scale * torch.pi * torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        elif freq_type == "exponential":
            self.freqs = freq_scale * torch.pi * (2.0 ** torch.arange(0, num_frequencies, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown freq_type '{freq_type}'. Choose 'linear' or 'exponential'.")

    def forward(self, inputs):
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        
        # Ensure frequencies are on the same device as input
        if self.freqs.device != t.device:
            self.freqs = self.freqs.to(t.device)
        
        spatial_features = self.spatial_net(x)
        
        # Compute temporal features: [sin(ω₁t), cos(ω₁t), sin(ω₂t), cos(ω₂t), ...]
        t_proj = t * self.freqs.unsqueeze(0)  # Broadcasting
        temporal_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output