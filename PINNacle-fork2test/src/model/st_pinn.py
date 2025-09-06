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
        
        # Determine spatial and output dimensions from layer_sizes
        self.total_input_dim = layer_sizes[0]  # Total input dimension (spatial + time)
        self.output_dim = layer_sizes[-1]      # Output dimension
        self.spatial_input_dim = self.total_input_dim - 1  # Spatial dims = total - 1 (time)
        
        # Spatial network outputs poly_degree features for each output component
        spatial_output_dim = self.poly_degree * self.output_dim
        
        modules = [nn.Linear(self.spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)

    def forward(self, inputs):
        # Split spatial and temporal coordinates
        spatial_coords = inputs[:, :-1]  # All columns except last
        t = inputs[:, -1:]               # Last column (time)
        
        # Get spatial features
        spatial_features = self.spatial_net(spatial_coords)  # [batch, poly_degree * output_dim]
        
        # Get temporal features
        temporal_features = torch.cat([t.pow(i) for i in range(self.poly_degree)], dim=1)  # [batch, poly_degree]
        
        # Reshape spatial features for tensor product
        spatial_features = spatial_features.view(-1, self.output_dim, self.poly_degree)  # [batch, output_dim, poly_degree]
        
        # Compute tensor product and sum over polynomial terms
        # spatial_features: [batch, output_dim, poly_degree]
        # temporal_features: [batch, poly_degree]
        output = torch.sum(spatial_features * temporal_features.unsqueeze(1), dim=2)  # [batch, output_dim]
        
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
        
        # Determine spatial and output dimensions from layer_sizes
        self.total_input_dim = layer_sizes[0]  # Total input dimension (spatial + time)
        self.output_dim = layer_sizes[-1]      # Output dimension
        self.spatial_input_dim = self.total_input_dim - 1  # Spatial dims = total - 1 (time)
        
        # Spatial network outputs 2*num_frequencies features for each output component
        spatial_output_dim = 2 * num_frequencies * self.output_dim
        
        modules = [nn.Linear(self.spatial_input_dim, spatial_layers[0])]
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
        # Split spatial and temporal coordinates
        spatial_coords = inputs[:, :-1]  # All columns except last
        t = inputs[:, -1:]               # Last column (time)
        
        # Ensure frequencies are on the same device as input
        if self.freqs.device != t.device:
            self.freqs = self.freqs.to(t.device)
        
        # Get spatial features
        spatial_features = self.spatial_net(spatial_coords)  # [batch, 2*num_frequencies*output_dim]
        
        # Compute temporal features: [sin(ω₁t), cos(ω₁t), sin(ω₂t), cos(ω₂t), ...]
        t_proj = t * self.freqs.unsqueeze(0)  # Broadcasting [batch, num_frequencies]
        temporal_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)  # [batch, 2*num_frequencies]
        
        # Reshape spatial features for tensor product
        spatial_features = spatial_features.view(-1, self.output_dim, 2 * self.num_frequencies)  # [batch, output_dim, 2*num_frequencies]
        
        # Compute tensor product and sum over Fourier modes
        # spatial_features: [batch, output_dim, 2*num_frequencies]
        # temporal_features: [batch, 2*num_frequencies]
        output = torch.sum(spatial_features * temporal_features.unsqueeze(1), dim=2)  # [batch, output_dim]
        
        return output