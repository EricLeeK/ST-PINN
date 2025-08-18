#导入库
import torch
import torch.nn as nn
import numpy as np


class SeparatedPINN_PolynomialTime(nn.Module):
  """
    一个PINN模型，时间分支使用固定的多项式基。
    解的形式: u(x,t) = Σ C_i(x) * t^i
    
    参数:
        spatial_layers (list): 空间MLP的层级结构, e.g., [64, 64].
        poly_degree (int): 时间多项式的项数 L (即 t^0 到 t^(L-1)).
"""
    def __init__(self, spatial_layers, poly_degree):
        super(SeparatedPINN_PolynomialTime, self).__init__()
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

    def forward(self, x, t):
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        spatial_features = self.spatial_net(x)
        temporal_features = torch.cat([t.pow(i) for i in range(self.poly_degree)], dim=1)
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output


class SeparatedPINN_FourierTime(nn.Module):
    """
    一个PINN模型，时间分支使用固定的傅里叶基。
    
    参数:
        spatial_layers (list): 空间MLP的层级结构.
        num_freqs (int): 傅里叶特征(频率)的数量 L.
        freq_dist (str): 频率分布方式。可选 'linear' 或 'exponential'.
        freq_scale (float): 频率的缩放因子.
    """
    # --- 核心修改 1: 在 __init__ 中添加 freq_dist 参数 ---
    def __init__(self, spatial_layers, num_freqs, freq_dist="linear", freq_scale=1.0):
        super(SeparatedPINN_FourierTime, self).__init__()
        self.num_freqs = num_freqs
        
        spatial_input_dim = 1
        spatial_output_dim = 2 * num_freqs
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)
        
        # --- 核心修改 2: 根据 freq_dist 的值选择频率生成方式 ---
        if freq_dist == "linear":
            # 线性增长: 1π, 2π, 3π, ...
            self.freqs = freq_scale * torch.pi * torch.arange(1, num_freqs + 1)
        elif freq_dist == "exponential":
            # 指数增长: 1π, 2π, 4π, ...
            self.freqs = freq_scale * torch.pi * (2.0 ** torch.arange(0, num_freqs))
        else:
            raise ValueError(f"未知的频率分布 '{freq_dist}'. 请选择 'linear' 或 'exponential'.")

    def forward(self, x, t):
        if self.freqs.device != t.device:
            self.freqs = self.freqs.to(t.device)
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        coeffs = self.spatial_net(x)
        t_proj = t @ self.freqs.unsqueeze(0)
        temporal_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        output = torch.sum(coeffs * temporal_features, dim=1, keepdim=True)
        return output