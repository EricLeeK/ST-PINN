# run_experiment_heat2d_multiscale.py
# Experiment for 2D Heat equation with multiscale features using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch
import numpy as np

from trainer import Trainer 

# Import PDE class and model
from src.pde.heat import Heat2D_Multiscale
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Heat equation with multiscale initial condition
    pde = Heat2D_Multiscale(
        bbox=[0, 1, 0, 1, 0, 5],                           # [x_min, x_max, y_min, y_max, t_min, t_max]
        pde_coef=(1 / np.square(500 * np.pi), 1 / np.square(np.pi)),  # Diffusion coefficients
        init_coef=(20 * np.pi, np.pi)                      # Multiscale initial condition coefficients
    )
    
    # Create separated network with Fourier time basis
    # Heat2D has input_dim=3 (x, y, t) and output_dim=1
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for multiscale features
        num_frequencies=25,                    # Many Fourier modes for complex temporal behavior
        freq_type="exponential",               # Exponential frequency distribution
        freq_scale=1.0                         # Standard frequency scaling
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=8e-4))
    
    return model

# Define training parameters
train_args = {
    'iterations': 20000,
    'callbacks': [TesterCallback(log_every=1000)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="Heat2D_Multiscale_Fourier", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D多尺度热方程 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")