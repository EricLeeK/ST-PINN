# run_experiment_grayscott.py
# Experiment for Gray-Scott reaction-diffusion system using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.chaotic import GrayScottEquation
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize Gray-Scott reaction-diffusion system
    pde = GrayScottEquation(
        datapath=r"PINNacle-fork2test/ref/grayscott.dat",
        bbox=[-1, 1, -1, 1, 0, 200],   # [x_min, x_max, y_min, y_max, t_min, t_max]
        b=0.04,                        # Feed rate
        d=0.1,                         # Kill rate  
        epsilon=(1e-5, 5e-6)           # Diffusion coefficients for u, v
    )
    
    # Create separated network with Fourier time basis
    # Gray-Scott has input_dim=3 (x, y, t) and output_dim=2 (u, v concentrations)
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for complex patterns
        num_frequencies=20,                    # Many Fourier modes for complex dynamics
        freq_type="exponential",               # Exponential frequency distribution
        freq_scale=0.5                         # Lower frequency scale for long-time dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Conservative learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 30000,  # Many iterations for complex chaotic system
    'callbacks': [TesterCallback(log_every=2000)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="GrayScott_Fourier_ReactionDiffusion", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！Gray-Scott反应扩散系统 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")