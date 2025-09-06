# run_experiment_kuramoto_sivashinsky.py
# Experiment for Kuramoto-Sivashinsky equation using polynomial time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch
import numpy as np

from trainer import Trainer 

# Import PDE class and model
from src.pde.chaotic import KuramotoSivashinskyEquation
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize Kuramoto-Sivashinsky equation
    pde = KuramotoSivashinskyEquation(
        datapath=r"PINNacle-fork2test/ref/Kuramoto_Sivashinsky.dat",
        bbox=[0, 2 * np.pi, 0, 1],     # [x_min, x_max, t_min, t_max]
        alpha=100 / 16,                # Nonlinear coefficient
        beta=100 / (16 * 16),          # Second-order diffusion coefficient
        gamma=100 / (16**4)            # Fourth-order dispersion coefficient
    )
    
    # Create separated network with polynomial time basis
    # KS equation has input_dim=2 (x, t) and output_dim=1
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for complex chaotic behavior
        poly_degree=30                         # High polynomial degree for complex temporal dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=3e-4))  # Conservative learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 25000,  # Many iterations for chaotic system
    'callbacks': [TesterCallback(log_every=1500)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="KuramotoSivashinsky_Polynomial_Chaotic", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！Kuramoto-Sivashinsky方程 + 多项式时间基")
    trainer.train_all()
    print(">>> 实验完成！")