# run_experiment_wave1d.py
# Example experiment for 1D Wave equation using standard neural network

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model  
from src.pde.wave import Wave1D
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 1D Wave equation
    pde = Wave1D(
        C=2,              # Wave speed
        bbox=[0, 1, 0, 1], # Domain bounds [x_min, x_max, t_min, t_max]
        scale=1,          # Spatial scaling
        a=4               # Frequency parameter for solution
    )
    
    # Create neural network
    # Note: Wave1D has input_dim=2 (x, t) and output_dim=1
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128], # Deeper network for wave equation
        poly_degree=25                   # Higher degree for wave dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Smaller learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 18000,
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
    trainer = Trainer(exp_name="Wave1D_Polynomial_TimeSpace", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！1D波动方程 + 多项式时间基")
    trainer.train_all()
    print(">>> 实验完成！")