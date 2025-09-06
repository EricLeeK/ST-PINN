# run_experiment_heat2d.py
# Example experiment for 2D Heat equation with varying coefficients using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.heat import Heat2D_VaryingCoef
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Heat equation with varying coefficients
    pde = Heat2D_VaryingCoef(
        datapath=r"ref/heat_darcy.dat",
        bbox=[0, 1, 0, 1, 0, 5],        # [x_min, x_max, y_min, y_max, t_min, t_max]
        A=200,                          # Source term amplitude
        m=(1, 5, 1)                     # Source term frequencies
    )
    
    # Create separated network with polynomial time basis
    # Note: Heat2D has input_dim=3 (x, y, t) and output_dim=1
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[64, 64, 64],    # Spatial network architecture
        poly_degree=15                   # Polynomial time basis degree
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
    
    return model

# Define training parameters
train_args = {
    'iterations': 15000,
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
    trainer = Trainer(exp_name="Heat2D_VaryingCoef_Polynomial", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D热方程（变系数）+ 多项式时间基")
    trainer.train_all()
    print(">>> 实验完成！")