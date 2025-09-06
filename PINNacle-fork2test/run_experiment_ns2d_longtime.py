# run_experiment_ns2d_longtime.py
# Experiment for 2D Navier-Stokes long-time dynamics using polynomial time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.ns import NS2D_LongTime
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Navier-Stokes equation for long-time dynamics
    pde = NS2D_LongTime(
        datapath=r"PINNacle-fork2test/ref/ns_long.dat",
        nu=1 / 100,                      # Kinematic viscosity (Reynolds number = 100)
        bbox=[0, 2, 0, 1, 0, 5]          # [x_min, x_max, y_min, y_max, t_min, t_max]
    )
    
    # Create separated network with polynomial time basis
    # NS2D has input_dim=3 (x, y, t) and output_dim=3 (u, v, p)
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128, 128],  # Very deep network for NS complexity
        poly_degree=20                             # Moderate polynomial degree for long-time dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Conservative learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 35000,  # Many iterations for complex NS dynamics
    'callbacks': [TesterCallback(log_every=2500)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="NS2D_LongTime_Polynomial", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D Navier-Stokes长时间动力学 + 多项式时间基")
    trainer.train_all()
    print(">>> 实验完成！")