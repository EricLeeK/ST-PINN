# run_experiment_burgers2d_fourier.py
# Example experiment for 2D Burgers equation using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.burgers import Burgers2D
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Burgers equation
    pde = Burgers2D(
        datapath=r"PINNacle-fork2test/ref/burgers2d_0.dat",
        icpath=(r"PINNacle-fork2test/ref/burgers2d_init_u_0.dat", 
                r"PINNacle-fork2test/ref/burgers2d_init_v_0.dat"),
        nu=0.001,  # Viscosity parameter
        L=4,       # Domain size
        T=1        # Time horizon
    )
    
    # Create separated network with Fourier time basis
    # Note: Burgers2D has input_dim=3 (x, y, t) and output_dim=2 (u, v components)
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deeper network for 2D problem
        num_frequencies=15,                    # Number of Fourier modes
        freq_type="linear",                    # Linear frequency distribution
        freq_scale=2.0,                         # Frequency scaling factor
        output_dim=2                         
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=8e-4))  # Careful learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 25000,  # More iterations for complex 2D problem
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
    trainer = Trainer(exp_name="Burgers2D_Fourier_TimeSpace", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D伯格斯方程 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")