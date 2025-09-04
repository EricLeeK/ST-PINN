# run_experiment_burgers1d_fourier.py
# Example experiment for 1D Burgers equation using Fourier time basis (comparison with original)

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.burgers import Burgers1D
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 1D Burgers equation with same parameters as original
    pde = Burgers1D(
        datapath="ref/burgers1d.dat",  # Correct relative path
        geom=[-1, 1],           # Spatial domain
        time=[0, 1],            # Time domain  
        nu=0.01 / 3.14159       # Viscosity parameter (same as original)
    )
    
    # Create separated network with Fourier time basis
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[64, 64, 64],      # Same architecture as polynomial version
        num_frequencies=12,                # Reasonable number of Fourier modes
        freq_type="exponential",           # Exponential frequency distribution
        freq_scale=1.5                     # Moderate frequency scaling
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))  # Same LR as original
    
    return model

# Define training parameters
train_args = {
    'iterations': 20000,  # Same as original experiment
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
    trainer = Trainer(exp_name="Burgers1D_Fourier_Comparison", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！1D伯格斯方程 + 傅里叶时间基（对比实验）")
    trainer.train_all()
    print(">>> 实验完成！")