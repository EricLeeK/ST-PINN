# run_experiment_wave2d_longtime.py
# Experiment for 2D Wave equation long-time dynamics using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch
import numpy as np

from trainer import Trainer 

# Import PDE class and model
from src.pde.wave import Wave2D_LongTime
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Wave equation for long-time dynamics
    pde = Wave2D_LongTime(
        bbox=[0, 1, 0, 1, 0, 100],       # [x_min, x_max, y_min, y_max, t_min, t_max] - very long time
        a=np.sqrt(2),                    # Wave speed parameter
        m1=1, m2=3,                      # Spatial frequency parameters for first mode
        n1=1, n2=2,                      # Spatial frequency parameters for second mode
        p1=1, p2=1                       # Temporal frequency parameters
    )
    
    # Create separated network with Fourier time basis
    # Wave2D has input_dim=3 (x, y, t) and output_dim=1
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for 2D wave patterns
        num_frequencies=30,                    # Many Fourier modes for long-time wave dynamics
        freq_type="linear",                    # Linear frequency distribution for wave harmonics
        freq_scale=0.5                         # Lower frequency scale for long-time dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=3e-4))  # Conservative learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 40000,  # Many iterations for long-time wave dynamics
    'callbacks': [TesterCallback(log_every=3000)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="Wave2D_LongTime_Fourier", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D波动方程长时间动力学 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")