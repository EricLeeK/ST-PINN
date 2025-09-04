# run_experiment_poisson1d.py  
# Example experiment for 1D Poisson equation (time-independent PDE)

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class
from src.pde.poisson import Poisson1D
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 1D Poisson equation
    pde = Poisson1D(
        a=1  # Parameter controlling frequency in the solution
    )
    
    # For time-independent PDEs, we use a standard feedforward network
    # instead of separated space-time network
    net = dde.nn.FNN(
        layer_sizes=[pde.input_dim] + [64, 64, 64, 64] + [pde.output_dim],
        activation="tanh",
        kernel_initializer="Glorot uniform"
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
    
    return model

# Define training parameters
train_args = {
    'iterations': 10000,  # Fewer iterations for time-independent problem
    'callbacks': [TesterCallback(log_every=500)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="Poisson1D_StandardFNN", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！1D泊松方程 + 标准前馈神经网络")
    trainer.train_all()
    print(">>> 实验完成！")