# run_experiment_helmholtz2d.py
# Example experiment for 2D Helmholtz equation (time-independent PDE)

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class
from src.pde.helmholtz import Helmholtz2D
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize 2D Helmholtz equation
    pde = Helmholtz2D(
        scale=1,      # Domain scaling
        A=(4, 4),     # Frequency parameters for analytical solution
        k=1           # Wave number parameter
    )
    
    # For 2D time-independent PDEs, use a standard deep network
    net = dde.nn.FNN(
        layer_sizes=[pde.input_dim] + [128, 128, 128, 128, 128] + [pde.output_dim],
        activation="tanh",
        kernel_initializer="Glorot uniform"
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Lower learning rate for stability
    
    return model

# Define training parameters
train_args = {
    'iterations': 12000,  # More iterations for 2D problem
    'callbacks': [TesterCallback(log_every=800)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="Helmholtz2D_StandardFNN", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D亥姆霍兹方程 + 标准前馈神经网络")
    trainer.train_all()
    print(">>> 实验完成！")