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
from src.utils.visualization_utils import generate_burgers_heatmaps
# Define model factory function
def get_model():
    # Initialize 1D Burgers equation with same parameters as original
    pde = Burgers1D(
        datapath=r"PINNacle-fork2test/ref/burgers1d.dat",
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
        freq_type="linear",           # Exponential frequency distribution
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
    print(">>> Experiment completed!")
    
    # =============================================================================
    # Visualization: generate analytical solution, model solution, and error heatmaps
    # =============================================================================
    print("\n>>> Starting visualization generation...")
    
    # Import visualization utilities
    import glob
    
    # Find the latest model checkpoint
    exp_name = "Burgers1D_Fourier_Comparison"
    checkpoint_pattern = f"runs/{exp_name}/*/*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint_dir = os.path.dirname(latest_checkpoint)
        print(f"Found checkpoint: {latest_checkpoint}")
        
        try:
            # Load the trained model for visualization
            # Note: We need to recreate the model architecture first
            test_model = get_model()
            
            # Load the saved weights
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Load from checkpoint dictionary
                if hasattr(test_model, 'net'):
                    test_model.net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    test_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Direct state dict
                if hasattr(test_model, 'net'):
                    test_model.net.load_state_dict(checkpoint)
                else:
                    test_model.load_state_dict(checkpoint)
            
            # Generate visualizations
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            results = generate_burgers_heatmaps(test_model, exp_name, device)
            
            if "error" not in results:
                print(f"Visualization completed! L2 error: {results['l2_error']:.6f}")
                print(f"Heatmap saved to: {results['heatmap_path']}")
                print(f"Time slices plot saved to: {results['slices_path']}")
            else:
                print(f"Visualization failed: {results['error']}")
                
        except Exception as e:
            print(f"Error loading model or generating visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model checkpoint not found, skipping visualization")
        print(f"Search path: {checkpoint_pattern}")
    
    print(">>> All tasks completed!")
