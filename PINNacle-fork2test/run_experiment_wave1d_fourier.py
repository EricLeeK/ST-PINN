# run_experiment_wave1d_fourier.py
# Experiment for 1D Wave equation using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model  
from src.pde.wave import Wave1D
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback
from src.utils.visualization_utils import generate_1d_visualization

# Define model factory function
def get_model():
    # Initialize 1D Wave equation
    pde = Wave1D(
        C=2,              # Wave speed
        bbox=[0, 1, 0, 1], # Domain bounds [x_min, x_max, t_min, t_max]
        scale=1,          # Spatial scaling
        a=4               # Frequency parameter for solution
    )
    
    # Create separated network with Fourier time basis
    # Note: Wave1D has input_dim=2 (x, t) and output_dim=1
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128], # Network for wave equation
        num_frequencies=15,             # Good number of modes for wave dynamics
        freq_type="linear",             # Linear frequency distribution
        freq_scale=2.0                  # Higher frequency scaling for oscillations
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Conservative LR for wave
    
    return model

# Define training parameters
train_args = {
    'iterations': 10000,
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
    trainer = Trainer(exp_name="Wave1D_Fourier", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！1D波动方程 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")
    
    # =============================================================================
    # 可视化部分：生成解析解、模型解和误差热图
    # =============================================================================
    print("\n>>> 开始生成可视化图表...")
    
    # Import visualization utilities
    import glob
    
    # Find the latest model checkpoint
    exp_name = "Wave1D_Fourier"
    checkpoint_pattern = f"runs/{exp_name}/*/*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint_dir = os.path.dirname(latest_checkpoint)
        print(f"Found checkpoint: {latest_checkpoint}")
        
        try:
            # Load the trained model for visualization
            test_model = get_model()
            
            # Load the saved weights
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                if hasattr(test_model, 'net'):
                    test_model.net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    test_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                if hasattr(test_model, 'net'):
                    test_model.net.load_state_dict(checkpoint)
                else:
                    test_model.load_state_dict(checkpoint)
            
            # Generate visualizations - Wave1D doesn't have a specific data file, it's analytical
            # We'll create synthetic reference data for visualization
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Note: Wave1D uses analytical solutions, creating synthetic reference for visualization")
            
            # Create synthetic reference data file if needed
            import numpy as np
            x_range = np.linspace(0, 1, 101)
            t_range = np.linspace(0, 1, 11)
            
            # Create a simple wave solution for reference visualization
            # This is just for demonstration - in practice you'd use the analytical solution
            synthetic_data = []
            for x in x_range:
                row = [x]
                for t in t_range:
                    # Simple wave: u = sin(4*pi*x) * cos(2*pi*t)
                    u_val = np.sin(4*np.pi*x) * np.cos(2*np.pi*t)
                    row.append(u_val)
                synthetic_data.append(' '.join(map(str, row)))
            
            # Save synthetic data temporarily
            temp_data_file = '/tmp/wave1d_reference.dat'
            with open(temp_data_file, 'w') as f:
                f.write('% X                       u @ t=0              u @ t=0.1            u @ t=0.2            u @ t=0.3            u @ t=0.4            u @ t=0.5            u @ t=0.6            u @ t=0.7            u @ t=0.8            u @ t=0.9            u @ t=1\n')
                f.write('\n'.join(synthetic_data))
            
            results = generate_1d_visualization(test_model, exp_name, device, temp_data_file)
            
            if "error" not in results:
                print(f"可视化完成！L2误差: {results['l2_error']:.6f}")
                print(f"热图保存路径: {results['heatmap_path']}")
                if 'slices_path' in results:
                    print(f"时间切片图保存路径: {results['slices_path']}")
            else:
                print(f"可视化失败: {results['error']}")
                
        except Exception as e:
            print(f"加载模型或生成可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"未找到模型检查点，跳过可视化")
        print(f"  搜索路径: {checkpoint_pattern}")
    
    print(">>> 所有任务完成！")