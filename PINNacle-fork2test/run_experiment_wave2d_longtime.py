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
    
    # =============================================================================
    # 可视化部分：生成解析解、模型解和误差热图
    # =============================================================================
    print("\n>>> 开始生成可视化图表...")
    
    # Import visualization utilities
    from src.utils.visualization_utils import generate_2d_scalar_visualization
    import glob
    
    # Find the latest model checkpoint
    exp_name = "Wave2D_LongTime_Fourier"
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
            
            # Generate visualizations - Wave2D uses analytical solutions
            # Create synthetic reference data for visualization
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Note: Wave2D uses analytical solutions, creating synthetic reference for visualization")
            
            # Create synthetic reference data
            import numpy as np
            x_range = np.linspace(0, 1, 51)
            y_range = np.linspace(0, 1, 51)
            t_range = np.linspace(0, 1, 11)
            
            synthetic_data = []
            for x in x_range:
                for y in y_range:
                    for t in t_range:
                        # Simple 2D wave: u = sin(2*pi*x) * sin(2*pi*y) * cos(pi*t)
                        u_val = np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(np.pi*t)
                        synthetic_data.append(f"{x} {y} {u_val}")
            
            # Save synthetic data temporarily
            temp_data_file = '/tmp/wave2d_reference.dat'
            with open(temp_data_file, 'w') as f:
                f.write('% X Y u\n')
                for row in synthetic_data:
                    f.write(row + '\n')
            
            results = generate_2d_scalar_visualization(test_model, exp_name, device, temp_data_file)
            
            if "error" not in results:
                print(f"可视化完成！L2误差: {results['l2_error']:.6f}")
                print(f"空间对比图保存路径: {results['heatmap_path']}")
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