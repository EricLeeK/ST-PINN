# run_experiment_grayscott.py
# Experiment for Gray-Scott reaction-diffusion system using Fourier time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.chaotic import GrayScottEquation
from src.model.st_pinn import SeparatedNetFourier
from src.utils.callbacks import TesterCallback

# Define model factory function
def get_model():
    # Initialize Gray-Scott reaction-diffusion system
    pde = GrayScottEquation(
        datapath=r"ref/grayscott.dat",
        bbox=[-1, 1, -1, 1, 0, 200],   # [x_min, x_max, y_min, y_max, t_min, t_max]
        b=0.04,                        # Feed rate
        d=0.1,                         # Kill rate  
        epsilon=(1e-5, 5e-6)           # Diffusion coefficients for u, v
    )
    
    # Create separated network with Fourier time basis
    # Gray-Scott has input_dim=3 (x, y, t) and output_dim=2 (u, v concentrations)
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for complex patterns
        num_frequencies=20,                    # Many Fourier modes for complex dynamics
        freq_type="exponential",               # Exponential frequency distribution
        freq_scale=0.5                         # Lower frequency scale for long-time dynamics
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=5e-4))  # Conservative learning rate
    
    return model

# Define training parameters
train_args = {
    'iterations': 30000,  # Many iterations for complex chaotic system
    'callbacks': [TesterCallback(log_every=2000)]
}

# Main execution
if __name__ == "__main__":
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # Initialize trainer
    trainer = Trainer(exp_name="GrayScott_Fourier_ReactionDiffusion", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！Gray-Scott反应扩散系统 + 傅里叶时间基")
    trainer.train_all()
    print(">>> 实验完成！")
    
    # =============================================================================
    # 可视化部分：生成解析解、模型解和误差热图
    # =============================================================================
    print("\n>>> 开始生成可视化图表...")
    
    # Import visualization utilities
    from src.utils.visualization_utils import generate_2d_vector_visualization
    import glob
    
    # Find the latest model checkpoint
    exp_name = "GrayScott_Fourier_ReactionDiffusion"
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
            
            # Generate visualizations
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            results = generate_2d_vector_visualization(test_model, exp_name, device, 'ref/grayscott.dat')
            
            if "error" not in results:
                print(f"可视化完成！L2误差: {results['l2_error']:.6f}")
                print(f"矢量场对比图保存路径: {results['heatmap_path']}")
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