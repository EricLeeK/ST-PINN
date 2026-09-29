# run_experiment_wave2d_longtime_polynomial.py
# Experiment for 2D Wave equation long-time dynamics using Polynomial time basis

# === Backend setup ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# Import required libraries
import deepxde as dde
import torch

from trainer import Trainer 

# Import PDE class and model
from src.pde.wave import Wave2D_Longtime
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback
from src.utils.visualization_utils import generate_2d_scalar_visualization

# Define model factory function
def get_model():
    # Initialize 2D Wave equation for long-time dynamics
    pde = Wave2D_Longtime(
        datapath=r"PINNacle-fork2test/ref/wave_darcy.dat",
        bbox=[0, 1, 0, 1, 0, 100],      # [x_min, x_max, y_min, y_max, t_min, t_max] - long time!
        a=1.4142,                       # Wave speed sqrt(2)
        modes=[(1,1), (1,2), (2,1)]     # Multiple spatial modes
    )
    
    # Create separated network with Polynomial time basis
    # Note: Wave2D has input_dim=3 (x, y, t) and output_dim=1
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, 
        kernel_initializer=None,
        spatial_layers=[128, 128, 128, 128],  # Deep network for long-time dynamics
        poly_degree=40                        # High degree for long-time coverage
    )
    
    # Create and compile model
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=3e-4))  # Very conservative LR
    
    return model

# Define training parameters
train_args = {
    'iterations': 40000,  # Many iterations for long-time problem
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
    trainer = Trainer(exp_name="Wave2D_Longtime_Polynomial", device="0")
    
    # Add experiment task
    trainer.add_task(get_model, train_args)

    print(">>> 开始实验！2D长时间波动方程 + 多项式时间基")
    trainer.train_all()
    print(">>> 实验完成！")
    
    # =============================================================================
    # 可视化部分：生成解析解、模型解和误差热图
    # =============================================================================
    print("\n>>> 开始生成可视化图表...")
    
    # Import visualization utilities
    import glob
    
    # Find the latest model checkpoint
    exp_name = "Wave2D_Longtime_Polynomial"
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
            results = generate_2d_scalar_visualization(test_model, exp_name, device, 'PINNacle-fork2test/ref/wave_darcy.dat')
            
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