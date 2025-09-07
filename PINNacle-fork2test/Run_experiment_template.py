# run_my_experiment.py

# === 关键修正1：在导入任何东西之前，设置后端 ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# 1. 导入我们需要的工具
# --------------------------------------------------------------------------
import deepxde as dde
import torch

from trainer import Trainer 

# === 关键修正2：导入正确的类名 ===
from src.pde.burgers import Burgers1D  # <-- 正确的类名是 Burgers
# ==================================
from src.utils.visualization_utils import generate_burgers_heatmaps
from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# 3. 定义“模型工厂”函数
# ==========================================================================
def get_model():
    # === 关键修正：使用完全正确的类来实例化 ===
    pde = Burgers1D()  # <-- THE KEY FIX!
    # ======================================
    
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, kernel_initializer=None,
        spatial_layers=[64, 64, 64], poly_degree=20
    )
    
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
    
    return model

# 4. 定义训练参数
# ==========================================================================
train_args = {
    'iterations': 20000,
    'callbacks': [TesterCallback(log_every=1000)]
}

# 5. 主程序：开始实验！
# ==========================================================================
if __name__ == "__main__":
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # 实例化 Trainer
    trainer = Trainer(exp_name="MyFinalExperiment_Poly_Burgers1D", device="0")
    
    # 添加任务
    trainer.add_task(get_model, train_args)

    print(">>> 实验开始！所有配置已修正，目标：Burgers1D。")
    trainer.train_all()
    print(">>> 实验完成！")
    
    # =============================================================================
    # 可视化部分：生成解析解、模型解和误差热图
    # =============================================================================
    print("\n>>> 开始生成可视化图表...")
    
    # Import visualization utilities
    from visualization_utils import generate_burgers_heatmaps
    import glob
    
    # Find the latest model checkpoint
    exp_name = "MyFinalExperiment_Poly_Burgers1D"
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
                print(f"✓ 可视化完成！L2误差: {results['l2_error']:.6f}")
                print(f"  热图保存路径: {results['heatmap_path']}")
                print(f"  时间切片图保存路径: {results['slices_path']}")
            else:
                print(f"✗ 可视化失败: {results['error']}")
                
        except Exception as e:
            print(f"✗ 加载模型或生成可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ 未找到模型检查点，跳过可视化")
        print(f"  搜索路径: {checkpoint_pattern}")
    
    print(">>> 所有任务完成！")