# demo_standard_fnn.py
# Quick demonstration of the standard feedforward network for Burgers equation

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import deepxde as dde
import torch
import sys

# Add path for imports
sys.path.insert(0, '/home/runner/work/ST-PINN/ST-PINN/PINNacle-fork2test')

from src.pde.burgers import Burgers1D
from src.model.fnn import FNN
from visualization_utils import generate_burgers_heatmaps

def demo_standard_fnn():
    """Quick demo of standard FNN for Burgers equation"""
    
    print("=== ST-PINN 标准前馈神经网络演示 ===\n")
    
    # Backend configuration
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
    
    # Create the model
    print("1. 创建标准前馈神经网络模型...")
    
    pde = Burgers1D(
        datapath=r"PINNacle-fork2test/ref/burgers1d.dat",
        geom=[-1, 1],           
        time=[0, 1],            
        nu=0.01 / 3.14159       
    )
    
    net = FNN(
        layer_sizes=[2, 64, 64, 64, 64, 1],  
        activation="tanh",                    
        kernel_initializer="Glorot normal"   
    )
    
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
    
    print(f"  ✓ 模型创建成功")
    print(f"  - 输入维度: {pde.input_dim} (x, t)")  
    print(f"  - 输出维度: {pde.output_dim} (u)")
    print(f"  - 网络架构: [2, 64, 64, 64, 64, 1]")
    print(f"  - 激活函数: tanh")
    
    # Quick training (very short for demo)
    print("\n2. 快速训练演示 (100次迭代)...")
    try:
        model.train(iterations=100, model_save_path="demo_fnn_model")
        print("  ✓ 训练完成")
    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        return False
    
    # Test visualization 
    print("\n3. 生成可视化图表...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = generate_burgers_heatmaps(model, 'Demo_Standard_FNN', device)
        
        if "error" not in results:
            print(f"  ✓ 可视化生成成功！")
            print(f"  - L2相对误差: {results['l2_error']:.6f}")
            print(f"  - 热图文件: {results['heatmap_path']}")
            print(f"  - 时间切片图: {results['slices_path']}")
            
            # Show comparison with existing Fourier model
            print(f"\n4. 性能对比:")
            print(f"  - 标准FNN L2误差: {results['l2_error']:.6f}")
            print(f"  - 已有傅里叶模型误差: ~1.008 (参考)")
            print(f"  - 注意: 仅100次迭代的演示结果，完整训练需要20000次迭代")
            
            return True
        else:
            print(f"  ✗ 可视化失败: {results['error']}")
            return False
            
    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_standard_fnn()
    
    print(f"\n=== 演示{'成功' if success else '失败'} ===")
    if success:
        print("\n🎉 标准前馈神经网络实验和可视化功能正常工作！")
        print("   请运行完整实验: python run_experiment_burgers1d_standard_fnn.py")
    else:
        print("\n❌ 演示过程中遇到问题，请检查依赖项和数据文件")