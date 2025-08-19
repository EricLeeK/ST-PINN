#本文件将进行AC方程的一切实验，历史实验请在git中查找
import sys
import os
from pathlib import Path 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io
import os
import json 

# =============================================================================
# 0. 动态路径设置 (放在所有其他代码之前)
# =============================================================================
# 获取当前文件(train_ac.py)的绝对路径
# __file__ 是一个内置变量，代表当前脚本的文件名
current_file_path = Path(__file__).resolve()
# 获取项目根目录 (即 train/ 文件夹的父目录)
project_root = current_file_path.parent.parent
# 将项目根目录添加到Python搜索路径
sys.path.insert(0, str(project_root)) # sys.path 需要字符串


# 从我们的模型库文件中导入模型类
from src.models import SeparatedPINN_PolynomialTime, SeparatedPINN_FourierTime

# =============================================================================
# 1. 实验配置区
# =============================================================================
config = {
    # --- 结果保存配置 ---
    "experiment_name": "AC_Fourier_Linear_Freqs_v2", # 将作为结果子文件夹名
    
    # --- 模型选择 ---
    "model_type": "fourier",  # 可选: "polynomial" 或 "fourier"
    
    # --- 网络架构参数 ---
    "spatial_layers": [128, 128, 128, 128],
    "poly_degree": 40,
    "num_frequencies": 40,
    "freq_type": "linear",     # 可选: 'linear', 'exponential'
    
    # --- 训练参数 ---
    "epochs": 3000,
    "learning_rate": 1e-3,
    "loss_weights": { "ic": 100.0, "bc": 100.0, "pde": 1.0 },
    
    # --- 采样点数量 ---
    "N_ic": 500, "N_bc": 500, "N_pde": 20000,
    
    # --- PDE物理参数 ---
    "d": 0.001
}

# =============================================================================
# 2. 结果保存设置
# =============================================================================
# 创建主结果文件夹
# 使用 'results' 而非 'Result' 是为了匹配 .gitignore 规则
main_results_dir = "results" 
os.makedirs(main_results_dir, exist_ok=True)

# 创建本次实验的子文件夹
experiment_path = os.path.join(main_results_dir, config["experiment_name"])
os.makedirs(experiment_path, exist_ok=True)

# 保存本次实验的配置，以便未来复现
with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# 定义日志和图片的文件路径
log_filepath = os.path.join(experiment_path, 'loss_log.csv')
plot_filepath = os.path.join(experiment_path, 'result_plot.png')

# =============================================================================
# 3. 环境设置与数据准备
# =============================================================================

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"实验 '{config['experiment_name']}' 开始, 使用设备: {device}")
print(f"结果将保存到: {experiment_path}")

x_domain = [-1.0, 1.0]; t_domain = [0.0, 1.0]
x_ic = (torch.rand((config['N_ic'], 1)) * 2 - 1).to(device)
t_ic = torch.zeros((config['N_ic'], 1)).to(device)
u_ic = x_ic**2 * torch.cos(np.pi * x_ic)
t_bc = torch.rand((config['N_bc'], 1)).to(device)
x_bc_left = torch.full((config['N_bc']//2, 1), -1.0).to(device)
x_bc_right = torch.full((config['N_bc']//2, 1), 1.0).to(device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.full((config['N_bc'], 1), -1.0, device=device)
x_pde = (torch.rand((config['N_pde'], 1)) * 2 - 1).to(device)
t_pde = torch.rand((config['N_pde'], 1)).to(device)
x_pde.requires_grad = True; t_pde.requires_grad = True

# =============================================================================
# 4. 模型初始化 (现在会使用新的freq_type参数)
# =============================================================================
pinn = None
if config["model_type"] == "polynomial":
    pinn = SeparatedPINN_PolynomialTime(spatial_layers=config["spatial_layers"], poly_degree=config["poly_degree"]).to(device)
elif config["model_type"] == "fourier":
    pinn = SeparatedPINN_FourierTime(
        spatial_layers=config["spatial_layers"], 
        num_freqs=config["num_frequencies"],
        freq_type=config["freq_type"] # 将配置传递给模型
    ).to(device)
else:
    raise ValueError(f"Unknown model_type: {config['model_type']}")
print(f"成功创建模型: {pinn.__class__.__name__} (频率模式: {config.get('freq_type', 'N/A')})")

optimizer = torch.optim.Adam(pinn.parameters(), lr=config["learning_rate"])
loss_fn = nn.MSELoss()

# =============================================================================
# 5. 训练循环 (带日志记录)
# =============================================================================
# 使用 with open 确保文件被正确关闭
with open(log_filepath, 'w') as log_file:
    # 写入CSV文件的表头
    log_file.write('epoch,total_loss,ic_loss,bc_loss,pde_loss\n')
    
    print(f"开始训练...")
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        
        u_pred_ic = pinn(x_ic, t_ic)
        loss_ic = loss_fn(u_pred_ic, u_ic)
        t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
        u_pred_bc = pinn(x_bc, t_bc_shuffled)
        loss_bc = loss_fn(u_pred_bc, u_bc)
        u_pred_pde = pinn(x_pde, t_pde)
        u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        pde_residual = u_t - config["d"] * u_xx - 5 * (u_pred_pde - u_pred_pde**3)
        loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
        
        total_loss = (config["loss_weights"]["ic"] * loss_ic + 
                      config["loss_weights"]["bc"] * loss_bc + 
                      config["loss_weights"]["pde"] * loss_pde)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            # 打印到控制台
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {total_loss.item():.4f}')
            # 写入到日志文件
            log_file.write(f'{epoch+1},{total_loss.item()},{loss_ic.item()},{loss_bc.item()},{loss_pde.item()}\n')

print("训练完成!")

# =============================================================================
# 6. 可视化与保存
# =============================================================================
# (加载参考解的代码与之前相同)
try:
    data = scipy.io.loadmat('data\Allen_Cahn.mat')
    x_exact = data['x']; t_exact = data['t']; U_exact = data['u']
except FileNotFoundError:
    print("警告: 未找到 Allen_Cahn.mat 参考解文件，将跳过精确解和误差的绘制。")
    U_exact = None

pinn.eval()
with torch.no_grad():
    T, X = np.meshgrid(t_exact.flatten(), x_exact.flatten())
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).to(device)
    
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(X.shape).cpu().numpy()
    
    if U_pred.shape != U_exact.shape:
        U_pred = U_pred.T

    

fig = plt.figure(figsize=(20, 5))
gs = GridSpec(1, 3)
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title(f'Prediction ({config["model_type"]})')
fig.colorbar(c1, ax=ax1)

if U_exact is not None:
    Error = np.abs(U_pred - U_exact)
    ax2 = fig.add_subplot(gs[0, 1])
    c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')
    fig.colorbar(c2, ax=ax2)
    ax3 = fig.add_subplot(gs[0, 2])
    c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
    fig.colorbar(c3, ax=ax3)
    l2_error = np.linalg.norm(U_pred.flatten() - U_exact.flatten()) / np.linalg.norm(U_exact.flatten())
    print(f'L2 Relative Error: {l2_error.item():.6f}')
    fig.suptitle(f"Experiment: {config['experiment_name']} | L2 Error: {l2_error:.4f}", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片到文件，而不是仅仅显示
plt.savefig(plot_filepath)
print(f"结果图已保存到: {plot_filepath}")
# 也可以继续显示
plt.show()