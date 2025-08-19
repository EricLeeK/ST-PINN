# train/train_diffusion.py

import sys
from pathlib import Path 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import json 

# =============================================================================
# 0. 动态路径设置
# =============================================================================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

from src.models import SeparatedPINN_PolynomialTime, SeparatedPINN_FourierTime

# =============================================================================
# 1. 实验配置区
# =============================================================================
config = {
    # --- 结果保存配置 ---
    "experiment_name": "Diffusion_Fourier_Linear_v1", # <--- 已修改

    # --- 模型选择 ---
    "model_type": "fourier",
    
    # --- 网络架构参数 ---
    "spatial_layers": [64, 64, 64],
    "num_frequencies": 30,
    "freq_type": "linear",
    
    # --- 训练参数 ---
    "epochs": 3000,
    "learning_rate": 1e-3,
    "loss_weights": { "ic": 10.0, "bc": 10.0, "pde": 1.0 },
    "use_scheduler": True,
    "scheduler_gamma": 0.995,
    
    # --- 采样点数量 ---
    "N_ic": 200, "N_bc": 200, "N_pde": 5000,
    
    # --- PDE物理参数 (扩散系数为1，已包含在PDE残差中) ---
    "pde_params": {}
}

# =============================================================================
# 2. 结果保存设置
# =============================================================================
main_results_dir = "results" 
os.makedirs(main_results_dir, exist_ok=True)
experiment_path = os.path.join(main_results_dir, config["experiment_name"])
os.makedirs(experiment_path, exist_ok=True)

with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

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

# 初始条件 (IC): u(x, 0) = sin(pi*x)
x_ic = (torch.rand((config['N_ic'], 1)) * 2 - 1).to(device)
t_ic = torch.zeros((config['N_ic'], 1)).to(device)
u_ic = torch.sin(np.pi * x_ic)

# 边界条件 (BC): u(-1, t) = 0, u(1, t) = 0
t_bc = torch.rand((config['N_bc'], 1)).to(device)
x_bc_left = torch.full((config['N_bc']//2, 1), -1.0, device=device)
x_bc_right = torch.full((config['N_bc']//2, 1), 1.0, device=device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.zeros((config['N_bc'], 1), device=device)

# PDE 配置点
x_pde = (torch.rand((config['N_pde'], 1)) * 2 - 1).to(device)
t_pde = torch.rand((config['N_pde'], 1)).to(device)
x_pde.requires_grad = True; t_pde.requires_grad = True

# =============================================================================
# 4. 模型初始化
# =============================================================================
pinn = None
if config["model_type"] == "polynomial":
    pinn = SeparatedPINN_PolynomialTime(
        spatial_layers=config["spatial_layers"], 
        poly_degree=config.get("poly_degree", 20)
    ).to(device)
elif config["model_type"] == "fourier":
    pinn = SeparatedPINN_FourierTime(
        spatial_layers=config["spatial_layers"], 
        num_freqs=config["num_frequencies"],
        freq_type=config["freq_type"]
    ).to(device)
else:
    raise ValueError(f"Unknown model_type: {config['model_type']}")

print(f"成功创建模型: {pinn.__class__.__name__} (频率模式: {config.get('freq_type', 'N/A')})")

optimizer = torch.optim.Adam(pinn.parameters(), lr=config["learning_rate"])
loss_fn = nn.MSELoss()

scheduler = None
if config.get("use_scheduler", False):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler_gamma"])
    print("已启用学习率调度器.")

# =============================================================================
# 5. 训练循环
# =============================================================================
with open(log_filepath, 'w') as log_file:
    log_file.write('epoch,total_loss,ic_loss,bc_loss,pde_loss,learning_rate\n')
    
    print("开始训练...")
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        
        # IC Loss
        u_pred_ic = pinn(x_ic, t_ic)
        loss_ic = loss_fn(u_pred_ic, u_ic)
        
        # BC Loss
        t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
        u_pred_bc = pinn(x_bc, t_bc_shuffled)
        loss_bc = loss_fn(u_pred_bc, u_bc)
        
        # PDE Loss: Diffusion Equation u_t - u_xx = source_term
        u_pred_pde = pinn(x_pde, t_pde)
        u_t = torch.autograd.grad(u_pred_pde, t_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_pde, x_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        
        source_term = -torch.exp(-t_pde) * (torch.sin(np.pi * x_pde) - np.pi**2 * torch.sin(np.pi * x_pde))
        pde_residual = u_t - u_xx - source_term
        loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
        
        total_loss = (config["loss_weights"]["ic"] * loss_ic + 
                      config["loss_weights"]["bc"] * loss_bc + 
                      config["loss_weights"]["pde"] * loss_pde)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()
                
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}')
            log_file.write(f'{epoch+1},{total_loss.item()},{loss_ic.item()},{loss_bc.item()},{loss_pde.item()},{current_lr}\n')

print("训练完成!")

# =============================================================================
# 6. 可视化与保存
# =============================================================================
pinn.eval()
with torch.no_grad():
    # 创建可视化网格
    x_vis = torch.linspace(x_domain[0], x_domain[1], 256, device=device)
    t_vis = torch.linspace(t_domain[0], t_domain[1], 101, device=device)
    T, X = torch.meshgrid(t_vis, x_vis, indexing='xy')
    x_flat = X.flatten().unsqueeze(1)
    t_flat = T.flatten().unsqueeze(1)
    
    # 获取模型预测
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(T.shape).cpu().numpy()
    
    # 计算精确解
    X_np, T_np = X.cpu().numpy(), T.cpu().numpy()
    U_exact = np.exp(-T_np) * np.sin(np.pi * X_np)
    
Error = np.abs(U_pred - U_exact)
l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
print(f'L2 Relative Error: {l2_error.item():.6f}')

fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title(f'Prediction ({config["model_type"]})')
fig.colorbar(c1, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(U_exact, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')
fig.colorbar(c2, ax=ax2)

ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=0)
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
fig.colorbar(c3, ax=ax3)

# --- 已修改图表总标题 ---
fig.suptitle(f"Diffusion Equation | Experiment: {config['experiment_name']} | L2 Error: {l2_error:.4f}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(plot_filepath)
print(f"结果图已保存到: {plot_filepath}")
plt.show()