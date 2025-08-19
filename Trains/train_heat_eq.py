# 描述: 使用多项式时间基的PINN求解一维热传导方程

import sys
import os
from pathlib import Path 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# =============================================================================
# 0. 动态路径设置
# =============================================================================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.insert(0, str(project_root))

# 从模型库导入所有模型
from src.models import SeparatedPINN_PolynomialTime, SeparatedPINN_FourierTime

# =============================================================================
# 1. 实验配置区
# =============================================================================
config = {
    # --- 结果保存配置 ---
    "experiment_name": "HeatEq_Polynomial_deg20_v1",

    # --- 模型选择 ---
    "model_type": "polynomial",  # 可选: "polynomial", "fourier", "learnable_time"

    # --- 网络架构参数 ---
    "spatial_layers": [64, 64, 64],
    "poly_degree": 20, # 核心超参数
    
    # (以下参数在此次实验中未使用，但保留以作模板)
    "temporal_layers": [], "feature_dim": 0,
    "num_frequencies": 0, "freq_type": "",
    
    # --- 训练参数 ---
    "epochs": 3000, # 从100000减少以便快速测试，您可以调回去
    "learning_rate": 1e-3,
    "loss_weights": { "ic": 1.0, "bc": 1.0, "pde": 1.0 },

    # --- 采样点数量 ---
    "N_ic": 100, "N_bc": 100, "N_pde": 2500,

    # --- PDE物理参数 ---
    "alpha": 1.0 / (np.pi**2)
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
torch.manual_seed(1234); np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"实验 '{config['experiment_name']}' 开始, 使用设备: {device}")
print(f"结果将保存到: {experiment_path}")

x_domain = [-1.0, 1.0]; t_domain = [0.0, 1.0]

# 初始条件 (t=0), u(x,0) = sin(pi*x)
x_ic = (torch.rand((config['N_ic'], 1)) * 2 - 1).to(device)
t_ic = torch.zeros_like(x_ic)
u_ic = torch.sin(np.pi * x_ic)

# 边界条件 (x=-1, x=1), u(-1,t)=0, u(1,t)=0
t_bc_full = torch.rand((config['N_bc'], 1)).to(device) # Re-sample for each side
x_bc_left = torch.full_like(t_bc_full, -1.0)
x_bc_right = torch.full_like(t_bc_full, 1.0)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
t_bc = torch.cat([t_bc_full, t_bc_full], dim=0)
u_bc = torch.zeros_like(x_bc)

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
        poly_degree=config["poly_degree"]
    ).to(device)
elif config["model_type"] == "fourier":
    pinn = SeparatedPINN_FourierTime(spatial_layers=config["spatial_layers"], num_freqs=config["num_frequencies"], freq_type=config["freq_type"]).to(device)
else:
    raise ValueError(f"Unknown model_type: {config['model_type']}")
print(f"成功创建模型: {pinn.__class__.__name__}")

optimizer = torch.optim.Adam(pinn.parameters(), lr=config["learning_rate"])
loss_fn = nn.MSELoss()

# =============================================================================
# 5. 训练循环
# =============================================================================
with open(log_filepath, 'w') as log_file:
    log_file.write('epoch,total_loss,ic_loss,bc_loss,pde_loss\n')
    
    print("开始训练...")
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        
        # IC Loss
        u_pred_ic = pinn(x_ic, t_ic)
        loss_ic = loss_fn(u_pred_ic, u_ic)
        
        # BC Loss
        u_pred_bc = pinn(x_bc, t_bc)
        loss_bc = loss_fn(u_pred_bc, u_bc)
        
        # PDE Residual Loss
        u_pred_pde = pinn(x_pde, t_pde)
        u_t = torch.autograd.grad(u_pred_pde, t_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_pde, x_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        
        pde_residual = u_t - config["alpha"] * u_xx
        loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
        
        total_loss = (config["loss_weights"]["ic"] * loss_ic + 
                      config["loss_weights"]["bc"] * loss_bc + 
                      config["loss_weights"]["pde"] * loss_pde)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {total_loss.item():.6f}')
            log_file.write(f'{epoch+1},{total_loss.item()},{loss_ic.item()},{loss_bc.item()},{loss_pde.item()}\n')

print("训练完成!")

# =============================================================================
# 6. 可视化与保存
# =============================================================================
pinn.eval()
with torch.no_grad():
    x_test = torch.linspace(x_domain[0], x_domain[1], 201)
    t_test = torch.linspace(t_domain[0], t_domain[1], 101)
    X, T = torch.meshgrid(x_test, t_test, indexing='ij')
    
    x_flat = X.flatten().unsqueeze(-1).to(device)
    t_flat = T.flatten().unsqueeze(-1).to(device)
    
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(X.shape).cpu().numpy()
    
    U_exact = np.exp(-T.numpy()) * np.sin(np.pi * X.numpy())
    Error = np.abs(U_pred - U_exact)

fig = plt.figure(figsize=(20, 5))
gs = GridSpec(1, 3)
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
fig.colorbar(c1, ax=ax1); ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction')

ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
fig.colorbar(c2, ax=ax2); ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')

ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
fig.colorbar(c3, ax=ax3); ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')

l2_error = np.linalg.norm(U_pred.flatten() - U_exact.flatten()) / np.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.6f}')
fig.suptitle(f"Experiment: {config['experiment_name']} | L2 Error: {l2_error:.4f}", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(plot_filepath)
print(f"结果图已保存到: {plot_filepath}")
plt.show()