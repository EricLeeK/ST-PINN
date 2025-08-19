# train/train_burgers.py

import sys
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
    "experiment_name": "Burgers_Fourier_Exponential_v1",

    # --- 模型选择 ---
    "model_type": "fourier",
    
    # --- 网络架构参数 ---
    "spatial_layers": [128, 128, 128, 128],
    # 使用指数频率，可以用较少的频率数覆盖很宽的频域，以捕捉激波
    "num_frequencies": 10, 
    "freq_type": "exponential", # 这是适配激波问题的关键
    
    # --- 训练参数 ---
    "epochs": 3000,
    "learning_rate": 1e-3,
    "loss_weights": { "ic": 100.0, "bc": 100.0, "pde": 1.0 },
    # Burgers方程训练可能不稳定，加入学习率调度和梯度裁剪作为可选项
    "use_scheduler": True,
    "scheduler_gamma": 0.995,
    "use_grad_clipping": True,
    "grad_clip_max_norm": 1.0,
    
    # --- 采样点数量 ---
    "N_ic": 500, "N_bc": 500, "N_pde": 20000,
    
    # --- PDE物理参数 ---
    "pde_params": {
        "nu": 0.01 / np.pi
    }
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
plot_heatmap_filepath = os.path.join(experiment_path, 'result_plot_heatmap.png')
plot_slices_filepath = os.path.join(experiment_path, 'result_plot_slices.png')

# =============================================================================
# 3. 环境设置与数据准备
# =============================================================================
torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"实验 '{config['experiment_name']}' 开始, 使用设备: {device}")
print(f"结果将保存到: {experiment_path}")

x_domain = [-1.0, 1.0]; t_domain = [0.0, 1.0]

# 初始条件 (t=0), u(x,0) = -sin(pi*x)
x_ic = (torch.rand((config['N_ic'], 1)) * (x_domain[1] - x_domain[0]) + x_domain[0]).to(device)
t_ic = torch.zeros((config['N_ic'], 1)).to(device)
u_ic = -torch.sin(np.pi * x_ic)

# 边界条件 (x=-1 and x=1), u=0
t_bc = (torch.rand((config['N_bc'], 1)) * (t_domain[1] - t_domain[0]) + t_domain[0]).to(device)
x_bc_left = torch.full((config['N_bc'] // 2, 1), x_domain[0], device=device)
x_bc_right = torch.full((config['N_bc'] // 2, 1), x_domain[1], device=device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.zeros((config['N_bc'], 1), device=device)

# PDE 配置点
x_pde = (torch.rand((config['N_pde'], 1)) * (x_domain[1] - x_domain[0]) + x_domain[0]).to(device)
t_pde = (torch.rand((config['N_pde'], 1)) * (t_domain[1] - t_domain[0]) + t_domain[0]).to(device)
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
        freq_type=config["freq_type"] # 将根据config选择 'exponential'
    ).to(device)
else:
    raise ValueError(f"Unknown model_type: {config['model_type']}")

print(f"成功创建模型: {pinn.__class__.__name__} (频率模式: {config.get('freq_type', 'N/A')})")
print(f"模型总参数量: {sum(p.numel() for p in pinn.parameters() if p.requires_grad)}")

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
        
        u_pred_ic = pinn(x_ic, t_ic)
        loss_ic = loss_fn(u_pred_ic, u_ic)
        
        t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
        u_pred_bc = pinn(x_bc, t_bc_shuffled)
        loss_bc = loss_fn(u_pred_bc, u_bc)
        
        u_pred_pde = pinn(x_pde, t_pde)
        u_t = torch.autograd.grad(u_pred_pde, t_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred_pde, x_pde, torch.ones_like(u_pred_pde), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_pde, torch.ones_like(u_x), create_graph=True)[0]
        
        nu = config["pde_params"]["nu"]
        pde_residual = u_t + u_pred_pde * u_x - nu * u_xx
        loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
        
        total_loss = (config["loss_weights"]["ic"] * loss_ic + 
                      config["loss_weights"]["bc"] * loss_bc + 
                      config["loss_weights"]["pde"] * loss_pde)
        
        total_loss.backward()

        if config.get("use_grad_clipping", False):
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=config["grad_clip_max_norm"])
        
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
try:
    # 确保参考解文件放在 data/ 目录下
    data = scipy.io.loadmat('data/burgers_shock.mat')
    x_exact_vec = data['x'].flatten()
    t_exact_vec = data['t'].flatten()
    U_exact = np.real(data['usol'])
except FileNotFoundError:
    print("警告: 未找到 data/burgers_shock.mat 参考解文件，将跳过可视化。")
    U_exact = None

if U_exact is not None:
    pinn.eval()
    with torch.no_grad():
        T_grid, X_grid = np.meshgrid(t_exact_vec, x_exact_vec)
        x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        
        u_pred_flat = pinn(x_flat, t_flat)
        U_pred = u_pred_flat.cpu().numpy().reshape(X_grid.shape)

    # --- 6.1 绘制热图 ---
    fig_hm = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3)
    extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

    ax1 = fig_hm.add_subplot(gs[0, 0])
    c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title(f'Prediction ({config["model_type"]})')
    fig_hm.colorbar(c1, ax=ax1)
    
    Error = np.abs(U_pred - U_exact)
    ax2 = fig_hm.add_subplot(gs[0, 1])
    c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')
    fig_hm.colorbar(c2, ax=ax2)
    
    ax3 = fig_hm.add_subplot(gs[0, 2])
    c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=0)
    ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
    fig_hm.colorbar(c3, ax=ax3)

    l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
    print(f'L2 Relative Error: {l2_error.item():.6f}')
    fig_hm.suptitle(f"Burgers' Equation | Experiment: {config['experiment_name']} | L2 Error: {l2_error:.4f}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_heatmap_filepath)
    print(f"热图已保存到: {plot_heatmap_filepath}")
    plt.close(fig_hm)

    # --- 6.2 绘制时间切片图 ---
    fig_slices = plt.figure(figsize=(10, 6))
    t_slices = [0.25, 0.50, 0.75, 0.99]
    for t_slice in t_slices:
        idx = np.argmin(np.abs(t_exact_vec - t_slice))
        plt.plot(x_exact_vec, U_exact[:, idx], 'b-', label=f'Exact t={t_slice}' if t_slice==t_slices[0] else None)
        plt.plot(x_exact_vec, U_pred[:, idx], 'r--', label=f'PINN t={t_slice}' if t_slice==t_slices[0] else None)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', lw=2, label='Exact'),
                       Line2D([0], [0], color='r', lw=2, linestyle='--', label='PINN')]
    plt.legend(handles=legend_elements)
    
    plt.title('Solution at Different Time Slices'); plt.xlabel('x'); plt.ylabel('u(x,t)')
    plt.grid(True); plt.tight_layout()
    plt.savefig(plot_slices_filepath)
    print(f"切片图已保存到: {plot_slices_filepath}")
    plt.show()