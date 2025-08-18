import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 傅里叶特征时间分支的网络
class SeparatedPINN_FourierTime(nn.Module):
    def __init__(self, spatial_layers, num_freqs):
        super(SeparatedPINN_FourierTime, self).__init__()
        self.num_freqs = num_freqs
        
        # 空间分支 (MLP)
        spatial_input_dim = 1
        spatial_output_dim = 2 * num_freqs
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)
        
        # --- 核心修正 1: 使用线性增长的频率 ---
        # 创建固定的频率 (非可学习)，从 1*pi 到 num_freqs*pi
        self.freqs = torch.pi * torch.arange(1, num_freqs + 1, dtype=torch.float32).to(device)

    def forward(self, x, t):
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        coeffs = self.spatial_net(x)
        t_proj = t @ self.freqs.unsqueeze(0)
        temporal_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        output = torch.sum(coeffs * temporal_features, dim=1, keepdim=True)
        return output

# 2. 问题参数和训练数据生成
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]
N_ic, N_bc, N_pde = 200, 200, 5000

# IC, BC, PDE 点的生成 (与之前相同)
x_ic = (torch.rand((N_ic, 1)) * 2 - 1).to(device)
t_ic = torch.zeros((N_ic, 1)).to(device)
y_ic = torch.sin(np.pi * x_ic)
t_bc = torch.rand((N_bc, 1)).to(device)
x_bc_left = torch.full((N_bc//2, 1), -1.0).to(device)
x_bc_right = torch.full((N_bc//2, 1), 1.0).to(device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
y_bc = torch.zeros((N_bc, 1)).to(device)
x_pde = (torch.rand((N_pde, 1)) * 2 - 1).to(device)
t_pde = torch.rand((N_pde, 1)).to(device)
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型、优化器和训练
# 网络参数
spatial_layers = [64, 64, 64]
num_frequencies = 30 # 可以适当增加傅里叶特征数量

pinn = SeparatedPINN_FourierTime(spatial_layers, num_frequencies).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# --- 新增修正 2: 添加学习率调度器来稳定训练 ---
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
epochs = 200000 # 增加训练轮次

for epoch in range(epochs):
    pinn.train()
    optimizer.zero_grad()
    
    # IC Loss
    y_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(y_pred_ic, y_ic)
    
    # BC Loss
    t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
    y_pred_bc = pinn(x_bc, t_bc_shuffled)
    loss_bc = loss_fn(y_pred_bc, y_bc)
    
    # PDE Loss
    y_pred_pde = pinn(x_pde, t_pde)
    y_t = torch.autograd.grad(y_pred_pde, t_pde, grad_outputs=torch.ones_like(y_pred_pde), create_graph=True)[0]
    y_x = torch.autograd.grad(y_pred_pde, x_pde, grad_outputs=torch.ones_like(y_pred_pde), create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, x_pde, grad_outputs=torch.ones_like(y_x), create_graph=True)[0]
    
    source_term = -torch.exp(-t_pde) * (torch.sin(np.pi * x_pde) - np.pi**2 * torch.sin(np.pi * x_pde))
    pde_residual = y_t - y_xx - source_term
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # --- 新增修正 3: 使用损失加权 ---
    total_loss = 10.0 * loss_ic + 10.0 * loss_bc + loss_pde
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        scheduler.step() # 更新学习率
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

# 4. 可视化
# 4. 可视化 (使用 imshow 替代 pcolormesh)
pinn.eval()
with torch.no_grad():
    x_test_vis = torch.linspace(x_domain[0], x_domain[1], 256)
    t_test_vis = torch.linspace(t_domain[0], t_domain[1], 100)
    X, T = torch.meshgrid(x_test_vis, t_test_vis, indexing='ij')
    
    x_flat = X.flatten().to(device)
    t_flat = T.flatten().to(device)
    
    y_pred_flat = pinn(x_flat, t_flat)
    Y_pred = y_pred_flat.reshape(X.shape).cpu().numpy() # 转换为numpy
    
    Y_exact = (torch.exp(-T) * torch.sin(np.pi * X)).numpy() # 转换为numpy
    Error = np.abs(Y_pred - Y_exact)

fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)

# 定义绘图范围
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

# 绘制预测图
ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(Y_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction')
fig.colorbar(c1, ax=ax1)

# 绘制精确解
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(Y_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')
fig.colorbar(c2, ax=ax2)

# 绘制误差图
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
fig.colorbar(c3, ax=ax3)

plt.tight_layout()
plt.show()

l2_error = np.linalg.norm(Y_pred.flatten() - Y_exact.flatten()) / np.linalg.norm(Y_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.4f}')