import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io # 用于加载参考解 .mat 文件

# 1. 环境与模型定义
# -----------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 使用我们之前修正好的多项式时间基模型
class SeparatedPINN_PolynomialTime(nn.Module):
    def __init__(self, spatial_layers, poly_degree):
        super(SeparatedPINN_PolynomialTime, self).__init__()
        self.poly_degree = poly_degree
        
        spatial_input_dim = 1
        spatial_output_dim = poly_degree
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        
        self.spatial_net = nn.Sequential(*modules)

    def forward(self, x, t):
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        spatial_features = self.spatial_net(x)
        temporal_features = torch.cat([t.pow(i) for i in range(self.poly_degree)], dim=1)
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output

# 2. 问题设置与数据准备
# -----------------------------------------------------------------------------
d = 0.001
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]
# 这是一个更难的问题，需要更多的点
N_ic, N_bc, N_pde = 500, 500, 20000

# 初始条件 (t=0): u(x,0) = x^2 * cos(pi*x)
x_ic = (torch.rand((N_ic, 1)) * 2 - 1).to(device)
t_ic = torch.zeros((N_ic, 1)).to(device)
u_ic = x_ic**2 * torch.cos(np.pi * x_ic)

# 边界条件 (x=-1, x=1): u = -1
t_bc = torch.rand((N_bc, 1)).to(device)
x_bc_left = torch.full((N_bc//2, 1), -1.0).to(device)
x_bc_right = torch.full((N_bc//2, 1), 1.0).to(device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.full((N_bc, 1), -1.0, device=device) # 关键: 边界值是-1

# 物理配置点 (内部点)
x_pde = (torch.rand((N_pde, 1)) * 2 - 1).to(device)
t_pde = torch.rand((N_pde, 1)).to(device)
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型训练
# -----------------------------------------------------------------------------
# 增加网络容量以应对更复杂的问题
spatial_layers = [128, 128, 128, 128, 128] 
poly_degree = 50 

pinn = SeparatedPINN_PolynomialTime(spatial_layers, poly_degree).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # 使用学习率衰减
loss_fn = nn.MSELoss()
epochs = 25000

print("开始使用多项式时间基模型求解 Allen-Cahn 方程...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 初始条件损失
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # 边界条件损失
    t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
    u_pred_bc = pinn(x_bc, t_bc_shuffled)
    loss_bc = loss_fn(u_pred_bc, u_bc)
    
    # PDE物理损失
    u_pred_pde = pinn(x_pde, t_pde)
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # PDE残差: u_t - d*u_xx - 5*(u - u^3) = 0
    pde_residual = u_t - d * u_xx - 5 * (u_pred_pde - u_pred_pde**3)
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 加权总损失，给IC和BC更高权重以稳定训练
    total_loss = 100 * loss_ic + 100 * loss_bc + loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        # 调整学习率
        if epoch > 5000: # 训练前期保持较高学习率
             scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Loss_IC: {loss_ic.item():.6f}, Loss_BC: {loss_bc.item():.6f}, Loss_PDE: {loss_pde.item():.6f}')

print("训练完成!")

# 4. 可视化 (使用您提供的代码)
# -----------------------------------------------------------------------------
# 加载参考解
try:
    data = scipy.io.loadmat('Allen_Cahn.mat')
except FileNotFoundError:
    print("\n错误: 未找到 'Allen_Cahn.mat' 文件。")
    print("请确保该文件与您的Python脚本在同一目录下。")
    exit()

t_exact_pts = data['t'].flatten() # (101,)
x_exact_pts = data['x'].flatten() # (512,)
U_exact = np.real(data['u']) # (512, 101)

# 创建与参考解匹配的网格进行预测
X_vis, T_vis = np.meshgrid(x_exact_pts, t_exact_pts)
x_flat = torch.tensor(X_vis.T.flatten(), dtype=torch.float32).to(device)
t_flat = torch.tensor(T_vis.T.flatten(), dtype=torch.float32).to(device)

pinn.eval()
with torch.no_grad():
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(U_exact.shape).cpu().numpy() # 转换为numpy

# 绘图
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

# 绘制预测图
ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction (Polynomial Basis)')
fig.colorbar(c1, ax=ax1)

# 绘制精确解
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Reference Solution')
fig.colorbar(c2, ax=ax2)

# 绘制误差图
Error = np.abs(U_pred - U_exact)
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
fig.colorbar(c3, ax=ax3)

plt.tight_layout()
plt.show()

l2_error = np.linalg.norm(U_pred.flatten() - U_exact.flatten()) / np.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.6f}')