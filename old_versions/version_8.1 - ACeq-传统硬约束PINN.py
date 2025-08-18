import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io

# 1. 环境与新的模型定义
# -----------------------------------------------------------------------------
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 新的模型: 一个标准的MLP，结合硬约束来处理边界条件
class PINN_HardBC(nn.Module):
    def __init__(self, layers):
        super(PINN_HardBC, self).__init__()
        
        # 构建一个标准的MLP
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        # 确保输入是 [N, 1]
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        # 将 x 和 t 拼接成 [N, 2] 作为输入
        x_t = torch.cat([x, t], dim=1)
        
        # 神经网络输出
        u_nn = self.net(x_t)
        
        # 应用硬约束 u = (1-x^2)*u_nn - 1
        u_final = (1 - x**2) * u_nn - 1
        
        return u_final

# 2. 问题设置与数据准备 (注意：我们不再需要BC数据点)
# -----------------------------------------------------------------------------
d = 0.001
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]
N_ic, N_pde = 500, 20000

# 初始条件 (t=0)
x_ic = (torch.rand((N_ic, 1)) * 2 - 1).to(device)
t_ic = torch.zeros((N_ic, 1)).to(device)
u_ic = x_ic**2 * torch.cos(np.pi * x_ic)

# 物理配置点 (内部点)
x_pde = (torch.rand((N_pde, 1)) * 2 - 1).to(device)
t_pde = torch.rand((N_pde, 1)).to(device)
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型训练 (注意: 损失函数中没有BC项)
# -----------------------------------------------------------------------------
# 网络层: 输入(x,t)是2维, 隐层... , 输出是1维
layers = [2, 128, 128, 128, 128, 1] 

pinn = PINN_HardBC(layers).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_fn = nn.MSELoss()
epochs = 30000 # 这个问题需要更多训练

print("开始使用带有硬约束的标准PINN模型求解...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 初始条件损失
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # PDE物理损失
    u_pred_pde = pinn(x_pde, t_pde)
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t - d * u_xx - 5 * (u_pred_pde - u_pred_pde**3)
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 总损失 (只有IC和PDE)
    total_loss = 100 * loss_ic + loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        if epoch > 8000:
             scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Loss_IC: {loss_ic.item():.6f}, Loss_PDE: {loss_pde.item():.6f}')

print("训练完成!")

# 4. 可视化 (代码与之前相同)
# -----------------------------------------------------------------------------
# 加载参考解
data = scipy.io.loadmat('Allen_Cahn.mat')
t_exact_pts = data['t'].flatten()
x_exact_pts = data['x'].flatten()
U_exact = np.real(data['u'])

X_vis, T_vis = np.meshgrid(x_exact_pts, t_exact_pts)
x_flat = torch.tensor(X_vis.T.flatten(), dtype=torch.float32).to(device)
t_flat = torch.tensor(T_vis.T.flatten(), dtype=torch.float32).to(device)

pinn.eval()
with torch.no_grad():
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(U_exact.shape).cpu().numpy()

fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction (Hard BC)')
fig.colorbar(c1, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Reference Solution')
fig.colorbar(c2, ax=ax2)

Error = np.abs(U_pred - U_exact)
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
fig.colorbar(c3, ax=ax3)

plt.tight_layout()
plt.show()

l2_error = np.linalg.norm(U_pred.flatten() - U_exact.flatten()) / np.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.6f}')