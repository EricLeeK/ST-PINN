import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io # 用于加载参考解

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
data = scipy.io.loadmat('burgers_shock.mat')

# 1. 网络结构 (使用我们之前修正好的版本)
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

# 2. 问题参数和训练数据生成
nu = 0.01 / np.pi
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]

# 训练点数量
N_ic = 200
N_bc = 200
N_pde = 10000 # 对于更复杂的问题，需要更多的配置点

# 初始条件 (t=0)
x_ic = torch.rand((N_ic, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_ic = torch.zeros((N_ic, 1), device=device)
u_ic = -torch.sin(np.pi * x_ic)

# 边界条件 (x=-1 and x=1)
t_bc = torch.rand((N_bc, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
x_bc_left = torch.full((N_bc // 2, 1), x_domain[0], device=device)
x_bc_right = torch.full((N_bc // 2, 1), x_domain[1], device=device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.zeros((N_bc, 1), device=device)


# 物理配置点 (Collocation points)
x_pde = torch.rand((N_pde, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_pde = torch.rand((N_pde, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型、优化器和训练循环
# 增加网络容量以应对更复杂的问题
spatial_layers = [128, 128, 128, 128] 
poly_degree = 40 # 可能需要更高的多项式阶数

pinn = SeparatedPINN_PolynomialTime(spatial_layers, poly_degree).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 使用学习率调度器，有助于在后期精细调整
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

epochs = 100000

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. 初始条件损失
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # 2. 边界条件损失
    # 将左右边界点合并在一个tensor中计算
    t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
    u_pred_bc = pinn(x_bc, t_bc_shuffled)
    loss_bc = loss_fn(u_pred_bc, u_bc)
    
    # 3. 物理损失 (核心变化在这里)
    u_pred_pde = pinn(x_pde, t_pde)
    
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t + u_pred_pde * u_x - nu * u_xx
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 加权总损失，可以给物理损失更高的权重
    loss_weight_pde = 1.0
    loss_weight_ic = 10.0 # 初始条件很重要，给更高权重
    loss_weight_bc = 10.0
    
    total_loss = loss_weight_ic * loss_ic + loss_weight_bc * loss_bc + loss_weight_pde * loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    # 每1000次迭代调整学习率
    if (epoch + 1) % 1000 == 0:
        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  Loss_IC: {loss_ic.item():.4f}, Loss_BC: {loss_bc.item():.4f}, Loss_PDE: {loss_pde.item():.4f}')




data = scipy.io.loadmat('burgers_shock.mat')

# --- 接下来的代码完全不变 ---
t_exact = data['t'].flatten()[:,None] # (100, 1)
x_exact = data['x'].flatten()[:,None] # (256, 1)
U_exact_grid = np.real(data['usol']) # (256, 100)

t_exact = data['t'].flatten()[:,None] # (100, 1)
x_exact = data['x'].flatten()[:,None] # (256, 1)
U_exact_grid = np.real(data['usol']) # (256, 100)

# 创建与参考解匹配的网格进行预测
X, T = np.meshgrid(x_exact, t_exact)
x_test = torch.tensor(X.T.flatten(), dtype=torch.float32).to(device)
t_test = torch.tensor(T.T.flatten(), dtype=torch.float32).to(device)

pinn.eval()
with torch.no_grad():
    u_pred_flat = pinn(x_test, t_test)
    U_pred = u_pred_flat.reshape(U_exact_grid.shape).cpu().numpy()

# 绘图
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)

ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.pcolormesh(T.T, X.T, U_pred, cmap='jet')
fig.colorbar(c1, ax=ax1)
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction u(x,t)')

ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.pcolormesh(T.T, X.T, U_exact_grid, cmap='jet')
fig.colorbar(c2, ax=ax2)
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution u(x,t)')

ax3 = fig.add_subplot(gs[0, 2])
Error = np.abs(U_pred - U_exact_grid)
c3 = ax3.pcolormesh(T.T, X.T, Error, cmap='jet')
fig.colorbar(c3, ax=ax3)
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# 打印L2相对误差
l2_error = np.linalg.norm(U_pred.flatten() - U_exact_grid.flatten()) / np.linalg.norm(U_exact_grid.flatten())
print(f'L2 Relative Error: {l2_error.item():.4f}')

# 绘制不同时刻的切片图，更清晰地看激波
plt.figure(figsize=(10, 6))
t_slices = [0.25, 0.5, 0.75]
for t_slice in t_slices:
    idx = np.argmin(np.abs(t_exact - t_slice))
    plt.plot(x_exact, U_exact_grid[:, idx], label=f'Exact t={t_slice}', linestyle='-')
    plt.plot(x_exact, U_pred[:, idx], label=f'PINN t={t_slice}', linestyle='--')
plt.title('Solution at different time slices')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()