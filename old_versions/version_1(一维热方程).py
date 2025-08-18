import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置随机种子以保证结果可复现
torch.manual_seed(1234)
np.random.seed(1234)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 网络结构定义 
class FactorizedPINN(nn.Module):
    def __init__(self, spatial_layers, temporal_layers, feature_dim):
        super(FactorizedPINN, self).__init__()
        
        # 空间分支 (Spatial Branch)
        spatial_modules = [nn.Linear(1, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            spatial_modules.append(nn.Tanh())
            spatial_modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        spatial_modules.append(nn.Tanh())
        spatial_modules.append(nn.Linear(spatial_layers[-1], feature_dim))
        self.spatial_net = nn.Sequential(*spatial_modules)

        # 时间分支 (Temporal Branch)
        temporal_modules = [nn.Linear(1, temporal_layers[0])]
        for i in range(len(temporal_layers) - 1):
            temporal_modules.append(nn.Tanh())
            temporal_modules.append(nn.Linear(temporal_layers[i], temporal_layers[i+1]))
        temporal_modules.append(nn.Tanh())
        temporal_modules.append(nn.Linear(temporal_layers[-1], feature_dim))
        self.temporal_net = nn.Sequential(*temporal_modules)

    def forward(self, x, t):
        # 确保输入是正确的形状 [N, 1]
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        # 获取空间和时间特征
        spatial_features = self.spatial_net(x) # 输出形状: [N, feature_dim]
        temporal_features = self.temporal_net(t) # 输出形状: [N, feature_dim]
        
        # 点积融合 (Element-wise product and sum)
        # u(x,t) = v_s(x) · v_t(t)
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output

# 2. 问题参数和训练数据生成
alpha = 1.0 / np.pi**2
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]

# 初始条件 (t=0)
N_ic = 100
x_ic = torch.rand((N_ic, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_ic = torch.zeros((N_ic, 1), device=device)
u_ic = torch.sin(np.pi * x_ic)

# 边界条件 (x=-1 and x=1)
N_bc = 100
t_bc = torch.rand((N_bc, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
x_bc_left = torch.full((N_bc, 1), x_domain[0], device=device)
x_bc_right = torch.full((N_bc, 1), x_domain[1], device=device)
u_bc = torch.zeros((N_bc, 1), device=device)

# 物理配置点 (Collocation points inside the domain)
N_pde = 2500
x_pde = torch.rand((N_pde, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_pde = torch.rand((N_pde, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]

# 允许计算梯度
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型、优化器和训练循环
# 网络参数
spatial_layers = [32, 32, 32]
temporal_layers = [32, 32, 32]
feature_dim = 50 # 两个分支输出的特征维度

pinn = FactorizedPINN(spatial_layers, temporal_layers, feature_dim).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 100000

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. 初始条件损失
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # 2. 边界条件损失
    u_pred_bc_left = pinn(x_bc_left, t_bc)
    u_pred_bc_right = pinn(x_bc_right, t_bc)
    loss_bc = loss_fn(u_pred_bc_left, u_bc) + loss_fn(u_pred_bc_right, u_bc)
    
    # 3. 物理损失 (PDE残差)
    u_pred_pde = pinn(x_pde, t_pde)
    
    # 使用autograd计算导数
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t - alpha * u_xx
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 总损失
    total_loss = loss_ic + loss_bc + loss_pde
    
    # 反向传播和优化
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.7f}, '
              f'Loss_IC: {loss_ic.item():.7f}, Loss_BC: {loss_bc.item():.7f}, Loss_PDE: {loss_pde.item():.7f}')

# 4. 结果可视化
pinn.eval()
with torch.no_grad():
    # 创建测试网格
    x_test = torch.linspace(x_domain[0], x_domain[1], 200)
    t_test = torch.linspace(t_domain[0], t_domain[1], 100)
    X, T = torch.meshgrid(x_test, t_test, indexing='ij')
    
    # 将网格展平以进行预测
    x_flat = X.flatten().to(device)
    t_flat = T.flatten().to(device)
    
    # PINN预测
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(X.shape).cpu()
    
    # 解析解
    U_exact = torch.exp(-T) * torch.sin(np.pi * X)
    
    # 误差
    Error = torch.abs(U_pred - U_exact)

# 绘图
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)

# PINN预测结果
ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.pcolormesh(T, X, U_pred, cmap='jet')
fig.colorbar(c1, ax=ax1)
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('PINN Prediction u(x,t)')

# 解析解
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.pcolormesh(T, X, U_exact, cmap='jet')
fig.colorbar(c2, ax=ax2)
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Exact Solution u(x,t)')

# 绝对误差
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.pcolormesh(T, X, Error, cmap='jet')
fig.colorbar(c3, ax=ax3)
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title('Absolute Error')

plt.tight_layout()
plt.show()

# 打印L2相对误差
l2_error = torch.linalg.norm(U_pred.flatten() - U_exact.flatten()) / torch.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.7f}')