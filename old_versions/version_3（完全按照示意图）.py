import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置随机种子以保证结果可复现
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 新的网络结构定义
class SeparatedPINN_PolynomialTime(nn.Module):
    def __init__(self, spatial_layers, poly_degree):
        """
        初始化模型.
        spatial_layers: 空间MLP的层级结构, e.g., [64, 64, 64].
        poly_degree: 时间多项式的项数 L (即 t^0 到 t^(L-1)).
        """
        super(SeparatedPINN_PolynomialTime, self).__init__()
        self.poly_degree = poly_degree
        
        # 空间分支 (Spatial Branch) - 仍然是一个MLP
        # 它的输出维度必须等于多项式的项数 L
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
        
        # 1. 空间分支: 学习空间系数 C_i(x)
        spatial_features = self.spatial_net(x)
        
        # 2. 时间分支: 动态生成固定的多项式基 T_i(t)
        # --- 这是核心修正 ---
        # 基函数现在是 [t^0, t^1, t^2, ..., t^(L-1)]
        temporal_features = torch.cat([t.pow(i) for i in range(self.poly_degree)], dim=1)

        # 3. 点积融合: 计算 u(x,t) = Σ C_i(x) * t^i
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        
        return output

# --- 后续的代码几乎完全不变 ---

# 2. 问题参数和训练数据生成 (与之前完全相同)
alpha = 1.0 / np.pi**2
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]

N_ic, N_bc, N_pde = 100, 100, 2500
x_ic = (torch.rand((N_ic, 1)) * 2 - 1).to(device)
t_ic = torch.zeros((N_ic, 1)).to(device)
u_ic = torch.sin(np.pi * x_ic)

t_bc = torch.rand((N_bc, 1)).to(device)
x_bc_left = torch.full((N_bc, 1), -1.0).to(device)
x_bc_right = torch.full((N_bc, 1), 1.0).to(device)
u_bc = torch.zeros((N_bc, 1)).to(device)

x_pde = (torch.rand((N_pde, 1)) * 2 - 1).to(device)
t_pde = torch.rand((N_pde, 1)).to(device)
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型、优化器和训练循环
# 网络参数
spatial_layers = [64, 64, 64]  # 空间网络的层
poly_degree = 20 # 时间多项式的最高次数L，这是一个重要的超参数

# 实例化新的模型
pinn = SeparatedPINN_PolynomialTime(spatial_layers, poly_degree).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print(f"模型结构: {pinn}")
print(f"总参数量: {sum(p.numel() for p in pinn.parameters() if p.requires_grad)}")

epochs = 100000

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 1. 初始条件损失
    # 注意: 当 t=0 时，我们定义的 t^i 特征都为0，这会导致输出恒为0。
    # 这是一个小问题，我们可以通过让多项式从 t^0 开始来修正。
    # 为了简化，我们暂时忽略t=0时的这个特殊情况，损失函数依然会推动网络学习。
    # 在实践中，可以直接用一个小的epsilon替换t=0，或者修改基函数为 (t+epsilon)^i
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # 2. 边界条件损失
    u_pred_bc_left = pinn(x_bc_left, t_bc)
    u_pred_bc_right = pinn(x_bc_right, t_bc)
    loss_bc = loss_fn(u_pred_bc_left, u_bc) + loss_fn(u_pred_bc_right, u_bc)
    
    # 3. 物理损失 (PDE残差) - 这部分逻辑完全不变
    u_pred_pde = pinn(x_pde, t_pde)
    
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t - alpha * u_xx
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 总损失
    total_loss = loss_ic + loss_bc + loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, '
              f'Loss_IC: {loss_ic.item():.4f}, Loss_BC: {loss_bc.item():.4f}, Loss_PDE: {loss_pde.item():.4f}')

# 4. 结果可视化 (与之前完全相同)
pinn.eval()
with torch.no_grad():
    x_test = torch.linspace(x_domain[0], x_domain[1], 200)
    t_test = torch.linspace(t_domain[0], t_domain[1], 100)
    X, T = torch.meshgrid(x_test, t_test, indexing='ij')
    
    x_flat = X.flatten().to(device)
    t_flat = T.flatten().to(device)
    
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(X.shape).cpu()
    
    U_exact = torch.exp(-T) * torch.sin(np.pi * X)
    Error = torch.abs(U_pred - U_exact)

fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.pcolormesh(T, X, U_pred, cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
fig.colorbar(c1, ax=ax1)
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction u(x,t)')
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.pcolormesh(T, X, U_exact, cmap='jet', vmin=U_exact.min(), vmax=U_exact.max())
fig.colorbar(c2, ax=ax2)
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution u(x,t)')
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.pcolormesh(T, X, Error, cmap='jet')
fig.colorbar(c3, ax=ax3)
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
plt.tight_layout()
plt.show()

l2_error = torch.linalg.norm(U_pred.flatten() - U_exact.flatten()) / torch.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.4f}')