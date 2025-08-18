import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io

# 1. 环境与模型定义
# -----------------------------------------------------------------------------
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 使用傅里叶时间基模型
class SeparatedPINN_FourierTime(nn.Module):
    def __init__(self, spatial_layers, num_freqs, freq_scale=1.0):
        super(SeparatedPINN_FourierTime, self).__init__()
        self.num_freqs = num_freqs
        
        spatial_input_dim = 1
        spatial_output_dim = 2 * num_freqs
        
        modules = [nn.Linear(spatial_input_dim, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            modules.append(nn.Tanh())
            modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        modules.append(nn.Tanh())
        modules.append(nn.Linear(spatial_layers[-1], spatial_output_dim))
        self.spatial_net = nn.Sequential(*modules)
        
        self.freqs = freq_scale * torch.pi * (2.0 ** torch.arange(0, num_freqs)).to(device)

    def forward(self, x, t):
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        coeffs = self.spatial_net(x)
        t_proj = t @ self.freqs.unsqueeze(0)
        temporal_features = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        output = torch.sum(coeffs * temporal_features, dim=1, keepdim=True)
        return output

# 2. 问题设置与数据准备
# -----------------------------------------------------------------------------
# Allen-Cahn 方程参数
d = 0.0001
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]

# 增加采样点数量以应对复杂问题
N_ic, N_bc, N_pde = 500, 500, 20000

# 初始条件 (t=0)
x_ic = (torch.rand((N_ic, 1)) * 2 - 1).to(device)
t_ic = torch.zeros((N_ic, 1)).to(device)
u_ic = x_ic**2 * torch.cos(np.pi * x_ic)

# 边界条件 (x=-1, x=1) -> u = -1
t_bc = torch.rand((N_bc, 1)).to(device)
x_bc_left = torch.full((N_bc//2, 1), -1.0).to(device)
x_bc_right = torch.full((N_bc//2, 1), 1.0).to(device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.full((N_bc, 1), -1.0, device=device) # 注意边界值为-1

# 物理配置点
x_pde = (torch.rand((N_pde, 1)) * 2 - 1).to(device)
t_pde = torch.rand((N_pde, 1)).to(device)
x_pde.requires_grad = True
t_pde.requires_grad = True

# 3. 模型训练
# -----------------------------------------------------------------------------
# 增加网络容量
spatial_layers = [128, 128, 128, 128]
num_frequencies = 40

pinn = SeparatedPINN_FourierTime(spatial_layers, num_frequencies).to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
epochs = 30000 # 增加训练轮次

print("开始使用傅里叶时间基模型求解 Allen-Cahn 方程...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # IC Loss
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    # BC Loss
    t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
    u_pred_bc = pinn(x_bc, t_bc_shuffled)
    loss_bc = loss_fn(u_pred_bc, u_bc)
    
    # PDE Loss - 核心变化
    u_pred_pde = pinn(x_pde, t_pde)
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # PDE: u_t = d*u_xx + 5*(u - u^3)  =>  Residual: u_t - d*u_xx - 5*(u - u^3)
    pde_residual = u_t - d * u_xx - 5 * (u_pred_pde - u_pred_pde**3)
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    # 加权损失，对于硬问题，给IC和BC更高权重通常有帮助
    total_loss = 100 * loss_ic + 100 * loss_bc + loss_pde
    
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, '
              f'Loss_IC: {loss_ic.item():.4f}, Loss_BC: {loss_bc.item():.4f}, Loss_PDE: {loss_pde.item():.4f}')

print("训练完成!")

# 4. 可视化
# -----------------------------------------------------------------------------
# 加载参考解
try:
    data = scipy.io.loadmat('Allen_Cahn.mat')
except FileNotFoundError:
    print("\n错误: 请先下载 'Allen_Cahn.mat' 文件到您的代码目录。")
    print("您可以运行: wget https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/Allen_Cahn.mat\n")
    exit()

x_exact = data['x']  # (512, 1)
t_exact = data['t']  # (201, 1)
U_exact = data['u']  # (512, 201)

pinn.eval()
with torch.no_grad():
    # 使用与参考解完全相同的网格进行预测
    T, X = np.meshgrid(t_exact.flatten(), x_exact.flatten())
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).to(device)
    
    u_pred_flat = pinn(x_flat, t_flat)
    U_pred = u_pred_flat.reshape(X.shape).cpu().numpy()
    
    Error = np.abs(U_pred - U_exact)

fig = plt.figure(figsize=(20, 5))
gs = GridSpec(1, 3)

# 定义绘图范围
extent = [t_domain[0], t_domain[1], x_domain[0], x_domain[1]]

# 绘制预测图
ax1 = fig.add_subplot(gs[0, 0])
c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction (Fourier Basis)')
fig.colorbar(c1, ax=ax1)

# 绘制精确解
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution')
fig.colorbar(c2, ax=ax2)

# 绘制误差图
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
fig.colorbar(c3, ax=ax3)

plt.tight_layout()
plt.show()

l2_error = np.linalg.norm(U_pred.flatten() - U_exact.flatten()) / np.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.6f}')