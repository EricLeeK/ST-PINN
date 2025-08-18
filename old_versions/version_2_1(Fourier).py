import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置随机种子
torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. 定义傅里叶特征编码器 (已修改)
class FourierFeatureEncoder:
    def __init__(self, input_dims, num_freqs, scale=1.0):
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        # --- 核心修正：使用线性增长的频率 ---
        # scale 参数可以进一步控制频率范围
        self.b_matrix = (torch.linspace(1.0, scale, num_freqs)).unsqueeze(0).to(device) * np.pi


    def __call__(self, x):
        proj = x @ self.b_matrix
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def output_dims(self):
        return self.num_freqs * 2

# 2. 新的网络结构 (不变)
class PINN_FourierBasis(nn.Module):
    def __init__(self, spatial_layers, time_encoder):
        super(PINN_FourierBasis, self).__init__()
        self.time_encoder = time_encoder
        spatial_input_dim = 1
        spatial_output_dim = self.time_encoder.output_dims
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
        temporal_features = self.time_encoder(t)
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output

# 3. 实例化新的模型
# --- 核心修正：调整编码器参数 ---
num_frequencies = 10  # 降低频率数量
frequency_scale = 10.0 # 控制最高频率
time_encoder = FourierFeatureEncoder(input_dims=1, num_freqs=num_frequencies, scale=frequency_scale)

spatial_layers = [64, 64, 64]
pinn = PINN_FourierBasis(spatial_layers, time_encoder).to(device)

print(f"模型结构: {pinn}")
print(f"总参数量: {sum(p.numel() for p in pinn.parameters() if p.requires_grad)}")


# 4. 优化器、损失函数和训练数据
# --- 核心修正：降低学习率 ---
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# 准备数据 (不变)
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

# 5. 训练循环 (已修改)
epochs = 400000 
for epoch in range(epochs):
    optimizer.zero_grad()
    
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    u_pred_bc_left = pinn(x_bc_left, t_bc)
    u_pred_bc_right = pinn(x_bc_right, t_bc)
    loss_bc = loss_fn(u_pred_bc_left, u_bc) + loss_fn(u_pred_bc_right, u_bc)
    
    u_pred_pde = pinn(x_pde, t_pde)
    
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t - alpha * u_xx
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    total_loss = loss_ic + loss_bc + loss_pde
    
    total_loss.backward()
    
    # --- 核心修正：加入梯度裁剪 ---
    torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.7f}')

# 6. 可视化 (不变)
# ... (可视化代码与之前相同) ...
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
c1 = ax1.pcolormesh(T, X, U_pred, cmap='jet'); fig.colorbar(c1, ax=ax1)
ax1.set_xlabel('t'); ax1.set_ylabel('x'); ax1.set_title('PINN Prediction u(x,t)')
ax2 = fig.add_subplot(gs[0, 1])
c2 = ax2.pcolormesh(T, X, U_exact, cmap='jet'); fig.colorbar(c2, ax=ax2)
ax2.set_xlabel('t'); ax2.set_ylabel('x'); ax2.set_title('Exact Solution u(x,t)')
ax3 = fig.add_subplot(gs[0, 2])
c3 = ax3.pcolormesh(T, X, Error, cmap='jet'); fig.colorbar(c3, ax=ax3)
ax3.set_xlabel('t'); ax3.set_ylabel('x'); ax3.set_title('Absolute Error')
plt.tight_layout(); plt.show()
l2_error = torch.linalg.norm(U_pred.flatten() - U_exact.flatten()) / torch.linalg.norm(U_exact.flatten())
print(f'L2 Relative Error: {l2_error.item():.7f}')