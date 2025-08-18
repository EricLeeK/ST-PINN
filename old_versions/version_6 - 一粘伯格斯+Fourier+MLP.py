import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io

data = scipy.io.loadmat('burgers_shock.mat')
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# ... (保持 FourierFeatureEncoder 类不变) ...
class FourierFeatureEncoder:
    def __init__(self, input_dims, num_freqs, scale=1.0):
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.b_matrix = (torch.linspace(1.0, scale, num_freqs)).unsqueeze(0).to(device) * np.pi
    def __call__(self, x):
        proj = x @ self.b_matrix
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    @property
    def output_dims(self):
        return self.num_freqs * 2

# 2. 切换回我们最开始讨论的、更强大的网络结构
class FactorizedPINN_Fourier(nn.Module):
    def __init__(self, spatial_layers, temporal_layers, feature_dim, time_encoder):
        super(FactorizedPINN_Fourier, self).__init__()
        self.time_encoder = time_encoder
        
        # 空间分支
        spatial_modules = [nn.Linear(1, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            spatial_modules.append(nn.Tanh())
            spatial_modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        spatial_modules.append(nn.Tanh())
        spatial_modules.append(nn.Linear(spatial_layers[-1], feature_dim))
        self.spatial_net = nn.Sequential(*spatial_modules)

        # 时间分支 (现在也是一个可学习的MLP)
        temporal_input_dim = self.time_encoder.output_dims
        temporal_modules = [nn.Linear(temporal_input_dim, temporal_layers[0])]
        for i in range(len(temporal_layers) - 1):
            temporal_modules.append(nn.Tanh())
            temporal_modules.append(nn.Linear(temporal_layers[i], temporal_layers[i+1]))
        temporal_modules.append(nn.Tanh())
        temporal_modules.append(nn.Linear(temporal_layers[-1], feature_dim))
        self.temporal_net = nn.Sequential(*temporal_modules)

    def forward(self, x, t):
        if x.dim() == 1: x = x.unsqueeze(1)
        if t.dim() == 1: t = t.unsqueeze(1)
        
        t_encoded = self.time_encoder(t)
        
        spatial_features = self.spatial_net(x)
        temporal_features = self.temporal_net(t_encoded)
        
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output

# --- 后续代码部分 ---

# 3. 实例化新的、更强大的模型
# 编码器参数
num_frequencies = 40
frequency_scale = 100.0
time_encoder = FourierFeatureEncoder(input_dims=1, num_freqs=num_frequencies, scale=frequency_scale)

# 网络参数 (为两个分支都配置强大的MLP)
spatial_layers = [128, 128, 128, 128] 
temporal_layers = [128, 128, 128, 128]
feature_dim = 100 # 空间和时间特征向量的维度

pinn = FactorizedPINN_Fourier(
    spatial_layers, 
    temporal_layers, 
    feature_dim, 
    time_encoder
).to(device)

print(f"模型总参数量: {sum(p.numel() for p in pinn.parameters() if p.requires_grad)}")

# --- 后续的优化器、数据准备、训练循环和可视化代码完全不变 ---
# (只需将下面的代码片段粘贴到您的文件中即可)

# 4. 优化器、损失函数和训练数据
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

nu = 0.01 / np.pi
x_domain = [-1.0, 1.0]
t_domain = [0.0, 1.0]
N_ic = 500
N_bc = 500
N_pde = 20000

x_ic = torch.rand((N_ic, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_ic = torch.zeros((N_ic, 1), device=device)
u_ic = -torch.sin(np.pi * x_ic)

t_bc = torch.rand((N_bc, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
x_bc_left = torch.full((N_bc // 2, 1), x_domain[0], device=device)
x_bc_right = torch.full((N_bc // 2, 1), x_domain[1], device=device)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
u_bc = torch.zeros((N_bc, 1), device=device)

x_pde = torch.rand((N_pde, 1), device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
t_pde = torch.rand((N_pde, 1), device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
x_pde.requires_grad = True
t_pde.requires_grad = True

# 5. 训练循环
epochs = 30000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    u_pred_ic = pinn(x_ic, t_ic)
    loss_ic = loss_fn(u_pred_ic, u_ic)
    
    t_bc_shuffled = t_bc[torch.randperm(t_bc.size()[0])]
    u_pred_bc = pinn(x_bc, t_bc_shuffled)
    loss_bc = loss_fn(u_pred_bc, u_bc)
    
    u_pred_pde = pinn(x_pde, t_pde)
    
    u_t = torch.autograd.grad(u_pred_pde, t_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred_pde, x_pde, grad_outputs=torch.ones_like(u_pred_pde), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    pde_residual = u_t + u_pred_pde * u_x - nu * u_xx
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))
    
    loss_weight_ic = 100.0
    loss_weight_bc = 100.0
    total_loss = loss_weight_ic * loss_ic + loss_weight_bc * loss_bc + loss_pde
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch + 1) % 2000 == 0:
        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  Loss_IC: {loss_ic.item():.6f}, Loss_BC: {loss_bc.item():.6f}, Loss_PDE: {loss_pde.item():.6f}')

# 6. 可视化
# 5. 结果可视化
# 下载并加载参考解
try:
    data = scipy.io.loadmat('burgers_shock.mat')
except FileNotFoundError:
    print("错误: 请先下载 'burgers_shock.mat' 文件。")
    print("请在终端运行: wget https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/burgers_shock.mat")
    exit()

t_exact = data['t'].flatten()[:,None] 
x_exact = data['x'].flatten()[:,None] 
U_exact_grid = np.real(data['usol']) 

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

l2_error = np.linalg.norm(U_pred.flatten() - U_exact_grid.flatten()) / np.linalg.norm(U_exact_grid.flatten())
print(f'L2 Relative Error: {l2_error.item():.4f}')

# 绘制不同时刻的切片图
plt.figure(figsize=(10, 6))
t_slices = [0.25, 0.5, 0.75, 0.9]
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