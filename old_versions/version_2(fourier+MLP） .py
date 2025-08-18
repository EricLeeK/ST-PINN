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


# 1. 定义傅里叶特征编码器
class FourierFeatureEncoder:
    def __init__(self, input_dims, num_freqs):
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        # 创建频率矩阵 B
        # self.b_matrix = torch.randn(input_dims, num_freqs).to(device) * 10 # 随机高斯频率
        # 对于1D输入，使用指数增长的频率更常见
        self.b_matrix = (2.0 ** torch.arange(0, num_freqs)).unsqueeze(0).to(device) * np.pi

    def __call__(self, x):
        # x 的形状是 [N, input_dims]
        # 投影到频率上
        proj = x @ self.b_matrix # [N, input_dims] @ [input_dims, num_freqs] -> [N, num_freqs]
        # 返回 sin 和 cos 特征
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    @property
    def output_dims(self):
        return self.num_freqs * 2

# 2. 修改后的网络结构
class FactorizedPINN_Fourier(nn.Module):
    def __init__(self, spatial_layers, temporal_layers, feature_dim, time_encoder):
        super(FactorizedPINN_Fourier, self).__init__()
        self.time_encoder = time_encoder
        
        # 空间分支 (不变)
        spatial_modules = [nn.Linear(1, spatial_layers[0])]
        for i in range(len(spatial_layers) - 1):
            spatial_modules.append(nn.Tanh())
            spatial_modules.append(nn.Linear(spatial_layers[i], spatial_layers[i+1]))
        spatial_modules.append(nn.Tanh())
        spatial_modules.append(nn.Linear(spatial_layers[-1], feature_dim))
        self.spatial_net = nn.Sequential(*spatial_modules)

        # 时间分支 (输入维度改变)
        # 输入维度现在是编码后的维度
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
        
        # 核心改动：先对时间t进行编码
        t_encoded = self.time_encoder(t)
        
        spatial_features = self.spatial_net(x)
        temporal_features = self.temporal_net(t_encoded) # 将编码后的t送入网络
        
        output = torch.sum(spatial_features * temporal_features, dim=1, keepdim=True)
        return output

# --- 后续的训练数据准备、训练循环、可视化代码与之前完全相同 ---
# --- 只需要在实例化模型时做出改变 ---

# 3. 实例化新的模型
# 定义编码器
num_frequencies = 10 # 超参数L
time_encoder = FourierFeatureEncoder(input_dims=1, num_freqs=num_frequencies)

# 网络参数
spatial_layers = [32, 32, 32]
temporal_layers = [32, 32, 32]
feature_dim = 50

# 实例化新的PINN
pinn = FactorizedPINN_Fourier(
    spatial_layers, 
    temporal_layers, 
    feature_dim, 
    time_encoder
).to(device)

# 之后的 optimizer, loss_fn, training loop, visualization 代码完全复用
# ... (将之前的训练和可视化代码粘贴到这里即可运行)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 准备数据 (与之前相同)
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

epochs = 100000
# 训练循环 (与之前相同)
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
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.7f}')

# 可视化

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