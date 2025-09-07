# ST-PINN 实验文档 / ST-PINN Experimental Documentation

## 路径格式说明 / Path Format Instructions

所有实验文件中的数据路径已统一为相对路径格式：
```python
datapath=r"ref/filename.dat"
icpath=(r"ref/init_u.dat", r"ref/init_v.dat")  # 对于需要初始条件文件的情况
```

**注意**: 
- 运行实验时需要在 `PINNacle-fork2test` 目录下执行
- 数据文件位于 `ref/` 子目录中
- 使用原始字符串 `r""` 避免路径分隔符问题

---

### 训练指标 / Training Metrics

训练过程中输出的各项指标及其物理/数学意义：

**损失函数组件 / Loss Function Components:**
- **PDE Loss**: 偏微分方程残差的均方误差，衡量网络解满足微分方程的程度
- **IC Loss**: 初始条件损失，衡量t=0时刻解与给定初值的匹配程度  
- **BC Loss**: 边界条件损失，衡量空间边界上解的约束满足程度
- **Data Loss**: 数据损失（如有参考数据），衡量网络预测与真实解的匹配程度

**误差指标 / Error Metrics:**
- **L2 Relative Error**: $\frac{||\mathbf{u}_{pred} - \mathbf{u}_{true}||_2}{||\mathbf{u}_{true}||_2}$，相对L2误差，无量纲化的解精度指标
- **Mean Squared Error (MSE)**: $\frac{1}{N}\sum_{i=1}^{N}(u_i^{pred} - u_i^{true})^2$，均方误差，绝对误差指标
- **Mean Absolute Percentage Error (MAPE)**: $\frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{u_i^{pred} - u_i^{true}}{u_i^{true}}\right|$，平均绝对百分比误差

**物理意义:**
- **收敛性**: 损失函数持续下降表明网络正在学习PDE的解
- **精度**: L2相对误差小于1%通常认为是高精度解
- **稳定性**: 损失函数平稳下降（无剧烈震荡）表明训练稳定

---

## 实验内容详述 / Detailed Experimental Content

### 1. run_experiment_burgers1d_fourier.py
**PDE问题**: 一维Burgers方程 (1D Burgers Equation)

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial u} = \nu \frac{\partial^2 u}{\partial x^2}$$

- **方程类型**: 抛物型非线性PDE
- **物理意义**: 描述粘性流体中的激波和扩散现象，是Navier-Stokes方程的简化版本
- **特征**: 非线性对流项 $u\frac{\partial u}{\partial x}$ 导致激波形成，粘性项 $\nu\frac{\partial^2 u}{\partial x^2}$ 提供扩散
- **ST-PINN架构**: SeparatedNetFourier (傅里叶时间基)
- **计算域**: $x \in [-1,1], t \in [0,1]$
- **参数**: 粘性系数 $\nu = 0.01/\pi$

### 2. run_experiment_burgers2d_fourier.py  
**PDE问题**: 二维Burgers方程 (2D Burgers Equation)

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = \nu \nabla^2 \mathbf{u}$$

其中 $\mathbf{u} = (u, v)^T$ 是速度场，具体形式为:
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

- **方程类型**: 抛物型非线性PDE系统
- **物理意义**: 二维粘性流体中的激波传播和扩散
- **特征**: 非线性对流、粘性扩散、二维空间耦合
- **ST-PINN架构**: SeparatedNetFourier (傅里叶时间基)
- **计算域**: $(x,y) \in [0,L]^2, t \in [0,T]$
- **输出**: 速度分量$(u, v)$

### 3. run_experiment_heat2d.py
**PDE问题**: 二维变系数热方程 (2D Heat Equation with Varying Coefficients)

$$\frac{\partial u}{\partial t} = \nabla \cdot (D(\mathbf{x}) \nabla u) + f(\mathbf{x}, t)$$

展开为:
$$\frac{\partial u}{\partial t} = D(x,y)\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + A\sin(m_1\pi x)\sin(m_2\pi y)\sin(m_3\pi t)$$

- **方程类型**: 抛物型线性PDE with 源项
- **物理意义**: 非均匀材料中的热传导，扩散系数$D(x,y)$空间变化
- **特征**: 变系数、周期性源项、复杂边界条件
- **ST-PINN架构**: SeparatedNetPolynomial (多项式时间基)  
- **计算域**: $(x,y) \in [0,1]^2, t \in [0,5]$

### 4. run_experiment_wave1d.py
**PDE问题**: 一维波动方程 (1D Wave Equation)

$$\frac{\partial^2 u}{\partial t^2} = C^2 \frac{\partial^2 u}{\partial x^2}$$

- **方程类型**: 双曲型线性PDE
- **物理意义**: 一维弦振动、声波传播
- **特征**: 二阶时间导数、无耗散、波速$C$
- **ST-PINN架构**: SeparatedNetPolynomial (多项式时间基)
- **计算域**: $x \in [0,1], t \in [0,1]$  
- **参数**: 波速 $C = 2$

### 5. run_experiment_grayscott.py
**PDE问题**: Gray-Scott反应扩散系统 (Gray-Scott Reaction-Diffusion System)

$$\frac{\partial u}{\partial t} = \epsilon_u \nabla^2 u + b(1-u) - uv^2$$
$$\frac{\partial v}{\partial t} = \epsilon_v \nabla^2 v - dv + uv^2$$

- **方程类型**: 抛物型非线性反应扩散PDE系统
- **物理意义**: 化学反应中的斑图形成，描述两种化学物质$u, v$的浓度演化
- **特征**: 非线性反应项$uv^2$、不同扩散系数、复杂时空斑图
- **ST-PINN架构**: SeparatedNetFourier (傅里叶时间基)
- **计算域**: $(x,y) \in [-1,1]^2, t \in [0,200]$
- **参数**: $b=0.04$ (进料率), $d=0.1$ (死亡率), $\epsilon=(10^{-5}, 5\times10^{-6})$

### 6. run_experiment_kuramoto_sivashinsky.py  
**PDE问题**: Kuramoto-Sivashinsky方程 (Kuramoto-Sivashinsky Equation)

$$\frac{\partial u}{\partial t} + \alpha u \frac{\partial u}{\partial x} + \beta \frac{\partial^2 u}{\partial x^2} + \gamma \frac{\partial^4 u}{\partial x^4} = 0$$

- **方程类型**: 四阶非线性色散-扩散PDE  
- **物理意义**: 火焰前锋不稳定性、薄膜流动中的混沌动力学
- **特征**: 四阶导数项、非线性、混沌解、多尺度结构
- **ST-PINN架构**: SeparatedNetPolynomial (多项式时间基)
- **计算域**: $x \in [0,2\pi], t \in [0,1]$
- **参数**: $\alpha=\frac{100}{16}, \beta=\frac{100}{16^2}, \gamma=\frac{100}{16^4}$

### 7. run_experiment_heat2d_multiscale.py
**PDE问题**: 二维多尺度热方程 (2D Multiscale Heat Equation)

$$\frac{\partial u}{\partial t} = D_x \frac{\partial^2 u}{\partial x^2} + D_y \frac{\partial^2 u}{\partial y^2}$$

其中扩散系数: $D_x = \frac{1}{(500\pi)^2}, D_y = \frac{1}{\pi^2}$

初始条件: $u(x,y,0) = \sin(20\pi x)\sin(\pi y)$

- **方程类型**: 抛物型线性PDE with 多尺度特征
- **物理意义**: 各向异性材料中的热传导，x方向扩散远慢于y方向
- **特征**: 极不同的扩散尺度、高频初值、快慢动力学分离
- **ST-PINN架构**: SeparatedNetFourier (傅里叶时间基)
- **计算域**: $(x,y) \in [0,1]^2, t \in [0,5]$

### 8. run_experiment_ns2d_longtime.py
**PDE问题**: 二维Navier-Stokes方程长时间动力学 (2D Navier-Stokes Long-time Dynamics)

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}$$
$$\nabla \cdot \mathbf{u} = 0$$

展开为速度-压力形式:
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

- **方程类型**: 抛物型非线性PDE系统 + 椭圆约束
- **物理意义**: 不可压缩粘性流体运动，涡量演化和能量耗散
- **特征**: 非线性对流、粘性扩散、不可压缩约束、长时间稳定性
- **ST-PINN架构**: SeparatedNetPolynomial (多项式时间基)  
- **计算域**: $(x,y) \in [0,2]\times[0,1], t \in [0,5]$
- **参数**: Reynolds数 $Re = 100$
- **输出**: 速度分量$(u, v)$和压力$p$

### 9. run_experiment_wave2d_longtime.py
**PDE问题**: 二维波动方程长时间动力学 (2D Wave Equation Long-time Dynamics)

$$\frac{\partial^2 u}{\partial t^2} = a^2 \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

- **方程类型**: 双曲型线性PDE
- **物理意义**: 二维膜振动、声波/地震波传播
- **特征**: 长时间波动、多模态振荡、能量守恒
- **ST-PINN架构**: SeparatedNetFourier (傅里叶时间基)
- **计算域**: $(x,y) \in [0,1]^2, t \in [0,100]$ (超长时间)
- **参数**: 波速 $a = \sqrt{2}$，多个空间频率模态

---

## ST-PINN架构特点 / ST-PINN Architecture Features

### 分离式网络结构 / Separated Network Structure

ST-PINN采用时空分离的网络架构：
$$u(\mathbf{x}, t) = \sum_{k=1}^{N} C_k(\mathbf{x}) \cdot T_k(t)$$

其中：
- $C_k(\mathbf{x})$: 空间特征网络，学习空间依赖性
- $T_k(t)$: 时间基函数，捕获时间演化模式

### 两种时间基函数 / Two Time Basis Types

**1. 多项式基 (Polynomial Basis):**
$$T_k(t) = t^{k-1}, \quad k = 1, 2, ..., N$$

- 适用于: 有限时间区间、光滑时间演化
- 优势: 简单、收敛快、适合短时间动力学
- 典型应用: Burgers方程、Kuramoto-Sivashinsky方程

**2. 傅里叶基 (Fourier Basis):**  
$$T_k(t) = \{\sin(\omega_k t), \cos(\omega_k t)\}, \quad \omega_k = k\omega_0$$

- 适用于: 周期性/准周期性、长时间动力学
- 优势: 频域表示、适合振荡解、长时间稳定
- 典型应用: Gray-Scott系统、长时间波动方程

### 架构优势 / Architectural Advantages

1. **参数效率**: 比传统PINN参数量更少
2. **时间外推**: 基函数的解析形式支持时间外推
3. **物理先验**: 时空分离符合许多物理系统的固有结构
4. **数值稳定性**: 避免了深度时间网络的梯度消失问题

---

---

## 可用实验总览 / Available Experiments Overview

本库现包含以下ST-PINN实验，均针对时间相关PDE设计：

| 实验文件 | PDE类型 | 空间维度 | 输出维度 | 时间基函数 | 复杂度 |
|----------|---------|----------|----------|------------|--------|
| `run_experiment_burgers1d_fourier.py` | 1D Burgers | 1D | 1 | Fourier | 中等 |
| `run_experiment_burgers2d_fourier.py` | 2D Burgers | 2D | 2 | Fourier | 高 |
| `run_experiment_heat2d.py` | 2D变系数热方程 | 2D | 1 | Polynomial | 中等 |
| `run_experiment_wave1d.py` | 1D波动方程 | 1D | 1 | Polynomial | 低 |
| `run_experiment_grayscott.py` | Gray-Scott反应扩散 | 2D | 2 | Fourier | 很高 |
| `run_experiment_kuramoto_sivashinsky.py` | Kuramoto-Sivashinsky | 1D | 1 | Polynomial | 很高 |
| `run_experiment_heat2d_multiscale.py` | 2D多尺度热方程 | 2D | 1 | Fourier | 高 |
| `run_experiment_ns2d_longtime.py` | 2D Navier-Stokes | 2D | 3 | Polynomial | 极高 |
| `run_experiment_wave2d_longtime.py` | 2D长时间波动 | 2D | 1 | Fourier | 高 |

**运行建议**:
1. 首先运行低复杂度实验验证环境配置
2. 根据计算资源选择合适的实验
3. 高复杂度实验建议使用GPU加速

**快速测试**:
```bash
cd PINNacle-fork2test
python run_experiment_wave1d.py  # 最简单的测试
```

---

### 基函数选择指导 / Basis Function Selection Guide

| PDE类型 | 推荐基函数 | 理由 |
|---------|------------|------|
| 抛物型 (Heat, Burgers) | Polynomial | 衰减性时间行为 |
| 双曲型 (Wave) | Fourier | 振荡性时间行为 |  
| 混沌系统 (KS, Gray-Scott) | Fourier | 复杂时间动力学 |
| 长时间动力学 | Fourier | 避免高阶多项式不稳定 |

### 网络参数调优 / Network Parameter Tuning

- **空间网络深度**: 复杂几何需要更深网络(4-5层)
- **时间基数量**: 多项式degree=15-30，傅里叶modes=10-30  
- **学习率**: 非线性PDE使用较小学习率(1e-4到1e-3)
- **训练轮数**: 混沌系统需要更多迭代(25k-40k)