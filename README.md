# 基于时空分解的物理信息神经网络（PINN）PDE求解器

## 摘要 (Abstract)

本项目实现了一种新颖的物理信息神经网络（PINN）架构，专门用于求解各类与时间相关的偏微分方程（PDEs）。其核心创新在于采用了**时空分解（Spatio-temporal Decomposition）**的思想，将待求解的函数 $u(x, t)$ 近似为一个由神经网络学习的**空间系数向量**与一个**预定义的、固定的时间基函数向量**的点积。这种设计将学习的复杂性主要集中在空间维度，而时间演化则由一组解析的基函数显式表达，为处理复杂的时间动态问题提供了一个高效且灵活的框架。

该项目代码结构清晰，实现了模型与实验的完全解耦。所有实验均由配置文件驱动，确保了结果的**完全可追溯与可复现性**，这对于科学研究至关重要。

## 核心思想：时空分解物理信息神经网络

传统的PINN通过一个单一的神经网络 $N(x, t; \theta)$ 来直接近似解 $u(x, t)$。其损失函数通常由初始条件（IC）、边界条件（BC）和PDE残差三部分构成：
$$
\mathcal{L}(\theta) = \lambda_{ic}\mathcal{L}_{ic} + \lambda_{bc}\mathcal{L}_{bc} + \lambda_{pde}\mathcal{L}_{pde}
$$
我们的方法则对解的结构进行了假设，将其分解为空间和时间两个部分：

$$
u(x, t) \approx \hat{u}(x, t; \theta) = \vec{C}(x; \theta) \cdot \vec{T}(t) = \sum_{i=1}^{L} C_i(x; \theta) T_i(t)
$$

其中：

*   **$\vec{C}(x; \theta)$**: 空间系数向量 (Spatial Coefficient Vector)。
    *   这是一个由深度神经网络（通常是多层感知机 MLP）建模的函数，其参数为 $\theta$。
    *   网络的输入是空间坐标 $x$（可以是多维的），输出是一个 $L$ 维的向量 $\vec{C}(x; \theta) = [C_1(x; \theta), \dots, C_L(x; \theta)]^T$。
    *   这个网络是模型中**唯一需要学习**的部分，它负责捕捉解在空间上的所有复杂性。

*   **$\vec{T}(t)$**: 时间基函数向量 (Temporal Basis Vector)。
    *   这是一个**固定的、非学习的**函数向量，由一组预先选择的基函数构成。
    *   它负责描述解在时间维度上的演化行为。
    *   通过选择不同的基函数，我们可以为不同类型的问题注入先验知识。

本项目中已实现了两种主要的时间基：

1.  **多项式基 (Polynomial Basis)**

$$
\vec{T}(t) = [1, t, t^2, \dots, t^{L-1}]^T
$$

3.  **傅里叶基 (Fourier Basis)**

$$
\vec{T}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \dots, \sin(\omega_{L/2} t), \cos(\omega_{L/2} t)]^T
$$
    
  其中，频率 $\omega_i$ 支持两种生成模式：
    *   **线性增长 (linear)**: `1, 2, 3, ...`
    *   **指数增长 (exponential)**: `1, 2, 4, 8, ...` (对于捕捉多尺度或突变现象特别有效)



## 环境搭建

## 环境搭建

1.  **克隆仓库**
    ```bash
    git clone <your-repository-url>
    cd pinn_project
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖**
    本项目的所有依赖项都已在 `requirements.txt` 文件中列出。运行以下命令即可一键安装：
    ```bash
    pip install -r requirements.txt
    ```
    
    > **💡 重要提示: 关于PyTorch的安装**
    > 
    > 为了确保您能使用 GPU 加速（如果可用），强烈建议您**首先访问 [PyTorch官网](https://pytorch.org/get-started/locally/)**，根据您的操作系统和 CUDA 版本获取最适合的安装命令，并单独执行它。
    >
    > 例如，您可能会先运行官网提供的命令：
    > `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    > 
    > 然后再运行 `pip install -r requirements.txt` 来安装其余的依赖库。`pip` 会智能地跳过已经安装的 PyTorch。

## 如何运行实验

本项目的核心是**配置驱动**。每一个 `train/train_*.py` 脚本的顶部都有一个名为 `config` 的Python字典，它是实验的唯一控制中心。

1.  **打开一个实验脚本**: 例如 `train/train_allen_cahn.py`。

2.  **修改 `config` 字典**:
    ```python
    config = {
        # --- 结果将保存在 "results/AC_Fourier_Test/" 文件夹 ---
        "experiment_name": "AC_Fourier_Test",
        
        # --- 选择傅里叶基模型 ---
        "model_type": "fourier",
        
        # --- 定义网络结构和时间基参数 ---
        "spatial_layers": [128, 128, 128, 128],
        "num_frequencies": 40,
        "freq_type": "linear",
        
        # --- 定义训练参数 ---
        "epochs": 5000,
        "learning_rate": 1e-3,
        "loss_weights": { "ic": 100.0, "bc": 100.0, "pde": 1.0 },
        
        # ... 其他参数
    }
    ```
    您可以自由调整任何参数，例如将 `model_type` 改为 `"polynomial"`，或者修改网络层数、学习率等。

3.  **运行脚本**:
    ```bash
    python train/train_allen_cahn.py
    ```

4.  **查看结果**: 脚本运行结束后，所有结果（配置快照、损失日志、结果图）都会自动保存在 `results/AC_Fourier_Test/` 文件夹中。

## 如何扩展项目

### 场景1: 求解一个新的偏微分方程 (PDE)

1.  **创建新脚本**: 复制一个现有的 `train_*.py` 文件（例如 `train_diffusion.py`），并重命名为 `train_new_pde.py`。
2.  **修改 `config`**: 在新脚本中，更新 `experiment_name` 和其他你需要的模型/训练参数。
3.  **定义数据点**: 修改 **数据准备** 部分的代码，以满足新PDE的初始条件（IC）和边界条件（BC）。
4.  **实现PDE残差**: 在 **训练循环** 中，修改 `pde_residual` 的计算方式，使其与你的新PDE的数学形式一致。
5.  **（可选）修改可视化**: 如果你有新PDE的参考解，请更新 **可视化** 部分的代码以加载并绘制它。
6.  **运行**: `python train/train_new_pde.py`

### 场景2: 添加一个新的时间基 (Time Basis)

1.  **修改 `src/models.py`**:
    *   创建一个新的模型类，例如 `SeparatedPINN_WaveletTime`。
    *   在新类的 `__init__` 方法中，定义空间网络和你的新时间基（如小波基）。
    *   在新类的 `forward` 方法中，实现空间系数和时间基的点积运算。
2.  **更新训练脚本**:
    *   在 `train_*.py` 的 **模型初始化** 部分，添加一个新的 `elif` 分支来处理 `model_type == "wavelet"` 的情况，使其能够正确地实例化你的新模型。
    *   在 `config` 字典中，添加新模型可能需要的特定参数（如小波类型等）。

## 已实现案例

-   **一维扩散方程 (Diffusion Equation)** 带源项
-   **一维粘性伯格斯方程 (Viscous Burgers' Equation)**
-   **一维艾伦-卡恩方程 (Allen-Cahn Equation)**

---
