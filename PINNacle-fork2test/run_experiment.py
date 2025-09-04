# run_my_experiment.py

# === 关键修正1：在导入任何东西之前，设置后端 ===
import os
os.environ['DDE_BACKEND'] = 'pytorch'

# 1. 导入我们需要的工具
# --------------------------------------------------------------------------
import deepxde as dde
import torch

from trainer import Trainer 

# === 关键修正2：导入正确的类名 ===
from src.pde.burgers import Burgers1D  # <-- 正确的类名是 Burgers
# ==================================

from src.model.st_pinn import SeparatedNetPolynomial
from src.utils.callbacks import TesterCallback

# 3. 定义“模型工厂”函数
# ==========================================================================
def get_model():
    # === 关键修正：使用完全正确的类来实例化 ===
    pde = Burgers1D()  # <-- THE KEY FIX!
    # ======================================
    
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, kernel_initializer=None,
        spatial_layers=[64, 64, 64], poly_degree=20
    )
    
    model = pde.create_model(net)
    model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
    
    return model

# 4. 定义训练参数
# ==========================================================================
train_args = {
    'iterations': 20000,
    'callbacks': [TesterCallback(log_every=1000)]
}

# 5. 主程序：开始实验！
# ==========================================================================
if __name__ == "__main__":
    if dde.backend.backend_name == "pytorch":
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    # 实例化 Trainer
    trainer = Trainer(exp_name="MyFinalExperiment_Poly_Burgers1D", device="0")
    
    # 添加任务
    trainer.add_task(get_model, train_args)

    print(">>> 实验开始！所有配置已修正，目标：Burgers1D。")
    trainer.train_all()
    print(">>> 实验完成！")