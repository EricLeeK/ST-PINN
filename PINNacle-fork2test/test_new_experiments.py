# test_new_experiments.py
# Simple test script to verify the new experimental setups work

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import sys
sys.path.append('../')

import torch
import deepxde as dde

# Test all new PDE classes and models
def test_experiment_setups():
    print("Testing new experimental setups...")
    
    # Test 1: Poisson1D
    print("\n1. Testing Poisson1D...")
    from src.pde.poisson import Poisson1D
    
    pde = Poisson1D(a=1)
    print(f"   ✓ PDE created: input_dim={pde.input_dim}, output_dim={pde.output_dim}")
    
    net = dde.nn.FNN([pde.input_dim] + [32, 32] + [pde.output_dim], 
                     activation="tanh", kernel_initializer="Glorot uniform")
    model = pde.create_model(net)
    print("   ✓ Model created successfully")
    
    # Test 2: Wave1D  
    print("\n2. Testing Wave1D...")
    from src.pde.wave import Wave1D
    from src.model.st_pinn import SeparatedNetPolynomial
    
    pde = Wave1D(C=2, bbox=[0, 1, 0, 1], scale=1, a=4)
    print(f"   ✓ PDE created: input_dim={pde.input_dim}, output_dim={pde.output_dim}")
    
    net = SeparatedNetPolynomial(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, kernel_initializer=None,
        spatial_layers=[32, 32], poly_degree=10
    )
    model = pde.create_model(net)
    print("   ✓ Model created successfully")
    
    # Test 3: Helmholtz2D
    print("\n3. Testing Helmholtz2D...")
    from src.pde.helmholtz import Helmholtz2D
    
    pde = Helmholtz2D(scale=1, A=(4, 4), k=1)
    print(f"   ✓ PDE created: input_dim={pde.input_dim}, output_dim={pde.output_dim}")
    
    net = dde.nn.FNN([pde.input_dim] + [64, 64, 64] + [pde.output_dim],
                     activation="tanh", kernel_initializer="Glorot uniform")
    model = pde.create_model(net)
    print("   ✓ Model created successfully")
    
    # Test 4: New Fourier model
    print("\n4. Testing SeparatedNetFourier...")
    from src.model.st_pinn import SeparatedNetFourier
    
    net = SeparatedNetFourier(
        layer_sizes=[2, 0, 1],
        activation=None, kernel_initializer=None,
        spatial_layers=[32, 32], num_frequencies=5, freq_type="linear"
    )
    
    # Test forward pass
    test_input = torch.randn(10, 2)
    output = net(test_input)
    print(f"   ✓ Fourier model works: input {test_input.shape} -> output {output.shape}")
    
    # Test 5: Burgers1D with Fourier
    print("\n5. Testing Burgers1D with Fourier...")
    from src.pde.burgers import Burgers1D

    pde = Burgers1D(datapath="D:/scientific_research/SelfCode/Spatio_Temporal_Neural_Network/PINNacle-fork2test/ref/burgers1d.dat", geom=[-1, 1], time=[0, 1], nu=0.01/3.14159)     # 注意此处的路径需要根据实际情况修改

    print(f"   ✓ PDE created: input_dim={pde.input_dim}, output_dim={pde.output_dim}")
    
    net = SeparatedNetFourier(
        layer_sizes=[pde.input_dim, 0, pde.output_dim],
        activation=None, kernel_initializer=None,
        spatial_layers=[32, 32], num_frequencies=8, freq_type="exponential"
    )
    model = pde.create_model(net)
    print("   ✓ Model created successfully")
    
    print("\n✅ All experimental setups test successfully!")
    print("🎯 Ready to run complete experiments")

if __name__ == "__main__":
    test_experiment_setups()