# simple_demo.py
# Simple demonstration of running a minimal experiment without the full trainer

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import sys
sys.path.append('../')

import torch
import deepxde as dde

def demo_poisson1d():
    """Simple demo of 1D Poisson equation solving"""
    print("🚀 Running simple Poisson1D demo...")
    
    # Import and create PDE
    from src.pde.poisson import Poisson1D
    pde = Poisson1D(a=1)
    
    # Create simple network
    net = dde.nn.FNN(
        layer_sizes=[pde.input_dim] + [32, 32, 32] + [pde.output_dim],
        activation="tanh",
        kernel_initializer="Glorot uniform"
    )
    
    # Create model
    model = pde.create_model(net)
    model.compile(optimizer="adam", lr=1e-3)
    
    print(f"✓ Model created: {pde.input_dim}D input -> {pde.output_dim}D output")
    print(f"✓ Network created successfully")
    
    # Train for a few steps to show it works
    print("⏳ Training for 100 steps...")
    losshistory, train_state = model.train(epochs=100, display_every=50)
    
    print(f"✓ Training completed!")
    print(f"  Final loss: {losshistory.loss_train[-1]:.6f}")
    
    # Make a prediction
    test_points = pde.geom.uniform_points(10, boundary=True)
    prediction = model.predict(test_points)
    
    print(f"✓ Made prediction on {len(test_points)} test points")
    print(f"  Sample prediction values: {prediction[:3].flatten()}")
    
    return model

def demo_fourier_model():
    """Simple demo of new Fourier model"""
    print("\n🎵 Running Fourier model demo...")
    
    from src.model.st_pinn import SeparatedNetFourier
    
    # Create Fourier model
    net = SeparatedNetFourier(
        layer_sizes=[2, 0, 1],
        activation=None, kernel_initializer=None,
        spatial_layers=[32, 32],
        num_frequencies=8,
        freq_type="linear",
        freq_scale=1.0
    )
    
    print(f"✓ Fourier model created with {net.num_frequencies} frequencies")
    
    # Test forward pass
    test_input = torch.rand(20, 2)  # 20 points in 2D space-time
    output = net(test_input)
    
    print(f"✓ Forward pass: {test_input.shape} -> {output.shape}")
    print(f"  Sample outputs: {output[:3].flatten().detach().numpy()}")
    
    return net

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ST-PINN新实验演示 / New Experiments Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Traditional PDE
        model1 = demo_poisson1d()
        
        # Demo 2: New Fourier model
        model2 = demo_fourier_model()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示成功完成！/ All demos completed successfully!")
        print("📝 现在可以运行完整的实验文件")
        print("📝 Now you can run the full experiment files")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check dependencies and file paths.")