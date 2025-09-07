# test_visualization.py
# Test script to verify visualization works with existing models

import os
os.environ['DDE_BACKEND'] = 'pytorch'

import torch
import glob
import sys

# Add the PINNacle-fork2test directory to path
sys.path.insert(0, '/home/runner/work/ST-PINN/ST-PINN/PINNacle-fork2test')

from visualization_utils import generate_burgers_heatmaps

def test_with_existing_model():
    """Test visualization with existing Fourier model"""
    
    try:
        # Import required modules
        from src.pde.burgers import Burgers1D
        from src.model.st_pinn import SeparatedNetFourier
        
        # Find existing model
        exp_name = 'Burgers1D_Fourier_Comparison'
        checkpoint_pattern = f'/home/runner/work/ST-PINN/ST-PINN/runs/{exp_name}/*/*.pt'
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("No existing checkpoints found for testing")
            return False
            
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Testing with checkpoint: {latest_checkpoint}")
        
        # Recreate the model architecture (same as in the experiment file)
        pde = Burgers1D(
            datapath=r"PINNacle-fork2test/ref/burgers1d.dat",
            geom=[-1, 1],           
            time=[0, 1],            
            nu=0.01 / 3.14159       
        )
        
        net = SeparatedNetFourier(
            layer_sizes=[pde.input_dim, 0, pde.output_dim],
            activation=None, 
            kernel_initializer=None,
            spatial_layers=[64, 64, 64],      
            num_frequencies=12,                
            freq_type="exponential",           
            freq_scale=1.5                     
        )
        
        model = pde.create_model(net)
        model.compile(optimizer=torch.optim.Adam(net.parameters(), lr=1e-3))
        
        # Load the saved weights
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.net.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.net.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
        
        # Test visualization 
        results = generate_burgers_heatmaps(model, 'Test_Visualization', 'cpu')
        
        if "error" not in results:
            print(f"✓ Visualization test successful! L2 error: {results['l2_error']:.6f}")
            print(f"  Generated files:")
            print(f"    {results['heatmap_path']}")
            print(f"    {results['slices_path']}")
            return True
        else:
            print(f"✗ Visualization failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing visualization with existing model...")
    success = test_with_existing_model()
    if success:
        print("\n✓ Visualization test passed!")
    else:
        print("\n✗ Visualization test failed!")