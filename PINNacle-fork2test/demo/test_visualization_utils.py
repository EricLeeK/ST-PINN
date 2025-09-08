#!/usr/bin/env python3
"""
Demo script to test visualization utilities without running full experiments.
This tests the data parsing and visualization generation functionality.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append('..')

# Import visualization utilities
from src.utils.visualization_utils import (
    parse_1d_data_file, 
    parse_2d_data_file,
    generate_1d_visualization,
    generate_2d_scalar_visualization,
    generate_2d_vector_visualization
)

class MockModel:
    """Mock model for testing visualization without training"""
    
    def __init__(self, data_shape):
        self.data_shape = data_shape
        
    def predict(self, inputs):
        """Generate synthetic predictions for testing"""
        n_points = inputs.shape[0]
        
        if len(self.data_shape) == 1:  # 1D output
            # Simple sinusoidal prediction based on x and t
            x, t = inputs[:, 0], inputs[:, 1] 
            predictions = np.sin(2*np.pi*x) * np.cos(np.pi*t) + 0.1*np.random.randn(n_points)
            return predictions.reshape(-1, 1)
        else:  # Multi-dimensional output
            # Create multiple outputs for vector fields
            predictions = []
            for i in range(self.data_shape[0]):
                if inputs.shape[1] == 2:  # 1D problem (x,t)
                    x, t = inputs[:, 0], inputs[:, 1]
                    pred = np.sin((i+1)*np.pi*x) * np.cos((i+1)*np.pi*t) + 0.1*np.random.randn(n_points)
                else:  # 2D problem (x,y,t)
                    x, y, t = inputs[:, 0], inputs[:, 1], inputs[:, 2]
                    pred = np.sin((i+1)*np.pi*x) * np.sin((i+1)*np.pi*y) * np.cos(np.pi*t) + 0.1*np.random.randn(n_points)
                predictions.append(pred)
            return np.column_stack(predictions)

def test_data_parsing():
    """Test data file parsing functionality"""
    print("=" * 60)
    print("Testing Data File Parsing")
    print("=" * 60)
    
    # Test 1D data parsing
    try:
        print("\n1. Testing 1D data parsing (burgers1d.dat)...")
        x_vec, t_vec, U_exact = parse_1d_data_file('../ref/burgers1d.dat')
        print(f"✓ Successfully parsed burgers1d.dat")
        print(f"  Shape: {U_exact.shape}, x: {len(x_vec)}, t: {len(t_vec)}")
        
        print("\n2. Testing Kuramoto-Sivashinsky data parsing...")
        x_vec_ks, t_vec_ks, U_exact_ks = parse_1d_data_file('../ref/Kuramoto_Sivashinsky.dat')
        print(f"✓ Successfully parsed Kuramoto_Sivashinsky.dat")
        print(f"  Shape: {U_exact_ks.shape}, x: {len(x_vec_ks)}, t: {len(t_vec_ks)}")
    except Exception as e:
        print(f"✗ Error parsing 1D data: {e}")
    
    # Test 2D data parsing
    try:
        print("\n3. Testing 2D data parsing (burgers2d_1.dat)...")
        x_vec, y_vec, t_vec, U_exact, var_names = parse_2d_data_file('../ref/burgers2d_1.dat')
        print(f"✓ Successfully parsed burgers2d_1.dat")
        print(f"  Shape: {U_exact.shape}, variables: {var_names}")
        
        print("\n4. Testing heat2d data parsing...")
        x_vec_h, y_vec_h, t_vec_h, U_exact_h, var_names_h = parse_2d_data_file('../ref/heat_multiscale.dat')
        print(f"✓ Successfully parsed heat_multiscale.dat")
        print(f"  Shape: {U_exact_h.shape}, variables: {var_names_h}")
        
        print("\n5. Testing grayscott data parsing...")
        x_vec_g, y_vec_g, t_vec_g, U_exact_g, var_names_g = parse_2d_data_file('../ref/grayscott.dat')
        print(f"✓ Successfully parsed grayscott.dat")
        print(f"  Shape: {U_exact_g.shape}, variables: {var_names_g}")
    except Exception as e:
        print(f"✗ Error parsing 2D data: {e}")
        import traceback
        traceback.print_exc()

def test_1d_visualization():
    """Test 1D visualization generation"""
    print("\n" + "=" * 60)
    print("Testing 1D Visualization Generation")
    print("=" * 60)
    
    try:
        # Create mock model
        mock_model = MockModel((1,))
        
        # Test with burgers1d data
        print("\n1. Testing 1D visualization with burgers1d data...")
        results = generate_1d_visualization(
            mock_model, 
            "Demo_Burgers1D", 
            device='cpu', 
            data_file='../ref/burgers1d.dat'
        )
        
        if "error" not in results:
            print(f"✓ Successfully generated 1D visualization")
            print(f"  L2 error: {results['l2_error']:.4f}")
            print(f"  Heatmap: {results['heatmap_path']}")
            print(f"  Slices: {results['slices_path']}")
        else:
            print(f"✗ Visualization failed: {results['error']}")
            
    except Exception as e:
        print(f"✗ Error in 1D visualization: {e}")
        import traceback
        traceback.print_exc()

def test_2d_scalar_visualization():
    """Test 2D scalar visualization generation"""
    print("\n" + "=" * 60)
    print("Testing 2D Scalar Visualization Generation")
    print("=" * 60)
    
    try:
        # Create mock model
        mock_model = MockModel((1,))
        
        # Test with heat2d data
        print("\n1. Testing 2D scalar visualization with heat data...")
        results = generate_2d_scalar_visualization(
            mock_model, 
            "Demo_Heat2D", 
            device='cpu', 
            data_file='../ref/heat_multiscale.dat'
        )
        
        if "error" not in results:
            print(f"✓ Successfully generated 2D scalar visualization")
            print(f"  L2 error: {results['l2_error']:.4f}")
            print(f"  Spatial comparison: {results['heatmap_path']}")
        else:
            print(f"✗ Visualization failed: {results['error']}")
            
    except Exception as e:
        print(f"✗ Error in 2D scalar visualization: {e}")
        import traceback
        traceback.print_exc()

def test_2d_vector_visualization():
    """Test 2D vector visualization generation"""
    print("\n" + "=" * 60)
    print("Testing 2D Vector Visualization Generation")
    print("=" * 60)
    
    try:
        # Create mock model with 2 outputs
        mock_model = MockModel((2,))
        
        # Test with burgers2d data
        print("\n1. Testing 2D vector visualization with burgers2d data...")
        results = generate_2d_vector_visualization(
            mock_model, 
            "Demo_Burgers2D", 
            device='cpu', 
            data_file='../ref/burgers2d_1.dat'
        )
        
        if "error" not in results:
            print(f"✓ Successfully generated 2D vector visualization")
            print(f"  L2 error: {results['l2_error']:.4f}")
            print(f"  Vector field comparison: {results['heatmap_path']}")
        else:
            print(f"✗ Visualization failed: {results['error']}")
            
    except Exception as e:
        print(f"✗ Error in 2D vector visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all visualization tests"""
    print("ST-PINN Visualization Utilities Demo")
    print("=" * 60)
    print("This script tests the visualization functionality")
    print("without requiring trained models or external dependencies.")
    print("=" * 60)
    
    # Change to demo directory for relative paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create visualization output directory
    os.makedirs("runs", exist_ok=True)
    
    # Run tests
    test_data_parsing()
    test_1d_visualization()
    test_2d_scalar_visualization()
    test_2d_vector_visualization()
    
    print("\n" + "=" * 60)
    print("Demo completed! Check the runs/ directory for generated visualizations.")
    print("=" * 60)

if __name__ == "__main__":
    main()