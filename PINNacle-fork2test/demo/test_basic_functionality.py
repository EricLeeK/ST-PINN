#!/usr/bin/env python3
"""
Simple test of data parsing functionality without external dependencies.
This verifies that the core data file parsing works correctly.
"""

import os
import sys

# Add parent directory to path
sys.path.append('..')

def test_basic_parsing():
    """Test basic data file parsing without numpy/matplotlib"""
    print("ST-PINN Data Parsing Test")
    print("=" * 40)
    
    # Test if we can read and parse the basic file structure
    test_files = [
        ('../ref/burgers1d.dat', '1D Burgers'),
        ('../ref/Kuramoto_Sivashinsky.dat', '1D Kuramoto-Sivashinsky'),
        ('../ref/burgers2d_1.dat', '2D Burgers'),
        ('../ref/heat_multiscale.dat', '2D Heat'),
        ('../ref/grayscott.dat', '2D Gray-Scott')
    ]
    
    for filepath, description in test_files:
        print(f"\nTesting {description}: {os.path.basename(filepath)}")
        
        if not os.path.exists(filepath):
            print(f"  ✗ File not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            print(f"  ✓ File readable, {len(lines)} lines")
            
            # Analyze file structure
            header_lines = 0
            data_lines = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('%') or line.startswith('#'):
                    header_lines += 1
                elif line and not line.startswith('%'):
                    try:
                        # Try to parse as numbers
                        values = line.split()
                        float_values = [float(x) for x in values]
                        data_lines += 1
                        if data_lines == 1:
                            print(f"  ✓ First data line: {len(float_values)} columns")
                    except ValueError:
                        continue
            
            print(f"  ✓ Structure: {header_lines} header lines, {data_lines} data lines")
            
            # Check for time information
            has_time_info = any('t=' in line for line in lines[:10])
            print(f"  ✓ Time information: {'Found' if has_time_info else 'Not found'}")
            
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")

def test_experiment_structure():
    """Test that all experiment files exist and have visualization code"""
    print("\n" + "=" * 40)
    print("Experiment File Structure Test")
    print("=" * 40)
    
    experiment_files = [
        'run_experiment_burgers1d_standard_fnn.py',
        'run_experiment_burgers1d_fourier.py', 
        'run_experiment_wave1d.py',
        'run_experiment_kuramoto_sivashinsky.py',
        'run_experiment_heat2d.py',
        'run_experiment_heat2d_multiscale.py',
        'run_experiment_burgers2d_fourier.py',
        'run_experiment_grayscott.py',
        'run_experiment_ns2d_longtime.py',
        'run_experiment_wave2d_longtime.py'
    ]
    
    for filename in experiment_files:
        filepath = f"../{filename}"
        print(f"\nChecking {filename}:")
        
        if not os.path.exists(filepath):
            print(f"  ✗ File not found")
            continue
            
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check for visualization imports
            has_viz_import = 'visualization_utils' in content
            has_viz_section = '可视化部分' in content or 'visualization' in content.lower()
            has_checkpoint_loading = 'checkpoint' in content and 'glob' in content
            
            print(f"  ✓ File exists ({len(content)} chars)")
            print(f"  {'✓' if has_viz_import else '✗'} Visualization import: {has_viz_import}")
            print(f"  {'✓' if has_viz_section else '✗'} Visualization section: {has_viz_section}")
            print(f"  {'✓' if has_checkpoint_loading else '✗'} Checkpoint loading: {has_checkpoint_loading}")
            
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")

def test_demo_files():
    """Test that demo files are properly created"""
    print("\n" + "=" * 40)
    print("Demo Files Test")
    print("=" * 40)
    
    demo_files = [
        ('test_visualization_utils.py', 'Visualization test script'),
        ('run_demo_experiment.py', 'Demo experiment runner'),
        ('README.md', 'Documentation')
    ]
    
    for filename, description in demo_files:
        print(f"\nChecking {description}: {filename}")
        
        if not os.path.exists(filename):
            print(f"  ✗ File not found")
            continue
            
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            print(f"  ✓ File exists ({len(content)} chars)")
            
            # Check for key content
            if filename.endswith('.py'):
                has_main = '__main__' in content
                has_imports = 'import' in content
                print(f"  {'✓' if has_main else '✗'} Has main section: {has_main}")
                print(f"  {'✓' if has_imports else '✗'} Has imports: {has_imports}")
            elif filename.endswith('.md'):
                has_usage = 'Usage' in content or 'usage' in content
                has_examples = 'example' in content.lower()
                print(f"  {'✓' if has_usage else '✗'} Has usage instructions: {has_usage}")
                print(f"  {'✓' if has_examples else '✗'} Has examples: {has_examples}")
                
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")

def main():
    """Run all tests"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    test_basic_parsing()
    test_experiment_structure()
    test_demo_files()
    
    print("\n" + "=" * 40)
    print("Basic Tests Completed")
    print("=" * 40)
    print("\nTo test full visualization functionality:")
    print("1. Install dependencies: numpy, matplotlib")
    print("2. Run: python test_visualization_utils.py")
    print("3. Or run any experiment file with visualization")

if __name__ == "__main__":
    main()