# visualization_utils.py
# Reusable visualization functions for PINN experiments

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io
import os
from pathlib import Path

def generate_burgers_heatmaps(model, exp_name, device='cpu', data_file=None):
    """
    Generate analytical solution, model solution, and error heatmaps for Burgers equation
    
    Args:
        model: Trained PINN model
        exp_name: Experiment name for saving plots
        device: Device to run computations on
        data_file: Path to reference solution file (optional)
    
    Returns:
        dict: Dictionary containing L2 error and plot file paths
    """
    
    # Default data file path - using PINNacle-fork2test data
    if data_file is None:
        data_file = 'PINNacle-fork2test/ref/burgers1d.dat'
    
    # Create results directory
    results_dir = f"runs/{exp_name}/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load reference solution from .dat file
        print(f"Loading reference solution from: {data_file}")
        
        # Read the .dat file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # Find the header line with time information
        header_line = None
        data_start_line = None
        for i, line in enumerate(lines):
            if line.startswith('% X') and 't=' in line:
                header_line = line
                data_start_line = i + 1
                break
        
        if header_line is None:
            raise ValueError("Could not find header line with time information")
        
        # Parse time values from header
        import re
        time_matches = re.findall(r't=([\d.]+)', header_line)
        t_exact_vec = np.array([float(t) for t in time_matches])
        
        # Read the data lines (skip comments)
        data_lines = []
        for line in lines[data_start_line:]:
            line = line.strip()
            if line and not line.startswith('%'):
                data_lines.append(line)
        
        # Parse the data
        data_matrix = []
        x_values = []
        for line in data_lines:
            values = [float(x) for x in line.split()]
            x_values.append(values[0])  # First column is x
            data_matrix.append(values[1:])  # Rest are u values at different times
        
        x_exact_vec = np.array(x_values)
        U_exact = np.array(data_matrix)  # Shape: (n_x, n_t)
        
        print(f"Reference solution shape: {U_exact.shape}")
        print(f"x range: [{x_exact_vec.min():.3f}, {x_exact_vec.max():.3f}]")
        print(f"t range: [{t_exact_vec.min():.3f}, {t_exact_vec.max():.3f}]")
        
    except FileNotFoundError:
        print(f"Error: Reference solution file not found at {data_file}")
        print("Please ensure the burgers1d.dat file is available.")
        return {"error": "Reference file not found"}
    except Exception as e:
        print(f"Error parsing reference solution file: {e}")
        return {"error": f"File parsing error: {e}"}
    
    # Generate prediction grid
    model.net.eval()
    with torch.no_grad():
        # Create coordinate grid matching the reference solution
        T_grid, X_grid = np.meshgrid(t_exact_vec, x_exact_vec)
        x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        
        # Create input tensor [x, t] for the model
        inputs = torch.cat([x_flat, t_flat], dim=1)
        
        # Get model predictions - use DeepXDE predict method
        u_pred_flat = model.predict(inputs.cpu().numpy())
        U_pred = u_pred_flat.reshape(X_grid.shape)
    
    # Calculate error
    Error = np.abs(U_pred - U_exact)
    l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
    print(f'L2 Relative Error: {l2_error:.6f}')
    
    # --- Generate heatmaps ---
    fig_hm = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3)
    extent = [t_exact_vec.min(), t_exact_vec.max(), x_exact_vec.min(), x_exact_vec.max()]
    
    # 1. Model prediction heatmap
    ax1 = fig_hm.add_subplot(gs[0, 0])
    c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x') 
    ax1.set_title('Model Prediction')
    fig_hm.colorbar(c1, ax=ax1)
    
    # 2. Analytical solution heatmap
    ax2 = fig_hm.add_subplot(gs[0, 1])
    c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='jet')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('Analytical Solution')
    fig_hm.colorbar(c2, ax=ax2)
    
    # 3. Error heatmap  
    ax3 = fig_hm.add_subplot(gs[0, 2])
    c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='jet', vmin=0)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_title('Absolute Error')
    fig_hm.colorbar(c3, ax=ax3)
    
    # Add overall title with error information
    fig_hm.suptitle(f"Burgers' Equation | {exp_name} | L2 Error: {l2_error:.4f}", fontsize=16)
    
    # Save heatmap
    heatmap_path = os.path.join(results_dir, 'heatmap_comparison.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close(fig_hm)
    
    # --- Generate time slice plots ---
    fig_slices = plt.figure(figsize=(10, 6))
    t_slices = [0.25, 0.50, 0.75, 0.99]
    
    for t_slice in t_slices:
        idx = np.argmin(np.abs(t_exact_vec - t_slice))
        plt.plot(x_exact_vec, U_exact[:, idx], 'b-', 
                label=f'Analytical t={t_slice}' if t_slice==t_slices[0] else None, alpha=0.8)
        plt.plot(x_exact_vec, U_pred[:, idx], 'r--', 
                label=f'Model t={t_slice}' if t_slice==t_slices[0] else None, alpha=0.8)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', lw=2, label='Analytical'),
                       Line2D([0], [0], color='r', lw=2, linestyle='--', label='Model')]
    plt.legend(handles=legend_elements)
    
    plt.title('Solution at Different Time Slices')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.grid(True, alpha=0.3)
    
    # Save time slice plot
    slices_path = os.path.join(results_dir, 'time_slices.png')
    plt.tight_layout()
    plt.savefig(slices_path, dpi=300, bbox_inches='tight')
    print(f"Time slices plot saved to: {slices_path}")
    plt.close(fig_slices)
    
    return {
        "l2_error": l2_error,
        "heatmap_path": heatmap_path,
        "slices_path": slices_path,
        "results_dir": results_dir
    }

def add_visualization_to_experiment(exp_name, model_checkpoint_path=None):
    """
    Add visualization code to an existing experiment
    
    Args:
        exp_name: Name of the experiment
        model_checkpoint_path: Path to saved model checkpoint (optional)
    """
    
    # Try to find the latest checkpoint if not provided
    if model_checkpoint_path is None:
        runs_dir = f"runs/{exp_name}"
        if os.path.exists(runs_dir):
            # Look for .pt files
            import glob
            checkpoints = glob.glob(os.path.join(runs_dir, "**/*.pt"), recursive=True)
            if checkpoints:
                model_checkpoint_path = max(checkpoints, key=os.path.getctime)
                print(f"Found checkpoint: {model_checkpoint_path}")
            else:
                print(f"No model checkpoints found in {runs_dir}")
                return None
        else:
            print(f"Experiment directory {runs_dir} not found")
            return None
    
    # Load the model
    try:
        # This is a placeholder - actual model loading would depend on how models are saved
        print(f"Would load model from: {model_checkpoint_path}")
        print("Note: Actual model loading implementation needed based on trainer.py saving format")
        
        # For now, return the visualization function that can be called
        return lambda model: generate_burgers_heatmaps(model, exp_name)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None