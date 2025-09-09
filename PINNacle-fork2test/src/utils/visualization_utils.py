# visualization_utils.py
# Reusable visualization functions for PINN experiments

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io
import os
from pathlib import Path
import re

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
    
    # Default data file path - relative to experiment run directory
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
    if hasattr(model, 'net'):
        model.net.eval()
    
    # Create coordinate grid matching the reference solution
    T_grid, X_grid = np.meshgrid(t_exact_vec, x_exact_vec)
    
    if TORCH_AVAILABLE and torch is not None:
        with torch.no_grad():
            x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            
            # Create input tensor [x, t] for the model
            inputs = torch.cat([x_flat, t_flat], dim=1)
            
            # Get model predictions - use DeepXDE predict method
            u_pred_flat = model.predict(inputs.cpu().numpy())
            U_pred = u_pred_flat.reshape(X_grid.shape)
    else:
        # Fallback for when torch is not available (testing)
        x_flat = X_grid.flatten().reshape(-1, 1)
        t_flat = T_grid.flatten().reshape(-1, 1)
        inputs = np.hstack([x_flat, t_flat])
        
        # Get model predictions
        u_pred_flat = model.predict(inputs)
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

def parse_1d_data_file(data_file):
    """
    Parse 1D time series data file format (e.g., burgers1d.dat, Kuramoto_Sivashinsky.dat)
    
    Returns:
        x_vec: 1D array of spatial coordinates
        t_vec: 1D array of time coordinates  
        U_exact: 2D array of shape (n_x, n_t) - solution values
    """
    print(f"Parsing 1D data file: {data_file}")
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # Check if this is a COMSOL format file (has header with time info)
    header_line = None
    data_start_line = None
    
    for i, line in enumerate(lines):
        if line.startswith('% X') and 't=' in line:
            header_line = line
            data_start_line = i + 1
            break
        elif line.startswith('% Description:'):
            # Check next few lines for time info
            for j in range(i+1, min(i+5, len(lines))):
                if 't=' in lines[j]:
                    header_line = lines[j]
                    data_start_line = j + 1
                    break
            break
    
    if header_line is not None:
        # COMSOL format with time info in header
        time_matches = re.findall(r't=([\d.]+)', header_line)
        t_vec = np.array([float(t) for t in time_matches])
        
        # Read data lines
        data_lines = []
        for line in lines[data_start_line:]:
            line = line.strip()
            if line and not line.startswith('%'):
                data_lines.append(line)
        
        # Parse data matrix
        data_matrix = []
        x_values = []
        for line in data_lines:
            values = [float(x) for x in line.split()]
            x_values.append(values[0])  # First column is x
            data_matrix.append(values[1:])  # Rest are u values at different times
        
        x_vec = np.array(x_values)
        U_exact = np.array(data_matrix)  # Shape: (n_x, n_t)
        
    else:
        # Simple format like Kuramoto-Sivashinsky: x, t, u columns
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                data_lines.append(line)
        
        # Parse all data points
        all_data = []
        for line in data_lines:
            values = [float(x) for x in line.split()]
            all_data.append(values)
        
        all_data = np.array(all_data)
        
        if all_data.shape[1] == 3:
            # Format: x, t, u
            x_vals = all_data[:, 0]
            t_vals = all_data[:, 1] 
            u_vals = all_data[:, 2]
            
            # Get unique coordinates
            x_vec = np.unique(x_vals)
            t_vec = np.unique(t_vals)
            
            # Create grid
            U_exact = np.zeros((len(x_vec), len(t_vec)))
            for i, (x, t, u) in enumerate(all_data):
                x_idx = np.argmin(np.abs(x_vec - x))
                t_idx = np.argmin(np.abs(t_vec - t))
                U_exact[x_idx, t_idx] = u
        else:
            raise ValueError(f"Unexpected data format: {all_data.shape[1]} columns")
    
    print(f"Parsed data shape: {U_exact.shape}")
    print(f"x range: [{x_vec.min():.3f}, {x_vec.max():.3f}]")
    print(f"t range: [{t_vec.min():.3f}, {t_vec.max():.3f}]")
    
    return x_vec, t_vec, U_exact


def parse_2d_data_file(data_file):
    """
    Parse 2D spatial + time data file format (e.g., burgers2d_*.dat, heat_*.dat, grayscott.dat)
    
    Returns:
        x_vec: 1D array of x coordinates
        y_vec: 1D array of y coordinates
        t_vec: 1D array of time coordinates
        U_exact: 4D array of shape (n_x, n_y, n_t, n_vars) - solution values
        var_names: List of variable names (e.g., ['u', 'v'])
    """
    print(f"Parsing 2D data file: {data_file}")
    
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # Find header with time information
    header_line = None
    data_start_line = None
    
    for i, line in enumerate(lines):
        if ('X' in line and 'Y' in line and '@' in line) or ('t=' in line and 'X' in line):
            header_line = line
            data_start_line = i + 1
            break
        elif line.startswith('% Description:'):
            # Look for header in next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if 'X' in lines[j] and 'Y' in lines[j] and '@' in lines[j]:
                    header_line = lines[j]
                    data_start_line = j + 1
                    break
    
    if header_line is None:
        # Try grayscott format (no header, just data)
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and not line.startswith('#'):
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 4:  # x, y, t, u (possibly v)
                        data_lines.append(values)
                except ValueError:
                    continue
        
        if len(data_lines) > 0:
            all_data = np.array(data_lines)
            x_vals = all_data[:, 0]
            y_vals = all_data[:, 1]
            t_vals = all_data[:, 2]
            
            x_vec = np.unique(x_vals)
            y_vec = np.unique(y_vals)
            t_vec = np.unique(t_vals)
            
            n_vars = all_data.shape[1] - 3  # Subtract x, y, t columns
            var_names = ['u', 'v'][:n_vars]
            
            U_exact = np.zeros((len(x_vec), len(y_vec), len(t_vec), n_vars))
            
            for row in all_data:
                x, y, t = row[0], row[1], row[2]
                x_idx = np.argmin(np.abs(x_vec - x))
                y_idx = np.argmin(np.abs(y_vec - y))
                t_idx = np.argmin(np.abs(t_vec - t))
                
                for var_idx in range(n_vars):
                    U_exact[x_idx, y_idx, t_idx, var_idx] = row[3 + var_idx]
            
            print(f"Parsed grayscott-style data shape: {U_exact.shape}")
            return x_vec, y_vec, t_vec, U_exact, var_names
    
    # Parse COMSOL format header
    if header_line is not None:
        # Extract time values from header
        time_matches = re.findall(r't=([\d.]+)', header_line)
        t_vec = np.array([float(t) for t in time_matches])
        
        # Determine number of variables from header
        if ' u ' in header_line and ' v ' in header_line:
            var_names = ['u', 'v']
        elif ' u ' in header_line:
            var_names = ['u']
        else:
            var_names = ['u']  # Default
        
        # Read data lines
        data_lines = []
        for line in lines[data_start_line:]:
            line = line.strip()
            if line and not line.startswith('%'):
                data_lines.append(line)
        
        # Parse data
        all_coords = []
        all_values = []
        
        for line in data_lines:
            values = [float(x) for x in line.split()]
            x, y = values[0], values[1]
            all_coords.append([x, y])
            all_values.append(values[2:])  # Skip x, y coordinates
        
        coords = np.array(all_coords)
        values = np.array(all_values)
        
        # Get unique coordinates
        x_vec = np.unique(coords[:, 0])
        y_vec = np.unique(coords[:, 1])
        
        n_vars = len(var_names)
        U_exact = np.zeros((len(x_vec), len(y_vec), len(t_vec), n_vars))
        
        # Fill the data array
        for i, (coord, val_row) in enumerate(zip(coords, values)):
            x, y = coord
            x_idx = np.argmin(np.abs(x_vec - x))
            y_idx = np.argmin(np.abs(y_vec - y))
            
            # Each variable has values at all time points
            for t_idx in range(len(t_vec)):
                for var_idx in range(n_vars):
                    col_idx = var_idx * len(t_vec) + t_idx
                    if col_idx < len(val_row):
                        U_exact[x_idx, y_idx, t_idx, var_idx] = val_row[col_idx]
    
    print(f"Parsed 2D data shape: {U_exact.shape}")
    print(f"x range: [{x_vec.min():.3f}, {x_vec.max():.3f}]")
    print(f"y range: [{y_vec.min():.3f}, {y_vec.max():.3f}]")
    print(f"t range: [{t_vec.min():.3f}, {t_vec.max():.3f}]")
    print(f"Variables: {var_names}")
    
    return x_vec, y_vec, t_vec, U_exact, var_names


def generate_1d_visualization(model, exp_name, device='cpu', data_file=None):
    """
    Generate visualization for 1D problems (Wave1D, Kuramoto-Sivashinsky)
    """
    # Create results directory
    results_dir = f"runs/{exp_name}/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Parse reference data
        x_exact_vec, t_exact_vec, U_exact = parse_1d_data_file(data_file)
        
    except Exception as e:
        print(f"Error parsing reference solution file: {e}")
        return {"error": f"File parsing error: {e}"}
    
    # Generate model predictions
    if hasattr(model, 'net'):
        model.net.eval() if hasattr(model.net, 'eval') else None
    
    # Create coordinate grid
    T_grid, X_grid = np.meshgrid(t_exact_vec, x_exact_vec)
    
    # Get model predictions
    if TORCH_AVAILABLE and torch is not None:
        with torch.no_grad():
            x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            inputs = torch.cat([x_flat, t_flat], dim=1)
            u_pred_flat = model.predict(inputs.cpu().numpy())
            U_pred = u_pred_flat.reshape(X_grid.shape)
    else:
        x_flat = X_grid.flatten().reshape(-1, 1)
        t_flat = T_grid.flatten().reshape(-1, 1)
        inputs = np.hstack([x_flat, t_flat])
        u_pred_flat = model.predict(inputs)
        U_pred = u_pred_flat.reshape(X_grid.shape)
    
    # Calculate error
    Error = np.abs(U_pred - U_exact)
    l2_error = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
    print(f'L2 Relative Error: {l2_error:.6f}')
    
    # Generate heatmaps
    fig_hm = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3)
    extent = [t_exact_vec.min(), t_exact_vec.max(), x_exact_vec.min(), x_exact_vec.max()]
    
    # Model prediction
    ax1 = fig_hm.add_subplot(gs[0, 0])
    c1 = ax1.imshow(U_pred.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x') 
    ax1.set_title('Model Prediction')
    fig_hm.colorbar(c1, ax=ax1)
    
    # Reference solution
    ax2 = fig_hm.add_subplot(gs[0, 1])
    c2 = ax2.imshow(U_exact.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_title('Reference Solution')
    fig_hm.colorbar(c2, ax=ax2)
    
    # Error
    ax3 = fig_hm.add_subplot(gs[0, 2])
    c3 = ax3.imshow(Error.T, origin='lower', extent=extent, aspect='auto', cmap='plasma', vmin=0)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_title('Absolute Error')
    fig_hm.colorbar(c3, ax=ax3)
    
    fig_hm.suptitle(f"{exp_name} | L2 Error: {l2_error:.4f}", fontsize=16)
    
    # Save heatmap
    heatmap_path = os.path.join(results_dir, 'heatmap_comparison.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close(fig_hm)
    
    # Generate time slice plots
    fig_slices = plt.figure(figsize=(10, 6))
    n_slices = min(4, len(t_exact_vec))
    t_indices = np.linspace(0, len(t_exact_vec)-1, n_slices, dtype=int)
    
    for i, t_idx in enumerate(t_indices):
        t_val = t_exact_vec[t_idx]
        plt.plot(x_exact_vec, U_exact[:, t_idx], 'b-', alpha=0.8,
                label='Reference' if i == 0 else None)
        plt.plot(x_exact_vec, U_pred[:, t_idx], 'r--', alpha=0.8,
                label='Model' if i == 0 else None)
    
    plt.legend()
    plt.title('Solution at Different Time Slices')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.grid(True, alpha=0.3)
    
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


def generate_2d_scalar_visualization(model, exp_name, device='cpu', data_file=None):
    """
    Generate visualization for 2D scalar problems (Heat2D, Heat2D_multiscale)
    """
    results_dir = f"runs/{exp_name}/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        x_vec, y_vec, t_vec, U_exact, var_names = parse_2d_data_file(data_file)
        # For scalar problems, we use only the first variable
        U_exact_scalar = U_exact[:, :, :, 0]  # Shape: (n_x, n_y, n_t)
        
    except Exception as e:
        print(f"Error parsing reference solution file: {e}")
        return {"error": f"File parsing error: {e}"}
    
    # Generate model predictions
    if hasattr(model, 'net'):
        model.net.eval() if hasattr(model.net, 'eval') else None
    
    # Create coordinate grid for a few time snapshots
    n_time_snapshots = min(4, len(t_vec))
    t_indices = np.linspace(0, len(t_vec)-1, n_time_snapshots, dtype=int)
    
    l2_errors = []
    
    # Generate comparison plots for each time snapshot
    fig, axes = plt.subplots(3, n_time_snapshots, figsize=(4*n_time_snapshots, 12))
    if n_time_snapshots == 1:
        axes = axes.reshape(-1, 1)
    
    for i, t_idx in enumerate(t_indices):
        t_val = t_vec[t_idx]
        
        # Create spatial grid at this time
        Y_grid, X_grid = np.meshgrid(y_vec, x_vec)
        T_grid = np.full_like(X_grid, t_val)
        
        # Get model predictions
        if TORCH_AVAILABLE and torch is not None:
            with torch.no_grad():
                x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                y_flat = torch.tensor(Y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
                u_pred_flat = model.predict(inputs.cpu().numpy())
                U_pred = u_pred_flat.reshape(X_grid.shape)
        else:
            x_flat = X_grid.flatten().reshape(-1, 1)
            y_flat = Y_grid.flatten().reshape(-1, 1)
            t_flat = T_grid.flatten().reshape(-1, 1)
            inputs = np.hstack([x_flat, y_flat, t_flat])
            u_pred_flat = model.predict(inputs)
            U_pred = u_pred_flat.reshape(X_grid.shape)
        
        U_exact_slice = U_exact_scalar[:, :, t_idx]
        Error = np.abs(U_pred - U_exact_slice)
        l2_error = np.linalg.norm(U_pred - U_exact_slice) / np.linalg.norm(U_exact_slice)
        l2_errors.append(l2_error)
        
        extent = [y_vec.min(), y_vec.max(), x_vec.min(), x_vec.max()]
        
        # Model prediction
        im1 = axes[0, i].imshow(U_pred, origin='lower', extent=extent, cmap='viridis')
        axes[0, i].set_title(f'Model t={t_val:.2f}')
        axes[0, i].set_xlabel('y')
        axes[0, i].set_ylabel('x')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Reference solution
        im2 = axes[1, i].imshow(U_exact_slice, origin='lower', extent=extent, cmap='viridis')
        axes[1, i].set_title(f'Reference t={t_val:.2f}')
        axes[1, i].set_xlabel('y')
        axes[1, i].set_ylabel('x')
        plt.colorbar(im2, ax=axes[1, i])
        
        # Error
        im3 = axes[2, i].imshow(Error, origin='lower', extent=extent, cmap='plasma', vmin=0)
        axes[2, i].set_title(f'Error t={t_val:.2f}')
        axes[2, i].set_xlabel('y')
        axes[2, i].set_ylabel('x')
        plt.colorbar(im3, ax=axes[2, i])
    
    avg_l2_error = np.mean(l2_errors)
    fig.suptitle(f"{exp_name} | Avg L2 Error: {avg_l2_error:.4f}", fontsize=16)
    
    heatmap_path = os.path.join(results_dir, 'spatial_comparison.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Spatial comparison saved to: {heatmap_path}")
    plt.close(fig)
    
    return {
        "l2_error": avg_l2_error,
        "heatmap_path": heatmap_path,
        "results_dir": results_dir
    }


def generate_2d_vector_visualization(model, exp_name, device='cpu', data_file=None):
    """
    Generate visualization for 2D vector problems (Burgers2D, GrayScott, NS2D)
    """
    results_dir = f"runs/{exp_name}/visualizations"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        x_vec, y_vec, t_vec, U_exact, var_names = parse_2d_data_file(data_file)
        
    except Exception as e:
        print(f"Error parsing reference solution file: {e}")
        return {"error": f"File parsing error: {e}"}
    
    # Generate model predictions for a few time snapshots
    if hasattr(model, 'net'):
        model.net.eval() if hasattr(model.net, 'eval') else None
    
    n_time_snapshots = min(3, len(t_vec))
    t_indices = np.linspace(0, len(t_vec)-1, n_time_snapshots, dtype=int)
    
    n_vars = len(var_names)
    l2_errors = []
    
    # Create figure with subplots for each variable and time
    fig, axes = plt.subplots(3*n_vars, n_time_snapshots, figsize=(5*n_time_snapshots, 4*3*n_vars))
    if n_time_snapshots == 1:
        axes = axes.reshape(-1, 1)
    
    for i, t_idx in enumerate(t_indices):
        t_val = t_vec[t_idx]
        
        # Create spatial grid
        Y_grid, X_grid = np.meshgrid(y_vec, x_vec)
        T_grid = np.full_like(X_grid, t_val)
        
        # Get model predictions
        if TORCH_AVAILABLE and torch is not None:
            with torch.no_grad():
                x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                y_flat = torch.tensor(Y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
                inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
                pred_flat = model.predict(inputs.cpu().numpy())
                if pred_flat.shape[1] != n_vars:
                    # Handle single output case
                    U_pred = pred_flat.reshape(X_grid.shape)
                    U_pred = np.stack([U_pred] * n_vars, axis=-1)
                else:
                    U_pred = pred_flat.reshape(X_grid.shape[0], X_grid.shape[1], n_vars)
        else:
            x_flat = X_grid.flatten().reshape(-1, 1)
            y_flat = Y_grid.flatten().reshape(-1, 1)
            t_flat = T_grid.flatten().reshape(-1, 1)
            inputs = np.hstack([x_flat, y_flat, t_flat])
            pred_flat = model.predict(inputs)
            if pred_flat.shape[1] != n_vars:
                U_pred = pred_flat.reshape(X_grid.shape)
                U_pred = np.stack([U_pred] * n_vars, axis=-1)
            else:
                U_pred = pred_flat.reshape(X_grid.shape[0], X_grid.shape[1], n_vars)
        
        extent = [y_vec.min(), y_vec.max(), x_vec.min(), x_vec.max()]
        
        # Plot each variable
        for var_idx, var_name in enumerate(var_names):
            row_base = var_idx * 3
            
            U_exact_slice = U_exact[:, :, t_idx, var_idx]
            U_pred_slice = U_pred[:, :, var_idx]
            Error = np.abs(U_pred_slice - U_exact_slice)
            
            l2_error = np.linalg.norm(U_pred_slice - U_exact_slice) / np.linalg.norm(U_exact_slice)
            l2_errors.append(l2_error)
            
            # Model prediction
            im1 = axes[row_base, i].imshow(U_pred_slice, origin='lower', extent=extent, cmap='viridis')
            axes[row_base, i].set_title(f'Model {var_name} t={t_val:.2f}')
            axes[row_base, i].set_xlabel('y')
            axes[row_base, i].set_ylabel('x')
            plt.colorbar(im1, ax=axes[row_base, i])
            
            # Reference solution
            im2 = axes[row_base+1, i].imshow(U_exact_slice, origin='lower', extent=extent, cmap='viridis')
            axes[row_base+1, i].set_title(f'Reference {var_name} t={t_val:.2f}')
            axes[row_base+1, i].set_xlabel('y')
            axes[row_base+1, i].set_ylabel('x')
            plt.colorbar(im2, ax=axes[row_base+1, i])
            
            # Error
            im3 = axes[row_base+2, i].imshow(Error, origin='lower', extent=extent, cmap='plasma', vmin=0)
            axes[row_base+2, i].set_title(f'Error {var_name} t={t_val:.2f}')
            axes[row_base+2, i].set_xlabel('y')
            axes[row_base+2, i].set_ylabel('x')
            plt.colorbar(im3, ax=axes[row_base+2, i])
    
    avg_l2_error = np.mean(l2_errors)
    fig.suptitle(f"{exp_name} | Avg L2 Error: {avg_l2_error:.4f}", fontsize=16)
    
    heatmap_path = os.path.join(results_dir, 'vector_field_comparison.png')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Vector field comparison saved to: {heatmap_path}")
    plt.close(fig)
    
    return {
        "l2_error": avg_l2_error,
        "heatmap_path": heatmap_path,
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