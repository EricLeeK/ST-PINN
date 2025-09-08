# ST-PINN Visualization Demo

This directory contains demonstration scripts for the ST-PINN visualization system.

## Overview

The visualization system has been enhanced to support all experiments in the repository:

### Supported Experiment Types

1. **1D Problems** (Wave1D, Kuramoto-Sivashinsky)
   - Heatmap comparisons (model vs reference)
   - Time slice plots showing evolution

2. **2D Scalar Problems** (Heat2D, Heat2D_multiscale)
   - Spatial field snapshots at different times
   - Side-by-side model vs reference comparison

3. **2D Vector Problems** (Burgers2D, GrayScott, NS2D)
   - Multi-variable field visualizations
   - Component-wise error analysis

## Demo Scripts

### `test_visualization_utils.py`
Tests the visualization utilities without requiring trained models:
- Data file parsing for different formats
- Mock model predictions
- Visualization generation for all types

**Usage:**
```bash
cd demo/
python test_visualization_utils.py
```

**Requirements:** 
- Only numpy and matplotlib (built-in parsing)
- No PyTorch or training required

### `run_demo_experiment.py`
Demonstrates the complete workflow:
- Mock experiment setup
- Checkpoint creation and loading
- Visualization generation
- File structure explanation

**Usage:**
```bash
cd demo/
python run_demo_experiment.py
```

## Data File Formats Supported

### 1D Data (COMSOL format)
```
% X    u @ t=0    u @ t=0.1    u @ t=0.2    ...
-1.0   0.0627     0.0476       0.0381       ...
-0.98  0.1253     0.0951       0.0762       ...
```

### 1D Data (Simple format)
```
x    t    u
0.0  0.0  1.0
0.0  0.1  0.98
```

### 2D Data (COMSOL format)
```
% X    Y    u @ t=0  v @ t=0  u @ t=0.1  v @ t=0.1  ...
0.0  0.0  1.389     1.952    0.436      0.917      ...
```

### 2D Data (Simple format)
```
x    y    t    u    v
0.0  0.0  0.0  1.0  1.0
```

## Integration with Experiments

All experiment files now include visualization code that:

1. **Automatically detects** the latest model checkpoint
2. **Loads the trained model** using the same architecture
3. **Generates appropriate visualizations** based on problem type
4. **Saves results** in the `runs/{experiment_name}/visualizations/` directory

### Example Output Structure
```
runs/
├── Burgers1D_Standard_FNN/
│   ├── checkpoints/
│   │   └── model_2000.pt
│   └── visualizations/
│       ├── heatmap_comparison.png
│       └── time_slices.png
├── Heat2D_Multiscale_Fourier/
│   └── visualizations/
│       └── spatial_comparison.png
└── GrayScott_Fourier_ReactionDiffusion/
    └── visualizations/
        └── vector_field_comparison.png
```

## Error Metrics

All visualizations include L2 relative error calculations:
```
L2_error = ||U_pred - U_exact|| / ||U_exact||
```

## Running Real Experiments

To run any experiment with visualization:

```bash
cd /path/to/PINNacle-fork2test
python run_experiment_burgers1d_fourier.py
```

The visualization will be generated automatically after training completes.

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure ref/ directory contains the required .dat files
2. **Import errors**: Run from PINNacle-fork2test directory with proper Python path
3. **Visualization failures**: Check that matplotlib is available and accessible

### Debug Mode

To test visualization without training, use the mock model approach shown in `test_visualization_utils.py`.

## File Dependencies

- `../src/utils/visualization_utils.py` - Core visualization functions
- `../ref/*.dat` - Reference solution data files
- Standard libraries: numpy, matplotlib, os, re