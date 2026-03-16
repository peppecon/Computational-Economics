#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Bivariate Function Approximation Plots
Generates 3D visualizations for Chebyshev bivariate approximations with default view angles
"""

import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functions_library import *

# Create output directory
os.makedirs('latex/teaching', exist_ok=True)

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

# Set scientific plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'patch.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold'
})

# ============================================================================
# Bivariate Function Approximation with Different Numbers of Points
# ============================================================================
print("Generating 3D Bivariate Function Approximation Plots (with default view angles)...")

def bivariate_function(x, y):
    """Bivariate function: exp(x) * exp(y)"""
    return np.exp(x) * np.exp(y)

x_min, x_max = 0, 1
y_min, y_max = -1, 0

# True function on fine grid
x_fine = np.linspace(x_min, x_max, 100)
y_fine = np.linspace(y_min, y_max, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z_true = bivariate_function(X_fine, Y_fine)

node_counts = [3, 5, 10]

def Approximating_Function(gamma, kron_xy):
    return gamma @ kron_xy

def Residual_Function(gamma, kron_xy, func, grid_xy):
    x = grid_xy[:, 0]
    y = grid_xy[:, 1]
    target_fun = func(x, y)
    residuals = target_fun - Approximating_Function(gamma, kron_xy)
    return np.sum(residuals**2)

# Generate separate figure for each node count
for n in node_counts:
    print(f"  Processing n={n}×{n}...")
    
    # Approximation
    kron_xy, cheb_nodes_x, cheb_nodes_y = Tenser_Product_bv(n, n, n, n)
    x_grid = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    y_grid = Change_Variable_Fromcheb(y_min, y_max, cheb_nodes_y)
    
    # Create grid for collocation points
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_xy = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Optimize coefficients
    gamma_0 = np.ones(n * n)
    q = lambda x: Residual_Function(x, kron_xy, bivariate_function, grid_xy)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    # Evaluate on fine grid
    # Need to create meshgrid properly for evaluation
    # Tenser_Product_new_points expects 1D arrays
    kron_xy_new = Tenser_Product_new_points(x_fine, y_fine, n, n)
    Z_approx_vec = gamma_star @ kron_xy_new
    
    # Reshape properly - need to match the meshgrid shape
    # The order matters: if meshgrid is (y_fine, x_fine), then reshape should match
    Z_approx = Z_approx_vec.reshape((len(y_fine), len(x_fine)))
    
    # Calculate error
    error = np.abs(Z_true - Z_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    
    # Print error statistics
    print(f"    Max Error: {max_error:.6e}, RMSE: {rmse:.6e}")
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'Bivariate Function Approximation: exp(x) · exp(y)\nwith {n}×{n} Chebyshev Points', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Approximation plot (left)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot surface
    surf1 = ax1.plot_surface(X_fine, Y_fine, Z_approx, cmap='plasma', 
                             alpha=0.85, linewidth=0, antialiased=True, shade=True)
    
    # Plot true function values at collocation points
    Z_colloc = bivariate_function(X_grid, Y_grid)
    ax1.scatter(X_grid, Y_grid, Z_colloc, c='red', s=150, marker='o', 
               edgecolors='black', linewidths=2, zorder=10, alpha=1.0, label='Chebyshev Points')
    
    ax1.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=12)
    ax1.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=12)
    ax1.set_zlabel('Approximation', fontsize=16, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='x', labelsize=14, width=1.5, length=6, pad=5)
    ax1.tick_params(axis='y', labelsize=14, width=1.5, length=6, pad=5)
    ax1.tick_params(axis='z', labelsize=14, width=1.5, length=6, pad=8)
    ax1.set_title(f'Chebyshev Approximation (n={n}×{n})\nMax Error: {max_error:.2e}\nRMSE: {rmse:.2e}', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Use default view angle (same as original)
    
    # Add colorbar
    cbar1 = plt.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20, pad=0.1)
    cbar1.set_label('Approximation Value', fontsize=12, fontweight='bold')
    
    # Error plot (right)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot error surface
    surf2 = ax2.plot_surface(X_fine, Y_fine, error, cmap='Reds', 
                             alpha=0.85, linewidth=0, antialiased=True, shade=True)
    
    ax2.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=12)
    ax2.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=12)
    ax2.set_zlabel('Absolute Error', fontsize=16, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='x', labelsize=14, width=1.5, length=6, pad=5)
    ax2.tick_params(axis='y', labelsize=14, width=1.5, length=6, pad=5)
    ax2.tick_params(axis='z', labelsize=14, width=1.5, length=6, pad=8)
    ax2.set_title(f'Error Distribution (n={n}×{n})', fontsize=14, fontweight='bold', pad=15)
    
    # Use default view angle (same as original)
    
    # Add colorbar
    cbar2 = plt.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
    cbar2.set_label('Absolute Error', fontsize=12, fontweight='bold')
    
    # Adjust spacing with better margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.15)
    
    # Save figure
    output_path = f'latex/teaching/05_bivariate_approximation_{n}points.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: 05_bivariate_approximation_{n}points.png")

# ============================================================================
# Additional: True Function vs Approximation Comparison
# ============================================================================
print("\nGenerating True Function vs Approximation Comparison...")

for n in node_counts:
    print(f"  Processing comparison plot for n={n}×{n}...")
    
    # Get approximation (reuse code from above)
    kron_xy, cheb_nodes_x, cheb_nodes_y = Tenser_Product_bv(n, n, n, n)
    x_grid = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    y_grid = Change_Variable_Fromcheb(y_min, y_max, cheb_nodes_y)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_xy = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    gamma_0 = np.ones(n * n)
    q = lambda x: Residual_Function(x, kron_xy, bivariate_function, grid_xy)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    kron_xy_new = Tenser_Product_new_points(x_fine, y_fine, n, n)
    Z_approx_vec = gamma_star @ kron_xy_new
    Z_approx = Z_approx_vec.reshape((len(y_fine), len(x_fine)))
    
    # Create comparison figure
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'True Function vs Chebyshev Approximation: exp(x) · exp(y)\nwith {n}×{n} Chebyshev Points', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # True function (left)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_fine, Y_fine, Z_true, cmap='viridis', 
                             alpha=0.85, linewidth=0, antialiased=True, shade=True)
    ax1.scatter(X_grid, Y_grid, bivariate_function(X_grid, Y_grid), c='red', s=150, 
               marker='o', edgecolors='black', linewidths=2, zorder=10, alpha=1.0)
    ax1.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=12)
    ax1.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=12)
    ax1.set_zlabel('True Function', fontsize=16, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='x', labelsize=14, width=1.5, length=6, pad=5)
    ax1.tick_params(axis='y', labelsize=14, width=1.5, length=6, pad=5)
    ax1.tick_params(axis='z', labelsize=14, width=1.5, length=6, pad=8)
    ax1.set_title('True Function: exp(x) · exp(y)', fontsize=14, fontweight='bold', pad=15)
    # Use default view angle (same as original)
    cbar1 = plt.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20, pad=0.1)
    cbar1.set_label('Function Value', fontsize=12, fontweight='bold')
    
    # Approximation (right)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_fine, Y_fine, Z_approx, cmap='plasma', 
                             alpha=0.85, linewidth=0, antialiased=True, shade=True)
    ax2.scatter(X_grid, Y_grid, bivariate_function(X_grid, Y_grid), c='red', s=150, 
               marker='o', edgecolors='black', linewidths=2, zorder=10, alpha=1.0)
    ax2.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=12)
    ax2.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=12)
    ax2.set_zlabel('Approximation', fontsize=16, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='x', labelsize=14, width=1.5, length=6, pad=5)
    ax2.tick_params(axis='y', labelsize=14, width=1.5, length=6, pad=5)
    ax2.tick_params(axis='z', labelsize=14, width=1.5, length=6, pad=8)
    error = np.abs(Z_true - Z_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    print(f"    Max Error: {max_error:.6e}, RMSE: {rmse:.6e}")
    ax2.set_title(f'Chebyshev Approximation (n={n}×{n})\nMax Error: {max_error:.2e}, RMSE: {rmse:.2e}', 
                  fontsize=14, fontweight='bold', pad=15)
    # Use default view angle (same as original)
    cbar2 = plt.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, pad=0.1)
    cbar2.set_label('Approximation Value', fontsize=12, fontweight='bold')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.15)
    
    output_path = f'latex/teaching/05_bivariate_comparison_{n}points.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓ Saved: 05_bivariate_comparison_{n}points.png")

print("\n" + "="*60)
print("All 3D bivariate plots generated successfully!")
print("="*60)
print(f"\nFigures saved in: {os.path.join(script_dir, 'latex/teaching/')}")
print("\nGenerated figures:")
print("  05_bivariate_approximation_3points.png")
print("  05_bivariate_approximation_5points.png")
print("  05_bivariate_approximation_10points.png")
print("  05_bivariate_comparison_3points.png")
print("  05_bivariate_comparison_5points.png")
print("  05_bivariate_comparison_10points.png")

