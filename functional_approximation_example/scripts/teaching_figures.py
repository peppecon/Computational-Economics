#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teaching Figures for Functional Approximation
Demonstrates Chebyshev polynomial approximation with various examples
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
# Figure 1: Chebyshev Nodes Visualization
# ============================================================================
print("Generating Figure 1: Chebyshev Nodes Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Chebyshev Nodes: Distribution and Convergence', fontsize=16, fontweight='bold')

node_counts = [5, 10, 15, 20]
for idx, n in enumerate(node_counts):
    ax = axes[idx // 2, idx % 2]
    cheb_nodes = Chebyshev_Nodes(n).ravel()
    
    # Plot nodes on [-1, 1] interval
    ax.scatter(cheb_nodes, np.zeros_like(cheb_nodes), s=100, c='red', 
               marker='o', zorder=3, label=f'n={n} nodes')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Chebyshev Domain [-1, 1]', fontsize=11)
    ax.set_title(f'Chebyshev Nodes (n={n})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add vertical lines to show clustering
    for node in cheb_nodes:
        ax.axvline(x=node, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('latex/teaching/01_chebyshev_nodes.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_chebyshev_nodes.png")

# ============================================================================
# Figure 2: Approximation Quality vs Number of Nodes
# ============================================================================
print("Generating Figure 2: Approximation Quality vs Number of Nodes...")

def smooth_function(x):
    """Smooth function: exp(x) * sin(2*pi*x)"""
    return np.exp(x) * np.sin(2 * np.pi * x)

x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = smooth_function(x_true)

node_counts = [3, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Approximation Quality: Effect of Number of Nodes', fontsize=16, fontweight='bold')

for idx, n in enumerate(node_counts):
    ax = axes[idx // 2, idx % 2]
    
    # Get Chebyshev approximation
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    # Define residual function
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        SSR = np.sum(residuals**2)
        return SSR
    
    # Optimize
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, smooth_function, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    # Evaluate on fine grid
    T_x_new, _ = Tx_new_points(x_true, n)
    y_approx = gamma_star @ T_x_new
    
    # Calculate error
    error = np.abs(y_true - y_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    
    # Plot
    ax.plot(x_true, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.8)
    ax.plot(x_true, y_approx, 'r--', linewidth=2, label='Approximation', alpha=0.8)
    ax.scatter(grid_x, smooth_function(grid_x), s=120, c='red', 
               marker='o', zorder=5, label='Collocation Points', edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('x', fontsize=16, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=16, fontweight='bold')
    ax.set_title(f'n={n} nodes | Max Error: {max_error:.2e} | RMSE: {rmse:.2e}', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax.legend(fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=1.2)

plt.tight_layout()
plt.savefig('latex/teaching/02_approximation_quality.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_approximation_quality.png")

# ============================================================================
# Figure 3: Smooth vs Non-Smooth Functions
# ============================================================================
print("Generating Figure 3: Smooth vs Non-Smooth Functions...")

def non_smooth_function(x):
    """Non-smooth function: max(0, x-1)"""
    return np.maximum(0, x - 1)

n = 15
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Chebyshev Approximation: Smooth vs Non-Smooth Functions', fontsize=16, fontweight='bold')

# Smooth function
ax1 = axes[0]
y_smooth_true = smooth_function(x_true)
T_x, cheb_nodes_x = Tx(n, n)
grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)

def Residual_Function_smooth(gamma, T_x, func, grid_x):
    target_fun = func(grid_x)
    residuals = target_fun - (gamma @ T_x)
    return np.sum(residuals**2)

gamma_0 = np.ones(n)
q = lambda x: Residual_Function_smooth(x, T_x, smooth_function, grid_x)
res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
gamma_star = res.x

T_x_new, _ = Tx_new_points(x_true, n)
y_smooth_approx = gamma_star @ T_x_new

ax1.plot(x_true, y_smooth_true, 'b-', linewidth=2.5, label='True Function', alpha=0.8)
ax1.plot(x_true, y_smooth_approx, 'r--', linewidth=2, label='Chebyshev Approximation', alpha=0.8)
ax1.scatter(grid_x, smooth_function(grid_x), s=120, c='red', marker='o', 
           zorder=5, label='Collocation Points', edgecolors='black', linewidths=1.5)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('f(x)', fontsize=11)
ax1.set_title('Smooth Function: exp(x) * sin(2πx)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Non-smooth function
ax2 = axes[1]
y_nonsmooth_true = non_smooth_function(x_true)

def Residual_Function_nonsmooth(gamma, T_x, func, grid_x):
    target_fun = func(grid_x)
    residuals = target_fun - (gamma @ T_x)
    return np.sum(residuals**2)

q = lambda x: Residual_Function_nonsmooth(x, T_x, non_smooth_function, grid_x)
res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
gamma_star = res.x

y_nonsmooth_approx = gamma_star @ T_x_new

ax2.plot(x_true, y_nonsmooth_true, 'b-', linewidth=2.5, label='True Function', alpha=0.8)
ax2.plot(x_true, y_nonsmooth_approx, 'r--', linewidth=2, label='Chebyshev Approximation', alpha=0.8)
ax2.scatter(grid_x, non_smooth_function(grid_x), s=120, c='red', marker='o', 
           zorder=5, label='Collocation Points', edgecolors='black', linewidths=1.5)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('f(x)', fontsize=11)
ax2.set_title('Non-Smooth Function: max(0, x-1)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latex/teaching/03_smooth_vs_nonsmooth.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_smooth_vs_nonsmooth.png")

# ============================================================================
# Figure 4: Error Analysis
# ============================================================================
print("Generating Figure 4: Error Analysis...")

node_counts = [3, 5, 7, 10, 15, 20, 25]
max_errors = []
rmse_errors = []

for n in node_counts:
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        return np.sum(residuals**2)
    
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, smooth_function, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    T_x_new, _ = Tx_new_points(x_true, n)
    y_approx = gamma_star @ T_x_new
    
    error = np.abs(y_true - y_approx)
    max_errors.append(np.max(error))
    rmse_errors.append(np.sqrt(np.mean(error**2)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Convergence Analysis: Error vs Number of Nodes', fontsize=16, fontweight='bold')

ax1 = axes[0]
ax1.semilogy(node_counts, max_errors, 'o-', linewidth=2.5, markersize=10, 
            color='blue', label='Max Error')
ax1.set_xlabel('Number of Nodes (n)', fontsize=11)
ax1.set_ylabel('Maximum Absolute Error (log scale)', fontsize=11)
ax1.set_title('Maximum Error Convergence', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=10)

ax2 = axes[1]
ax2.semilogy(node_counts, rmse_errors, 's-', linewidth=2.5, markersize=10, 
            color='red', label='RMSE')
ax2.set_xlabel('Number of Nodes (n)', fontsize=11)
ax2.set_ylabel('Root Mean Squared Error (log scale)', fontsize=11)
ax2.set_title('RMSE Convergence', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('latex/teaching/04_error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_error_analysis.png")

# ============================================================================
# Figure 5: Bivariate Function Approximation with Different Numbers of Points
# ============================================================================
print("Generating Figure 5: Bivariate Function Approximation with Different Numbers of Points...")

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
    # Approximation
    kron_xy, cheb_nodes_x, cheb_nodes_y = Tenser_Product_bv(n, n, n, n)
    x_grid = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    y_grid = Change_Variable_Fromcheb(y_min, y_max, cheb_nodes_y)
    xy, yx = np.meshgrid(x_grid, y_grid)
    grid_xy = np.array((xy.ravel(), yx.ravel())).T
    
    gamma_0 = np.ones(n * n)
    q = lambda x: Residual_Function(x, kron_xy, bivariate_function, grid_xy)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    # Evaluate on fine grid
    kron_xy_new = Tenser_Product_new_points(x_fine, y_fine, n, n)
    Z_approx_vec = gamma_star @ kron_xy_new
    Z_approx = Z_approx_vec.reshape((len(x_fine), len(y_fine)))
    
    # Calculate error
    error = np.abs(Z_true - Z_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    
    # Create separate figure (1x2 layout) with proper margins
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Bivariate Function Approximation: exp(x) · exp(y)\nwith {n}×{n} Chebyshev Points', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    # Approximation plot (left)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_fine, Y_fine, Z_approx, cmap='plasma', alpha=0.85, linewidth=0)
    ax1.scatter(xy, yx, bivariate_function(xy, yx), c='red', s=120, marker='o', 
               edgecolors='black', linewidths=2.5, zorder=5, alpha=0.9)
    ax1.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=10)
    ax1.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Approximation', fontsize=16, fontweight='bold', labelpad=15, rotation=90)
    ax1.tick_params(axis='x', labelsize=15, width=1.5, length=7, pad=5)
    ax1.tick_params(axis='y', labelsize=15, width=1.5, length=7, pad=5)
    ax1.tick_params(axis='z', labelsize=15, width=1.5, length=7, pad=8)
    ax1.set_title(f'Chebyshev Approximation (n={n}×{n})\nMax Error: {max_error:.2e}\nRMSE: {rmse:.2e}', 
                  fontsize=15, fontweight='bold', pad=10)
    ax1.xaxis._axinfo['grid']['linewidth'] = 0.8
    ax1.yaxis._axinfo['grid']['linewidth'] = 0.8
    ax1.zaxis._axinfo['grid']['linewidth'] = 0.8
    
    # Error plot (right)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_fine, Y_fine, error, cmap='Reds', alpha=0.85, linewidth=0)
    ax2.set_xlabel('x', fontsize=16, fontweight='bold', labelpad=10)
    ax2.set_ylabel('y', fontsize=16, fontweight='bold', labelpad=10)
    ax2.set_zlabel('Absolute Error', fontsize=16, fontweight='bold', labelpad=15, rotation=90)
    ax2.tick_params(axis='x', labelsize=15, width=1.5, length=7, pad=5)
    ax2.tick_params(axis='y', labelsize=15, width=1.5, length=7, pad=5)
    ax2.tick_params(axis='z', labelsize=15, width=1.5, length=7, pad=8)
    ax2.set_title(f'Error Distribution (n={n}×{n})', fontsize=15, fontweight='bold', pad=10)
    ax2.xaxis._axinfo['grid']['linewidth'] = 0.8
    ax2.yaxis._axinfo['grid']['linewidth'] = 0.8
    ax2.zaxis._axinfo['grid']['linewidth'] = 0.8
    
    # Adjust spacing: more white space outside, plots closer together, ensure nothing is cut
    # Increased right margin to ensure "Absolute Error" label is visible
    plt.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.15, wspace=0.1)
    plt.savefig(f'latex/teaching/05_bivariate_approximation_{n}points.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 05_bivariate_approximation_{n}points.png")

# ============================================================================
# Figure 6: Chebyshev Polynomials Visualization
# ============================================================================
print("Generating Figure 6: Chebyshev Polynomials Visualization...")

x_cheb = np.linspace(-1, 1, 1000)
p_max = 6

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('First Six Chebyshev Polynomials', fontsize=16, fontweight='bold')
axes = axes.flatten()

for p in range(1, p_max + 1):
    ax = axes[p - 1]
    # Need at least 2 for recursion to work
    T = Chebyshev_Polynomials_Recursion_mv(x_cheb, max(p, 2))
    
    # Plot all polynomials up to order p
    for i in range(p):
        if i == 0:
            ax.plot(x_cheb, T[i, :], linewidth=2.5, label=f'T₀(x) = 1', alpha=0.8)
        elif i == 1:
            ax.plot(x_cheb, T[i, :], linewidth=2.5, label=f'T₁(x) = x', alpha=0.8)
        else:
            ax.plot(x_cheb, T[i, :], linewidth=2, label=f'T{i}(x)', alpha=0.7)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Tₙ(x)', fontsize=11)
    ax.set_title(f'Polynomials up to Order {p-1}', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('latex/teaching/06_chebyshev_polynomials.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_chebyshev_polynomials.png")

# ============================================================================
# Figure 7: Approximation with Different Functions
# ============================================================================
print("Generating Figure 7: Approximation with Different Functions...")

functions = [
    (lambda x: np.exp(x), 'exp(x)', 'Exponential'),
    (lambda x: np.sin(2 * np.pi * x), 'sin(2πx)', 'Sinusoidal'),
    (lambda x: x**3 - 2*x**2 + x, 'x³ - 2x² + x', 'Polynomial'),
    (lambda x: 1 / (1 + x**2), '1/(1+x²)', 'Rational')
]

n = 12
x_min, x_max = -1, 1
x_true = np.linspace(x_min, x_max, 1000)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Chebyshev Approximation: Different Function Types', fontsize=16, fontweight='bold')
axes = axes.flatten()

for idx, (func, func_str, func_name) in enumerate(functions):
    ax = axes[idx]
    
    y_true = func(x_true)
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        return np.sum(residuals**2)
    
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, func, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    T_x_new, _ = Tx_new_points(x_true, n)
    y_approx = gamma_star @ T_x_new
    
    error = np.abs(y_true - y_approx)
    max_error = np.max(error)
    
    ax.plot(x_true, y_true, 'b-', linewidth=2.5, label='True Function', alpha=0.8)
    ax.plot(x_true, y_approx, 'r--', linewidth=2, label='Approximation', alpha=0.8)
    ax.scatter(grid_x, func(grid_x), s=100, c='red', marker='o', 
              zorder=5, label='Collocation Points', edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('f(x)', fontsize=11)
    ax.set_title(f'{func_name}: {func_str}\nMax Error: {max_error:.2e}', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latex/teaching/07_different_functions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_different_functions.png")

# ============================================================================
# Figure 8: Exponential Function Approximation with Different Chebyshev Points
# ============================================================================
print("Generating Figure 8: Exponential Function Approximation with Different Chebyshev Points...")

def exp_function(x):
    """Exponential function: exp(x)"""
    return np.exp(x)

x_min, x_max = -1, 1
x_true = np.linspace(x_min, x_max, 1000)
y_true = exp_function(x_true)

node_counts = [3, 5, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('Exponential Function Approximation: exp(x)\nwith Different Chebyshev Points', 
             fontsize=18, fontweight='bold', y=0.995)

for idx, n in enumerate(node_counts):
    ax = axes[idx]
    
    # Get Chebyshev approximation
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    # Define residual function
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        SSR = np.sum(residuals**2)
        return SSR
    
    # Optimize
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, exp_function, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    # Evaluate on fine grid
    T_x_new, _ = Tx_new_points(x_true, n)
    y_approx = gamma_star @ T_x_new
    
    # Calculate error
    error = np.abs(y_true - y_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    
    # Plot
    ax.plot(x_true, y_true, 'b-', linewidth=3, label='True Function: exp(x)', alpha=0.9)
    ax.plot(x_true, y_approx, 'r--', linewidth=2.5, label='Chebyshev Approximation', alpha=0.8)
    ax.scatter(grid_x, exp_function(grid_x), s=150, c='red', 
               marker='o', zorder=5, label='Chebyshev Points', edgecolors='black', linewidths=2)
    
    ax.set_xlabel('x', fontsize=16, fontweight='bold')
    ax.set_ylabel('f(x)', fontsize=16, fontweight='bold')
    ax.set_title(f'n={n} Chebyshev Points | Max Error: {max_error:.2e}\nRMSE: {rmse:.2e}', 
                 fontsize=15, fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax.legend(fontsize=12, framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.2)

plt.tight_layout(rect=[0, 0, 1, 0.98], pad=1.5)
plt.savefig('latex/teaching/08_exp_approximation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_exp_approximation.png")

# ============================================================================
# Figure 9: Chebyshev vs Polynomial Approximations
# ============================================================================
print("Generating Figure 9: Chebyshev vs Polynomial Approximations (exp(x) and curvy function)...")

def compute_derivative_analytical_curvy(x0, n):
    """
    Compute analytical derivatives for f(x) = exp(x) * sin(2πx)
    Using product rule: (uv)' = u'v + uv'
    """
    exp_x = np.exp(x0)
    sin_2pi_x = np.sin(2 * np.pi * x0)
    cos_2pi_x = np.cos(2 * np.pi * x0)
    two_pi = 2 * np.pi
    two_pi_sq = (2 * np.pi)**2
    two_pi_cub = (2 * np.pi)**3
    two_pi_4 = (2 * np.pi)**4
    two_pi_5 = (2 * np.pi)**5
    
    if n == 1:
        return exp_x * (sin_2pi_x + two_pi * cos_2pi_x)
    elif n == 2:
        return exp_x * (sin_2pi_x + 2*two_pi*cos_2pi_x - two_pi_sq*sin_2pi_x)
    elif n == 3:
        return exp_x * (sin_2pi_x + 3*two_pi*cos_2pi_x - 3*two_pi_sq*sin_2pi_x - two_pi_cub*cos_2pi_x)
    elif n == 4:
        return exp_x * (sin_2pi_x + 4*two_pi*cos_2pi_x - 6*two_pi_sq*sin_2pi_x - 4*two_pi_cub*cos_2pi_x + two_pi_4*sin_2pi_x)
    elif n == 5:
        return exp_x * (sin_2pi_x + 5*two_pi*cos_2pi_x - 10*two_pi_sq*sin_2pi_x - 10*two_pi_cub*cos_2pi_x + 5*two_pi_4*sin_2pi_x + two_pi_5*cos_2pi_x)
    else:
        raise ValueError(f"Analytical derivative order {n} not implemented")

def compute_derivative(func, x0, n=1, dx=1e-6):
    """
    Compute numerical derivative using finite differences
    For the curvy function exp(x)*sin(2πx), uses analytical derivatives to avoid numerical errors
    """
    # Check if this is the curvy function by testing at a known point
    # This is a heuristic approach - in practice you'd pass function metadata
    try:
        test_x = 0.5
        test_val = func(test_x)
        expected = np.exp(test_x) * np.sin(2 * np.pi * test_x)
        # If function matches exp(x)*sin(2πx) at test point, use analytical derivatives
        if abs(test_val - expected) < 1e-10:
            return compute_derivative_analytical_curvy(x0, n)
    except:
        pass  # Fall back to numerical method
    
    # Standard finite difference method for other functions
    if n == 1:
        return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
    elif n == 2:
        return (func(x0 + dx) - 2 * func(x0) + func(x0 - dx)) / (dx**2)
    elif n == 3:
        return (func(x0 + 2*dx) - 2*func(x0 + dx) + 2*func(x0 - dx) - func(x0 - 2*dx)) / (2 * dx**3)
    elif n == 4:
        return (func(x0 + 2*dx) - 4*func(x0 + dx) + 6*func(x0) - 4*func(x0 - dx) + func(x0 - 2*dx)) / (dx**4)
    elif n == 5:
        # For 5th derivative, use a more stable approach with smaller step
        # Use Richardson extrapolation for better accuracy
        h1 = dx * 0.8
        h2 = dx * 1.2
        d1 = (func(x0 + 3*h1) - 5*func(x0 + 2*h1) + 10*func(x0 + h1) - 10*func(x0 - h1) + 5*func(x0 - 2*h1) - func(x0 - 3*h1)) / (2 * h1**5)
        d2 = (func(x0 + 3*h2) - 5*func(x0 + 2*h2) + 10*func(x0 + h2) - 10*func(x0 - h2) + 5*func(x0 - 2*h2) - func(x0 - 3*h2)) / (2 * h2**5)
        # Richardson extrapolation to reduce error
        return (h2**5 * d1 - h1**5 * d2) / (h2**5 - h1**5)
    elif n <= 10:
        # For higher order derivatives (6-10), use analytical derivatives for curvy function
        # or fall back to recursive finite differences
        # For now, use a general finite difference approach
        h = dx * (0.3 if n > 7 else 0.5)
        # Use central differences with more points for higher orders
        if n == 6:
            return (func(x0 + 3*h) - 6*func(x0 + 2*h) + 15*func(x0 + h) - 20*func(x0) + 15*func(x0 - h) - 6*func(x0 - 2*h) + func(x0 - 3*h)) / (h**6)
        elif n == 7:
            return (func(x0 + 4*h) - 7*func(x0 + 3*h) + 21*func(x0 + 2*h) - 35*func(x0 + h) + 35*func(x0 - h) - 21*func(x0 - 2*h) + 7*func(x0 - 3*h) - func(x0 - 4*h)) / (2 * h**7)
        elif n == 8:
            return (func(x0 + 4*h) - 8*func(x0 + 3*h) + 28*func(x0 + 2*h) - 56*func(x0 + h) + 70*func(x0) - 56*func(x0 - h) + 28*func(x0 - 2*h) - 8*func(x0 - 3*h) + func(x0 - 4*h)) / (h**8)
        elif n == 9:
            return (func(x0 + 5*h) - 9*func(x0 + 4*h) + 36*func(x0 + 3*h) - 84*func(x0 + 2*h) + 126*func(x0 + h) - 126*func(x0 - h) + 84*func(x0 - 2*h) - 36*func(x0 - 3*h) + 9*func(x0 - 4*h) - func(x0 - 5*h)) / (2 * h**9)
        elif n == 10:
            return (func(x0 + 5*h) - 10*func(x0 + 4*h) + 45*func(x0 + 3*h) - 120*func(x0 + 2*h) + 210*func(x0 + h) - 252*func(x0) + 210*func(x0 - h) - 120*func(x0 - 2*h) + 45*func(x0 - 3*h) - 10*func(x0 - 4*h) + func(x0 - 5*h)) / (h**10)
    else:
        raise ValueError(f"Derivative order {n} not implemented (max: 10)")

def polynomial_approximation(x, func, x_center, order):
    """
    Compute polynomial (Taylor series) approximation of func around x_center
    up to given order
    """
    result = func(x_center) * np.ones_like(x)
    
    if order >= 1:
        # First order: f'(x0) * (x - x0)
        df1 = compute_derivative(func, x_center, n=1)
        result += df1 * (x - x_center)
    
    if order >= 2:
        # Second order: (1/2!) * f''(x0) * (x - x0)^2
        df2 = compute_derivative(func, x_center, n=2)
        result += (df2 / 2.0) * (x - x_center)**2
    
    if order >= 3:
        # Third order: (1/3!) * f'''(x0) * (x - x0)^3
        df3 = compute_derivative(func, x_center, n=3)
        result += (df3 / 6.0) * (x - x_center)**3
    
    if order >= 4:
        # Fourth order: (1/4!) * f''''(x0) * (x - x0)^4
        df4 = compute_derivative(func, x_center, n=4)
        result += (df4 / 24.0) * (x - x_center)**4
    
    if order >= 5:
        # Fifth order: (1/5!) * f'''''(x0) * (x - x0)^5
        df5 = compute_derivative(func, x_center, n=5)
        result += (df5 / 120.0) * (x - x_center)**5
    
    if order >= 6:
        df6 = compute_derivative(func, x_center, n=6)
        result += (df6 / 720.0) * (x - x_center)**6
    
    if order >= 7:
        df7 = compute_derivative(func, x_center, n=7)
        result += (df7 / 5040.0) * (x - x_center)**7
    
    if order >= 8:
        df8 = compute_derivative(func, x_center, n=8)
        result += (df8 / 40320.0) * (x - x_center)**8
    
    if order >= 9:
        df9 = compute_derivative(func, x_center, n=9)
        result += (df9 / 362880.0) * (x - x_center)**9
    
    if order >= 10:
        df10 = compute_derivative(func, x_center, n=10)
        result += (df10 / 3628800.0) * (x - x_center)**10
    
    return result

# Define functions to approximate
def curvy_function(x):
    """Curvy function: exp(x) * sin(2*pi*x)"""
    return np.exp(x) * np.sin(2 * np.pi * x)

functions_to_compare = [
    (exp_function, 'exp(x)', -1, 1, '09a_chebyshev_vs_polynomial_exp'),
    (curvy_function, 'exp(x) · sin(2πx)', 0, 2, '09b_chebyshev_vs_polynomial_curvy')
]

# Use Chebyshev with n=3 nodes for exp(x) (to match 2nd order polynomial = 3 parameters), 
# and n=3,5,10 for curvy function
n_cheb = 3

for func, func_name, x_min, x_max, filename in functions_to_compare:
    x_true = np.linspace(x_min, x_max, 1000)
    y_true = func(x_true)
    x_center = (x_min + x_max) / 2.0  # Center point for Taylor expansion
    
    # Check if this is the curvy function - use multiple Chebyshev nodes
    is_curvy = (func_name == 'exp(x) · sin(2πx)')
    
    if is_curvy:
        # For curvy function: use n=3, 6, 10
        n_cheb_list = [3, 6, 10]
        y_cheb_dict = {}
        error_cheb_dict = {}
        max_error_cheb_dict = {}
        grid_x_dict = {}
        
        for n in n_cheb_list:
            T_x, cheb_nodes_x = Tx(n, n)
            grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
            
            def Residual_Function(gamma, T_x, func, grid_x):
                target_fun = func(grid_x)
                residuals = target_fun - (gamma @ T_x)
                return np.sum(residuals**2)
            
            gamma_0 = np.ones(n)
            q = lambda x: Residual_Function(x, T_x, func, grid_x)
            res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
            gamma_star = res.x
            
            T_x_new, _ = Tx_new_points(x_true, n)
            y_cheb = gamma_star @ T_x_new
            error_cheb = np.abs(y_true - y_cheb)
            
            y_cheb_dict[n] = y_cheb
            error_cheb_dict[n] = error_cheb
            max_error_cheb_dict[n] = np.max(error_cheb)
            grid_x_dict[n] = grid_x
    else:
        # For exp(x): use n=3 only
        T_x, cheb_nodes_x = Tx(n_cheb, n_cheb)
        grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
        
        def Residual_Function(gamma, T_x, func, grid_x):
            target_fun = func(grid_x)
            residuals = target_fun - (gamma @ T_x)
            return np.sum(residuals**2)
        
        gamma_0 = np.ones(n_cheb)
        q = lambda x: Residual_Function(x, T_x, func, grid_x)
        res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
        gamma_star = res.x
        
        T_x_new, _ = Tx_new_points(x_true, n_cheb)
        y_cheb = gamma_star @ T_x_new
        error_cheb = np.abs(y_true - y_cheb)
        max_error_cheb = np.max(error_cheb)
    
    # Polynomial approximations
    # For curvy function, use analytical derivatives directly
    if is_curvy:
        # Use analytical derivatives for exp(x)*sin(2πx)
        def polynomial_approximation_curvy(x, x_center, order):
            result = func(x_center) * np.ones_like(x)
            if order >= 1:
                df1 = compute_derivative_analytical_curvy(x_center, 1)
                result += df1 * (x - x_center)
            if order >= 2:
                df2 = compute_derivative_analytical_curvy(x_center, 2)
                result += (df2 / 2.0) * (x - x_center)**2
            if order >= 3:
                df3 = compute_derivative_analytical_curvy(x_center, 3)
                result += (df3 / 6.0) * (x - x_center)**3
            if order >= 4:
                df4 = compute_derivative_analytical_curvy(x_center, 4)
                result += (df4 / 24.0) * (x - x_center)**4
            if order >= 5:
                df5 = compute_derivative_analytical_curvy(x_center, 5)
                result += (df5 / 120.0) * (x - x_center)**5
            return result
        
        y_poly_1 = polynomial_approximation_curvy(x_true, x_center, 1)
        y_poly_2 = polynomial_approximation_curvy(x_true, x_center, 2)
        y_poly_3 = polynomial_approximation_curvy(x_true, x_center, 3)
    else:
        y_poly_1 = polynomial_approximation(x_true, func, x_center, 1)
        y_poly_2 = polynomial_approximation(x_true, func, x_center, 2)
        y_poly_3 = polynomial_approximation(x_true, func, x_center, 3)
    
    # Calculate errors and convert to log10 (following Judd and Guu 1997)
    # Add small epsilon to avoid log(0)
    eps = 1e-16
    error_poly1 = np.log10(np.abs(y_true - y_poly_1) + eps)
    error_poly2 = np.log10(np.abs(y_true - y_poly_2) + eps)
    error_poly3 = np.log10(np.abs(y_true - y_poly_3) + eps)
    
    max_error_poly1 = np.max(error_poly1)
    max_error_poly2 = np.max(error_poly2)
    max_error_poly3 = np.max(error_poly3)
    
    # Create separate figure for this function
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Chebyshev vs Polynomial Approximations: {func_name}', 
                 fontsize=18, fontweight='bold')
    
    # Plot function approximations (left subplot)
    ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
    
    if is_curvy:
        # Plot multiple Chebyshev approximations
        ax1.plot(x_true, y_cheb_dict[3], 'r--', linewidth=2.5, label='Chebyshev (n=3)', alpha=0.8)
        ax1.plot(x_true, y_cheb_dict[6], 'r:', linewidth=2.5, label='Chebyshev (n=6)', alpha=0.8)
        ax1.plot(x_true, y_cheb_dict[10], 'r-.', linewidth=2.5, label='Chebyshev (n=10)', alpha=0.8)
        # Show points for n=10 (most nodes)
        ax1.scatter(grid_x_dict[10], func(grid_x_dict[10]), s=100, c='red', marker='o', 
                   zorder=5, label='Chebyshev Points (n=10)', edgecolors='black', linewidths=2, alpha=0.7)
    else:
        ax1.plot(x_true, y_cheb, 'r--', linewidth=3, label=f'Chebyshev (n={n_cheb})', alpha=0.8)
        ax1.scatter(grid_x, func(grid_x), s=140, c='red', marker='o', 
                   zorder=5, label='Chebyshev Points', edgecolors='black', linewidths=2.5, alpha=0.8)
    
    ax1.plot(x_true, y_poly_1, 'b:', linewidth=2.5, label='1st Order Taylor', alpha=0.8)
    ax1.plot(x_true, y_poly_2, 'g:', linewidth=2.5, label='2nd Order Taylor', alpha=0.8)
    ax1.scatter([x_center], [func(x_center)], s=220, c='blue', marker='*', 
               zorder=6, label='Taylor Center', edgecolors='black', linewidths=3, alpha=0.9)
    
    ax1.set_xlabel('x', fontsize=16, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
    ax1.set_title(f'{func_name}: Approximations', fontsize=15, fontweight='bold', pad=15)
    ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax1.legend(fontsize=11, framealpha=0.95, loc='best')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    
    # Plot errors (right subplot) - log10 of absolute errors (following Judd and Guu 1997)
    eps = 1e-16
    if is_curvy:
        error_cheb_log3 = np.log10(error_cheb_dict[3] + eps)
        error_cheb_log6 = np.log10(error_cheb_dict[6] + eps)
        error_cheb_log10 = np.log10(error_cheb_dict[10] + eps)
        ax2.plot(x_true, error_cheb_log3, 'r--', linewidth=2.5, label=f'Chebyshev n=3 (max: {np.max(error_cheb_log3):.2f})', alpha=0.8)
        ax2.plot(x_true, error_cheb_log6, 'r:', linewidth=2.5, label=f'Chebyshev n=6 (max: {np.max(error_cheb_log6):.2f})', alpha=0.8)
        ax2.plot(x_true, error_cheb_log10, 'r-.', linewidth=2.5, label=f'Chebyshev n=10 (max: {np.max(error_cheb_log10):.2f})', alpha=0.8)
    else:
        error_cheb_log = np.log10(error_cheb + eps)
        ax2.plot(x_true, error_cheb_log, 'r-', linewidth=3, label=f'Chebyshev n={n_cheb} (max: {np.max(error_cheb_log):.2f})', alpha=0.8)
    ax2.plot(x_true, error_poly1, 'b:', linewidth=2.5, label=f'1st Order Taylor (max: {max_error_poly1:.2f})', alpha=0.8)
    ax2.plot(x_true, error_poly2, 'g:', linewidth=2.5, label=f'2nd Order Taylor (max: {max_error_poly2:.2f})', alpha=0.8)
    
    ax2.set_xlabel('x', fontsize=16, fontweight='bold')
    ax2.set_ylabel('log(Absolute Error)', fontsize=16, fontweight='bold')
    ax2.set_title(f'{func_name}: Approximation Errors (log scale)', fontsize=15, fontweight='bold', pad=15)
    ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    ax2.legend(fontsize=11, framealpha=0.95, loc='best', ncol=1)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(f'latex/teaching/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}.png")

# ============================================================================
# Figure 10: Curvy Function with 5th Order Polynomial and 5 Chebyshev Nodes
# ============================================================================
print("Generating Figure 10: Curvy Function with 5th Order Polynomial and 5 Chebyshev Nodes...")

func = curvy_function
func_name = 'exp(x) · sin(2πx)'
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = func(x_true)
x_center = (x_min + x_max) / 2.0  # Center point for Taylor expansion

# Chebyshev approximation with n=5
n_cheb = 5
T_x, cheb_nodes_x = Tx(n_cheb, n_cheb)
grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)

def Residual_Function(gamma, T_x, func, grid_x):
    target_fun = func(grid_x)
    residuals = target_fun - (gamma @ T_x)
    return np.sum(residuals**2)

gamma_0 = np.ones(n_cheb)
q = lambda x: Residual_Function(x, T_x, func, grid_x)
res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
gamma_star = res.x

T_x_new, _ = Tx_new_points(x_true, n_cheb)
y_cheb = gamma_star @ T_x_new

# 5th order polynomial approximation - use analytical derivatives
def polynomial_approximation_curvy_5th(x, x_center):
    result = func(x_center) * np.ones_like(x)
    df1 = compute_derivative_analytical_curvy(x_center, 1)
    result += df1 * (x - x_center)
    df2 = compute_derivative_analytical_curvy(x_center, 2)
    result += (df2 / 2.0) * (x - x_center)**2
    df3 = compute_derivative_analytical_curvy(x_center, 3)
    result += (df3 / 6.0) * (x - x_center)**3
    df4 = compute_derivative_analytical_curvy(x_center, 4)
    result += (df4 / 24.0) * (x - x_center)**4
    df5 = compute_derivative_analytical_curvy(x_center, 5)
    result += (df5 / 120.0) * (x - x_center)**5
    return result

y_poly_5 = polynomial_approximation_curvy_5th(x_true, x_center)

# Calculate errors and convert to log10 (following Judd and Guu 1997)
eps = 1e-16
error_cheb_abs = np.abs(y_true - y_cheb)
error_poly5_abs = np.abs(y_true - y_poly_5)
error_cheb = np.log10(error_cheb_abs + eps)  # log10 of absolute error
error_poly5 = np.log10(error_poly5_abs + eps)  # log10 of absolute error

max_error_cheb = np.max(error_cheb)
max_error_poly5 = np.max(error_poly5)
# For display in title, also compute absolute max errors
max_error_cheb_abs = np.max(error_cheb_abs)
max_error_poly5_abs = np.max(error_poly5_abs)
rmse_cheb = np.sqrt(np.mean(error_cheb_abs**2))
rmse_poly5 = np.sqrt(np.mean(error_poly5_abs**2))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Chebyshev vs 5th Order Polynomial: {func_name}', 
             fontsize=18, fontweight='bold')

# Plot function approximations (left subplot)
ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
ax1.plot(x_true, y_cheb, 'r--', linewidth=3, label=f'Chebyshev (n={n_cheb})', alpha=0.8)
ax1.scatter(grid_x, func(grid_x), s=140, c='red', marker='o', 
           zorder=5, label='Chebyshev Points', edgecolors='black', linewidths=2.5, alpha=0.8)
ax1.scatter([x_center], [func(x_center)], s=220, c='blue', marker='*', 
           zorder=6, label='Taylor Center', edgecolors='black', linewidths=3, alpha=0.9)

# Secondary axis for 5th order polynomial
ax1_twin = ax1.twinx()
ax1_twin.plot(x_true, y_poly_5, 'b:', linewidth=2.5, label='5th Order Taylor', alpha=0.8)
ax1_twin.set_ylabel('5th Order Taylor (secondary axis)', fontsize=14, color='b', fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='b', labelsize=13, width=1.5, length=6)

ax1.set_xlabel('x', fontsize=16, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
ax1.set_title(f'{func_name}: Approximations\nChebyshev: Max Error={max_error_cheb_abs:.2e}, RMSE={rmse_cheb:.2e}\n5th Order: Max Error={max_error_poly5_abs:.2e}, RMSE={rmse_poly5:.2e}', 
              fontsize=15, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, framealpha=0.95, loc='best')
ax1.grid(True, alpha=0.3, linewidth=1.5)

# Plot errors (right subplot) - log10 of absolute errors (following Judd and Guu 1997)
ax2.plot(x_true, error_cheb, 'r-', linewidth=3, label=f'Chebyshev n={n_cheb} (max: {max_error_cheb:.2f})', alpha=0.8)
ax2.plot(x_true, error_poly5, 'b:', linewidth=2.5, label=f'5th Order Taylor (max: {max_error_poly5:.2f})', alpha=0.8)

ax2.set_xlabel('x', fontsize=16, fontweight='bold')
ax2.set_ylabel('log₁₀(Absolute Error)', fontsize=16, fontweight='bold')
ax2.set_title(f'{func_name}: Approximation Errors (log₁₀ scale)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax2.legend(fontsize=12, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3, which='both', linewidth=1.5)

plt.tight_layout()
plt.savefig('latex/teaching/10_chebyshev_vs_polynomial_curvy_5th_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 10_chebyshev_vs_polynomial_curvy_5th_order.png")

# ============================================================================
# Figure 11: Polynomial Expansions (Orders 2, 3, 5, 10) vs Chebyshev (n=10)
# ============================================================================
print("Generating Figure 11: Polynomial Expansions (Orders 2, 3, 5) vs Chebyshev (n=10)...")

func = curvy_function
func_name = 'exp(x) · sin(2πx)'
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = func(x_true)
x_center = (x_min + x_max) / 2.0

orders = [2, 3, 5]
colors = ['b', 'g', 'm']
linestyles = ['--', '-.', ':']
y_poly = {}
error_poly = {}
max_error_poly = {}

# Use analytical derivatives for curvy function
def polynomial_approximation_curvy_orders(x, x_center, order):
    result = func(x_center) * np.ones_like(x)
    if order >= 1:
        df1 = compute_derivative_analytical_curvy(x_center, 1)
        result += df1 * (x - x_center)
    if order >= 2:
        df2 = compute_derivative_analytical_curvy(x_center, 2)
        result += (df2 / 2.0) * (x - x_center)**2
    if order >= 3:
        df3 = compute_derivative_analytical_curvy(x_center, 3)
        result += (df3 / 6.0) * (x - x_center)**3
    if order >= 4:
        df4 = compute_derivative_analytical_curvy(x_center, 4)
        result += (df4 / 24.0) * (x - x_center)**4
    if order >= 5:
        df5 = compute_derivative_analytical_curvy(x_center, 5)
        result += (df5 / 120.0) * (x - x_center)**5
    return result

for order, color, ls in zip(orders, colors, linestyles):
    y_poly_order = polynomial_approximation_curvy_orders(x_true, x_center, order)
    y_poly[order] = y_poly_order
    err_abs = np.abs(y_true - y_poly_order)
    eps = 1e-16
    err = np.log10(err_abs + eps)  # log10 of absolute error
    error_poly[order] = err
    max_error_poly[order] = np.max(err)

# Add Chebyshev approximation with n=10 (10th degree polynomial)
n_cheb = 10
T_x, cheb_nodes_x = Tx(n_cheb, n_cheb)
grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)

def Residual_Function(gamma, T_x, func, grid_x):
    target_fun = func(grid_x)
    residuals = target_fun - (gamma @ T_x)
    return np.sum(residuals**2)

gamma_0 = np.ones(n_cheb)
q = lambda x: Residual_Function(x, T_x, func, grid_x)
res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
gamma_star = res.x

T_x_new, _ = Tx_new_points(x_true, n_cheb)
y_cheb = gamma_star @ T_x_new
error_cheb_abs = np.abs(y_true - y_cheb)
eps = 1e-16
error_cheb = np.log10(error_cheb_abs + eps)  # log10 of absolute error
max_error_cheb = np.max(error_cheb)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Taylor Expansions (Orders 2, 3, 5) vs Chebyshev (n=10): exp(x) · sin(2πx)', fontsize=18, fontweight='bold')

ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
for order, color, ls in zip(orders, colors, linestyles):
    ax1.plot(x_true, y_poly[order], color=color, linestyle=ls, linewidth=2.5,
             label=f'{order}{"nd" if order == 2 else "rd" if order == 3 else "th"} Order Taylor')

# Add Chebyshev approximation to left plot
ax1.plot(x_true, y_cheb, 'r-', linewidth=3, label=f'Chebyshev (n={n_cheb})', alpha=0.9)

ax1.scatter([x_center], [func(x_center)], s=220, c='blue', marker='*',
           zorder=6, label='Taylor Center', edgecolors='black', linewidths=3, alpha=0.9)
ax1.set_xlabel('x', fontsize=16, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
ax1.set_title('Approximations (All on Same Scale)', fontsize=15, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax1.legend(fontsize=11, framealpha=0.95, loc='best', ncol=2)
ax1.grid(True, alpha=0.3, linewidth=1.5)

for order, color, ls in zip(orders, colors, linestyles):
    ax2.plot(x_true, error_poly[order], color=color, linestyle=ls, linewidth=2.5,
             label=f'{order}{"nd" if order == 2 else "rd" if order == 3 else "th"} Order Taylor (max: {max_error_poly[order]:.2f})',
             alpha=0.8)

# Add Chebyshev error in red
ax2.semilogy(x_true, error_cheb, 'r-', linewidth=3,
             label=f'Chebyshev (n={n_cheb}) (max: {max_error_cheb:.2e})',
             alpha=0.9)

ax2.set_xlabel('x', fontsize=16, fontweight='bold')
ax2.set_ylabel('log(Absolute Error)', fontsize=16, fontweight='bold')
ax2.set_title('Approximation Errors (log scale)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax2.legend(fontsize=11, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3, which='both', linewidth=1.5)

plt.tight_layout()
plt.savefig('latex/teaching/11_taylor_up_to_5th_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 11_taylor_up_to_5th_order.png")

# ============================================================================
# Figure 12: Chebyshev Approximations for Curvy Function (n=3, 6, 10)
# ============================================================================
print("Generating Figure 12: Chebyshev Approximations for Curvy Function (n=3, 6, 10)...")

func = curvy_function
func_name = 'exp(x) · sin(2πx)'
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = func(x_true)

n_cheb_list = [3, 6, 10]
y_cheb_dict = {}
error_cheb_dict = {}
max_error_cheb_dict = {}
grid_x_dict = {}

for n in n_cheb_list:
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        return np.sum(residuals**2)
    
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, func, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    T_x_new, _ = Tx_new_points(x_true, n)
    y_cheb = gamma_star @ T_x_new
    error_cheb_abs = np.abs(y_true - y_cheb)
    eps = 1e-16
    error_cheb = np.log10(error_cheb_abs + eps)  # log10 of absolute error
    
    y_cheb_dict[n] = y_cheb
    error_cheb_dict[n] = error_cheb
    max_error_cheb_dict[n] = np.max(error_cheb)
    grid_x_dict[n] = grid_x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Chebyshev Approximations: {func_name}', fontsize=18, fontweight='bold')

# Left plot: Function and approximations
ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
colors_cheb = ['r', 'b', 'g']
linestyles_cheb = ['--', '-.', ':']
for idx, n in enumerate(n_cheb_list):
    ax1.plot(x_true, y_cheb_dict[n], color=colors_cheb[idx], linestyle=linestyles_cheb[idx], 
             linewidth=2.5, label=f'Chebyshev (n={n})', alpha=0.8)
    # Show points for n=10 (most nodes)
    if n == 10:
        ax1.scatter(grid_x_dict[n], func(grid_x_dict[n]), s=100, c=colors_cheb[idx], 
                   marker='o', zorder=5, label=f'Chebyshev Points (n={n})', 
                   edgecolors='black', linewidths=2, alpha=0.7)

ax1.set_xlabel('x', fontsize=16, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
ax1.set_title('Function and Chebyshev Approximations', fontsize=15, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax1.legend(fontsize=11, framealpha=0.95, loc='best')
ax1.grid(True, alpha=0.3, linewidth=1.5)

# Right plot: Approximation errors (log10 scale)
for idx, n in enumerate(n_cheb_list):
    ax2.plot(x_true, error_cheb_dict[n], color=colors_cheb[idx], linestyle=linestyles_cheb[idx], 
             linewidth=2.5, label=f'Chebyshev n={n} (max: {max_error_cheb_dict[n]:.2f})', alpha=0.8)

ax2.set_xlabel('x', fontsize=16, fontweight='bold')
ax2.set_ylabel('log(Absolute Error)', fontsize=16, fontweight='bold')
ax2.set_title('Approximation Errors (log scale)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax2.legend(fontsize=11, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3, which='both', linewidth=1.5)

plt.tight_layout()
plt.savefig('latex/teaching/12_chebyshev_curvy_function.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 12_chebyshev_curvy_function.png")

# ============================================================================
# Figure 13: Taylor Polynomial Expansions Only (Diagnostic)
# ============================================================================
print("Generating Figure 13: Taylor Polynomial Expansions Only (Diagnostic)...")

func = curvy_function
func_name = 'exp(x) · sin(2πx)'
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = func(x_true)
x_center = (x_min + x_max) / 2.0  # Center point = 1.0

# Use analytical derivatives for Taylor expansion
def taylor_expansion_curvy(x, x_center, order):
    """Taylor expansion using analytical derivatives"""
    result = func(x_center) * np.ones_like(x)
    if order >= 1:
        df1 = compute_derivative_analytical_curvy(x_center, 1)
        result += df1 * (x - x_center)
    if order >= 2:
        df2 = compute_derivative_analytical_curvy(x_center, 2)
        result += (df2 / 2.0) * (x - x_center)**2
    if order >= 3:
        df3 = compute_derivative_analytical_curvy(x_center, 3)
        result += (df3 / 6.0) * (x - x_center)**3
    if order >= 4:
        df4 = compute_derivative_analytical_curvy(x_center, 4)
        result += (df4 / 24.0) * (x - x_center)**4
    if order >= 5:
        df5 = compute_derivative_analytical_curvy(x_center, 5)
        result += (df5 / 120.0) * (x - x_center)**5
    return result

orders = [1, 2, 3, 4, 5]
colors = ['b', 'g', 'm', 'c', 'y']
linestyles = ['--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
y_poly = {}
error_poly = {}
max_error_poly = {}

eps = 1e-16
dx = x_true[1] - x_true[0]  # spacing for total error calculation
total_error_poly = {}
for order, color, ls in zip(orders, colors, linestyles):
    y_poly_order = taylor_expansion_curvy(x_true, x_center, order)
    y_poly[order] = y_poly_order
    err_abs = np.abs(y_true - y_poly_order)
    err = np.log10(err_abs + eps)  # log10 of absolute error
    error_poly[order] = err
    max_error_poly[order] = np.max(err)
    total_error_poly[order] = np.sum(err_abs) * dx  # total error (integral)

# Add Chebyshev approximations (n=6, 10, 20) - n=6 matches Taylor order 5 (both have 6 parameters)
n_cheb_list = [6, 10, 20]
error_cheb_dict = {}
max_error_cheb_dict = {}
colors_cheb = ['r', 'orange', 'darkred']
linestyles_cheb = ['-', '-.', '--']

for n in n_cheb_list:
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        return np.sum(residuals**2)
    
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, func, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    T_x_new, _ = Tx_new_points(x_true, n)
    y_cheb = gamma_star @ T_x_new
    error_cheb_abs = np.abs(y_true - y_cheb)
    error_cheb = np.log10(error_cheb_abs + eps)  # log10 of absolute error
    # Compute total error (integral of absolute error)
    total_error_cheb = np.sum(error_cheb_abs) * dx
    
    error_cheb_dict[n] = error_cheb
    max_error_cheb_dict[n] = np.max(error_cheb)
    # Store total error for reporting
    if 'total_error_cheb_dict' not in locals():
        total_error_cheb_dict = {}
    total_error_cheb_dict[n] = total_error_cheb

# Print total errors for LaTeX
print(f"  Total errors - Chebyshev: n=6: {total_error_cheb_dict[6]:.4f}, n=10: {total_error_cheb_dict[10]:.4f}, n=20: {total_error_cheb_dict[20]:.4f}")
print(f"  Total errors - Taylor: 1st: {total_error_poly[1]:.4f}, 2nd: {total_error_poly[2]:.4f}, 5th: {total_error_poly[5]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Taylor Polynomial Expansions: {func_name}', fontsize=18, fontweight='bold')

# Left plot: Function and Taylor approximations (all on same axis)
ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
for order, color, ls in zip(orders, colors, linestyles):
    ax1.plot(x_true, y_poly[order], color=color, linestyle=ls, linewidth=2.5,
             label=f'{order}{"st" if order == 1 else "nd" if order == 2 else "rd" if order == 3 else "th"} Order Taylor',
             alpha=0.8)

ax1.scatter([x_center], [func(x_center)], s=220, c='blue', marker='*',
           zorder=6, label='Taylor Center', edgecolors='black', linewidths=3, alpha=0.9)
ax1.set_xlabel('x', fontsize=16, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
ax1.set_title('Taylor Polynomial Approximations', fontsize=15, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax1.legend(fontsize=11, framealpha=0.95, loc='best', ncol=2)
ax1.grid(True, alpha=0.3, linewidth=1.5)

# Right plot: Errors on log10 scale (all on same axis)
for order, color, ls in zip(orders, colors, linestyles):
    ax2.plot(x_true, error_poly[order], color=color, linestyle=ls, linewidth=2.5,
             label=f'{order}{"st" if order == 1 else "nd" if order == 2 else "rd" if order == 3 else "th"} Order (max: {max_error_poly[order]:.2f})',
             alpha=0.8)

# Add Chebyshev errors
for idx, n in enumerate(n_cheb_list):
    ax2.plot(x_true, error_cheb_dict[n], color=colors_cheb[idx], linestyle=linestyles_cheb[idx], 
             linewidth=2.5, label=f'Chebyshev n={n} (max: {max_error_cheb_dict[n]:.2f})', alpha=0.8)

ax2.set_xlabel('x', fontsize=16, fontweight='bold')
ax2.set_ylabel('log(Absolute Error)', fontsize=16, fontweight='bold')
ax2.set_title('Approximation Errors (log scale)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax2.legend(fontsize=11, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig('latex/teaching/13_taylor_expansions_only.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 13_taylor_expansions_only.png")

# ============================================================================
# Figure 14: Chebyshev Approximations Only (n=3, 6, 10, 20)
# ============================================================================
print("Generating Figure 14: Chebyshev Approximations Only (n=3, 6, 10, 20)...")

func = curvy_function
func_name = 'exp(x) · sin(2πx)'
x_min, x_max = 0, 2
x_true = np.linspace(x_min, x_max, 1000)
y_true = func(x_true)

n_cheb_list = [3, 6, 10, 20]
colors_cheb = ['r', 'b', 'g', 'm']
linestyles_cheb = ['--', '-.', ':', '-']
y_cheb_dict = {}
error_cheb_dict = {}
max_error_cheb_dict = {}
grid_x_dict = {}

for n in n_cheb_list:
    T_x, cheb_nodes_x = Tx(n, n)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    def Residual_Function(gamma, T_x, func, grid_x):
        target_fun = func(grid_x)
        residuals = target_fun - (gamma @ T_x)
        return np.sum(residuals**2)
    
    gamma_0 = np.ones(n)
    q = lambda x: Residual_Function(x, T_x, func, grid_x)
    res = minimize(q, gamma_0, method='BFGS', options={'disp': False})
    gamma_star = res.x
    
    T_x_new, _ = Tx_new_points(x_true, n)
    y_cheb = gamma_star @ T_x_new
    error_cheb_abs = np.abs(y_true - y_cheb)
    eps = 1e-16
    error_cheb = np.log10(error_cheb_abs + eps)  # log10 of absolute error
    
    y_cheb_dict[n] = y_cheb
    error_cheb_dict[n] = error_cheb
    max_error_cheb_dict[n] = np.max(error_cheb)
    grid_x_dict[n] = grid_x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'Chebyshev Approximations: {func_name}', fontsize=18, fontweight='bold')

# Left plot: Function and Chebyshev approximations
ax1.plot(x_true, y_true, 'k-', linewidth=3.5, label='True Function', alpha=0.9)
for idx, n in enumerate(n_cheb_list):
    ax1.plot(x_true, y_cheb_dict[n], color=colors_cheb[idx], linestyle=linestyles_cheb[idx], 
             linewidth=2.5, label=f'Chebyshev (n={n})', alpha=0.8)
    # Show points for n=20 (most nodes)
    if n == 20:
        ax1.scatter(grid_x_dict[n], func(grid_x_dict[n]), s=80, c=colors_cheb[idx], 
                   marker='o', zorder=5, label=f'Chebyshev Points (n={n})', 
                   edgecolors='black', linewidths=1.5, alpha=0.7)

ax1.set_xlabel('x', fontsize=16, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=16, fontweight='bold')
ax1.set_title('Chebyshev Polynomial Approximations', fontsize=15, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax1.legend(fontsize=11, framealpha=0.95, loc='best')
ax1.grid(True, alpha=0.3, linewidth=1.5)

# Right plot: Errors on log10 scale (all on same axis)
for idx, n in enumerate(n_cheb_list):
    ax2.plot(x_true, error_cheb_dict[n], color=colors_cheb[idx], linestyle=linestyles_cheb[idx], 
             linewidth=2.5, label=f'Chebyshev n={n} (max: {max_error_cheb_dict[n]:.2f})', alpha=0.8)

ax2.set_xlabel('x', fontsize=16, fontweight='bold')
ax2.set_ylabel('log(Absolute Error)', fontsize=16, fontweight='bold')
ax2.set_title('Approximation Errors (log scale)', fontsize=15, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14, width=1.5, length=6)
ax2.legend(fontsize=11, framealpha=0.95, loc='best')
ax2.grid(True, alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig('latex/teaching/14_chebyshev_only.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 14_chebyshev_only.png")

print("\n" + "="*60)
print("All teaching figures generated successfully!")
print("="*60)
print(f"\nFigures saved in: {os.path.join(script_dir, 'latex/teaching/')}")
print("\nGenerated figures:")
print("  01_chebyshev_nodes.png - Chebyshev nodes distribution")
print("  02_approximation_quality.png - Effect of number of nodes")
print("  03_smooth_vs_nonsmooth.png - Smooth vs non-smooth functions")
print("  04_error_analysis.png - Convergence analysis")
print("  05_bivariate_approximation.png - 2D function approximation with 3, 5, and 10 points per dimension")
print("  06_chebyshev_polynomials.png - Chebyshev polynomials visualization")
print("  07_different_functions.png - Various function types")
print("  08_exp_approximation.png - Exponential function with different Chebyshev points")
print("  09a_chebyshev_vs_polynomial_exp.png - Chebyshev vs polynomial: exp(x)")
print("  09b_chebyshev_vs_polynomial_curvy.png - Chebyshev vs polynomial: curvy function")
print("  10_chebyshev_vs_polynomial_curvy_5th_order.png - Chebyshev (n=5) vs 5th order polynomial: curvy function")
print("  11_taylor_up_to_5th_order.png - Taylor expansions up to 5th order (same scale)")
print("  12_chebyshev_curvy_function.png - Chebyshev approximations (n=3, 6, 10) for curvy function")
print("  13_taylor_expansions_only.png - Taylor polynomial expansions only (diagnostic)")
print("  14_chebyshev_only.png - Chebyshev approximations only (n=3, 6, 10, 20)")

