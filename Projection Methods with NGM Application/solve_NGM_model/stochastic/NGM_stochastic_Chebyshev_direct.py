#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Neoclassical Growth Model with Productivity Shocks
State variables: capital k and productivity z
Uses Chebyshev polynomial projection method
DIRECT APPROXIMATION: Approximates c(k,z) directly without log-exp transformation
"""

import numpy as np
from scipy import optimize as opt
import sys
import os

# Add parent directory to path to import functions_library
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'scripts'))
from functions_library import *

# Graphics imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (9, 6)
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33      # Capital share
δ = 0.025     # Depreciation rate
γ = 1         # Risk aversion (CRRA parameter)
ρ = 0.95      # Persistence of productivity shock
σ = 0.007       # Standard deviation of productivity shock

# Damping parameter for iteration
dampening = 1  # Damping parameter for consumption updates

# ============================================================================
# CHEBYSHEV APPROXIMATION SETUP
# ============================================================================
n_k = 10      # Number of Chebyshev nodes for capital
n_z = 10      # Number of Chebyshev nodes for productivity
p_k = n_k     # Polynomial order for capital
p_z = n_z     # Polynomial order for productivity

# Calculate steady state capital (assuming z=1)
z_ss = 1.0
k_ss = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))

# Capital domain bounds
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

# Productivity domain bounds (in log space, 3 std deviations)
z_low_log = -3 * np.sqrt(σ**2 / (1 - ρ**2))
z_high_log = 3 * np.sqrt(σ**2 / (1 - ρ**2))

# Get Chebyshev nodes in [-1, 1] domain
cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()
cheb_nodes_z_log = Chebyshev_Nodes(n_z).ravel()

# Map Chebyshev nodes to economic domain
k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
z_grid_log = Change_Variable_Fromcheb(z_low_log, z_high_log, cheb_nodes_z_log)
z_grid = np.exp(z_grid_log)  # Convert from log space to level

# Number of coefficients (tensor product)
n_coeffs = p_k * p_z

# Gauss-Hermite quadrature for expectations
n_q = 5  # Number of quadrature nodes
gh_quad = np.polynomial.hermite.hermgauss(n_q)
q_nodes, q_weights = gh_quad

# ============================================================================
# CONSUMPTION POLICY FUNCTION (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, z, gamma, k_low, k_high, z_low_log, z_high_log, p_k, p_z):
    """
    Returns Chebyshev approximation of consumption as a function of (k, z)
    DIRECT APPROXIMATION: c(k,z) = gamma @ T_kz (no log-exp transformation)
    
    Parameters:
    -----------
    k : scalar
        Capital value
    z : scalar
        Productivity value
    gamma : array
        Chebyshev coefficients (length p_k * p_z) - approximates c(k,z) directly
    k_low, k_high : scalars
        Capital domain bounds
    z_low_log, z_high_log : scalars
        Productivity domain bounds (in log space)
    p_k, p_z : int
        Polynomial orders
    
    Returns:
    --------
    c : scalar
        Consumption value (ensured to be positive)
    """
    # Transform to Chebyshev domain [-1, 1]
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    
    # Clip k_cheb to [-1, 1] if it goes beyond bounds
    k_cheb = np.clip(k_cheb, -1.0, 1.0)
    
    # For productivity: transform log(z) to [-1, 1] domain
    z_cheb = Change_Variable_Tocheb(z_low_log, z_high_log, np.log(z))
    
    # Clip z_cheb to [-1, 1] if it goes beyond bounds
    z_cheb = np.clip(z_cheb, -1.0, 1.0)
    
    # Evaluate Chebyshev polynomials
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), p_z)
    
    # Tensor product: kron_kz shape is (p_k * p_z, 1)
    # np.kron orders as: (k0,z0), (k1,z0), ..., (k_{p_k-1},z0), (k0,z1), (k1,z1), ...
    kron_kz = np.kron(T_k, T_z)
    
    # Direct approximation: c(k,z) = gamma @ kron_kz
    # gamma is (p_k * p_z,), kron_kz is (p_k * p_z, 1), so gamma @ kron_kz gives scalar
    c = float(gamma @ kron_kz)
    
    # Ensure consumption is positive (enforce lower bound)
    c = max(c, 1e-10)
    
    return c

# ============================================================================
# COMPUTE EULER ERRORS AND UPDATE CONSUMPTION
# ============================================================================
def compute_euler_errors_and_update(c_values, k_grid, z_grid, k_low, k_high, 
                                     z_low_log, z_high_log, p_k, p_z, gamma):
    """
    Computes Euler errors and returns updated consumption values
    Uses Gauss-Hermite quadrature for expectations over productivity shocks
    
    Parameters:
    -----------
    c_values : array (n_k * n_z,)
        Current consumption values at grid points (flattened)
    k_grid : array
        Capital grid points
    z_grid : array
        Productivity grid points
    k_low, k_high : scalars
        Capital domain bounds
    z_low_log, z_high_log : scalars
        Productivity domain bounds (in log space)
    p_k, p_z : int
        Polynomial orders
    gamma : array
        Current Chebyshev coefficients (used to evaluate c at k_prime, z_prime)
    
    Returns:
    --------
    c_new : array
        Updated consumption values
    euler_errors : array
        Euler errors at each grid point
    """
    n_k_grid = len(k_grid)
    n_z_grid = len(z_grid)
    n = n_k_grid * n_z_grid
    c_new = np.zeros(n)
    euler_errors = np.zeros(n)
    
    idx = 0
    for i_z in range(n_z_grid):
        z = z_grid[i_z]
        for i_k in range(n_k_grid):
            k = k_grid[i_k]
            c = c_values[idx]
            
            # Compute next period capital from resource constraint
            k_prime = (1 - δ) * k + z * k**α - c
            
            # No penalties - just compute consumption even if k_prime is outside bounds
            
            # Compute expectation over productivity shocks using Gauss-Hermite quadrature
            E = 0.0
            for i_q in range(len(q_nodes)):
                # Transform GH node to shock
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]
                # Next period productivity: AR(1) in logs
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                
                # Evaluate consumption at (k_prime, z_prime)
                c_prime = c_cheb(k_prime, z_prime, gamma, k_low, k_high, 
                                 z_low_log, z_high_log, p_k, p_z)
                
                # Return on capital
                R_prime = α * z_prime * k_prime**(α - 1) + (1 - δ)
                
                # Accumulate expectation
                E += q_weights[i_q] * β * c_prime**(-γ) * R_prime
            
            # Normalize expectation
            E = E / np.sqrt(np.pi)
            
            # Euler error: E - c^(-γ) = 0
            euler_error = E - c**(-γ)
            euler_errors[idx] = euler_error
            
            # Update consumption using the Euler equation
            # From Euler: E = c^(-γ), so c = E^(-1/γ)
            if E > 0:
                c_target = E**(-1/γ)
                # Ensure target is reasonable
                c_target = max(c_target, 1e-10)
                c_target = min(c_target, z * k**α + (1-δ)*k)
                
                # Damped update: c_new = (1-λ)*c_old + λ*c_target
                c_new[idx] = (1 - dampening) * c + dampening * c_target
            else:
                c_new[idx] = c  # Keep current if E invalid
            
            # Ensure consumption stays positive and reasonable
            c_new[idx] = max(c_new[idx], 1e-10)
            c_new[idx] = min(c_new[idx], z * k**α + (1-δ)*k)  # Can't consume more than available
            
            idx += 1
    
    return c_new, euler_errors

# ============================================================================
# INVERT CONSUMPTION TO GET GAMMA COEFFICIENTS
# ============================================================================
def invert_consumption_to_gamma(c_values, k_grid, z_grid, k_low, k_high, 
                                 z_low_log, z_high_log, p_k, p_z):
    """
    Inverts consumption values at grid points to get Chebyshev coefficients
    DIRECT APPROXIMATION: c(k_i, z_j) = gamma @ T_kz_ij
    
    Parameters:
    -----------
    c_values : array (n_k * n_z,)
        Consumption values at grid points (flattened)
    k_grid : array
        Capital grid points
    z_grid : array
        Productivity grid points
    k_low, k_high : scalars
        Capital domain bounds
    z_low_log, z_high_log : scalars
        Productivity domain bounds (in log space)
    p_k, p_z : int
        Polynomial orders
    
    Returns:
    --------
    gamma : array
        Chebyshev coefficients
    """
    n_k_grid = len(k_grid)
    n_z_grid = len(z_grid)
    n = n_k_grid * n_z_grid
    
    # Build matrix of Chebyshev polynomials evaluated at grid points
    T_matrix = np.zeros((n, p_k * p_z))
    idx = 0
    for i_z in range(n_z_grid):
        z = z_grid[i_z]
        z_cheb = Change_Variable_Tocheb(z_low_log, z_high_log, np.log(z))
        z_cheb = np.clip(z_cheb, -1.0, 1.0)
        T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), p_z)
        
        for i_k in range(n_k_grid):
            k = k_grid[i_k]
            k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
            k_cheb = np.clip(k_cheb, -1.0, 1.0)
            T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
            
            # Tensor product
            kron_kz = np.kron(T_k, T_z)
            T_matrix[idx, :] = kron_kz.ravel()
            idx += 1
    
    # Solve for gamma: c_values = T_matrix @ gamma
    # Use least squares if n != p_k * p_z, direct solve if n == p_k * p_z
    if n == p_k * p_z:
        gamma = np.linalg.solve(T_matrix, c_values)
    else:
        gamma = np.linalg.lstsq(T_matrix, c_values, rcond=None)[0]
    
    return gamma

# ============================================================================
# SOLVE THE MODEL
# ============================================================================
print("="*80)
print("STOCHASTIC NEOCLASSICAL GROWTH MODEL - CHEBYSHEV PROJECTION (DIRECT)")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}")
print(f"  α = {α}")
print(f"  δ = {δ}")
print(f"  γ = {γ}")
print(f"  ρ = {ρ}")
print(f"  σ = {σ}")
print(f"\nSteady-state capital: k_ss = {k_ss:.6f}")
print(f"Steady-state productivity: z_ss = {z_ss:.6f}")
print(f"Capital domain: [{k_low:.6f}, {k_high:.6f}]")
print(f"Productivity domain (log): [{z_low_log:.6f}, {z_high_log:.6f}]")
print(f"Number of Chebyshev nodes: k={n_k}, z={n_z}")
print(f"Number of coefficients: {n_coeffs}")
print(f"Gauss-Hermite quadrature nodes: {n_q}")
print(f"\nNOTE: Using DIRECT approximation c(k,z) = gamma @ T_kz (no log-exp transformation)")
print(f"NOTE: Clipping k_cheb and z_cheb to [-1, 1] when computing c_cheb")

# ============================================================================
# FIXED-POINT ITERATION ALGORITHM
# ============================================================================
c_ss = z_ss * k_ss**α - δ * k_ss  # Steady-state consumption

# Initialize consumption at grid points (constant at steady state)
# Flatten to match grid ordering: z outer, k inner
c_current = np.full(n_k * n_z, c_ss)

# Initialize gamma to approximate constant function c(k,z) = c_ss
gamma_current = invert_consumption_to_gamma(c_current, k_grid, z_grid, 
                                            k_low, k_high, z_low_log, z_high_log, 
                                            p_k, p_z)

print(f"\nInitial consumption: c = c_ss = {c_ss:.6f} everywhere")
print(f"Initial gamma (first 5): {gamma_current[:5]}")
print(f"\nStarting fixed-point iteration...")
print("="*80)

# Fixed-point iteration parameters
max_iter = 2000
tol = 1e-8

for iter in range(max_iter):
    # Compute Euler errors and update consumption
    c_new, euler_errors = compute_euler_errors_and_update(
        c_current, k_grid, z_grid, k_low, k_high, 
        z_low_log, z_high_log, p_k, p_z, gamma_current)
    
    # Check convergence: max Euler error
    max_error = np.max(np.abs(euler_errors))
    mean_error = np.mean(np.abs(euler_errors))
    
    # Print progress every 10 iterations
    if iter % 10 == 0 or max_error < tol:
        print(f"Iteration {iter:4d}: max |Euler error| = {max_error:.6e}, mean = {mean_error:.6e}")
    
    # Check convergence
    if max_error < tol:
        print(f"\nConverged after {iter} iterations!")
        break
    
    # Update consumption with damping (only update if errors are reasonable)
    if max_error < 1.0:  # Only update if errors aren't too large
        c_current = (1 - dampening) * c_current + dampening * c_new
    else:
        # If errors are too large, use smaller step
        c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
    
    # Invert consumption to get new gamma coefficients
    gamma_current = invert_consumption_to_gamma(c_current, k_grid, z_grid, 
                                                 k_low, k_high, z_low_log, z_high_log, 
                                                 p_k, p_z)
    
    # Recompute consumption from gamma to ensure consistency
    idx = 0
    for i_z in range(n_z):
        z = z_grid[i_z]
        for i_k in range(n_k):
            k = k_grid[i_k]
            c_current[idx] = c_cheb(k, z, gamma_current, k_low, k_high, 
                                    z_low_log, z_high_log, p_k, p_z)
            idx += 1

if iter == max_iter - 1:
    print(f"\nWarning: Reached maximum iterations ({max_iter})")

gamma_opt = gamma_current
print(f"\nFinal max |Euler error|: {max_error:.6e}")
print(f"Final mean |Euler error|: {mean_error:.6e}")
print(f"Optimal coefficients (first 5): {gamma_opt[:5]}")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("\nGenerating plots...")

# Fine grid for plotting
k_grid_fine = np.linspace(k_low, k_high, 200)
z_grid_fine = np.linspace(z_grid.min(), z_grid.max(), 50)

# Evaluate consumption policy function on fine grid
# Plot for different productivity levels
z_plot_values = [z_grid[0], z_grid[n_z//2], z_grid[-1]]  # Low, middle, high z

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Consumption policy function for different z values
ax1 = axes[0, 0]
for z_plot in z_plot_values:
    c_policy = np.zeros_like(k_grid_fine)
    for i in range(len(k_grid_fine)):
        c_policy[i] = c_cheb(k_grid_fine[i], z_plot, gamma_opt, k_low, k_high, 
                            z_low_log, z_high_log, p_k, p_z)
    ax1.plot(k_grid_fine, c_policy, linewidth=2, 
            label=f'z = {z_plot:.3f}')

# Add grid points
for z_plot in z_plot_values:
    c_grid = [c_cheb(k, z_plot, gamma_opt, k_low, k_high, 
                    z_low_log, z_high_log, p_k, p_z) for k in k_grid]
    ax1.scatter(k_grid, c_grid, s=50, marker='o', alpha=0.5)

ax1.axvline(k_ss, color='green', linestyle='--', linewidth=2, 
           label=f'Steady-state (k={k_ss:.3f})')
ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax1.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
ax1.set_title('Consumption Policy Function (Direct Approximation)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Consumption policy function as 3D surface
ax2 = axes[0, 1]
K_fine, Z_fine = np.meshgrid(k_grid_fine[::10], z_grid_fine)  # Subsample for speed
C_fine = np.zeros_like(K_fine)
for i in range(K_fine.shape[0]):
    for j in range(K_fine.shape[1]):
        C_fine[i, j] = c_cheb(K_fine[i, j], Z_fine[i, j], gamma_opt, 
                              k_low, k_high, z_low_log, z_high_log, p_k, p_z)

surf = ax2.contourf(K_fine, Z_fine, C_fine, levels=20, cmap='viridis')
ax2.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax2.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
ax2.set_title('Consumption Policy Function (3D)', fontsize=14, fontweight='bold')
plt.colorbar(surf, ax=ax2, label='Consumption')

# Plot 3: Euler errors at collocation points
ax3 = axes[1, 0]
euler_errors_reshaped = euler_errors.reshape(n_z, n_k)
im = ax3.contourf(k_grid, z_grid, euler_errors_reshaped, levels=20, cmap='RdBu_r')
ax3.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax3.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
ax3.set_title('Euler Errors at Collocation Points', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='Euler Error')

# Plot 4: Consumption vs Capital for steady-state productivity
ax4 = axes[1, 1]
z_ss_plot = z_ss
c_policy_ss = np.zeros_like(k_grid_fine)
for i in range(len(k_grid_fine)):
    c_policy_ss[i] = c_cheb(k_grid_fine[i], z_ss_plot, gamma_opt, k_low, k_high, 
                           z_low_log, z_high_log, p_k, p_z)

ax4.plot(k_grid_fine, c_policy_ss, 'b-', linewidth=2, label='Policy function')
ax4.axvline(k_ss, color='green', linestyle='--', linewidth=2)
ax4.axhline(c_ss, color='green', linestyle='--', linewidth=2, 
           label=f'Steady-state (c={c_ss:.3f})')
ax4.scatter([k_ss], [c_ss], c='green', s=200, marker='*', edgecolors='black', 
           linewidths=2, zorder=10, label='Steady-state point')
ax4.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax4.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
ax4.set_title(f'Policy Function at z={z_ss_plot:.3f}', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = '../NGM_figures/stochastic/stochastic_Chebyshev_results_direct.png'
os.makedirs('../NGM_figures/stochastic', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# ============================================================================
# CREATE 3D SURFACE PLOT OF CONSUMPTION POLICY FUNCTION
# ============================================================================
print("\nGenerating 3D surface plot...")

# Create fine grid for 3D plot
k_grid_3d = np.linspace(k_low, k_high, 50)
z_grid_3d = np.exp(np.linspace(z_low_log, z_high_log, 50))

# Generate meshgrid
K_3d, Z_3d = np.meshgrid(k_grid_3d, z_grid_3d)
C_3d = np.zeros_like(K_3d)

# Evaluate consumption policy function on 3D grid
for i in range(K_3d.shape[0]):
    for j in range(K_3d.shape[1]):
        C_3d[i, j] = c_cheb(K_3d[i, j], Z_3d[i, j], gamma_opt, 
                            k_low, k_high, z_low_log, z_high_log, p_k, p_z)

# Compute capital accumulation: k' = (1-δ)k + z*k^α - c
K_prime_3d = np.zeros_like(K_3d)
for i in range(K_3d.shape[0]):
    for j in range(K_3d.shape[1]):
        k_val = K_3d[i, j]
        z_val = Z_3d[i, j]
        c_val = C_3d[i, j]
        K_prime_3d[i, j] = (1 - δ) * k_val + z_val * k_val**α - c_val

# Create 3D figure with two subplots side by side
fig_3d = plt.figure(figsize=(20, 10))

# Plot 1: Consumption policy function
ax1_3d = fig_3d.add_subplot(121, projection='3d')
surf1_3d = ax1_3d.plot_surface(K_3d, Z_3d, C_3d, cmap='viridis', 
                                alpha=0.9, linewidth=0, antialiased=True)

# Add scatter points for Chebyshev nodes
idx = 0
for i_z in range(n_z):
    z_node = z_grid[i_z]
    for i_k in range(n_k):
        k_node = k_grid[i_k]
        c_node = c_cheb(k_node, z_node, gamma_opt, k_low, k_high, 
                       z_low_log, z_high_log, p_k, p_z)
        ax1_3d.scatter([k_node], [z_node], [c_node], 
                      c='red', s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=10)

# Add steady-state point
c_ss_actual = c_cheb(k_ss, z_ss, gamma_opt, k_low, k_high, 
                     z_low_log, z_high_log, p_k, p_z)
ax1_3d.scatter([k_ss], [z_ss], [c_ss_actual], 
              c='green', s=300, marker='*', edgecolors='black', 
              linewidths=2, zorder=15, label='Steady-state')

# Labels and title
ax1_3d.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax1_3d.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
ax1_3d.set_zlabel('c (Consumption)', fontsize=12, fontweight='bold')
ax1_3d.set_title('Consumption Policy Function', fontsize=14, fontweight='bold')

# Add colorbar
fig_3d.colorbar(surf1_3d, ax=ax1_3d, shrink=0.5, aspect=20, label='Consumption')

# Set viewing angle
ax1_3d.view_init(elev=30, azim=45)

# Plot 2: Capital accumulation function
ax2_3d = fig_3d.add_subplot(122, projection='3d')
surf2_3d = ax2_3d.plot_surface(K_3d, Z_3d, K_prime_3d, cmap='plasma', 
                                alpha=0.9, linewidth=0, antialiased=True)

# Add scatter points for Chebyshev nodes (showing k')
idx = 0
for i_z in range(n_z):
    z_node = z_grid[i_z]
    for i_k in range(n_k):
        k_node = k_grid[i_k]
        c_node = c_cheb(k_node, z_node, gamma_opt, k_low, k_high, 
                       z_low_log, z_high_log, p_k, p_z)
        k_prime_node = (1 - δ) * k_node + z_node * k_node**α - c_node
        ax2_3d.scatter([k_node], [z_node], [k_prime_node], 
                      c='red', s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=10)

# Add steady-state point (where k' = k)
k_prime_ss = (1 - δ) * k_ss + z_ss * k_ss**α - c_ss_actual
ax2_3d.scatter([k_ss], [z_ss], [k_ss], 
              c='green', s=300, marker='*', edgecolors='black', 
              linewidths=2, zorder=15, label='Steady-state')

# Add 45-degree line reference (k' = k plane)
k_line = np.linspace(k_low, k_high, 20)
z_line = np.linspace(z_grid_3d.min(), z_grid_3d.max(), 20)
K_line, Z_line = np.meshgrid(k_line, z_line)
K_prime_line = K_line  # k' = k line
ax2_3d.plot_wireframe(K_line, Z_line, K_prime_line, alpha=0.3, 
                      color='gray', linewidth=0.5, label='k\' = k')

# Labels and title
ax2_3d.set_xlabel('k (Current Capital)', fontsize=12, fontweight='bold')
ax2_3d.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
ax2_3d.set_zlabel("k' (Next Period Capital)", fontsize=12, fontweight='bold')
ax2_3d.set_title('Capital Accumulation Function', fontsize=14, fontweight='bold')

# Add colorbar
fig_3d.colorbar(surf2_3d, ax=ax2_3d, shrink=0.5, aspect=20, label="k'")

# Set viewing angle (same as consumption plot)
ax2_3d.view_init(elev=30, azim=45)

plt.tight_layout()

# Save 3D figure
output_path_3d = '../NGM_figures/stochastic/stochastic_Chebyshev_3d_surface.png'
plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
print(f"✓ Saved 3D plot: {output_path_3d}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Max Euler error: {np.max(np.abs(euler_errors)):.6e}")
print(f"Mean Euler error: {np.mean(np.abs(euler_errors)):.6e}")
print(f"RMS Euler error: {np.sqrt(np.mean(euler_errors**2)):.6e}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  z_ss = {z_ss:.6f}")
print(f"  c_ss (deterministic) = {c_ss:.6f}")
c_ss_policy = c_cheb(k_ss, z_ss, gamma_opt, k_low, k_high, z_low_log, z_high_log, p_k, p_z)
print(f"  Policy function at (k_ss, z_ss): c({k_ss:.6f}, {z_ss:.6f}) = {c_ss_policy:.6f}")
print(f"  Difference: {abs(c_ss_policy - c_ss):.6e}")
print(f"  NOTE: In stochastic model, steady-state consumption may differ from deterministic")
print(f"        due to precautionary savings effects. Policy function uses standard Chebyshev nodes.")

print("\nDone!")

