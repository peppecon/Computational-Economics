#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence Study: Stochastic Neoclassical Growth Model with Labor
Tests different grid sizes (n_k = n_z = 3, 5, 10, 20) and tracks:
- Euler errors
- Intratemporal errors
- Computational time
"""

import numpy as np
from scipy import optimize as opt
from scipy.optimize import root
import sys
import os
import time

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
rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33     # Capital share
δ = 0.025    # Depreciation rate
χ = 2.0      # Labor disutility parameter
ν = 1.0      # Frisch elasticity parameter
ρ = 0.95     # Persistence of productivity shock
σ = 0.007    # Standard deviation of productivity shock

# Damping parameter for iteration
dampening = 1.0

# ============================================================================
# POLICY FUNCTIONS (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, z, gamma_c, k_low, k_high, z_low_log, z_high_log, p_k, p_z):
    """Returns Chebyshev approximation of consumption"""
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    k_cheb = np.clip(k_cheb, -1.0, 1.0)
    z_cheb = Change_Variable_Tocheb(z_low_log, z_high_log, np.log(z))
    z_cheb = np.clip(z_cheb, -1.0, 1.0)
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), p_z)
    kron_kz = np.kron(T_k, T_z)
    c = float(gamma_c @ kron_kz)
    c = max(c, 1e-10)
    return c

def l_from_c(k, z, c, χ, α, ν):
    """Computes labor from consumption using intratemporal FOC"""
    if c <= 1e-10:
        return 1e-6
    ℓ = ((1-α) * z * k**α / (χ * c))**(ν / (1 + α*ν))
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    return ℓ

def invert_to_gamma(values, k_grid, z_grid, k_low, k_high, 
                    z_low_log, z_high_log, p_k, p_z):
    """Inverts values at grid points to get Chebyshev coefficients"""
    n_k_grid = len(k_grid)
    n_z_grid = len(z_grid)
    n = n_k_grid * n_z_grid
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
            kron_kz = np.kron(T_k, T_z)
            T_matrix[idx, :] = kron_kz.ravel()
            idx += 1
    if n == p_k * p_z:
        gamma = np.linalg.solve(T_matrix, values)
    else:
        gamma = np.linalg.lstsq(T_matrix, values, rcond=None)[0]
    return gamma

def solve_model(n_k, n_z, max_iter=2000, tol=1e-8, verbose=False):
    """
    Solve the stochastic neoclassical growth model with labor
    Returns: (euler_errors, intratemporal_errors, computation_time, iterations)
    """
    start_time = time.time()
    
    p_k = n_k
    p_z = n_z
    
    # Calculate steady state
    z_ss = 1.0
    def steady_state_system(x):
        k, ℓ = x
        k = max(k, 1e-6)
        ℓ = max(ℓ, 1e-6)
        ℓ = min(ℓ, 1.0)
        c = z_ss * k**α * ℓ**(1-α) - δ * k
        if c <= 1e-10:
            return [1e10, 1e10]
        R = α * z_ss * k**(α-1) * ℓ**(1-α) + (1 - δ)
        euler_err = 1 - β * R
        lhs = χ * ℓ**(1/ν)
        rhs = (1-α) * z_ss * k**α * ℓ**(-α) / c
        intratemporal_err = lhs - rhs
        return [euler_err, intratemporal_err]
    
    k_guess = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
    ℓ_guess = 0.33
    result = root(steady_state_system, [k_guess, ℓ_guess], method='hybr', options={'xtol': 1e-12})
    
    if result.success:
        k_ss, ℓ_ss = result.x[0], result.x[1]
        k_ss = max(k_ss, 1e-6)
        ℓ_ss = max(ℓ_ss, 1e-6)
        ℓ_ss = min(ℓ_ss, 1.0)
    else:
        from scipy.optimize import minimize
        def obj(x):
            errs = steady_state_system(x)
            return errs[0]**2 + errs[1]**2
        result_min = minimize(obj, [k_guess, ℓ_guess], method='BFGS', bounds=[(1e-6, 100), (1e-6, 1.0)])
        k_ss, ℓ_ss = result_min.x[0], result_min.x[1]
    
    c_ss = z_ss * k_ss**α * ℓ_ss**(1-α) - δ * k_ss
    
    # Domain setup
    k_low = 0.5 * k_ss
    k_high = 1.5 * k_ss
    z_low_log = -3 * np.sqrt(σ**2 / (1 - ρ**2))
    z_high_log = 3 * np.sqrt(σ**2 / (1 - ρ**2))
    
    # Chebyshev nodes
    cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()
    cheb_nodes_z_log = Chebyshev_Nodes(n_z).ravel()
    k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
    z_grid_log = Change_Variable_Fromcheb(z_low_log, z_high_log, cheb_nodes_z_log)
    z_grid = np.exp(z_grid_log)
    
    # Gauss-Hermite quadrature
    n_q = 5
    gh_quad = np.polynomial.hermite.hermgauss(n_q)
    q_nodes, q_weights = gh_quad
    
    # Initialize
    c_current = np.full(n_k * n_z, c_ss)
    gamma_c_current = invert_to_gamma(c_current, k_grid, z_grid, 
                                      k_low, k_high, z_low_log, z_high_log, p_k, p_z)
    l_current = np.zeros(n_k * n_z)
    idx = 0
    for i_z in range(n_z):
        z = z_grid[i_z]
        for i_k in range(n_k):
            k = k_grid[i_k]
            l_current[idx] = l_from_c(k, z, c_current[idx], χ, α, ν)
            idx += 1
    
    # Fixed-point iteration
    for iter in range(max_iter):
        # Compute Euler errors and update
        n_k_grid = len(k_grid)
        n_z_grid = len(z_grid)
        n = n_k_grid * n_z_grid
        c_new = np.zeros(n)
        l_new = np.zeros(n)
        euler_errors = np.zeros(n)
        intratemporal_errors = np.zeros(n)
        
        idx = 0
        for i_z in range(n_z_grid):
            z = z_grid[i_z]
            for i_k in range(n_k_grid):
                k = k_grid[i_k]
                c = c_current[idx]
                ℓ = l_from_c(k, z, c, χ, α, ν)
                k_prime = (1 - δ) * k + z * k**α * ℓ**(1-α) - c
                
                E = 0.0
                for i_q in range(len(q_nodes)):
                    e_prime = np.sqrt(2) * σ * q_nodes[i_q]
                    z_prime = np.exp(ρ * np.log(z) + e_prime)
                    c_prime = c_cheb(k_prime, z_prime, gamma_c_current, k_low, k_high, 
                                     z_low_log, z_high_log, p_k, p_z)
                    ℓ_prime = l_from_c(k_prime, z_prime, c_prime, χ, α, ν)
                    R_prime = α * z_prime * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
                    E += q_weights[i_q] * β * (1/c_prime) * R_prime
                
                E = E / np.sqrt(np.pi)
                euler_error = E - (1/c)
                euler_errors[idx] = euler_error
                
                if E > 0:
                    c_target = 1 / E
                    c_target = max(c_target, 1e-10)
                    c_target = min(c_target, z * k**α * ℓ**(1-α) + (1-δ)*k)
                    c_new[idx] = (1 - dampening) * c + dampening * c_target
                else:
                    c_new[idx] = c
                
                c_new[idx] = max(c_new[idx], 1e-10)
                c_new[idx] = min(c_new[idx], z * k**α * ℓ**(1-α) + (1-δ)*k)
                l_new[idx] = l_from_c(k, z, c_new[idx], χ, α, ν)
                
                lhs = χ * l_new[idx]**(1/ν)
                rhs = (1-α) * z * k**α * l_new[idx]**(-α) / c_new[idx]
                intratemporal_error = lhs - rhs
                intratemporal_errors[idx] = intratemporal_error
                
                idx += 1
        
        max_euler_error = np.max(np.abs(euler_errors))
        max_intratemporal_error = np.max(np.abs(intratemporal_errors))
        max_error = max(max_euler_error, max_intratemporal_error)
        
        if verbose and (iter % 50 == 0 or max_error < tol):
            print(f"  Iteration {iter:4d}: max |Euler error| = {max_euler_error:.6e}, "
                  f"max |Intratemporal error| = {max_intratemporal_error:.6e}")
        
        if max_error < tol:
            if verbose:
                print(f"  Converged after {iter} iterations!")
            break
        
        if max_error < 1.0:
            c_current = (1 - dampening) * c_current + dampening * c_new
            l_current = (1 - dampening) * l_current + dampening * l_new
        else:
            c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
            l_current = (1 - dampening * 0.1) * l_current + dampening * 0.1 * l_new
        
        gamma_c_current = invert_to_gamma(c_current, k_grid, z_grid, 
                                          k_low, k_high, z_low_log, z_high_log, p_k, p_z)
        
        idx = 0
        for i_z in range(n_z):
            z = z_grid[i_z]
            for i_k in range(n_k):
                k = k_grid[i_k]
                c_current[idx] = c_cheb(k, z, gamma_c_current, k_low, k_high, 
                                       z_low_log, z_high_log, p_k, p_z)
                l_current[idx] = l_from_c(k, z, c_current[idx], χ, α, ν)
                idx += 1
    
    computation_time = time.time() - start_time
    
    # Also compute final gamma for evaluation
    gamma_c_final = invert_to_gamma(c_current, k_grid, z_grid, 
                                    k_low, k_high, z_low_log, z_high_log, p_k, p_z)
    
    return euler_errors, intratemporal_errors, computation_time, iter + 1, k_grid, z_grid, gamma_c_final, k_low, k_high, z_low_log, z_high_log

# ============================================================================
# CONVERGENCE STUDY
# ============================================================================
print("="*80)
print("CONVERGENCE STUDY: Stochastic Neoclassical Growth Model with Labor")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}, α = {α}, δ = {δ}, χ = {χ}, ν = {ν}")
print(f"  ρ = {ρ}, σ = {σ}")
print(f"\nTesting grid sizes: n_k = n_z = 3, 5, 10, 20")
print("="*80)

grid_sizes = [3, 5, 10, 20]
results = []
euler_errors_by_grid = {}  # Store errors and grids for plotting

for n in grid_sizes:
    print(f"\n{'='*80}")
    print(f"Solving with n_k = n_z = {n}")
    print(f"{'='*80}")
    
    euler_errors, intratemporal_errors, comp_time, iterations, k_grid, z_grid, gamma_c_final, k_low, k_high, z_low_log, z_high_log = solve_model(
        n, n, max_iter=2000, tol=1e-8, verbose=True)
    
    # Store for plotting
    euler_errors_by_grid[n] = {
        'errors': euler_errors,
        'k_grid': k_grid,
        'z_grid': z_grid,
        'gamma_c': gamma_c_final,
        'k_low': k_low,
        'k_high': k_high,
        'z_low_log': z_low_log,
        'z_high_log': z_high_log
    }
    
    max_euler = np.max(np.abs(euler_errors))
    mean_euler = np.mean(np.abs(euler_errors))
    rms_euler = np.sqrt(np.mean(euler_errors**2))
    
    max_intra = np.max(np.abs(intratemporal_errors))
    mean_intra = np.mean(np.abs(intratemporal_errors))
    rms_intra = np.sqrt(np.mean(intratemporal_errors**2))
    
    results.append({
        'n': n,
        'n_coeffs': n * n,
        'max_euler': max_euler,
        'mean_euler': mean_euler,
        'rms_euler': rms_euler,
        'max_intra': max_intra,
        'mean_intra': mean_intra,
        'rms_intra': rms_intra,
        'time': comp_time,
        'iterations': iterations
    })
    
    print(f"\nResults for n = {n}:")
    print(f"  Max |Euler error|: {max_euler:.6e}")
    print(f"  Mean |Euler error|: {mean_euler:.6e}")
    print(f"  RMS Euler error: {rms_euler:.6e}")
    print(f"  Max |Intratemporal error|: {max_intra:.6e}")
    print(f"  Mean |Intratemporal error|: {mean_intra:.6e}")
    print(f"  Computation time: {comp_time:.2f} seconds")
    print(f"  Iterations: {iterations}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'n':<6} {'Coeffs':<8} {'Max|EE|':<12} {'Mean|EE|':<12} {'RMS EE':<12} "
      f"{'Max|IE|':<12} {'Time(s)':<10} {'Iter':<6}")
print("-"*80)

for r in results:
    print(f"{r['n']:<6} {r['n_coeffs']:<8} {r['max_euler']:<12.6e} {r['mean_euler']:<12.6e} "
          f"{r['rms_euler']:<12.6e} {r['max_intra']:<12.6e} {r['time']:<10.2f} {r['iterations']:<6}")

# ============================================================================
# PLOTS
# ============================================================================
print("\nGenerating convergence plots...")
os.makedirs('../NGM_figures/stochastic', exist_ok=True)

# Create figure with subplots for Euler errors at collocation nodes
fig_euler, axes_euler = plt.subplots(2, 2, figsize=(16, 12))
axes_euler = axes_euler.flatten()

for idx, n in enumerate(grid_sizes):
    ax = axes_euler[idx]
    data = euler_errors_by_grid[n]
    errors = data['errors']
    k_grid = data['k_grid']
    z_grid = data['z_grid']
    
    # Reshape errors to match grid structure
    errors_2d = errors.reshape(len(z_grid), len(k_grid))
    
    # Create contour plot
    K_mesh, Z_mesh = np.meshgrid(k_grid, z_grid)
    
    # Use log scale for errors (take absolute value and add small epsilon to avoid log(0))
    log_errors = np.log10(np.abs(errors_2d) + 1e-16)
    
    # Create contour plot
    contour = ax.contourf(K_mesh, Z_mesh, log_errors, levels=20, cmap='viridis')
    
    # Add scatter points for collocation nodes
    idx_flat = 0
    for i_z in range(len(z_grid)):
        for i_k in range(len(k_grid)):
            k_node = k_grid[i_k]
            z_node = z_grid[i_z]
            error_val = np.abs(errors[idx_flat])
            # Color by error magnitude
            ax.scatter([k_node], [z_node], c='red', s=100, marker='o', 
                      edgecolors='black', linewidths=1.5, zorder=10, alpha=0.8)
            idx_flat += 1
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('log10(|Euler Error|)', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('k (Capital)', fontsize=11, fontweight='bold')
    ax.set_ylabel('z (Productivity)', fontsize=11, fontweight='bold')
    ax.set_title(f'Euler Errors at Collocation Nodes (n={n}, {n*n} nodes)', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path_euler = '../NGM_figures/stochastic/stochastic_labor_euler_errors_collocation.png'
plt.savefig(output_path_euler, dpi=300, bbox_inches='tight')
print(f"✓ Saved Euler errors plot: {output_path_euler}")
plt.close()

# Create plot: Euler errors vs capital (k) at steady-state productivity (z=1)
# Similar to the reference image - showing errors along k dimension
fig_euler_k, ax_euler_k = plt.subplots(1, 1, figsize=(12, 8))

# Get steady-state k from first grid (all grids use same steady-state)
data_first = euler_errors_by_grid[grid_sizes[0]]
k_low = data_first['k_low']
k_high = data_first['k_high']
k_ss = (k_low + k_high) / 2  # Approximate steady-state (middle of domain)
z_ss_plot = 1.0

# Fine grid for evaluation
k_fine = np.linspace(0.5 * k_ss, 1.5 * k_ss, 500)
z_ss_plot = 1.0

# Colors and styles for different n
colors = ['blue', 'green', 'orange', 'red']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']

for idx, n in enumerate(grid_sizes):
    data = euler_errors_by_grid[n]
    gamma_c = data['gamma_c']
    k_low = data['k_low']
    k_high = data['k_high']
    z_low_log = data['z_low_log']
    z_high_log = data['z_high_log']
    k_grid = data['k_grid']
    z_grid = data['z_grid']
    
    # Evaluate Euler errors on fine grid at z=z_ss
    errors_fine = []
    for k_val in k_fine:
        # Compute consumption and labor at (k_val, z_ss)
        c_val = c_cheb(k_val, z_ss_plot, gamma_c, k_low, k_high, 
                       z_low_log, z_high_log, n, n)
        ℓ_val = l_from_c(k_val, z_ss_plot, c_val, χ, α, ν)
        
        # Compute next period capital
        k_prime = (1 - δ) * k_val + z_ss_plot * k_val**α * ℓ_val**(1-α) - c_val
        
        # Compute expectation
        E = 0.0
        n_q = 5
        gh_quad = np.polynomial.hermite.hermgauss(n_q)
        q_nodes, q_weights = gh_quad
        
        for i_q in range(len(q_nodes)):
            e_prime = np.sqrt(2) * σ * q_nodes[i_q]
            z_prime = np.exp(ρ * np.log(z_ss_plot) + e_prime)
            c_prime = c_cheb(k_prime, z_prime, gamma_c, k_low, k_high, 
                            z_low_log, z_high_log, n, n)
            ℓ_prime = l_from_c(k_prime, z_prime, c_prime, χ, α, ν)
            R_prime = α * z_prime * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
            E += q_weights[i_q] * β * (1/c_prime) * R_prime
        
        E = E / np.sqrt(np.pi)
        euler_error = abs(E - (1/c_val))
        errors_fine.append(euler_error)
    
    errors_fine = np.array(errors_fine)
    
    # Plot line
    ax_euler_k.semilogy(k_fine, errors_fine, linestyle=linestyles[idx], 
                       color=colors[idx], linewidth=2, 
                       label=f'n = {n}', alpha=0.8)
    
    # Mark collocation nodes (where z = z_ss or closest)
    # Find z_grid point closest to z_ss
    z_idx_ss = np.argmin(np.abs(z_grid - z_ss_plot))
    z_node_ss = z_grid[z_idx_ss]
    
    # Get errors at collocation nodes for this z
    errors_at_nodes = []
    k_nodes_plot = []
    for i_k in range(len(k_grid)):
        node_idx = z_idx_ss * len(k_grid) + i_k
        errors_at_nodes.append(abs(data['errors'][node_idx]))
        k_nodes_plot.append(k_grid[i_k])
    
    # Plot markers at collocation nodes
    ax_euler_k.semilogy(k_nodes_plot, errors_at_nodes, 
                       marker=markers[idx], linestyle='None', 
                       color=colors[idx], markersize=8, 
                       markeredgecolor='black', markeredgewidth=1,
                       label=f'n = {n} (nodes)', alpha=0.9, zorder=10)

# Add steady-state line
ax_euler_k.axvline(k_ss, color='black', linestyle=':', linewidth=2, 
                  alpha=0.6, label='Steady-state')

ax_euler_k.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax_euler_k.set_ylabel('|Euler Error| (log scale)', fontsize=12, fontweight='bold')
ax_euler_k.set_title('Euler Errors vs Capital (at z = 1.0)', fontsize=14, fontweight='bold')
ax_euler_k.legend(loc='best', fontsize=10)
ax_euler_k.grid(True, alpha=0.3)
ax_euler_k.set_ylim([1e-14, 1e-4])

plt.tight_layout()
output_path_euler_k = '../NGM_figures/stochastic/stochastic_labor_euler_errors_vs_k.png'
plt.savefig(output_path_euler_k, dpi=300, bbox_inches='tight')
print(f"✓ Saved Euler errors vs k plot: {output_path_euler_k}")
plt.close()

# ============================================================================
# 3D PLOT: Euler Errors across (k, z) space with collocation nodes
# ============================================================================
print("\nGenerating 3D Euler error plots...")

# Create one 3D plot for each grid size
for n in grid_sizes:
    data = euler_errors_by_grid[n]
    gamma_c = data['gamma_c']
    k_low = data['k_low']
    k_high = data['k_high']
    z_low_log = data['z_low_log']
    z_high_log = data['z_high_log']
    k_grid = data['k_grid']
    z_grid = data['z_grid']
    
    # Fine grid for 3D surface
    k_fine_3d = np.linspace(k_low, k_high, 50)
    z_fine_3d = np.exp(np.linspace(z_low_log, z_high_log, 50))
    K_3d, Z_3d = np.meshgrid(k_fine_3d, z_fine_3d)
    Errors_3d = np.zeros_like(K_3d)
    
    # Evaluate Euler errors on fine grid
    n_q = 5
    gh_quad = np.polynomial.hermite.hermgauss(n_q)
    q_nodes, q_weights = gh_quad
    
    for i in range(K_3d.shape[0]):
        for j in range(K_3d.shape[1]):
            k_val = K_3d[i, j]
            z_val = Z_3d[i, j]
            
            # Compute consumption and labor at (k_val, z_val)
            c_val = c_cheb(k_val, z_val, gamma_c, k_low, k_high, 
                          z_low_log, z_high_log, n, n)
            ℓ_val = l_from_c(k_val, z_val, c_val, χ, α, ν)
            
            # Compute next period capital
            k_prime = (1 - δ) * k_val + z_val * k_val**α * ℓ_val**(1-α) - c_val
            
            # Compute expectation
            E = 0.0
            for i_q in range(len(q_nodes)):
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]
                z_prime = np.exp(ρ * np.log(z_val) + e_prime)
                c_prime = c_cheb(k_prime, z_prime, gamma_c, k_low, k_high, 
                                z_low_log, z_high_log, n, n)
                ℓ_prime = l_from_c(k_prime, z_prime, c_prime, χ, α, ν)
                R_prime = α * z_prime * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
                E += q_weights[i_q] * β * (1/c_prime) * R_prime
            
            E = E / np.sqrt(np.pi)
            euler_error = abs(E - (1/c_val))
            Errors_3d[i, j] = euler_error
    
    # Create 3D figure
    fig_3d = plt.figure(figsize=(14, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot surface (using log scale for better visualization)
    log_errors_3d = np.log10(Errors_3d + 1e-16)
    surf = ax_3d.plot_surface(K_3d, Z_3d, log_errors_3d, 
                              cmap='viridis', alpha=0.7, 
                              linewidth=0, antialiased=True)
    
    # Mark collocation nodes at their exact (k, z) positions
    # Interpolate the surface to get the exact height at each node coordinate
    # This ensures nodes "touch exactly the surface"
    from scipy.interpolate import griddata
    
    k_nodes_3d = []
    z_nodes_3d = []
    
    # Collect all node coordinates
    for i_z in range(len(z_grid)):
        z_node = z_grid[i_z]
        for i_k in range(len(k_grid)):
            k_node = k_grid[i_k]
            k_nodes_3d.append(k_node)
            z_nodes_3d.append(z_node)
    
    k_nodes_array = np.array(k_nodes_3d)
    z_nodes_array = np.array(z_nodes_3d)
    
    # Prepare surface data for interpolation
    # Flatten the surface grid points
    k_surface_flat = K_3d.flatten()
    z_surface_flat = Z_3d.flatten()
    log_errors_surface_flat = log_errors_3d.flatten()
    
    # Interpolate surface values at node coordinates
    # This gives the exact height of the surface at each node
    errors_at_nodes = griddata(
        (k_surface_flat, z_surface_flat), 
        log_errors_surface_flat,
        (k_nodes_array, z_nodes_array),
        method='linear'  # Use linear interpolation to match surface rendering
    )
    
    errors_array = np.array(errors_at_nodes)
    
    # Verify the errors are being used as z-coordinate
    if n == grid_sizes[0]:  # Only print for first grid to avoid clutter
        print(f"\n  Verifying node placement (n={n}):")
        print(f"    k_nodes shape: {k_nodes_array.shape}")
        print(f"    z_nodes shape: {z_nodes_array.shape}")
        print(f"    errors shape: {errors_array.shape}")
        print(f"    Node error range: [{np.min(errors_array):.4f}, {np.max(errors_array):.4f}]")
        print(f"    Surface z range: [{np.min(log_errors_3d):.4f}, {np.max(log_errors_3d):.4f}]")
        for i in range(min(3, len(k_nodes_array))):
            print(f"    Node {i}: (k={k_nodes_array[i]:.4f}, z={z_nodes_array[i]:.4f}, "
                  f"surface_height={errors_array[i]:.4f})")
    
    # Plot nodes at their exact surface heights
    # The nodes will now "touch exactly the surface" since we interpolated the surface values
    ax_3d.scatter(k_nodes_array, z_nodes_array, errors_array,
                 c='red', s=200, marker='o', edgecolors='black', 
                 linewidths=2.5, zorder=10, alpha=1.0, label='Collocation nodes')
    
    # Add steady-state point
    # Calculate steady-state k
    z_ss = 1.0
    def steady_state_system(x):
        k, ℓ = x
        k = max(k, 1e-6)
        ℓ = max(ℓ, 1e-6)
        ℓ = min(ℓ, 1.0)
        c = z_ss * k**α * ℓ**(1-α) - δ * k
        if c <= 1e-10:
            return [1e10, 1e10]
        R = α * z_ss * k**(α-1) * ℓ**(1-α) + (1 - δ)
        euler_err = 1 - β * R
        lhs = χ * ℓ**(1/ν)
        rhs = (1-α) * z_ss * k**α * ℓ**(-α) / c
        intratemporal_err = lhs - rhs
        return [euler_err, intratemporal_err]
    
    k_guess = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
    ℓ_guess = 0.33
    result = root(steady_state_system, [k_guess, ℓ_guess], method='hybr', options={'xtol': 1e-12})
    if result.success:
        k_ss = result.x[0]
    else:
        from scipy.optimize import minimize
        def obj(x):
            errs = steady_state_system(x)
            return errs[0]**2 + errs[1]**2
        result_min = minimize(obj, [k_guess, ℓ_guess], method='BFGS', bounds=[(1e-6, 100), (1e-6, 1.0)])
        k_ss = result_min.x[0]
    
    # Evaluate error at steady-state
    c_ss_val = c_cheb(k_ss, z_ss, gamma_c, k_low, k_high, 
                      z_low_log, z_high_log, n, n)
    ℓ_ss_val = l_from_c(k_ss, z_ss, c_ss_val, χ, α, ν)
    k_prime_ss = (1 - δ) * k_ss + z_ss * k_ss**α * ℓ_ss_val**(1-α) - c_ss_val
    
    E_ss = 0.0
    for i_q in range(len(q_nodes)):
        e_prime = np.sqrt(2) * σ * q_nodes[i_q]
        z_prime = np.exp(ρ * np.log(z_ss) + e_prime)
        c_prime = c_cheb(k_prime_ss, z_prime, gamma_c, k_low, k_high, 
                        z_low_log, z_high_log, n, n)
        ℓ_prime = l_from_c(k_prime_ss, z_prime, c_prime, χ, α, ν)
        R_prime = α * z_prime * k_prime_ss**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
        E_ss += q_weights[i_q] * β * (1/c_prime) * R_prime
    E_ss = E_ss / np.sqrt(np.pi)
    error_ss = abs(E_ss - (1/c_ss_val))
    log_error_ss = np.log10(error_ss + 1e-16)
    
    ax_3d.scatter([k_ss], [z_ss], [log_error_ss],
                 c='green', s=300, marker='*', edgecolors='black', 
                 linewidths=2, zorder=15, label='Steady-state')
    
    # Labels and title
    ax_3d.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
    ax_3d.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
    ax_3d.set_zlabel('log10(|Euler Error|)', fontsize=12, fontweight='bold')
    ax_3d.set_title(f'Euler Errors - 3D Surface (n={n}, {n*n} collocation nodes)', 
                    fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.6, aspect=20, label='log10(|Euler Error|)')
    
    # Set viewing angle
    ax_3d.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    output_path_3d = f'../NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n{n}.png'
    plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 3D Euler errors plot: {output_path_3d}")
    plt.close()

# ============================================================================
# COMBINED 3D PLOT: All 4 surfaces together, each with its own nodes
# ============================================================================
print("\nGenerating combined 3D plot with all 4 surfaces...")

# Create combined 3D figure
fig_combined = plt.figure(figsize=(18, 14))
ax_combined = fig_combined.add_subplot(111, projection='3d')

# Use same colormap for all surfaces, but differentiate by style
common_colormap = 'viridis'  # Same colormap for all surfaces

# Node colors and markers (keep these different)
node_colors = {
    3: 'red',
    5: 'blue', 
    10: 'orange',
    20: 'purple'
}
node_markers = {
    3: 'o',
    5: 's',
    10: '^',
    20: 'D'
}

# Surface styles to differentiate them (same colormap, different visual style)
# Use different linewidths and edgecolors to distinguish surfaces
surface_styles = {
    3: {'alpha': 0.4, 'linewidth': 0.3, 'edgecolor': 'red'},
    5: {'alpha': 0.5, 'linewidth': 0.3, 'edgecolor': 'blue'},
    10: {'alpha': 0.6, 'linewidth': 0.3, 'edgecolor': 'orange'},
    20: {'alpha': 0.7, 'linewidth': 0.3, 'edgecolor': 'purple'}
}

# Find common domain for all surfaces
all_k_lows = [euler_errors_by_grid[n]['k_low'] for n in grid_sizes]
all_k_highs = [euler_errors_by_grid[n]['k_high'] for n in grid_sizes]
all_z_low_logs = [euler_errors_by_grid[n]['z_low_log'] for n in grid_sizes]
all_z_high_logs = [euler_errors_by_grid[n]['z_high_log'] for n in grid_sizes]

k_low_common = min(all_k_lows)
k_high_common = max(all_k_highs)
z_low_log_common = min(all_z_low_logs)
z_high_log_common = max(all_z_high_logs)

# Fine grid for 3D surface (common for all)
k_fine_3d = np.linspace(k_low_common, k_high_common, 50)
z_fine_3d = np.exp(np.linspace(z_low_log_common, z_high_log_common, 50))
K_3d, Z_3d = np.meshgrid(k_fine_3d, z_fine_3d)

# Setup quadrature
n_q = 5
gh_quad = np.polynomial.hermite.hermgauss(n_q)
q_nodes, q_weights = gh_quad

# Store z-ranges for colorbar
all_z_mins = []
all_z_maxs = []

# Plot each surface with its own nodes
for n in grid_sizes:
    data = euler_errors_by_grid[n]
    gamma_c = data['gamma_c']
    k_low = data['k_low']
    k_high = data['k_high']
    z_low_log = data['z_low_log']
    z_high_log = data['z_high_log']
    k_grid = data['k_grid']
    z_grid = data['z_grid']
    
    # Evaluate Euler errors on fine grid for this surface
    Errors_3d = np.zeros_like(K_3d)
    
    for i in range(K_3d.shape[0]):
        for j in range(K_3d.shape[1]):
            k_val = K_3d[i, j]
            z_val = Z_3d[i, j]
            
            # Skip if outside this surface's domain
            if k_val < k_low or k_val > k_high:
                Errors_3d[i, j] = np.nan
                continue
            
            z_val_log = np.log(z_val)
            if z_val_log < z_low_log or z_val_log > z_high_log:
                Errors_3d[i, j] = np.nan
                continue
            
            # Compute consumption and labor at (k_val, z_val)
            c_val = c_cheb(k_val, z_val, gamma_c, k_low, k_high, 
                          z_low_log, z_high_log, n, n)
            ℓ_val = l_from_c(k_val, z_val, c_val, χ, α, ν)
            
            # Compute next period capital
            k_prime = (1 - δ) * k_val + z_val * k_val**α * ℓ_val**(1-α) - c_val
            
            # Compute expectation
            E = 0.0
            for i_q in range(len(q_nodes)):
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]
                z_prime = np.exp(ρ * np.log(z_val) + e_prime)
                c_prime = c_cheb(k_prime, z_prime, gamma_c, k_low, k_high, 
                                z_low_log, z_high_log, n, n)
                ℓ_prime = l_from_c(k_prime, z_prime, c_prime, χ, α, ν)
                R_prime = α * z_prime * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
                E += q_weights[i_q] * β * (1/c_prime) * R_prime
            
            E = E / np.sqrt(np.pi)
            euler_error = abs(E - (1/c_val))
            Errors_3d[i, j] = euler_error
    
    # Plot surface (using log scale for better visualization)
    log_errors_3d = np.log10(Errors_3d + 1e-16)
    
    # Store z-range for colorbar
    valid_errors = log_errors_3d[~np.isnan(log_errors_3d)]
    if len(valid_errors) > 0:
        all_z_mins.append(np.min(valid_errors))
        all_z_maxs.append(np.max(valid_errors))
    
    style = surface_styles[n]
    surf = ax_combined.plot_surface(K_3d, Z_3d, log_errors_3d, 
                                     cmap=common_colormap, 
                                     alpha=style['alpha'], 
                                     linewidth=style['linewidth'],
                                     edgecolors=style['edgecolor'],
                                     antialiased=True, 
                                     label=f'n = {n} surface')
    
    # Prepare surface data for interpolation
    k_surface_flat = K_3d.flatten()
    z_surface_flat = Z_3d.flatten()
    log_errors_surface_flat = log_errors_3d.flatten()
    
    # Collect node coordinates
    k_nodes_3d = []
    z_nodes_3d = []
    for i_z in range(len(z_grid)):
        z_node = z_grid[i_z]
        for i_k in range(len(k_grid)):
            k_node = k_grid[i_k]
            k_nodes_3d.append(k_node)
            z_nodes_3d.append(z_node)
    
    k_nodes_array = np.array(k_nodes_3d)
    z_nodes_array = np.array(z_nodes_3d)
    
    # Interpolate THIS surface's values at node coordinates
    errors_at_nodes = griddata(
        (k_surface_flat, z_surface_flat), 
        log_errors_surface_flat,
        (k_nodes_array, z_nodes_array),
        method='linear'
    )
    
    errors_array = np.array(errors_at_nodes)
    
    # Plot nodes at their exact heights on THIS surface
    ax_combined.scatter(k_nodes_array, z_nodes_array, errors_array,
                        c=node_colors[n], s=200, marker=node_markers[n], 
                        edgecolors='black', linewidths=2.5, zorder=10, 
                        alpha=1.0, label=f'n = {n} ({n*n} nodes)')

# Labels and title
ax_combined.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax_combined.set_ylabel('z (Productivity)', fontsize=12, fontweight='bold')
ax_combined.set_zlabel('log10(|Euler Error|)', fontsize=12, fontweight='bold')
ax_combined.set_title('Euler Errors - All 4 Surfaces Combined\n(Each surface with its own nodes)', 
                     fontsize=14, fontweight='bold')

# Add single colorbar for all surfaces
# Use the z-ranges we collected while plotting surfaces
z_min_global = min(all_z_mins) if all_z_mins else -8
z_max_global = max(all_z_maxs) if all_z_maxs else -2

# Create a scalar mappable for the colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sm = ScalarMappable(cmap=common_colormap, norm=Normalize(vmin=z_min_global, vmax=z_max_global))
sm.set_array([])
cbar = fig_combined.colorbar(sm, ax=ax_combined, shrink=0.6, aspect=20, label='log10(|Euler Error|)')

# Add legend
ax_combined.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Set viewing angle
ax_combined.view_init(elev=30, azim=45)

plt.tight_layout()
output_path_combined = '../NGM_figures/stochastic/stochastic_labor_euler_errors_3d_combined.png'
plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
print(f"✓ Saved combined 3D Euler errors plot: {output_path_combined}")
plt.close()

# Create convergence summary plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extract data
n_vals = [r['n'] for r in results]
n_coeffs = [r['n_coeffs'] for r in results]
max_euler = [r['max_euler'] for r in results]
mean_euler = [r['mean_euler'] for r in results]
rms_euler = [r['rms_euler'] for r in results]
max_intra = [r['max_intra'] for r in results]
comp_times = [r['time'] for r in results]

# Plot 1: Euler errors vs grid size
ax1 = axes[0, 0]
ax1.semilogy(n_vals, max_euler, 'o-', linewidth=2, markersize=8, label='Max |Euler error|')
ax1.semilogy(n_vals, mean_euler, 's-', linewidth=2, markersize=8, label='Mean |Euler error|')
ax1.semilogy(n_vals, rms_euler, '^-', linewidth=2, markersize=8, label='RMS Euler error')
ax1.set_xlabel('Grid Size (n_k = n_z)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Euler Error (log scale)', fontsize=12, fontweight='bold')
ax1.set_title('Euler Error Convergence', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(n_vals)

# Plot 2: Intratemporal errors vs grid size
ax2 = axes[0, 1]
ax2.semilogy(n_vals, max_intra, 'o-', linewidth=2, markersize=8, label='Max |Intratemporal error|', color='orange')
ax2.set_xlabel('Grid Size (n_k = n_z)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Intratemporal Error (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('Intratemporal Error Convergence', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_vals)

# Plot 3: Computation time vs grid size
ax3 = axes[1, 0]
ax3.plot(n_vals, comp_times, 'o-', linewidth=2, markersize=8, color='green')
ax3.set_xlabel('Grid Size (n_k = n_z)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
ax3.set_title('Computational Cost', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(n_vals)

# Plot 4: Euler errors vs number of coefficients
ax4 = axes[1, 1]
ax4.loglog(n_coeffs, max_euler, 'o-', linewidth=2, markersize=8, label='Max |Euler error|')
ax4.loglog(n_coeffs, mean_euler, 's-', linewidth=2, markersize=8, label='Mean |Euler error|')
ax4.set_xlabel('Number of Coefficients', fontsize=12, fontweight='bold')
ax4.set_ylabel('Euler Error (log scale)', fontsize=12, fontweight='bold')
ax4.set_title('Error vs Number of Coefficients', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = '../NGM_figures/stochastic/stochastic_labor_convergence_study.png'
os.makedirs('../NGM_figures/stochastic', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# Save results to file
results_file = '../NGM_figures/convergence_study_results.txt'
with open(results_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CONVERGENCE STUDY RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'n':<6} {'Coeffs':<8} {'Max|EE|':<12} {'Mean|EE|':<12} {'RMS EE':<12} "
            f"{'Max|IE|':<12} {'Time(s)':<10} {'Iter':<6}\n")
    f.write("-"*80 + "\n")
    for r in results:
        f.write(f"{r['n']:<6} {r['n_coeffs']:<8} {r['max_euler']:<12.6e} {r['mean_euler']:<12.6e} "
                f"{r['rms_euler']:<12.6e} {r['max_intra']:<12.6e} {r['time']:<10.2f} {r['iterations']:<6}\n")

print(f"✓ Saved results: {results_file}")

# ============================================================================
# POLICY FUNCTIONS AT z=1 (STEADY-STATE PRODUCTIVITY)
# ============================================================================
print("\nGenerating policy function plots at z=1...")

# Use the finest grid (n=20) for the policy function plot
n_policy = grid_sizes[-1]  # Use n=20
data_policy = euler_errors_by_grid[n_policy]
gamma_c_policy = data_policy['gamma_c']
k_low_policy = data_policy['k_low']
k_high_policy = data_policy['k_high']
z_low_log_policy = data_policy['z_low_log']
z_high_log_policy = data_policy['z_high_log']

# Calculate steady-state
z_ss = 1.0
def steady_state_system(x):
    k, ℓ = x
    k = max(k, 1e-6)
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    c = z_ss * k**α * ℓ**(1-α) - δ * k
    if c <= 1e-10:
        return [1e10, 1e10]
    R = α * z_ss * k**(α-1) * ℓ**(1-α) + (1 - δ)
    euler_err = 1 - β * R
    lhs = χ * ℓ**(1/ν)
    rhs = (1-α) * z_ss * k**α * ℓ**(-α) / c
    intratemporal_err = lhs - rhs
    return [euler_err, intratemporal_err]

k_guess = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
ℓ_guess = 0.33
result = root(steady_state_system, [k_guess, ℓ_guess], method='hybr', options={'xtol': 1e-12})
if result.success:
    k_ss = result.x[0]
    ℓ_ss = result.x[1]
else:
    from scipy.optimize import minimize
    def obj(x):
        errs = steady_state_system(x)
        return errs[0]**2 + errs[1]**2
    result_min = minimize(obj, [k_guess, ℓ_guess], method='BFGS', bounds=[(1e-6, 100), (1e-6, 1.0)])
    k_ss = result_min.x[0]
    ℓ_ss = result_min.x[1]

c_ss = z_ss * k_ss**α * ℓ_ss**(1-α) - δ * k_ss

# Fine grid for capital
k_fine = np.linspace(k_low_policy, k_high_policy, 200)

# Evaluate policy functions
c_policy = []
k_prime_policy = []
l_policy = []

for k_val in k_fine:
    # Consumption
    c_val = c_cheb(k_val, z_ss, gamma_c_policy, k_low_policy, k_high_policy, 
                   z_low_log_policy, z_high_log_policy, n_policy, n_policy)
    c_policy.append(c_val)
    
    # Labor (from intratemporal FOC)
    ℓ_val = l_from_c(k_val, z_ss, c_val, χ, α, ν)
    l_policy.append(ℓ_val)
    
    # Next period capital
    k_prime = (1 - δ) * k_val + z_ss * k_val**α * ℓ_val**(1-α) - c_val
    k_prime_policy.append(k_prime)

c_policy = np.array(c_policy)
k_prime_policy = np.array(k_prime_policy)
l_policy = np.array(l_policy)

# Create 1x3 figure
fig_policy, axes_policy = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Consumption policy function
ax1 = axes_policy[0]
ax1.plot(k_fine, c_policy, 'b-', linewidth=2.5, label=f'n = {n_policy}')
ax1.axvline(k_ss, color='black', linestyle=':', linewidth=2, alpha=0.6, label='Steady-state k')
# Find steady-state consumption from policy function
c_ss_policy = c_cheb(k_ss, z_ss, gamma_c_policy, k_low_policy, k_high_policy, 
                     z_low_log_policy, z_high_log_policy, n_policy, n_policy)
ax1.axhline(c_ss_policy, color='black', linestyle=':', linewidth=2, alpha=0.6)
ax1.scatter([k_ss], [c_ss_policy], s=200, marker='*', color='gold', 
           edgecolors='black', linewidths=2, zorder=10, label='Steady-state')
ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax1.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
ax1.set_title('Consumption Policy Function\n(z = 1)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Capital accumulation policy function
ax2 = axes_policy[1]
ax2.plot(k_fine, k_prime_policy, 'g-', linewidth=2.5, label=f'n = {n_policy}')
ax2.plot(k_fine, k_fine, 'k--', linewidth=1.5, alpha=0.5, label='45° line')
ax2.axvline(k_ss, color='black', linestyle=':', linewidth=2, alpha=0.6)
# Compute steady-state next period capital
ℓ_ss_policy = l_from_c(k_ss, z_ss, c_ss_policy, χ, α, ν)
k_prime_ss = (1 - δ) * k_ss + z_ss * k_ss**α * ℓ_ss_policy**(1-α) - c_ss_policy
ax2.axhline(k_prime_ss, color='black', linestyle=':', linewidth=2, alpha=0.6)
ax2.scatter([k_ss], [k_prime_ss], s=200, marker='*', color='gold', 
           edgecolors='black', linewidths=2, zorder=10, label='Steady-state')
ax2.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax2.set_ylabel("k' (Next Period Capital)", fontsize=12, fontweight='bold')
ax2.set_title('Capital Accumulation Policy Function\n(z = 1)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Plot 3: Labor policy function
ax3 = axes_policy[2]
ax3.plot(k_fine, l_policy, 'r-', linewidth=2.5, label=f'n = {n_policy}')
ax3.axvline(k_ss, color='black', linestyle=':', linewidth=2, alpha=0.6)
ax3.axhline(ℓ_ss_policy, color='black', linestyle=':', linewidth=2, alpha=0.6)
ax3.scatter([k_ss], [ℓ_ss_policy], s=200, marker='*', color='gold', 
           edgecolors='black', linewidths=2, zorder=10, label='Steady-state')
ax3.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax3.set_ylabel('ℓ (Labor)', fontsize=12, fontweight='bold')
ax3.set_title('Labor Policy Function\n(z = 1)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path_policy = '../NGM_figures/stochastic/stochastic_labor_policy_functions_z1.png'
plt.savefig(output_path_policy, dpi=300, bbox_inches='tight')
print(f"✓ Saved policy functions plot: {output_path_policy}")
plt.close()

print("\n" + "="*80)
print("CONVERGENCE STUDY COMPLETE")
print("="*80)

