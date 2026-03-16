#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of Chebyshev Labor Model for Different Numbers of Nodes
Solves NGM model with labor supply for n = 3, 5, 10, and 20 nodes and plots Euler errors
"""

import numpy as np
from scipy import optimize as opt
from scipy.optimize import root
from scipy.interpolate import interp1d
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
rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33     # Capital share
δ = 0.025    # Depreciation rate
χ = 2.0      # Labor disutility parameter (fixed)
ν = 1.0      # Frisch elasticity parameter
# Damping parameter (adaptive: starts at 1.0, decreases if needed)
dampening = 1.0  # Start with no damping (use new guess fully)

# ============================================================================
# STEADY-STATE CALCULATION
# ============================================================================
def steady_state_system(x):
    """System of equations for steady state: x = [k, ℓ]"""
    k, ℓ = x
    k = max(k, 1e-6)
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    
    c = k**α * ℓ**(1-α) - δ * k
    
    if c <= 1e-10:
        return [1e10, 1e10]
    
    R = α * k**(α-1) * ℓ**(1-α) + (1 - δ)
    euler_err = 1 - β * R
    
    lhs = χ * ℓ**(1/ν)
    rhs = (1-α) * k**α * ℓ**(-α) / c
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

c_ss = k_ss**α * ℓ_ss**(1-α) - δ * k_ss
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

# ============================================================================
# POLICY FUNCTIONS
# ============================================================================
def c_cheb(k, gamma_c, k_low, k_high, p_k):
    """Chebyshev approximation of consumption"""
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    c = float(gamma_c @ T_k)
    c = max(c, 1e-10)
    return c

def l_from_c(k, c, χ, α, ν):
    """Compute labor from consumption using intratemporal FOC"""
    if c <= 1e-10:
        return 1e-6
    ℓ = ((1-α) * k**α / (χ * c))**(ν/(1+α*ν))
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    return ℓ

def invert_policy_to_gamma(policy_values, k_grid, k_low, k_high, p_k):
    """Invert policy function values to get Chebyshev coefficients"""
    n = len(k_grid)
    T_matrix = np.zeros((n, p_k))
    for i in range(n):
        k = k_grid[i]
        k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
        T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
        T_matrix[i, :] = T_k.ravel()
    
    if n == p_k:
        gamma = np.linalg.solve(T_matrix, policy_values)
    else:
        gamma = np.linalg.lstsq(T_matrix, policy_values, rcond=None)[0]
    
    return gamma

def compute_euler_errors_and_update(c_values, k_grid, k_low, k_high, p_k, gamma_c):
    """Compute Euler errors and update consumption"""
    n = len(k_grid)
    c_new = np.zeros(n)
    euler_errors = np.zeros(n)
    
    for i_k in range(n):
        k = k_grid[i_k]
        c = c_values[i_k]
        ℓ = l_from_c(k, c, χ, α, ν)
        y = k**α * ℓ**(1-α)
        k_prime = (1 - δ) * k + y - c
        
        if k_prime <= 0 or k_prime > 2 * k_high:
            c_new[i_k] = c
            euler_errors[i_k] = 1e10
            continue
        
        c_prime = c_cheb(k_prime, gamma_c, k_low, k_high, p_k)
        ℓ_prime = l_from_c(k_prime, c_prime, χ, α, ν)
        y_prime = k_prime**α * ℓ_prime**(1-α)
        R_prime = α * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
        consumption_ratio = c / c_prime
        euler_error = 1 - β * consumption_ratio * R_prime
        euler_errors[i_k] = euler_error
        
        if R_prime > 0 and c_prime > 1e-10:
            c_target = c_prime / (β * R_prime)
            c_target = max(c_target, 1e-10)
            y_max = k**α * 1.0**(1-α)
            c_target = min(c_target, y_max + (1-δ)*k)
            # Use full update (no damping here - damping applied in main loop)
            c_new[i_k] = c_target
        else:
            c_new[i_k] = c
        
        c_new[i_k] = max(c_new[i_k], 1e-10)
        ℓ_new = l_from_c(k, c_new[i_k], χ, α, ν)
        y_new = k**α * ℓ_new**(1-α)
        c_new[i_k] = min(c_new[i_k], y_new + (1-δ)*k)
    
    return c_new, euler_errors

# ============================================================================
# SOLVE MODEL FOR GIVEN n_k
# ============================================================================
def solve_model(n_k):
    """Solve the model for a given number of Chebyshev nodes"""
    p_k = n_k
    
    # Get Chebyshev nodes
    cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()
    k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
    
    # Initialize
    c_current = np.full(n_k, c_ss)
    gamma_c_current = invert_policy_to_gamma(c_current, k_grid, k_low, k_high, p_k)
    
    # Fixed-point iteration
    max_iter = 20000
    tol = 1e-8
    
    # Adaptive dampening: start at 1 (no damping, always use new guess)
    # Only reduce dampening if errors increase significantly
    dampening_current = 1.0
    prev_max_error = np.inf
    error_increase_count = 0
    
    for iter in range(max_iter):
        c_new, euler_errors = compute_euler_errors_and_update(
            c_current, k_grid, k_low, k_high, p_k, gamma_c_current)
        
        max_euler_error = np.max(np.abs(euler_errors))
        
        # Adaptive dampening: only reduce if error increases significantly
        if iter > 0:
            if max_euler_error > prev_max_error * 1.1:  # Error increased by more than 10%
                error_increase_count += 1
                # Only reduce dampening after multiple consecutive increases
                if error_increase_count >= 3:
                    dampening_current = max(0.1, dampening_current * 0.9)
                    error_increase_count = 0
            elif max_euler_error < prev_max_error:
                # Error decreased, reset counter and potentially increase dampening
                error_increase_count = 0
                if iter > 50 and dampening_current < 1.0:
                    dampening_current = min(1.0, dampening_current * 1.05)
            else:
                # Error stable, reset counter
                error_increase_count = 0
        
        prev_max_error = max_euler_error
        
        if iter % 100 == 0 or max_euler_error < tol:
            print(f"  Iteration {iter:4d}: max |Euler error| = {max_euler_error:.6e}, "
                  f"dampening = {dampening_current:.4f}")
        
        if max_euler_error < tol:
            print(f"  Converged after {iter} iterations!")
            break
        
        # Update consumption with adaptive damping (always use dampening_current)
        c_current = (1 - dampening_current) * c_current + dampening_current * c_new
        
        gamma_c_current = invert_policy_to_gamma(c_current, k_grid, k_low, k_high, p_k)
        c_current = np.array([c_cheb(k, gamma_c_current, k_low, k_high, p_k) for k in k_grid])
    
    # Compute Euler errors on fine grid
    k_grid_fine = np.linspace(k_low, k_high, 500)
    euler_errors_fine = []
    
    for k in k_grid_fine:
        c = c_cheb(k, gamma_c_current, k_low, k_high, p_k)
        ℓ = l_from_c(k, c, χ, α, ν)
        y = k**α * ℓ**(1-α)
        k_prime = (1 - δ) * k + y - c
        
        if k_prime <= 0 or k_prime > 2 * k_high:
            euler_errors_fine.append(1e10)
            continue
        
        c_prime = c_cheb(k_prime, gamma_c_current, k_low, k_high, p_k)
        ℓ_prime = l_from_c(k_prime, c_prime, χ, α, ν)
        y_prime = k_prime**α * ℓ_prime**(1-α)
        R_prime = α * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
        consumption_ratio = c / c_prime
        euler_error = 1 - β * consumption_ratio * R_prime
        euler_errors_fine.append(euler_error)
    
    # Get errors at nodes
    errors_at_nodes = []
    k_positions_at_nodes = []
    for k in k_grid:
        c = c_cheb(k, gamma_c_current, k_low, k_high, p_k)
        ℓ = l_from_c(k, c, χ, α, ν)
        y = k**α * ℓ**(1-α)
        k_prime = (1 - δ) * k + y - c
        
        if k_prime <= 0 or k_prime > 2 * k_high:
            errors_at_nodes.append(1e10)
            k_positions_at_nodes.append(k)
            continue
        
        c_prime = c_cheb(k_prime, gamma_c_current, k_low, k_high, p_k)
        ℓ_prime = l_from_c(k_prime, c_prime, χ, α, ν)
        y_prime = k_prime**α * ℓ_prime**(1-α)
        R_prime = α * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
        consumption_ratio = c / c_prime
        euler_error = 1 - β * consumption_ratio * R_prime
        errors_at_nodes.append(euler_error)
        k_positions_at_nodes.append(k)
    
    return {
        'gamma_c': gamma_c_current,
        'k_grid_fine': k_grid_fine,
        'euler_errors_fine': np.array(euler_errors_fine),
        'k_grid': k_grid,
        'errors_at_nodes': np.array(errors_at_nodes),
        'k_positions_at_nodes': np.array(k_positions_at_nodes),
        'max_error': max(np.abs(euler_errors_fine)),
        'mean_error': np.mean(np.abs(euler_errors_fine)),
        'rms_error': np.sqrt(np.mean(np.array(euler_errors_fine)**2))
    }

# ============================================================================
# MAIN: SOLVE FOR DIFFERENT n_k VALUES
# ============================================================================
print("="*80)
print("DETERMINISTIC NGM WITH LABOR - CHEBYSHEV COMPARISON")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}, α = {α}, δ = {δ}, χ = {χ}, ν = {ν}")
print(f"\nSteady-state: k_ss = {k_ss:.6f}, ℓ_ss = {ℓ_ss:.6f}, c_ss = {c_ss:.6f}")
print(f"Capital domain: [{k_low:.6f}, {k_high:.6f}]")

node_counts = [3, 5, 10, 20]
results = {}

for n_k in node_counts:
    print(f"\n{'='*80}")
    print(f"Solving for n_k = {n_k}")
    print(f"{'='*80}")
    results[n_k] = solve_model(n_k)
    print(f"Saved n_k={n_k}: {n_k} nodes, error range=[{results[n_k]['max_error']:.2e}, {results[n_k]['mean_error']:.2e}]")

# ============================================================================
# PLOT EULER ERRORS COMPARISON
# ============================================================================
print("\nGenerating comparison plots...")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Colors and styles for different n_k values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']

for i, n_k_val in enumerate(node_counts):
    res_this = results[n_k_val]
    
    # Get fine grid data for THIS n_k
    k_fine_this = res_this['k_grid_fine']
    errors_fine_this = np.abs(res_this['euler_errors_fine'])
    
    # Filter out invalid values
    valid_fine = ~np.isnan(errors_fine_this) & (errors_fine_this > 0)
    k_fine_valid = k_fine_this[valid_fine]
    errors_fine_valid = errors_fine_this[valid_fine]
    
    # Plot fine grid error line for THIS n_k
    ax.plot(k_fine_valid, errors_fine_valid, color=colors[i], linestyle=linestyles[i], 
           linewidth=2.5, alpha=0.8, label=f'$n={n_k_val}$')
    
    # Get node positions for THIS n_k
    k_nodes_this = res_this['k_positions_at_nodes']
    
    # Interpolate fine-grid errors at exact node positions to ensure nodes are on the line
    if len(k_fine_valid) > 1 and len(k_nodes_this) > 0:
        # Interpolate errors at node positions
        interp_func = interp1d(k_fine_valid, errors_fine_valid, kind='linear',
                              bounds_error=False, fill_value=np.nan)
        errors_at_nodes_interp = interp_func(k_nodes_this)
        
        # Filter valid interpolated values
        valid_nodes = ~np.isnan(errors_at_nodes_interp) & (errors_at_nodes_interp > 0)
        
        # Plot nodes with interpolated errors (ensures they're exactly on the line)
        ax.scatter(k_nodes_this[valid_nodes], errors_at_nodes_interp[valid_nodes],
                  color=colors[i], s=150, marker=markers[i],
                  edgecolors='black', linewidths=1.5, zorder=10, alpha=0.9)

ax.set_xlabel('Capital, $k$', fontsize=14, fontweight='bold')
ax.set_ylabel('|Euler Error| (log scale)', fontsize=14, fontweight='bold')
ax.set_title('Euler Errors: Comparison Across Node Counts', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_yscale('log')
ax.legend(fontsize=12, framealpha=0.95, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure
output_dir = 'NGM_deterministic_labor'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'euler_errors_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"{'n_nodes':<10} {'Max Error':<15} {'Mean Error':<15} {'RMS Error':<15}")
print("-" * 80)
for n_k_val in node_counts:
    res = results[n_k_val]
    print(f"{n_k_val:<10} {res['max_error']:<15.6e} {res['mean_error']:<15.6e} {res['rms_error']:<15.6e}")

print("\nDone!")

