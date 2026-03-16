#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of Chebyshev Direct Approximation for Different Numbers of Nodes
Solves NGM model with 3, 5, 10, and 20 nodes and plots results
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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33      # Capital share
δ = 0.025     # Depreciation rate
γ = 1         # Risk aversion (CRRA parameter)
dampening = 1  # Damping parameter for consumption updates

# Calculate steady state capital
k_ss = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
c_ss = k_ss**α - δ * k_ss  # Steady-state consumption
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

# ============================================================================
# CONSUMPTION POLICY FUNCTION (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, gamma, k_low, k_high, p_k):
    """Direct Chebyshev approximation: c(k) = gamma @ T_k"""
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    c = float(gamma @ T_k)
    c = max(c, 1e-10)  # Ensure positivity
    return c

# ============================================================================
# COMPUTE EULER ERRORS AND UPDATE CONSUMPTION
# ============================================================================
def compute_euler_errors_and_update(c_values, k_grid, k_low, k_high, p_k, gamma):
    """Computes Euler errors and returns updated consumption values"""
    n = len(k_grid)
    c_new = np.zeros(n)
    euler_errors = np.zeros(n)
    
    for i_k in range(n):
        k = k_grid[i_k]
        c = c_values[i_k]
        k_prime = (1 - δ) * k + k**α - c
        
        if k_prime <= 0 or k_prime > 2 * k_high:
            c_new[i_k] = c
            euler_errors[i_k] = 1e10
            continue
        
        c_prime = c_cheb(k_prime, gamma, k_low, k_high, p_k)
        R_prime = α * k_prime**(α - 1) + (1 - δ)
        consumption_ratio = c_prime / c
        euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
        euler_errors[i_k] = euler_error
        
        if R_prime > 0 and c_prime > 1e-10:
            c_target = c_prime / (β * R_prime)**(1/γ)
            c_target = max(c_target, 1e-10)
            c_target = min(c_target, k**α + (1-δ)*k)
            c_new[i_k] = (1 - dampening) * c + dampening * c_target
        else:
            c_new[i_k] = c
        
        c_new[i_k] = max(c_new[i_k], 1e-10)
        c_new[i_k] = min(c_new[i_k], k**α + (1-δ)*k)
    
    return c_new, euler_errors

# ============================================================================
# INVERT CONSUMPTION TO GET GAMMA COEFFICIENTS
# ============================================================================
def invert_consumption_to_gamma(c_values, k_grid, k_low, k_high, p_k):
    """Inverts consumption values to get Chebyshev coefficients"""
    n = len(k_grid)
    T_matrix = np.zeros((n, p_k))
    for i in range(n):
        k = k_grid[i]
        k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
        T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
        T_matrix[i, :] = T_k.ravel()
    
    if n == p_k:
        gamma = np.linalg.solve(T_matrix, c_values)
    else:
        gamma = np.linalg.lstsq(T_matrix, c_values, rcond=None)[0]
    
    return gamma

# ============================================================================
# SOLVE MODEL FOR GIVEN NUMBER OF NODES
# ============================================================================
def solve_model(n_k):
    """Solves the model for a given number of Chebyshev nodes"""
    print(f"\n{'='*80}")
    print(f"Solving with n_k = {n_k} nodes")
    print(f"{'='*80}")
    
    p_k = n_k
    n_coeffs = p_k
    
    # Get Chebyshev nodes
    cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()
    k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
    
    # Initialize
    c_current = np.full(n_k, c_ss)
    gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)
    
    # Fixed-point iteration
    max_iter = 2000
    tol = 1e-12
    
    for iter in range(max_iter):
        c_new, euler_errors = compute_euler_errors_and_update(
            c_current, k_grid, k_low, k_high, p_k, gamma_current)
        
        max_error = np.max(np.abs(euler_errors))
        mean_error = np.mean(np.abs(euler_errors))
        
        if iter % 50 == 0:
            print(f"  Iteration {iter:4d}: max |Euler error| = {max_error:.6e}")
        
        if max_error < tol:
            print(f"  Converged after {iter} iterations!")
            break
        
        if max_error < 1.0:
            c_current = (1 - dampening) * c_current + dampening * c_new
        else:
            c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
        
        gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)
        c_current = np.array([c_cheb(k, gamma_current, k_low, k_high, p_k) for k in k_grid])
    
    if iter == max_iter - 1:
        print(f"  Warning: Reached maximum iterations ({max_iter})")
    
    # Compute Euler errors on fine grid for plotting
    k_grid_fine = np.linspace(k_low, k_high, 500)
    euler_errors_fine = []
    c_policy_fine = []
    
    for k in k_grid_fine:
        c = c_cheb(k, gamma_current, k_low, k_high, p_k)
        c_policy_fine.append(c)
        k_prime = (1 - δ) * k + k**α - c
        if k_prime > 0 and k_prime <= 2 * k_high:
            c_prime = c_cheb(k_prime, gamma_current, k_low, k_high, p_k)
            R_prime = α * k_prime**(α - 1) + (1 - δ)
            consumption_ratio = c_prime / c
            euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
            euler_errors_fine.append(abs(euler_error))
        else:
            euler_errors_fine.append(np.nan)
    
    # Euler errors at nodes
    euler_errors_nodes = []
    for k in k_grid:
        c = c_cheb(k, gamma_current, k_low, k_high, p_k)
        k_prime = (1 - δ) * k + k**α - c
        if k_prime > 0 and k_prime <= 2 * k_high:
            c_prime = c_cheb(k_prime, gamma_current, k_low, k_high, p_k)
            R_prime = α * k_prime**(α - 1) + (1 - δ)
            consumption_ratio = c_prime / c
            euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
            euler_errors_nodes.append(abs(euler_error))
        else:
            euler_errors_nodes.append(np.nan)
    
    return {
        'n_k': n_k,
        'gamma': gamma_current,
        'k_grid': k_grid,
        'c_policy_fine': np.array(c_policy_fine),
        'k_grid_fine': k_grid_fine,
        'euler_errors_fine': np.array(euler_errors_fine),
        'euler_errors_nodes': np.array(euler_errors_nodes),
        'max_error': max_error,
        'mean_error': mean_error
    }

# ============================================================================
# MAIN: SOLVE FOR DIFFERENT NUMBERS OF NODES
# ============================================================================
print("="*80)
print("DETERMINISTIC NGM - CHEBYSHEV DIRECT APPROXIMATION COMPARISON")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}, α = {α}, δ = {δ}, γ = {γ}")
print(f"  Steady-state: k_ss = {k_ss:.6f}, c_ss = {c_ss:.6f}")
print(f"  Domain: [{k_low:.6f}, {k_high:.6f}]")

# Solve for different numbers of nodes
node_counts = [3, 5, 10, 20]
results = {}
# Store errors and node positions explicitly for each n_k
errors_at_nodes = {}  # errors_at_nodes[n_k] = array of errors
k_positions_at_nodes = {}  # k_positions_at_nodes[n_k] = array of k positions

for n_k in node_counts:
    res = solve_model(n_k)
    results[n_k] = res
    
    # Save k positions
    k_positions_at_nodes[n_k] = np.array(res['k_grid']).copy()
    
    # Interpolate fine grid errors at exact node positions
    from scipy.interpolate import interp1d
    valid_fine = ~np.isnan(res['euler_errors_fine'])
    k_fine_valid = res['k_grid_fine'][valid_fine]
    errors_fine_valid = res['euler_errors_fine'][valid_fine]
    
    # Interpolate (use linear interpolation on log scale for better accuracy)
    log_errors_fine = np.log10(errors_fine_valid + 1e-16)
    interp_func = interp1d(k_fine_valid, log_errors_fine, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    log_errors_at_nodes = interp_func(k_positions_at_nodes[n_k])
    errors_interp = 10**log_errors_at_nodes - 1e-16
    errors_interp = np.maximum(errors_interp, 1e-16)  # Ensure positive
    errors_at_nodes[n_k] = errors_interp
    
    print(f"Saved n_k={n_k}: {len(errors_at_nodes[n_k])} nodes, error range=[{np.nanmin(errors_at_nodes[n_k]):.2e}, {np.nanmax(errors_at_nodes[n_k]):.2e}]")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("\nGenerating comparison plots...")

# Fine grid for plotting
k_grid_fine = np.linspace(k_low, k_high, 500)

# Create fancy side-by-side plot
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # More vibrant colors
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']  # Different markers for each n_k
marker_sizes = [100, 100, 80, 60]  # Larger markers for fewer nodes

# Create figure with side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ============================================================================
# LEFT PLOT: Consumption Policy Functions
# ============================================================================
for i, n_k_val in enumerate(node_counts):
    res = results[n_k_val]
    ax1.plot(res['k_grid_fine'], res['c_policy_fine'], 
            color=colors[i], linestyle=linestyles[i], linewidth=2.5, 
            alpha=0.8, label=f'n = {n_k_val}', zorder=5-i)

# Find region with largest errors (for n=3, errors are largest near boundaries)
# Use a much smaller zoom region - focus on left boundary where errors are highest
res_3 = results[3]
valid_errors_3 = ~np.isnan(res_3['euler_errors_fine'])
k_errors_3 = res_3['k_grid_fine'][valid_errors_3]
errors_3 = res_3['euler_errors_fine'][valid_errors_3]
# Find peak error location (left boundary)
peak_error_idx = np.argmax(errors_3)
peak_k = k_errors_3[peak_error_idx]
# Zoom in much more - use a very small region around peak error
zoom_width = 0.03 * (k_high - k_low)  # Much smaller zoom - only 3% of domain
zoom_k_low = max(k_low, peak_k - zoom_width/2)
zoom_k_high = min(k_high, peak_k + zoom_width/2)

# Get consumption values in zoom region for all n_k
zoom_c_values = []
for n_k in node_counts:
    zoom_mask = (results[n_k]['k_grid_fine'] >= zoom_k_low) & (results[n_k]['k_grid_fine'] <= zoom_k_high)
    zoom_c_values.extend(results[n_k]['c_policy_fine'][zoom_mask].tolist())
zoom_c_min = np.min(zoom_c_values)
zoom_c_max = np.max(zoom_c_values)
zoom_c_range = zoom_c_max - zoom_c_min
zoom_c_low = zoom_c_min - 0.1 * zoom_c_range
zoom_c_high = zoom_c_max + 0.1 * zoom_c_range

# Add zoom box on main plot
zoom_box = Rectangle((zoom_k_low, zoom_c_low), zoom_k_high - zoom_k_low, 
                     zoom_c_high - zoom_c_low,
                     linewidth=2.5, edgecolor='red', facecolor='none', 
                     linestyle='--', alpha=0.8, zorder=10)
ax1.add_patch(zoom_box)
ax1.text(zoom_k_low, zoom_c_high, 'Zoom', fontsize=11, fontweight='bold', 
         color='red', va='bottom', ha='left', zorder=11)

# Add zoomed inset INSIDE the consumption plot (left plot) - use simple positioning
axins = ax1.inset_axes([0.06, 0.55, 0.4, 0.4])  # [x0, y0, width, height] in axes coordinates

for i, n_k_val in enumerate(node_counts):
    res = results[n_k_val]
    zoom_mask_inset = (res['k_grid_fine'] >= zoom_k_low) & (res['k_grid_fine'] <= zoom_k_high)
    axins.plot(res['k_grid_fine'][zoom_mask_inset], res['c_policy_fine'][zoom_mask_inset],
              color=colors[i], linestyle=linestyles[i], linewidth=2.5, alpha=0.9)

axins.set_xlim(zoom_k_low, zoom_k_high)
axins.set_ylim(zoom_c_low, zoom_c_high)
axins.set_xlabel('k', fontsize=10, fontweight='bold')
axins.set_ylabel('c', fontsize=10, fontweight='bold')
axins.tick_params(labelsize=9)
axins.grid(True, alpha=0.4, linestyle='--')
axins.set_title('Zoom: High Error Region', fontsize=11, fontweight='bold', color='red', pad=8)
# Make inset stand out
for spine in axins.spines.values():
    spine.set_edgecolor('red')
    spine.set_linewidth(2)

# Main plot styling
ax1.axvline(k_ss, color='black', linestyle=':', linewidth=2, alpha=0.6, label='Steady-state')
ax1.axhline(c_ss, color='black', linestyle=':', linewidth=2, alpha=0.6)
# Add fancy marker at steady-state point (no label to avoid duplicate in legend)
ax1.scatter([k_ss], [c_ss], s=300, marker='*', color='gold', 
           edgecolors='black', linewidths=2.5, zorder=15, alpha=0.95)
ax1.set_xlabel('k (Capital)', fontsize=13, fontweight='bold')
ax1.set_ylabel('c (Consumption)', fontsize=13, fontweight='bold')
ax1.set_title('Consumption Policy Functions', fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=11, framealpha=0.9, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ============================================================================
# RIGHT PLOT: Euler Errors
# ============================================================================
for i, n_k_val in enumerate(node_counts):
    res_this = results[n_k_val]
    
    # Plot error line for THIS n_k
    valid_fine = ~np.isnan(res_this['euler_errors_fine'])
    k_fine_this = res_this['k_grid_fine'][valid_fine]
    errors_fine_this = res_this['euler_errors_fine'][valid_fine]
    ax2.plot(k_fine_this, errors_fine_this, 
            color=colors[i], linestyle=linestyles[i], linewidth=2.5, 
            alpha=0.8, label=f'n = {n_k_val}', zorder=5-i)
    
    # Use saved errors and k positions for THIS n_k
    k_nodes_this_nk = k_positions_at_nodes[n_k_val]
    errors_nodes_this_nk = errors_at_nodes[n_k_val]
    valid_nodes_this = ~np.isnan(errors_nodes_this_nk)
    
    # Plot nodes with saved errors
    ax2.scatter(k_nodes_this_nk[valid_nodes_this], errors_nodes_this_nk[valid_nodes_this],
            c=colors[i], marker=markers[i], s=marker_sizes[i], 
            edgecolors='black', linewidths=2, zorder=15, alpha=1.0)

ax2.axvline(k_ss, color='black', linestyle=':', linewidth=2, alpha=0.6)
ax2.set_xlabel('k (Capital)', fontsize=13, fontweight='bold')
ax2.set_ylabel('|Euler Error| (log scale)', fontsize=13, fontweight='bold')
ax2.set_title('Euler Errors', 
             fontsize=15, fontweight='bold', pad=15)
ax2.set_yscale('log')
# No legend for right plot
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure
output_path = '../NGM_figures/deterministic_Chebyshev_direct_comparison.png'
os.makedirs('NGM_figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"{'n_nodes':<10} {'Max Error':<15} {'Mean Error':<15} {'RMS Error':<15}")
print("-" * 60)

for n_k in node_counts:
    res = results[n_k]
    valid_errors = res['euler_errors_fine'][~np.isnan(res['euler_errors_fine'])]
    max_err = np.max(valid_errors)
    mean_err = np.mean(valid_errors)
    rms_err = np.sqrt(np.mean(valid_errors**2))
    print(f"{n_k:<10} {max_err:<15.6e} {mean_err:<15.6e} {rms_err:<15.6e}")

print("\nDone!")

