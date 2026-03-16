#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to check Euler errors at collocation nodes vs fine grid
"""

import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'scripts'))
from functions_library import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model parameters
β = 0.99
α = 0.33
δ = 0.025
γ = 1
dampening = 1

k_ss = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
c_ss = k_ss**α - δ * k_ss
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

def c_cheb(k, gamma, k_low, k_high, p_k):
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    c = float(gamma @ T_k)
    c = max(c, 1e-10)
    return c

def compute_euler_errors_and_update(c_values, k_grid, k_low, k_high, p_k, gamma):
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

def invert_consumption_to_gamma(c_values, k_grid, k_low, k_high, p_k):
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

def solve_model(n_k):
    p_k = n_k
    cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()
    k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
    
    c_current = np.full(n_k, c_ss)
    gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)
    
    max_iter = 2000
    tol = 1e-8
    
    for iter in range(max_iter):
        c_new, euler_errors = compute_euler_errors_and_update(
            c_current, k_grid, k_low, k_high, p_k, gamma_current)
        
        max_error = np.max(np.abs(euler_errors))
        
        if max_error < tol:
            break
        
        if max_error < 1.0:
            c_current = (1 - dampening) * c_current + dampening * c_new
        else:
            c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
        
        gamma_current = invert_consumption_to_gamma(c_current, k_grid, k_low, k_high, p_k)
        c_current = np.array([c_cheb(k, gamma_current, k_low, k_high, p_k) for k in k_grid])
    
    return gamma_current, k_grid

# Solve for n=20
n_k = 20
gamma, k_grid = solve_model(n_k)

# Compute Euler errors at nodes
euler_errors_at_nodes = []
for k in k_grid:
    c = c_cheb(k, gamma, k_low, k_high, n_k)
    k_prime = (1 - δ) * k + k**α - c
    if k_prime > 0 and k_prime <= 2 * k_high:
        c_prime = c_cheb(k_prime, gamma, k_low, k_high, n_k)
        R_prime = α * k_prime**(α - 1) + (1 - δ)
        consumption_ratio = c_prime / c
        euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
        euler_errors_at_nodes.append(abs(euler_error))
    else:
        euler_errors_at_nodes.append(np.nan)

# Compute Euler errors on fine grid
k_grid_fine = np.linspace(k_low, k_high, 1000)
euler_errors_fine = []
for k in k_grid_fine:
    c = c_cheb(k, gamma, k_low, k_high, n_k)
    k_prime = (1 - δ) * k + k**α - c
    if k_prime > 0 and k_prime <= 2 * k_high:
        c_prime = c_cheb(k_prime, gamma, k_low, k_high, n_k)
        R_prime = α * k_prime**(α - 1) + (1 - δ)
        consumption_ratio = c_prime / c
        euler_error = 1 - β * (consumption_ratio)**(-γ) * R_prime
        euler_errors_fine.append(abs(euler_error))
    else:
        euler_errors_fine.append(np.nan)

euler_errors_at_nodes = np.array(euler_errors_at_nodes)
euler_errors_fine = np.array(euler_errors_fine)

# Print statistics
print("="*80)
print(f"EULER ERROR DIAGNOSTICS FOR n={n_k}")
print("="*80)
print(f"\nAt collocation nodes:")
print(f"  Min error: {np.nanmin(euler_errors_at_nodes):.6e}")
print(f"  Max error: {np.nanmax(euler_errors_at_nodes):.6e}")
print(f"  Mean error: {np.nanmean(euler_errors_at_nodes):.6e}")
print(f"  Errors at nodes: {euler_errors_at_nodes}")

print(f"\nOn fine grid (1000 points):")
print(f"  Min error: {np.nanmin(euler_errors_fine):.6e}")
print(f"  Max error: {np.nanmax(euler_errors_fine):.6e}")
print(f"  Mean error: {np.nanmean(euler_errors_fine):.6e}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Errors at nodes vs fine grid
ax1 = axes[0]
ax1.plot(k_grid_fine, euler_errors_fine, 'b-', linewidth=1.5, alpha=0.7, label='Fine grid')
ax1.scatter(k_grid, euler_errors_at_nodes, c='red', s=150, marker='o', 
           edgecolors='black', linewidths=2, zorder=10, label='Collocation nodes')
ax1.axvline(k_ss, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Steady-state')
ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax1.set_ylabel('|Euler Error| (log scale)', fontsize=12, fontweight='bold')
ax1.set_title(f'Euler Errors: Nodes vs Fine Grid (n={n_k})', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Zoomed in around nodes
ax2 = axes[1]
# Find indices of fine grid points closest to nodes
node_indices = []
for k_node in k_grid:
    idx = np.argmin(np.abs(k_grid_fine - k_node))
    node_indices.append(idx)

# Plot fine grid errors
ax2.plot(k_grid_fine, euler_errors_fine, 'b-', linewidth=1.5, alpha=0.7, label='Fine grid')
# Highlight nodes
ax2.scatter(k_grid, euler_errors_at_nodes, c='red', s=200, marker='o', 
           edgecolors='black', linewidths=2, zorder=10, label='Collocation nodes')
# Mark node positions on fine grid
for i, k_node in enumerate(k_grid):
    idx = node_indices[i]
    ax2.plot([k_grid_fine[idx], k_grid_fine[idx]], 
            [euler_errors_fine[idx], euler_errors_at_nodes[i]], 
            'r--', linewidth=1, alpha=0.5)
    # Annotate difference
    diff = abs(euler_errors_fine[idx] - euler_errors_at_nodes[i])
    if diff > 1e-10:
        ax2.annotate(f'{diff:.2e}', 
                    xy=(k_node, euler_errors_at_nodes[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.axvline(k_ss, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
ax2.set_ylabel('|Euler Error| (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('Zoom: Errors at Nodes', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '../NGM_figures/euler_error_diagnostic.png'
os.makedirs('NGM_figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved diagnostic plot: {output_path}")
plt.close()

print("\nDone!")

