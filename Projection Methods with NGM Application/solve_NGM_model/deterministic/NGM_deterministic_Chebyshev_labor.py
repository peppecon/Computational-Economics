#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neoclassical Growth Model - Deterministic Version WITH LABOR SUPPLY
Only capital k as state variable (no productivity shocks)
Uses Chebyshev polynomial projection method
DIRECT APPROXIMATION: Approximates c(k) and ℓ(k) directly without log-exp transformation

Utility: U(c, ℓ) = log(c) - χ * ℓ^(1+1/ν)/(1+1/ν)
Production: y = k^α * ℓ^(1-α)
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
rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33     # Capital share
δ = 0.025    # Depreciation rate
χ = 2.0      # Labor disutility parameter (fixed)
ν = 1.0      # Frisch elasticity parameter (1/ν is the inverse Frisch elasticity)

# Damping parameter for iteration (adaptive: starts at 1, decreases if needed)
dampening = 1.0  # Start with no damping (use new guess fully)

# ============================================================================
# CHEBYSHEV APPROXIMATION SETUP
# ============================================================================
n_k = 10      # Number of Chebyshev nodes for capital
p_k = n_k     # Polynomial order (typically equals number of nodes)

# Calculate steady state (with labor)
# We need to solve the system simultaneously:
# 1. Euler: 1 = β * (α * k^(α-1) * ℓ^(1-α) + 1 - δ)
# 2. Intratemporal: χ * ℓ^(1/ν) = (1-α) * k^α * ℓ^(-α) / c
# 3. Resource: c = k^α * ℓ^(1-α) - δ*k

# From Euler equation: α * k^(α-1) * ℓ^(1-α) = (1/β - 1 + δ)
# So: k^(α-1) * ℓ^(1-α) = (1/β - 1 + δ) / α
# Taking both sides to power 1/(α-1): k * ℓ^((1-α)/(α-1)) = [(1/β - 1 + δ) / α]^(1/(α-1))
# Note: (1-α)/(α-1) = -(1-α)/(1-α) = -1
# So: k / ℓ = [(1/β - 1 + δ) / α]^(1/(α-1))
# Or: k = ℓ * [(1/β - 1 + δ) / α]^(1/(α-1))

# From intratemporal FOC: χ * ℓ^(1/ν) * c = (1-α) * k^α * ℓ^(-α)
# And from resource: c = k^α * ℓ^(1-α) - δ*k
# Substituting: χ * ℓ^(1/ν) * (k^α * ℓ^(1-α) - δ*k) = (1-α) * k^α * ℓ^(-α)
# Simplifying: χ * ℓ^(1/ν) * k^α * ℓ^(1-α) - χ * ℓ^(1/ν) * δ*k = (1-α) * k^α * ℓ^(-α)
# Dividing by k^α: χ * ℓ^(1/ν + 1-α) - χ * ℓ^(1/ν) * δ * k^(1-α) = (1-α) * ℓ^(-α)
# This is complex. Let's use a numerical solver instead.

def steady_state_system(x):
    """System of equations for steady state: x = [k, ℓ]"""
    k, ℓ = x
    # Ensure positive values
    k = max(k, 1e-6)
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    
    # Resource constraint
    c = k**α * ℓ**(1-α) - δ * k
    
    if c <= 1e-10:
        return [1e10, 1e10]
    
    # Euler equation error: 1 = β * R
    # R = α * k^(α-1) * ℓ^(1-α) + (1 - δ)
    R = α * k**(α-1) * ℓ**(1-α) + (1 - δ)
    euler_err = 1 - β * R
    
    # Intratemporal FOC error: χ * ℓ^(1/ν) = (1-α) * k^α * ℓ^(-α) / c
    lhs = χ * ℓ**(1/ν)
    rhs = (1-α) * k**α * ℓ**(-α) / c
    intratemporal_err = lhs - rhs
    
    return [euler_err, intratemporal_err]

# Initial guess: use no-labor steady state for k, reasonable labor
k_guess = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
ℓ_guess = 0.33

# Solve for steady state using root finding
from scipy.optimize import root
result = root(steady_state_system, [k_guess, ℓ_guess], method='hybr', options={'xtol': 1e-12})

if result.success:
    k_ss, ℓ_ss = result.x[0], result.x[1]
    k_ss = max(k_ss, 1e-6)
    ℓ_ss = max(ℓ_ss, 1e-6)
    ℓ_ss = min(ℓ_ss, 1.0)
else:
    # If root finding fails, try with different initial guess
    print("Warning: Root finding failed, trying alternative method...")
    from scipy.optimize import minimize
    def obj(x):
        errs = steady_state_system(x)
        return errs[0]**2 + errs[1]**2
    result_min = minimize(obj, [k_guess, ℓ_guess], method='BFGS', bounds=[(1e-6, 100), (1e-6, 1.0)])
    k_ss, ℓ_ss = result_min.x[0], result_min.x[1]

# Compute steady state consumption
c_ss = k_ss**α * ℓ_ss**(1-α) - δ * k_ss

# Verify steady state satisfies all conditions
R_ss = α * k_ss**(α-1) * ℓ_ss**(1-α) + (1 - δ)
euler_check = 1 - β * R_ss
intratemporal_check = χ * ℓ_ss**(1/ν) - (1-α) * k_ss**α * ℓ_ss**(-α) / c_ss

print(f"\nSteady-state solved numerically:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  ℓ_ss = {ℓ_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"  χ = {χ:.6f} (fixed parameter)")
print(f"  Euler check: {euler_check:.2e} (should be ~0)")
print(f"  Intratemporal check: {intratemporal_check:.2e} (should be ~0)")

# Capital domain bounds
k_low = 0.5 * k_ss
k_high = 1.5 * k_ss

# Get Chebyshev nodes in [-1, 1] domain
cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()

# Map Chebyshev nodes to economic domain [k_low, k_high]
k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)

# Number of coefficients (only for consumption, labor computed from c)
n_coeffs_c = p_k  # Consumption coefficients

# ============================================================================
# POLICY FUNCTIONS (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, gamma_c, k_low, k_high, p_k):
    """
    Returns Chebyshev approximation of consumption as a function of capital k
    DIRECT APPROXIMATION: c(k) = gamma_c @ T_k
    """
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    c = float(gamma_c @ T_k)
    c = max(c, 1e-10)  # Ensure positive
    return c

def l_from_c(k, c, χ, α, ν):
    """
    Computes labor from consumption using intratemporal FOC
    From intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * k^α * ℓ^(-α) / c
    Rearranging: ℓ^(1/ν + α) = (1-α) * k^α / (χ * c)
    So: ℓ = [(1-α) * k^α / (χ * c)]^(ν/(1+αν))
    """
    if c <= 1e-10:
        return 1e-6
    ℓ = ((1-α) * k**α / (χ * c))**(ν/(1+α*ν))
    ℓ = max(ℓ, 1e-6)   # Ensure positive (minimum labor)
    ℓ = min(ℓ, 1.0)    # Ensure labor <= 1
    return ℓ

# ============================================================================
# COMPUTE EULER ERRORS AND UPDATE POLICIES
# ============================================================================
def compute_euler_errors_and_update(c_values, k_grid, k_low, k_high, 
                                     p_k, gamma_c):
    """
    Computes Euler errors and returns updated consumption values
    Labor is computed from consumption using intratemporal FOC
    
    For each k:
    1. Compute labor from c: ℓ = [(1-α) * k^α / (χ * c)]^(ν/(1+αν))
    2. Compute output: y = k^α * ℓ^(1-α)
    3. Compute next capital: k' = (1-δ)*k + y - c
    4. Evaluate c' at k', compute ℓ' from c'
    5. Check intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * k^α * ℓ^(-α) / c
    6. Check Euler equation: 1 = β * (c/c') * (α * y'/k' + 1 - δ)
    """
    n = len(k_grid)
    c_new = np.zeros(n)
    euler_errors = np.zeros(n)
    intratemporal_errors = np.zeros(n)
    
    for i_k in range(n):
        k = k_grid[i_k]
        c = c_values[i_k]
        
        # Compute labor from consumption using intratemporal FOC
        ℓ = l_from_c(k, c, χ, α, ν)
        
        # Compute output
        y = k**α * ℓ**(1-α)
        
        # Compute next period capital from resource constraint
        k_prime = (1 - δ) * k + y - c
        
        # Ensure k_prime is positive and within reasonable bounds
        if k_prime <= 0 or k_prime > 2 * k_high:
            c_new[i_k] = c
            euler_errors[i_k] = 1e10
            intratemporal_errors[i_k] = 1e10
            continue
        
        # Evaluate consumption at k_prime, then compute labor from it
        c_prime = c_cheb(k_prime, gamma_c, k_low, k_high, p_k)
        ℓ_prime = l_from_c(k_prime, c_prime, χ, α, ν)
        
        # Compute next period output
        y_prime = k_prime**α * ℓ_prime**(1-α)
        
        # Euler equation: 1 = β * (c/c') * R'
        # R' = α * y'/k' + (1 - δ) = α * k'^(α-1) * ℓ'^(1-α) + (1 - δ)
        R_prime = α * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
        consumption_ratio = c / c_prime
        
        # Euler error: 1 - β * (c/c') * R'
        euler_error = 1 - β * consumption_ratio * R_prime
        euler_errors[i_k] = euler_error
        
        # Intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * y/ℓ * 1/c
        # Rearranging: χ * ℓ^(1+1/ν) * c = (1-α) * y
        # Or: χ * ℓ^(1/ν) = (1-α) * k^α * ℓ^(-α) / c
        lhs = χ * ℓ**(1/ν)
        rhs = (1-α) * k**α * ℓ**(-α) / c
        intratemporal_error = lhs - rhs
        intratemporal_errors[i_k] = intratemporal_error
        
        # Update consumption using Euler equation
        # From Euler: 1 = β * (c/c') * R'
        # So: c/c' = 1/(β * R')
        # This gives: c = c' / (β * R')
        # So: c_target = c' / (β * R')
        if R_prime > 0 and c_prime > 1e-10:
            c_target = c_prime / (β * R_prime)
            c_target = max(c_target, 1e-10)
            # Can't consume more than available (need to account for labor choice)
            y_max = k**α * 1.0**(1-α)  # Maximum output (labor = 1)
            c_target = min(c_target, y_max + (1-δ)*k)
            
            # Use full update (no damping here - damping applied in main loop)
            c_new[i_k] = c_target
        else:
            c_new[i_k] = c
        
        # Ensure consumption stays reasonable
        c_new[i_k] = max(c_new[i_k], 1e-10)
        # Recompute labor from new consumption to ensure consistency
        ℓ_new = l_from_c(k, c_new[i_k], χ, α, ν)
        y_new = k**α * ℓ_new**(1-α)
        c_new[i_k] = min(c_new[i_k], y_new + (1-δ)*k)
    
    return c_new, euler_errors, intratemporal_errors

# ============================================================================
# INVERT POLICIES TO GET GAMMA COEFFICIENTS
# ============================================================================
def invert_policy_to_gamma(policy_values, k_grid, k_low, k_high, p_k):
    """
    Inverts policy function values at grid points to get Chebyshev coefficients
    """
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

# ============================================================================
# SOLVE THE MODEL
# ============================================================================
print("="*80)
print("DETERMINISTIC NEOCLASSICAL GROWTH MODEL WITH LABOR - CHEBYSHEV PROJECTION")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}")
print(f"  α = {α}")
print(f"  δ = {δ}")
print(f"  χ = {χ:.6f}")
print(f"  ν = {ν}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  ℓ_ss = {ℓ_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"Capital domain: [{k_low:.6f}, {k_high:.6f}]")
print(f"Number of Chebyshev nodes: {n_k}")
print(f"Number of coefficients (c only, ℓ computed from c): {n_coeffs_c}")

# ============================================================================
# FIXED-POINT ITERATION ALGORITHM
# ============================================================================
# Initialize consumption at grid points (constant at steady state)
c_current = np.full(n_k, c_ss)

# Initialize gamma coefficients (only for consumption)
gamma_c_current = invert_policy_to_gamma(c_current, k_grid, k_low, k_high, p_k)

print(f"\nInitial consumption: c = c_ss = {c_ss:.6f} everywhere")
print(f"Labor computed from consumption using intratemporal FOC")
print(f"\nStarting fixed-point iteration...")
print("="*80)

# Fixed-point iteration parameters
max_iter = 20000
tol = 1e-8

# Adaptive dampening: start at 1 (no damping, always use new guess)
# Only reduce dampening if errors increase significantly
dampening_current = 1.0
prev_max_error = np.inf
error_increase_count = 0

for iter in range(max_iter):
    # Compute Euler errors and update consumption
    c_new, euler_errors, intratemporal_errors = compute_euler_errors_and_update(
        c_current, k_grid, k_low, k_high, p_k, gamma_c_current)
    
    # Check convergence: max Euler error
    max_euler_error = np.max(np.abs(euler_errors))
    max_intratemporal_error = np.max(np.abs(intratemporal_errors))
    mean_euler_error = np.mean(np.abs(euler_errors))
    
    # Adaptive dampening: only reduce if error increases significantly
    # Start with dampening = 1.0 (always use new guess)
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
    
    # Print progress every 10 iterations
    if iter % 10 == 0 or max_euler_error < tol:
        print(f"Iteration {iter:4d}: max |Euler error| = {max_euler_error:.6e}, "
              f"max |Intratemporal error| = {max_intratemporal_error:.6e}, "
              f"dampening = {dampening_current:.4f}")
    
    # Check convergence
    if max_euler_error < tol and max_intratemporal_error < tol:
        print(f"\nConverged after {iter} iterations!")
        break
    
    # Update consumption with adaptive damping (always use dampening_current)
    c_current = (1 - dampening_current) * c_current + dampening_current * c_new
    
    # Invert consumption to get new gamma coefficients
    gamma_c_current = invert_policy_to_gamma(c_current, k_grid, k_low, k_high, p_k)
    
    # Recompute consumption from gamma to ensure consistency
    c_current = np.array([c_cheb(k, gamma_c_current, k_low, k_high, p_k) for k in k_grid])

if iter == max_iter - 1:
    print(f"\nWarning: Reached maximum iterations ({max_iter})")

gamma_c_opt = gamma_c_current
print(f"\nFinal max |Euler error|: {max_euler_error:.6e}")
print(f"Final max |Intratemporal error|: {max_intratemporal_error:.6e}")
print(f"Final mean |Euler error|: {mean_euler_error:.6e}")

# ============================================================================
# PLOT RESULTS
# ============================================================================
print("\nGenerating plots...")

# Fine grid for plotting
k_grid_fine = np.linspace(k_low, k_high, 200)

# Evaluate policy functions on fine grid
c_policy = np.zeros_like(k_grid_fine)
l_policy = np.zeros_like(k_grid_fine)
k_prime_policy = np.zeros_like(k_grid_fine)

for i in range(len(k_grid_fine)):
    k = k_grid_fine[i]
    c_policy[i] = c_cheb(k, gamma_c_opt, k_low, k_high, p_k)
    l_policy[i] = l_from_c(k, c_policy[i], χ, α, ν)
    y = k**α * l_policy[i]**(1-α)
    k_prime_policy[i] = (1 - δ) * k + y - c_policy[i]

# Create 1x3 figure for policy functions (fancy scientific plot)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Consumption policy function
ax1 = axes[0]
ax1.plot(k_grid_fine, c_policy, 'b-', linewidth=3, label='Policy Function', alpha=0.9)
ax1.scatter(k_grid, [c_cheb(k, gamma_c_opt, k_low, k_high, p_k) for k in k_grid], 
           c='darkred', s=120, marker='o', edgecolors='black', linewidths=2, 
           label=f'Collocation Nodes (n={n_k})', zorder=5, alpha=0.8)
ax1.axvline(k_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7, label='Steady-State')
ax1.axhline(c_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.scatter([k_ss], [c_ss], s=400, marker='*', color='gold', 
           edgecolors='black', linewidths=3, zorder=15, alpha=0.95)
ax1.set_xlabel('Capital, $k$', fontsize=14, fontweight='bold')
ax1.set_ylabel('Consumption, $c(k)$', fontsize=14, fontweight='bold')
ax1.set_title('(a) Consumption Policy Function', fontsize=15, fontweight='bold', pad=10)
ax1.legend(fontsize=11, framealpha=0.95, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Labor policy function
ax2 = axes[1]
ax2.plot(k_grid_fine, l_policy, 'r-', linewidth=3, label='Policy Function', alpha=0.9)
ax2.scatter(k_grid, [l_from_c(k, c_cheb(k, gamma_c_opt, k_low, k_high, p_k), χ, α, ν) for k in k_grid], 
           c='darkred', s=120, marker='o', edgecolors='black', linewidths=2, 
           label=f'Collocation Nodes (n={n_k})', zorder=5, alpha=0.8)
ax2.axvline(k_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7, label='Steady-State')
ax2.axhline(ℓ_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7)
ax2.scatter([k_ss], [ℓ_ss], s=400, marker='*', color='gold', 
           edgecolors='black', linewidths=3, zorder=15, alpha=0.95)
ax2.set_xlabel('Capital, $k$', fontsize=14, fontweight='bold')
ax2.set_ylabel('Labor Supply, $\\ell(k)$', fontsize=14, fontweight='bold')
ax2.set_title('(b) Labor Policy Function', fontsize=15, fontweight='bold', pad=10)
ax2.legend(fontsize=11, framealpha=0.95, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: Capital transition function
ax3 = axes[2]
ax3.plot(k_grid_fine, k_prime_policy, 'b-', linewidth=3, label="Policy Function $k' = f(k)$", alpha=0.9)
ax3.plot([k_low, k_high], [k_low, k_high], 'r--', linewidth=2.5, label='45° Line', alpha=0.7)
ax3.scatter(k_grid, [(1-δ)*k + k**α * l_from_c(k, c_cheb(k, gamma_c_opt, k_low, k_high, p_k), χ, α, ν)**(1-α) 
                    - c_cheb(k, gamma_c_opt, k_low, k_high, p_k) for k in k_grid], 
           c='darkred', s=120, marker='o', edgecolors='black', linewidths=2, 
           label=f'Collocation Nodes (n={n_k})', zorder=5, alpha=0.8)
ax3.axvline(k_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7)
ax3.axhline(k_ss, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7, label='Steady-State')
ax3.scatter([k_ss], [k_ss], s=400, marker='*', color='gold', 
           edgecolors='black', linewidths=3, zorder=15, alpha=0.95)
ax3.set_xlabel('Current Capital, $k$', fontsize=14, fontweight='bold')
ax3.set_ylabel("Next Period Capital, $k'$", fontsize=14, fontweight='bold')
ax3.set_title('(c) Capital Transition Function', fontsize=15, fontweight='bold', pad=10)
ax3.legend(fontsize=11, framealpha=0.95, loc='lower right')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()

# Create output directory
output_dir = 'NGM_deterministic_labor'
os.makedirs(output_dir, exist_ok=True)

# Save figure
output_path = os.path.join(output_dir, 'policy_functions.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: {output_path}")
plt.close()

# Also compute Euler errors for later use
euler_errors_fine = []
for k in k_grid:
    c = c_cheb(k, gamma_c_opt, k_low, k_high, p_k)
    ℓ = l_from_c(k, c, χ, α, ν)
    y = k**α * ℓ**(1-α)
    k_prime = (1 - δ) * k + y - c
    c_prime = c_cheb(k_prime, gamma_c_opt, k_low, k_high, p_k)
    ℓ_prime = l_from_c(k_prime, c_prime, χ, α, ν)
    y_prime = k_prime**α * ℓ_prime**(1-α)
    R_prime = α * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
    consumption_ratio = c / c_prime
    euler_error = 1 - β * consumption_ratio * R_prime
    euler_errors_fine.append(euler_error)

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Max Euler error: {max(np.abs(euler_errors_fine)):.6e}")
print(f"Mean Euler error: {np.mean(np.abs(euler_errors_fine)):.6e}")
print(f"RMS Euler error: {np.sqrt(np.mean(np.array(euler_errors_fine)**2)):.6e}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  ℓ_ss = {ℓ_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"  Policy functions at k_ss:")
c_ss_approx = c_cheb(k_ss, gamma_c_opt, k_low, k_high, p_k)
ℓ_ss_approx = l_from_c(k_ss, c_ss_approx, χ, α, ν)
print(f"    c({k_ss:.6f}) = {c_ss_approx:.6f} (error: {abs(c_ss_approx - c_ss):.6e})")
print(f"    ℓ({k_ss:.6f}) = {ℓ_ss_approx:.6f} (error: {abs(ℓ_ss_approx - ℓ_ss):.6e})")

print("\nDone!")

