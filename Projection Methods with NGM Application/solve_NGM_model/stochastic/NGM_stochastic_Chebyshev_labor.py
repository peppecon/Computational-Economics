#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Neoclassical Growth Model with Productivity Shocks and Labor Supply
State variables: capital k and productivity z
Uses Chebyshev polynomial projection method
DIRECT APPROXIMATION: Approximates c(k,z) and ℓ(k,z) directly without log-exp transformation

Utility: U(c, ℓ) = log(c) - χ * ℓ^(1+1/ν)/(1+1/ν)
Production: y = z * k^α * ℓ^(1-α)
"""

import numpy as np
from scipy import optimize as opt
from scipy.optimize import root
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
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
β = 0.99     # Discount factor
α = 0.33     # Capital share
δ = 0.025    # Depreciation rate
χ = 8.042775      # Labor disutility parameter
ν = 1.0      # Frisch elasticity parameter (1/ν is the inverse Frisch elasticity)
ρ = 0.95     # Persistence of productivity shock
σ = 0.02    # Standard deviation of productivity shock

# Damping parameter for iteration
dampening = 1.0  # Damping parameter for consumption updates

# ============================================================================
# CHEBYSHEV APPROXIMATION SETUP
# ============================================================================
n_k = 10      # Number of Chebyshev nodes for capital
n_z = 10      # Number of Chebyshev nodes for productivity
p_k = n_k     # Polynomial order for capital
p_z = n_z     # Polynomial order for productivity

# Calculate steady state (with labor, assuming z=1)
z_ss = 1.0

def steady_state_system(x):
    """System of equations for steady state: x = [k, ℓ]"""
    k, ℓ = x
    # Ensure positive values
    k = max(k, 1e-6)
    ℓ = max(ℓ, 1e-6)
    ℓ = min(ℓ, 1.0)
    
    # Resource constraint
    c = z_ss * k**α * ℓ**(1-α) - δ * k
    
    if c <= 1e-10:
        return [1e10, 1e10]
    
    # Euler equation error: 1 = β * R
    # R = α * z * k^(α-1) * ℓ^(1-α) + (1 - δ)
    R = α * z_ss * k**(α-1) * ℓ**(1-α) + (1 - δ)
    euler_err = 1 - β * R
    
    # Intratemporal FOC error: χ * ℓ^(1/ν) = (1-α) * z * k^α * ℓ^(-α) / c
    lhs = χ * ℓ**(1/ν)
    rhs = (1-α) * z_ss * k**α * ℓ**(-α) / c
    intratemporal_err = lhs - rhs
    
    return [euler_err, intratemporal_err]

# Initial guess: use no-labor steady state for k, reasonable labor
k_guess = (β * α / (1 - β * (1 - δ)))**(1 / (1 - α))
ℓ_guess = 0.33

# Solve for steady state using root finding
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
c_ss = z_ss * k_ss**α * ℓ_ss**(1-α) - δ * k_ss

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

# Number of coefficients (tensor product) - only for consumption
# Labor is computed from consumption using intratemporal FOC
n_coeffs = p_k * p_z

# Gauss-Hermite quadrature for expectations
n_q = 5  # Number of quadrature nodes
gh_quad = np.polynomial.hermite.hermgauss(n_q)
q_nodes, q_weights = gh_quad

# ============================================================================
# POLICY FUNCTIONS (Chebyshev approximation - DIRECT)
# ============================================================================
def c_cheb(k, z, gamma_c, k_low, k_high, z_low_log, z_high_log, p_k, p_z):
    """
    Returns Chebyshev approximation of consumption as a function of (k, z)
    DIRECT APPROXIMATION: c(k,z) = gamma_c @ T_kz
    """
    # Transform to Chebyshev domain [-1, 1]
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    k_cheb = np.clip(k_cheb, -1.0, 1.0)
    
    z_cheb = Change_Variable_Tocheb(z_low_log, z_high_log, np.log(z))
    z_cheb = np.clip(z_cheb, -1.0, 1.0)
    
    # Evaluate Chebyshev polynomials
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), p_k)
    T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), p_z)
    kron_kz = np.kron(T_k, T_z)
    
    # Direct approximation
    c = float(gamma_c @ kron_kz)
    c = max(c, 1e-10)  # Ensure positive
    
    return c

def l_from_c(k, z, c, χ, α, ν):
    """
    Computes labor from consumption using intratemporal FOC
    From intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * z * k^α * ℓ^(-α) / c
    Rearranging: χ * ℓ^(1/ν + α) = (1-α) * z * k^α / c
    So: ℓ = [(1-α) * z * k^α / (χ * c)]^(ν/(1+αν))
    """
    if c <= 1e-10:
        return 1e-6
    ℓ = ((1-α) * z * k**α / (χ * c))**(ν / (1 + α*ν))
    ℓ = max(ℓ, 1e-6)   # Ensure positive
    ℓ = min(ℓ, 1.0)    # Ensure ≤ 1
    return ℓ

# ============================================================================
# COMPUTE EULER ERRORS AND UPDATE CONSUMPTION/LABOR
# ============================================================================
def compute_euler_errors_and_update(c_values, k_grid, z_grid, k_low, k_high, 
                                     z_low_log, z_high_log, p_k, p_z, gamma_c):
    """
    Computes Euler errors and returns updated consumption and labor values
    Uses Gauss-Hermite quadrature for expectations over productivity shocks
    """
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
            c = c_values[idx]
            # Compute labor from consumption to ensure intratemporal FOC is satisfied
            ℓ = ((1-α) * z * k**α / (χ * c))**(ν / (1 + α*ν))
            ℓ = max(ℓ, 1e-6)
            ℓ = min(ℓ, 1.0)
            
            # Compute next period capital from resource constraint
            k_prime = (1 - δ) * k + z * k**α * ℓ**(1-α) - c
            
            # Compute expectation over productivity shocks using Gauss-Hermite quadrature
            E = 0.0
            for i_q in range(len(q_nodes)):
                # Transform GH node to shock
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]
                # Next period productivity: AR(1) in logs
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                
                # Evaluate consumption at (k_prime, z_prime), then compute labor from it
                c_prime = c_cheb(k_prime, z_prime, gamma_c, k_low, k_high, 
                                 z_low_log, z_high_log, p_k, p_z)
                ℓ_prime = l_from_c(k_prime, z_prime, c_prime, χ, α, ν)
                
                # Return on capital
                R_prime = α * z_prime * k_prime**(α - 1) * ℓ_prime**(1-α) + (1 - δ)
                
                # Accumulate expectation (using log utility: u'(c) = 1/c)
                E += q_weights[i_q] * β * (1/c_prime) * R_prime
            
            # Normalize expectation
            E = E / np.sqrt(np.pi)
            
            # Euler error: E - u'(c) = 0, where u'(c) = 1/c for log utility
            euler_error = E - (1/c)
            euler_errors[idx] = euler_error
            
            # Update consumption using Euler equation
            if E > 0:
                c_target = 1 / E
                c_target = max(c_target, 1e-10)
                c_target = min(c_target, z * k**α * ℓ**(1-α) + (1-δ)*k)
                c_new[idx] = (1 - dampening) * c + dampening * c_target
            else:
                c_new[idx] = c
            
            # Ensure consumption stays positive and reasonable
            c_new[idx] = max(c_new[idx], 1e-10)
            c_new[idx] = min(c_new[idx], z * k**α * ℓ**(1-α) + (1-δ)*k)
            
            # Update labor using intratemporal FOC
            # Labor is computed directly from consumption to satisfy FOC exactly
            # From intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * z * k^α * ℓ^(-α) / c
            # Rearranging: χ * ℓ^(1/ν + α) = (1-α) * z * k^α / c
            # So: ℓ = [(1-α) * z * k^α / (χ * c)]^(ν/(1+αν))
            if c_new[idx] > 1e-10:
                l_new[idx] = ((1-α) * z * k**α / (χ * c_new[idx]))**(ν / (1 + α*ν))
                l_new[idx] = max(l_new[idx], 1e-6)
                l_new[idx] = min(l_new[idx], 1.0)
            else:
                l_new[idx] = ℓ
            
            # Compute intratemporal error AFTER updating consumption and labor
            # Since labor is computed from consumption, this should be zero (up to clipping effects)
            # NOTE: If labor is clipped to bounds (ℓ = 1e-6 or ℓ = 1.0), the FOC may be violated
            # This is expected behavior when constraints bind - the error reflects the shadow cost
            lhs = χ * l_new[idx]**(1/ν)
            rhs = (1-α) * z * k**α * l_new[idx]**(-α) / c_new[idx]
            intratemporal_error = lhs - rhs
            intratemporal_errors[idx] = intratemporal_error
            
            # If labor is at bounds, the error is expected to be non-zero
            # This is the shadow cost of the constraint
            if abs(l_new[idx] - 1.0) < 1e-6 or abs(l_new[idx] - 1e-6) < 1e-6:
                # Labor is at a bound - error is expected
                pass
            
            idx += 1
    
    return c_new, l_new, euler_errors, intratemporal_errors

# ============================================================================
# INVERT CONSUMPTION/LABOR TO GET GAMMA COEFFICIENTS
# ============================================================================
def invert_to_gamma(values, k_grid, z_grid, k_low, k_high, 
                    z_low_log, z_high_log, p_k, p_z):
    """
    Inverts consumption or labor values at grid points to get Chebyshev coefficients
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
    
    # Solve for gamma
    if n == p_k * p_z:
        gamma = np.linalg.solve(T_matrix, values)
    else:
        gamma = np.linalg.lstsq(T_matrix, values, rcond=None)[0]
    
    return gamma

# ============================================================================
# SOLVE THE MODEL
# ============================================================================
print("="*80)
print("STOCHASTIC NEOCLASSICAL GROWTH MODEL WITH LABOR - CHEBYSHEV PROJECTION")
print("="*80)
print(f"\nModel Parameters:")
print(f"  β = {β}")
print(f"  α = {α}")
print(f"  δ = {δ}")
print(f"  χ = {χ}")
print(f"  ν = {ν}")
print(f"  ρ = {ρ}")
print(f"  σ = {σ}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  ℓ_ss = {ℓ_ss:.6f}")
print(f"  z_ss = {z_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"Capital domain: [{k_low:.6f}, {k_high:.6f}]")
print(f"Productivity domain (log): [{z_low_log:.6f}, {z_high_log:.6f}]")
print(f"Number of Chebyshev nodes: k={n_k}, z={n_z}")
print(f"Number of coefficients: {n_coeffs} (for consumption only)")
print(f"Gauss-Hermite quadrature nodes: {n_q}")
print(f"\nNOTE: Using DIRECT approximation c(k,z) = gamma_c @ T_kz")
print(f"NOTE: Labor ℓ(k,z) is computed from consumption using intratemporal FOC")
print(f"NOTE: Clipping k_cheb and z_cheb to [-1, 1] when computing policy functions")

# ============================================================================
# FIXED-POINT ITERATION ALGORITHM
# ============================================================================
# Initialize consumption at grid points (constant at steady state)
c_current = np.full(n_k * n_z, c_ss)

# Initialize gamma coefficients (only for consumption)
gamma_c_current = invert_to_gamma(c_current, k_grid, z_grid, 
                                  k_low, k_high, z_low_log, z_high_log, p_k, p_z)

# Compute labor from consumption using intratemporal FOC
l_current = np.zeros(n_k * n_z)
idx = 0
for i_z in range(n_z):
    z = z_grid[i_z]
    for i_k in range(n_k):
        k = k_grid[i_k]
        l_current[idx] = l_from_c(k, z, c_current[idx], χ, α, ν)
        idx += 1

print(f"\nInitial consumption: c = c_ss = {c_ss:.6f} everywhere")
print(f"Initial labor: ℓ = ℓ_ss = {ℓ_ss:.6f} everywhere")
print(f"\nStarting fixed-point iteration...")
print("="*80)

# Fixed-point iteration parameters
max_iter = 2000
tol = 1e-8

for iter in range(max_iter):
    # Compute Euler errors and update consumption (labor computed from consumption)
    c_new, l_new, euler_errors, intratemporal_errors = compute_euler_errors_and_update(
        c_current, k_grid, z_grid, k_low, k_high, 
        z_low_log, z_high_log, p_k, p_z, gamma_c_current)
    
    # Check convergence: max Euler error
    max_euler_error = np.max(np.abs(euler_errors))
    max_intratemporal_error = np.max(np.abs(intratemporal_errors))
    max_error = max(max_euler_error, max_intratemporal_error)
    mean_error = np.mean(np.abs(euler_errors)) + np.mean(np.abs(intratemporal_errors))
    
    # Print progress every 10 iterations
    if iter % 10 == 0 or max_error < tol:
        print(f"Iteration {iter:4d}: max |Euler error| = {max_euler_error:.6e}, "
              f"max |Intratemporal error| = {max_intratemporal_error:.6e}")
    
    # Check convergence
    if max_error < tol:
        print(f"\nConverged after {iter} iterations!")
        break
    
    # Update consumption and labor with damping
    if max_error < 1.0:
        c_current = (1 - dampening) * c_current + dampening * c_new
        l_current = (1 - dampening) * l_current + dampening * l_new
    else:
        c_current = (1 - dampening * 0.1) * c_current + dampening * 0.1 * c_new
        l_current = (1 - dampening * 0.1) * l_current + dampening * 0.1 * l_new
    
    # Invert consumption to get new gamma coefficients
    gamma_c_current = invert_to_gamma(c_current, k_grid, z_grid, 
                                      k_low, k_high, z_low_log, z_high_log, p_k, p_z)
    
    # Recompute consumption from gamma, then compute labor from consumption
    # This ensures intratemporal FOC is always satisfied exactly
    idx = 0
    for i_z in range(n_z):
        z = z_grid[i_z]
        for i_k in range(n_k):
            k = k_grid[i_k]
            c_current[idx] = c_cheb(k, z, gamma_c_current, k_low, k_high, 
                                   z_low_log, z_high_log, p_k, p_z)
            # Compute labor from consumption using intratemporal FOC
            # This ensures the FOC is satisfied exactly
            l_current[idx] = l_from_c(k, z, c_current[idx], χ, α, ν)
            idx += 1

if iter == max_iter - 1:
    print(f"\nWarning: Reached maximum iterations ({max_iter})")

gamma_c_opt = gamma_c_current
print(f"\nFinal max |Euler error|: {max_euler_error:.6e}")
print(f"Final max |Intratemporal error|: {max_intratemporal_error:.6e}")
print(f"Optimal coefficients (first 5): gamma_c = {gamma_c_opt[:5]}")

# ============================================================================
# PLOT RESULTS - 3D SURFACE PLOTS
# ============================================================================
print("\nGenerating 3D surface plots...")

# Fine grid for 3D plots
k_grid_3d = np.linspace(k_low, k_high, 50)
z_grid_3d = np.exp(np.linspace(z_low_log, z_high_log, 50))

# Generate meshgrid
K_3d, Z_3d = np.meshgrid(k_grid_3d, z_grid_3d)
C_3d = np.zeros_like(K_3d)
L_3d = np.zeros_like(K_3d)
K_prime_3d = np.zeros_like(K_3d)

# Evaluate policy functions on 3D grid
for i in range(K_3d.shape[0]):
    for j in range(K_3d.shape[1]):
        k_val = K_3d[i, j]
        z_val = Z_3d[i, j]
        C_3d[i, j] = c_cheb(k_val, z_val, gamma_c_opt, k_low, k_high, 
                           z_low_log, z_high_log, p_k, p_z)
        L_3d[i, j] = l_from_c(k_val, z_val, C_3d[i, j], χ, α, ν)
        # Compute capital accumulation
        K_prime_3d[i, j] = (1 - δ) * k_val + z_val * k_val**α * L_3d[i, j]**(1-α) - C_3d[i, j]

# Create figure with three 3D subplots
fig_3d = plt.figure(figsize=(24, 8))

# Plot 1: Consumption policy function
ax1_3d = fig_3d.add_subplot(131, projection='3d')
surf1_3d = ax1_3d.plot_surface(K_3d, Z_3d, C_3d, cmap='viridis', 
                                alpha=0.9, linewidth=0, antialiased=True)

# Add scatter points for Chebyshev nodes
idx = 0
for i_z in range(n_z):
    z_node = z_grid[i_z]
    for i_k in range(n_k):
        k_node = k_grid[i_k]
        c_node = c_cheb(k_node, z_node, gamma_c_opt, k_low, k_high, 
                       z_low_log, z_high_log, p_k, p_z)
        ax1_3d.scatter([k_node], [z_node], [c_node], 
                      c='red', s=80, marker='o', edgecolors='black', 
                      linewidths=1, zorder=10, alpha=0.7)

# Add steady-state point
c_ss_actual = c_cheb(k_ss, z_ss, gamma_c_opt, k_low, k_high, 
                     z_low_log, z_high_log, p_k, p_z)
ax1_3d.scatter([k_ss], [z_ss], [c_ss_actual], 
              c='green', s=300, marker='*', edgecolors='black', 
              linewidths=2, zorder=15)

ax1_3d.set_xlabel('k (Capital)', fontsize=11, fontweight='bold')
ax1_3d.set_ylabel('z (Productivity)', fontsize=11, fontweight='bold')
ax1_3d.set_zlabel('c (Consumption)', fontsize=11, fontweight='bold')
ax1_3d.set_title('Consumption Policy Function', fontsize=12, fontweight='bold')
fig_3d.colorbar(surf1_3d, ax=ax1_3d, shrink=0.6, aspect=15, label='Consumption')
ax1_3d.view_init(elev=30, azim=45)

# Plot 2: Labor policy function
ax2_3d = fig_3d.add_subplot(132, projection='3d')
surf2_3d = ax2_3d.plot_surface(K_3d, Z_3d, L_3d, cmap='plasma', 
                                alpha=0.9, linewidth=0, antialiased=True)

# Add scatter points for Chebyshev nodes
idx = 0
for i_z in range(n_z):
    z_node = z_grid[i_z]
    for i_k in range(n_k):
        k_node = k_grid[i_k]
        c_node = c_cheb(k_node, z_node, gamma_c_opt, k_low, k_high, 
                       z_low_log, z_high_log, p_k, p_z)
        l_node = l_from_c(k_node, z_node, c_node, χ, α, ν)
        ax2_3d.scatter([k_node], [z_node], [l_node], 
                      c='red', s=80, marker='o', edgecolors='black', 
                      linewidths=1, zorder=10, alpha=0.7)

# Add steady-state point
l_ss_actual = l_from_c(k_ss, z_ss, c_ss_actual, χ, α, ν)
ax2_3d.scatter([k_ss], [z_ss], [l_ss_actual], 
              c='green', s=300, marker='*', edgecolors='black', 
              linewidths=2, zorder=15)

ax2_3d.set_xlabel('k (Capital)', fontsize=11, fontweight='bold')
ax2_3d.set_ylabel('z (Productivity)', fontsize=11, fontweight='bold')
ax2_3d.set_zlabel('ℓ (Labor)', fontsize=11, fontweight='bold')
ax2_3d.set_title('Labor Policy Function', fontsize=12, fontweight='bold')
fig_3d.colorbar(surf2_3d, ax=ax2_3d, shrink=0.6, aspect=15, label='Labor')
ax2_3d.view_init(elev=30, azim=45)

# Plot 3: Capital accumulation function
ax3_3d = fig_3d.add_subplot(133, projection='3d')
surf3_3d = ax3_3d.plot_surface(K_3d, Z_3d, K_prime_3d, cmap='coolwarm', 
                                alpha=0.9, linewidth=0, antialiased=True)

# Add scatter points for Chebyshev nodes
idx = 0
for i_z in range(n_z):
    z_node = z_grid[i_z]
    for i_k in range(n_k):
        k_node = k_grid[i_k]
        c_node = c_cheb(k_node, z_node, gamma_c_opt, k_low, k_high, 
                       z_low_log, z_high_log, p_k, p_z)
        l_node = l_from_c(k_node, z_node, c_node, χ, α, ν)
        k_prime_node = (1 - δ) * k_node + z_node * k_node**α * l_node**(1-α) - c_node
        ax3_3d.scatter([k_node], [z_node], [k_prime_node], 
                      c='red', s=80, marker='o', edgecolors='black', 
                      linewidths=1, zorder=10, alpha=0.7)

# Add steady-state point (where k' = k)
k_prime_ss = (1 - δ) * k_ss + z_ss * k_ss**α * l_ss_actual**(1-α) - c_ss_actual
ax3_3d.scatter([k_ss], [z_ss], [k_ss], 
              c='green', s=300, marker='*', edgecolors='black', 
              linewidths=2, zorder=15)

# Add 45-degree line reference (k' = k plane)
k_line = np.linspace(k_low, k_high, 20)
z_line = np.linspace(z_grid_3d.min(), z_grid_3d.max(), 20)
K_line, Z_line = np.meshgrid(k_line, z_line)
K_prime_line = K_line  # k' = k line
ax3_3d.plot_wireframe(K_line, Z_line, K_prime_line, alpha=0.3, 
                      color='gray', linewidth=0.5)

ax3_3d.set_xlabel('k (Current Capital)', fontsize=11, fontweight='bold')
ax3_3d.set_ylabel('z (Productivity)', fontsize=11, fontweight='bold')
ax3_3d.set_zlabel("k' (Next Period Capital)", fontsize=11, fontweight='bold')
ax3_3d.set_title('Capital Accumulation Function', fontsize=12, fontweight='bold')
fig_3d.colorbar(surf3_3d, ax=ax3_3d, shrink=0.6, aspect=15, label="k'")
ax3_3d.view_init(elev=30, azim=45)

plt.tight_layout()

# Save 3D figure
output_path_3d = '../NGM_figures/stochastic/stochastic_Chebyshev_labor_3d_surface.png'
os.makedirs('../NGM_figures/stochastic', exist_ok=True)
plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
print(f"✓ Saved 3D plot: {output_path_3d}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Max Euler error: {np.max(np.abs(euler_errors)):.6e}")
print(f"Mean Euler error: {np.mean(np.abs(euler_errors)):.6e}")
print(f"Max Intratemporal error: {np.max(np.abs(intratemporal_errors)):.6e}")
print(f"Mean Intratemporal error: {np.mean(np.abs(intratemporal_errors)):.6e}")
print(f"\nSteady-state values:")
print(f"  k_ss = {k_ss:.6f}")
print(f"  ℓ_ss = {ℓ_ss:.6f}")
print(f"  z_ss = {z_ss:.6f}")
print(f"  c_ss = {c_ss:.6f}")
print(f"  Policy function at (k_ss, z_ss):")
print(f"    c({k_ss:.6f}, {z_ss:.6f}) = {c_ss_actual:.6f}")
print(f"    ℓ({k_ss:.6f}, {z_ss:.6f}) = {l_ss_actual:.6f}")

print("\nDone!")

