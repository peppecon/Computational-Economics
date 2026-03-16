#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Computational imports
import numpy as np
from scipy import optimize as opt
from numba import njit
import sys
import os

# Add parent directory to path to import functions_library (for Chebyshev)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'scripts'))
try:
    from functions_library import *
except ImportError:
    pass  # Will only be needed for Chebyshev version

# Graphics imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # Better quality figures
except ImportError:
    pass
from matplotlib import rcParams
rcParams['figure.figsize'] = (9, 6)  # Sets the size of the figures in the notebook
from matplotlib import cm # for 3d poltting
from mpl_toolkits.mplot3d.axes3d import Axes3D # for 3d poltting

# ============================================================================
# METHOD 1: LOG-POLYNOMIAL APPROXIMATION (ORIGINAL)
# ============================================================================

@njit
def c_poly(k,z,η):
    """
    Returns a polynomial approximation of consumption as a function of the state (k,z)
    """
    
    c_log = η[0] + η[1]*np.log(k) + η[2]*np.log(z) + η[3]*np.log(k)**2 + η[4]*np.log(z)**2 + η[5]*np.log(k)*np.log(z)
    
    return np.exp(c_log)

n_q = 5  # Number of nodes and weights for the Gauss-Hermite quadrature

# Use the hermgauss function to get the nodes and the weights for the Gauss-Hermite quadrature
gh_quad = np.polynomial.hermite.hermgauss(n_q)

@njit
def euler_err(η, quad, k_grid, z_grid):
    """
    Returns the sum of squared Euler errors at all grid points
    
    """
    
    q_nodes, q_weights = quad
    β = 0.99
    α = 0.33
    δ = 0.025
    γ = 1
    ρ = 0.95
    σ = 0.1
    ssr      =  0  # Initialize the sum of squared errors
    
    for i_k in range(len(k_grid)):  # Iterate over k and z grids
        
        for i_z in range(len(z_grid)):
            
            k       = k_grid[i_k]
            z       = z_grid[i_z]
            c       = c_poly(k,z,η)
            k_prime = z * k**α + (1-δ) * k - c;
            
            # Calculating the expectation over the GH nodes for every (k,z) weighted by the GH weights
            # We use the Gauss-Hermite formula with a change of variable
            
            E  = 0
            
            for i_q in range(len(q_nodes)):
                
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # The errors are normally distributed with mean 0 and std σ
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                c_prime = c_poly(k_prime,z_prime,η)
                
                E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))            
                
            E = E / np.sqrt(np.pi)      
            ssr += (E - c**(-γ))**2

    return ssr

β = 0.99
α = 0.33
δ = 0.025
γ = 4
ρ = 0.95
σ = 0.1

# Calculate the steady state level of capital
k_ss = (β * α/(1-β*(1-δ)))**(1/(1-α))

# Setting up the capital grid
k_low    =  0.5 * k_ss
k_high   =  1.5 * k_ss
n_k =  10
k_grid = np.linspace(k_low,k_high,n_k)

# Setting up the productivity grid (3 std)
z_low    = -3 * np.sqrt(σ**2/(1-ρ**2))
z_high   =  3 * np.sqrt(σ**2/(1-ρ**2))
n_z =  10 
z_grid  = np.exp(np.linspace(z_low,z_high,n_z))

# Set initial values for the coefficients
η_init = np.zeros(6)

# Find solution by minimizing the errors on the grid
η_opt = opt.minimize(euler_err, η_init, args=(gh_quad, k_grid, z_grid), 
                     method='Nelder-Mead', options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print("Optimal coefficients:", η_opt)


k_grid_fine = np.linspace(k_low,k_high,100)
z_grid_fine = np.exp(np.linspace(z_low,z_high,100))

# Generate meshgrid coordinates for 3d plot
kg, zg = np.meshgrid(k_grid_fine, z_grid_fine)

# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kg,
                zg,
                c_poly(kg,zg,η_opt),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
ax.set_zlabel('c', fontsize=14)
ax.set_title('Consumption Policy Function (QE Original)', fontsize=14, fontweight='bold')

# Save figure instead of showing
output_path = '../NGM_figures/QE_original_policy_function.png'
os.makedirs('NGM_figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot: {output_path}")
plt.close()

# ============================================================================
# METHOD 2: CHEBYSHEV POLYNOMIAL APPROXIMATION
# ============================================================================

print("\n" + "="*80)
print("CHEBYSHEV POLYNOMIAL APPROXIMATION")
print("="*80)

def c_cheb(k, z, gamma, k_low, k_high, z_low_log, z_high_log, n_k_cheb, n_z_cheb):
    """
    Returns Chebyshev approximation of consumption as a function of the state (k,z)
    Approximates consumption directly (no log/exp transformation)
    """
    # Transform to Chebyshev domain [-1, 1]
    k_cheb = Change_Variable_Tocheb(k_low, k_high, k)
    # For productivity: transform log(z) to [-1, 1] domain (since productivity is AR(1) in logs)
    z_cheb = Change_Variable_Tocheb(z_low_log, z_high_log, np.log(z))
    
    # Evaluate Chebyshev polynomials
    T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), n_k_cheb)
    T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), n_z_cheb)
    
    # Tensor product: kron_kz shape is (n_k_cheb * n_z_cheb, 1)
    # np.kron orders as: (k0,z0), (k1,z0), ..., (k_{n_k-1},z0), (k0,z1), (k1,z1), ...
    kron_kz = np.kron(T_k, T_z)
    
    # Compute consumption directly: kron_kz @ gamma (transpose operation)
    # kron_kz is (n_coeffs, 1), gamma is (n_coeffs,), so kron_kz.T @ gamma gives scalar
    c = float(kron_kz.T @ gamma)
    
    # Ensure consumption is positive
    c = max(c, 1e-10)
    
    return c

# Use same quadrature setup
n_q_cheb = 5  # Number of nodes and weights for the Gauss-Hermite quadrature
gh_quad_cheb = np.polynomial.hermite.hermgauss(n_q_cheb)

def euler_err_cheb(gamma, quad, k_grid, z_grid, k_low, k_high, z_low_log, z_high_log, n_k_cheb, n_z_cheb):
    """
    Returns the sum of squared Euler errors at all grid points (Chebyshev version)
    
    """
    
    q_nodes, q_weights = quad
    β = 0.99
    α = 0.33
    δ = 0.025
    γ = 1
    ρ = 0.95
    σ = 0.1
    ssr      =  0  # Initialize the sum of squared errors
    
    # CRITICAL: Loop order must match np.kron(T_k, T_z) ordering
    # np.kron orders as: (k0,z0), (k1,z0), ..., (k_{n_k-1},z0), (k0,z1), (k1,z1), ...
    # So iterate: z outer, k inner
    for i_z in range(len(z_grid)):  # Iterate over z first (outer loop)
        
        for i_k in range(len(k_grid)):  # Then k (inner loop)
            
            k       = k_grid[i_k]
            z       = z_grid[i_z]
            c       = c_cheb(k, z, gamma, k_low, k_high, z_low_log, z_high_log, n_k_cheb, n_z_cheb)
            k_prime = z * k**α + (1-δ) * k - c
            
            # Ensure k_prime is positive
            if k_prime <= 0:
                return 1e10
            
            # Calculating the expectation over the GH nodes for every (k,z) weighted by the GH weights
            # We use the Gauss-Hermite formula with a change of variable
            
            E  = 0
            
            for i_q in range(len(q_nodes)):
                
                e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # The errors are normally distributed with mean 0 and std σ
                z_prime = np.exp(ρ * np.log(z) + e_prime)
                c_prime = c_cheb(k_prime, z_prime, gamma, k_low, k_high, z_low_log, z_high_log, n_k_cheb, n_z_cheb)
                
                E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))            
                
            E = E / np.sqrt(np.pi)      
            ssr += (E - c**(-γ))**2

    return ssr

# Use same parameters (module level, but euler_err_cheb defines its own inside)
β_cheb = 0.99
α_cheb = 0.33
δ_cheb = 0.025
γ_cheb = 4
ρ_cheb = 0.95
σ_cheb = 0.1

# Calculate the steady state level of capital (same as original)
k_ss_cheb = (β_cheb * α_cheb/(1-β_cheb*(1-δ_cheb)))**(1/(1-α_cheb))

# Setting up the capital domain (same bounds as original)
k_low_cheb    =  0.5 * k_ss_cheb
k_high_cheb   =  1.5 * k_ss_cheb
n_k_cheb =  10

# Setting up the productivity domain (3 std) - same as original
z_low_log_cheb    = -3 * np.sqrt(σ_cheb**2/(1-ρ_cheb**2))
z_high_log_cheb   =  3 * np.sqrt(σ_cheb**2/(1-ρ_cheb**2))
n_z_cheb =  10

# Get Chebyshev nodes in [-1, 1] domain
cheb_nodes_k = Chebyshev_Nodes(n_k_cheb).ravel()  # Nodes in [-1, 1]
cheb_nodes_z_log = Chebyshev_Nodes(n_z_cheb).ravel()  # Nodes in [-1, 1] for log(z)

# Map Chebyshev nodes from [-1, 1] to economic domain
# For capital: map directly to [k_low, k_high]
k_grid_cheb = Change_Variable_Fromcheb(k_low_cheb, k_high_cheb, cheb_nodes_k)
# For productivity: map to log space [z_low_log, z_high_log], then exponentiate (same as original)
z_grid_log_cheb = Change_Variable_Fromcheb(z_low_log_cheb, z_high_log_cheb, cheb_nodes_z_log)
z_grid_cheb  = np.exp(z_grid_log_cheb)

# Number of coefficients
n_coeffs_cheb = n_k_cheb * n_z_cheb

# Set initial values for the coefficients
η_init_cheb = np.ones(n_coeffs_cheb) * 0.00001

# Find solution by minimizing the errors on the grid (same optimizer settings as original)
η_opt_cheb = opt.minimize(euler_err_cheb, η_init_cheb, args=(gh_quad_cheb, k_grid_cheb, z_grid_cheb, k_low_cheb, k_high_cheb, z_low_log_cheb, z_high_log_cheb, n_k_cheb, n_z_cheb), 
                     method='Nelder-Mead', options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print("Optimal coefficients (Chebyshev, first 10):", η_opt_cheb[:10])


k_grid_fine_cheb = np.linspace(k_low_cheb,k_high_cheb,100)
z_grid_fine_cheb = np.exp(np.linspace(z_low_log_cheb,z_high_log_cheb,100))

# Generate meshgrid coordinates for 3d plot
kg_cheb, zg_cheb = np.meshgrid(k_grid_fine_cheb, z_grid_fine_cheb)

# Evaluate consumption policy function on fine grid
c_policy_cheb = np.zeros_like(kg_cheb)
for i in range(kg_cheb.shape[0]):
    for j in range(kg_cheb.shape[1]):
        k_val = kg_cheb[i, j]
        z_val = zg_cheb[i, j]
        c_policy_cheb[i, j] = c_cheb(k_val, z_val, η_opt_cheb, k_low_cheb, k_high_cheb, z_low_log_cheb, z_high_log_cheb, n_k_cheb, n_z_cheb)

# Plot policy function approximation
fig_cheb = plt.figure(figsize=(12,9))
ax_cheb = fig_cheb.add_subplot(111, projection='3d')
ax_cheb.plot_surface(kg_cheb,
                zg_cheb,
                c_policy_cheb,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax_cheb.set_xlabel('k', fontsize=14)
ax_cheb.set_ylabel('z', fontsize=14)
ax_cheb.set_zlabel('c', fontsize=14)
ax_cheb.set_title('Consumption Policy Function (Chebyshev)', fontsize=14, fontweight='bold')

# Save figure instead of showing
output_path_cheb = '../NGM_figures/Chebyshev_policy_function.png'
plt.savefig(output_path_cheb, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot: {output_path_cheb}")
plt.close()

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)