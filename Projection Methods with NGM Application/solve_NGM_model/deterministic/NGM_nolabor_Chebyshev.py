#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Computational imports
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

n_q = 5  # Number of nodes and weights for the Gauss-Hermite quadrature

# Use the hermgauss function to get the nodes and the weights for the Gauss-Hermite quadrature
gh_quad = np.polynomial.hermite.hermgauss(n_q)

def euler_err(gamma, quad, k_grid, z_grid, k_low, k_high, z_low_log, z_high_log, n_k_cheb, n_z_cheb):
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

β = 0.99
α = 0.33
δ = 0.025
γ = 4
ρ = 0.95
σ = 0.1

# Calculate the steady state level of capital
k_ss = (β * α/(1-β*(1-δ)))**(1/(1-α))

# Setting up the capital domain
k_low    =  0.5 * k_ss
k_high   =  1.5 * k_ss
n_k =  10

# Setting up the productivity domain (3 std)
z_low_log    = -3 * np.sqrt(σ**2/(1-ρ**2))
z_high_log   =  3 * np.sqrt(σ**2/(1-ρ**2))
n_z =  10

# Get Chebyshev nodes in [-1, 1] domain
cheb_nodes_k = Chebyshev_Nodes(n_k).ravel()  # Nodes in [-1, 1]
cheb_nodes_z_log = Chebyshev_Nodes(n_z).ravel()  # Nodes in [-1, 1] for log(z)

# Map Chebyshev nodes from [-1, 1] to economic domain
# For capital: map directly to [k_low, k_high]
k_grid = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)
# For productivity: map to log space [z_low_log, z_high_log], then exponentiate (same as original)
z_grid_log = Change_Variable_Fromcheb(z_low_log, z_high_log, cheb_nodes_z_log)
z_grid  = np.exp(z_grid_log)

# Number of coefficients
n_coeffs = n_k * n_z

# Set initial values for the coefficients
η_init = np.ones(n_coeffs) * 0.00001

# Find solution by minimizing the errors on the grid
η_opt = opt.minimize(euler_err, η_init, args=(gh_quad, k_grid, z_grid, k_low, k_high, z_low_log, z_high_log, n_k, n_z), 
                     method='Nelder-Mead', options={'disp':True,'maxiter':100000,'xatol': 1e-10,'fatol': 1e-10}).x
print("Optimal coefficients:", η_opt)


k_grid_fine = np.linspace(k_low,k_high,100)
z_grid_fine = np.exp(np.linspace(z_low_log,z_high_log,100))

# Generate meshgrid coordinates for 3d plot
kg, zg = np.meshgrid(k_grid_fine, z_grid_fine)

# Evaluate consumption policy function on fine grid
c_policy = np.zeros_like(kg)
for i in range(kg.shape[0]):
    for j in range(kg.shape[1]):
        k_val = kg[i, j]
        z_val = zg[i, j]
        c_policy[i, j] = c_cheb(k_val, z_val, η_opt, k_low, k_high, z_low_log, z_high_log, n_k, n_z)

# Plot policy function approximation
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kg,
                zg,
                c_policy,
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.5,
                linewidth=0.25)
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('z', fontsize=14)
ax.set_zlabel('c', fontsize=14)
ax.set_title('Consumption Policy Function (Chebyshev)', fontsize=14, fontweight='bold')

# Save figure instead of showing
output_path = '../NGM_figures/Chebyshev_policy_function.png'
os.makedirs('NGM_figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot: {output_path}")
plt.close()
