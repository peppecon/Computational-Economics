#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
STOCHASTIC NEOCLASSICAL GROWTH MODEL WITH LABOR - CHEBYSHEV PROJECTION METHOD
================================================================================

This code solves a stochastic neoclassical growth model with endogenous labor
supply using Chebyshev polynomial projection.

MODEL SETUP:
-----------
- State variables: capital k and productivity z
- Control variables: consumption c and labor l
- Utility: U(c, l) = log(c) - chi * l^(1+1/nu)/(1+1/nu)
- Production: y = z * k^alpha * l^(1-alpha)
- Productivity follows AR(1) in logs: log(z') = rho*log(z) + eps', eps' ~ N(0,sigma^2)

SOLUTION METHOD:
---------------
1. Approximate consumption policy function c(k,z) using Chebyshev polynomials
2. Compute labor l(k,z) from consumption using intratemporal FOC
3. Use fixed-point iteration to satisfy Euler equation at Chebyshev nodes
4. Use Gauss-Hermite quadrature to compute expectations over productivity shocks

KEY INSIGHT:
-----------
We only approximate c(k,z) with Chebyshev polynomials. Labor is computed directly
from consumption using the intratemporal FOC, ensuring this condition is always
satisfied exactly.
"""

import numpy as np
import os

# Script is now self-contained - no external library needed for Chebyshev functions

# Graphics imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (12, 8)
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ============================================================================
# CLASS 1: CHEBYSHEV APPROXIMATION
# ============================================================================

class ChebyshevApproximation:
    """
    Class for Chebyshev polynomial approximation of functions
    
    This class handles:
    - Setting up Chebyshev nodes and grids
    - Evaluating Chebyshev polynomials
    - Converting between function values and coefficients
    """
    
    def chebyshev_nodes(self, n):
        """
        Generate n Chebyshev nodes in [-1, 1]
        
        Chebyshev nodes: x_i = cos(π*(2*i-1)/(2*n)) for i=1,...,n
        
        Parameters:
        -----------
        n : int
            Number of nodes
        
        Returns:
        --------
        array : Chebyshev nodes in [-1, 1]
        """
        i = np.arange(1, n + 1)
        return np.cos(np.pi * (2 * i - 1) / (2 * n))
    
    def change_variable_to_cheb(self, a, b, x):
        """
        Transform variable from economic domain [a, b] to Chebyshev domain [-1, 1]
        
        Linear transformation: x_cheb = 2*(x - a)/(b - a) - 1
        
        Parameters:
        -----------
        a, b : float
            Lower and upper bounds of economic domain
        x : float or array
            Value(s) in economic domain
        
        Returns:
        --------
        float or array : Value(s) in Chebyshev domain [-1, 1]
        """
        return 2 * (x - a) / (b - a) - 1
    
    def change_variable_from_cheb(self, a, b, x_cheb):
        """
        Transform variable from Chebyshev domain [-1, 1] to economic domain [a, b]
        
        Inverse transformation: x = a + (b - a)*(x_cheb + 1)/2
        
        Parameters:
        -----------
        a, b : float
            Lower and upper bounds of economic domain
        x_cheb : float or array
            Value(s) in Chebyshev domain [-1, 1]
        
        Returns:
        --------
        float or array : Value(s) in economic domain
        """
        return a + (b - a) * (x_cheb + 1) / 2
    
    def chebyshev_polynomials(self, x, order):
        """
        Evaluate Chebyshev polynomials T_0(x), T_1(x), ..., T_{order-1}(x)
        
        Uses recursion: T_0(x) = 1, T_1(x) = x
                        T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        
        Parameters:
        -----------
        x : float or array
            Point(s) in [-1, 1] where to evaluate polynomials
        order : int
            Number of polynomials to evaluate (orders 0 to order-1)
        
        Returns:
        --------
        array : Chebyshev polynomials [T_0(x), T_1(x), ..., T_{order-1}(x)]
                Shape: (order,) for scalar x, (n_points, order) for array x
        """
        x = np.asarray(x)
        is_scalar = x.ndim == 0
        if is_scalar:
            x = x.reshape(1)
        
        n_points = len(x)
        T = np.zeros((n_points, order))
        
        if order >= 1:
            # T_0(x) = 1
            T[:, 0] = 1.0
        
        if order >= 2:
            # T_1(x) = x
            T[:, 1] = x
        
        # Recurrence relation: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        for n in range(2, order):
            T[:, n] = 2 * x * T[:, n-1] - T[:, n-2]
        
        # Return 1D array for scalar input to match original function behavior
        if is_scalar:
            return T[0, :]
        return T
    
    def __init__(self, n_nodes_k, n_nodes_z, k_low, k_high, z_low_log, z_high_log):
        """
        Initialize Chebyshev approximation
        
        Parameters:
        -----------
        n_nodes_k : int
            Number of Chebyshev nodes for capital dimension
        n_nodes_z : int
            Number of Chebyshev nodes for productivity dimension
        k_low, k_high : float
            Bounds for capital domain
        z_low_log, z_high_log : float
            Bounds for log-productivity domain
        """
        self.n_k = n_nodes_k
        self.n_z = n_nodes_z
        self.p_k = n_nodes_k  # Polynomial order = number of nodes
        self.p_z = n_nodes_z
        
        self.k_low = k_low
        self.k_high = k_high
        self.z_low_log = z_low_log
        self.z_high_log = z_high_log
        
        # Get Chebyshev nodes in [-1, 1] and map to economic domain
        self.k_grid = self.change_variable_from_cheb(
            k_low, k_high, self.chebyshev_nodes(n_nodes_k))
        self.z_grid_log = self.change_variable_from_cheb(
            z_low_log, z_high_log, self.chebyshev_nodes(n_nodes_z))
        self.z_grid = np.exp(self.z_grid_log)  # Convert to level
        
        self.n_coeffs = self.p_k * self.p_z
    
    def evaluate(self, k, z, coefficients):
        """
        Evaluate Chebyshev approximation at point (k, z)
        
        Parameters:
        -----------
        k, z : float
            Capital and productivity values
        coefficients : array
            Chebyshev coefficients (gamma)
        
        Returns:
        --------
        float : Approximated function value
        """
        # Transform to Chebyshev domain [-1, 1]
        k_cheb = self.change_variable_to_cheb(self.k_low, self.k_high, k)
        z_cheb = self.change_variable_to_cheb(self.z_low_log, self.z_high_log, np.log(z))
        
        # Evaluate Chebyshev polynomials
        T_k = self.chebyshev_polynomials(k_cheb, self.p_k)
        T_z = self.chebyshev_polynomials(z_cheb, self.p_z)
        
        # Tensor product and evaluate: f(k,z) = coefficients @ (T_k ⊗ T_z)
        return float(coefficients @ np.kron(T_k, T_z))
    
    def coefficients_from_values(self, values):
        """
        Compute Chebyshev coefficients from function values at grid points
        
        Solves: T_matrix @ coefficients = values
        
        Parameters:
        -----------
        values : array
            Function values at grid points (flattened)
        
        Returns:
        --------
        array : Chebyshev coefficients
        """
        n = len(self.k_grid) * len(self.z_grid)
        T_matrix = np.zeros((n, self.p_k * self.p_z))
        
        idx = 0
        for z in self.z_grid:
            z_cheb = self.change_variable_to_cheb(self.z_low_log, self.z_high_log, np.log(z))
            T_z = self.chebyshev_polynomials(z_cheb, self.p_z)
            for k in self.k_grid:
                k_cheb = self.change_variable_to_cheb(self.k_low, self.k_high, k)
                T_k = self.chebyshev_polynomials(k_cheb, self.p_k)
                T_matrix[idx, :] = np.kron(T_k, T_z).ravel()
                idx += 1
        
        # Solve linear system: T_matrix @ coefficients = values
        # Since n = n_k * n_z = p_k * p_z, the system is always square
        return np.linalg.solve(T_matrix, values)

# ============================================================================
# CLASS 2: NEOCLASSICAL GROWTH MODEL
# ============================================================================

class NeoclassicalGrowthModel:
    """
    Stochastic Neoclassical Growth Model with Labor Supply
    
    This class organizes the model into clear sections:
    - Production technology
    - Utility function
    - Equilibrium conditions (Euler, intratemporal FOC)
    - Dynamics (capital and productivity evolution)
    """
    
    def __init__(self, beta=0.99, alpha=0.33, delta=0.025, nu=1.0, 
                 rho=0.95, sigma=0.02, l_ss_target=1/3):
        """
        Initialize model with parameters
        
        Parameters:
        -----------
        beta : float
            Discount factor
        alpha : float
            Capital share in production
        delta : float
            Depreciation rate
        nu : float
            Frisch elasticity parameter
        rho : float
            Persistence of productivity shock
        sigma : float
            Standard deviation of productivity shock
        l_ss_target : float
            Target steady-state labor (chi will be calibrated)
        """
        # Model parameters
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.nu = nu
        self.rho = rho
        self.sigma = sigma
        self.l_ss_target = l_ss_target
        
        # Calibrated parameter (set in compute_steady_state)
        self.chi = None
        
        # Steady state values (computed in compute_steady_state)
        self.k_ss = None
        self.l_ss = None
        self.c_ss = None
        self.z_ss = 1.0
        
        # Quadrature setup for expectations
        self.n_q = 5
        self.q_nodes, self.q_weights = np.polynomial.hermite.hermgauss(self.n_q)
    
    # ========================================================================
    # PRODUCTION TECHNOLOGY
    # ========================================================================
    
    def production(self, k, z, l):
        """
        Production function: y = z * k^alpha * l^(1-alpha)
        
        Parameters:
        -----------
        k, z, l : float
            Capital, productivity, and labor
        
        Returns:
        --------
        float : Output
        """
        return z * k**self.alpha * l**(1 - self.alpha)
    
    def marginal_product_capital(self, k, z, l):
        """
        Marginal product of capital: MPK = alpha * z * k^(alpha-1) * l^(1-alpha)
        
        Parameters:
        -----------
        k, z, l : float
            Capital, productivity, and labor
        
        Returns:
        --------
        float : Marginal product of capital
        """
        return self.alpha * z * k**(self.alpha - 1) * l**(1 - self.alpha)
    
    def marginal_product_labor(self, k, z, l):
        """
        Marginal product of labor: MPL = (1-alpha) * z * k^alpha * l^(-alpha)
        
        Parameters:
        -----------
        k, z, l : float
            Capital, productivity, and labor
        
        Returns:
        --------
        float : Marginal product of labor
        """
        return (1 - self.alpha) * z * k**self.alpha * l**(-self.alpha)
    
    def return_on_capital(self, k, z, l):
        """
        Return on capital: R = MPK + (1-delta) = alpha*z*k^(alpha-1)*l^(1-alpha) + (1-delta)
        
        Parameters:
        -----------
        k, z, l : float
            Capital, productivity, and labor
        
        Returns:
        --------
        float : Return on capital
        """
        return self.marginal_product_capital(k, z, l) + (1 - self.delta)
    
    # ========================================================================
    # UTILITY FUNCTION
    # ========================================================================
    
    def utility(self, c, l):
        """
        Utility function: U(c, l) = log(c) - chi * l^(1+1/nu) / (1+1/nu)
        
        Parameters:
        -----------
        c, l : float
            Consumption and labor
        
        Returns:
        --------
        float : Utility
        """
        return np.log(c) - self.chi * l**(1 + 1/self.nu) / (1 + 1/self.nu)
    
    def marginal_utility_consumption(self, c):
        """
        Marginal utility of consumption: u'(c) = 1/c (for log utility)
        
        Parameters:
        -----------
        c : float
            Consumption
        
        Returns:
        --------
        float : Marginal utility
        """
        return 1 / c
    
    def marginal_utility_labor(self, l):
        """
        Marginal disutility of labor: -chi * l^(1/nu)
        
        Parameters:
        -----------
        l : float
            Labor
        
        Returns:
        --------
        float : Marginal disutility (negative)
        """
        return -self.chi * l**(1/self.nu)
    
    # ========================================================================
    # EQUILIBRIUM CONDITIONS
    # ========================================================================
    
    def intratemporal_foc(self, k, z, c, l):
        """
        Intratemporal FOC: -u_l / u_c = MPL
        
        For our utility: chi * l^(1/nu) = MPL / c
        Rearranging: chi * l^(1/nu) = (1-alpha) * z * k^alpha * l^(-alpha) / c
        
        Parameters:
        -----------
        k, z, c, l : float
            Capital, productivity, consumption, and labor
        
        Returns:
        --------
        float : FOC error (should be zero in equilibrium)
        """
        lhs = -self.marginal_utility_labor(l)  # chi * l^(1/nu)
        rhs = self.marginal_product_labor(k, z, l) / c
        return lhs - rhs
    
    def labor_from_consumption(self, k, z, c):
        """
        Solve intratemporal FOC for labor given consumption
        
        From FOC: chi * l^(1/nu) = (1-alpha) * z * k^alpha * l^(-alpha) / c
        Rearranging: l = [(1-alpha) * z * k^alpha / (chi * c)]^(nu/(1+alpha*nu))
        
        Parameters:
        -----------
        k, z, c : float
            Capital, productivity, and consumption
        
        Returns:
        --------
        float : Labor supply
        """
        return ((1-self.alpha) * z * k**self.alpha / (self.chi * c))**(self.nu / (1 + self.alpha*self.nu))
    
    def euler_equation(self, k, z, c, c_prime, k_prime, z_prime, l_prime):
        """
        Euler equation: u'(c) = beta * E[u'(c') * R']
        
        For log utility: 1/c = beta * E[(1/c') * R']
        where R' = return on capital next period
        
        Parameters:
        -----------
        k, z, c : float
            Current capital, productivity, consumption
        c_prime, k_prime, z_prime, l_prime : float
            Next period values
        
        Returns:
        --------
        float : Euler equation value (should equal u'(c) in equilibrium)
        """
        R_prime = self.return_on_capital(k_prime, z_prime, l_prime)
        return self.beta * self.marginal_utility_consumption(c_prime) * R_prime
    
    # ========================================================================
    # DYNAMICS
    # ========================================================================
    
    def capital_evolution(self, k, z, c, l):
        """
        Capital accumulation: k' = (1-delta)*k + y - c
        
        Parameters:
        -----------
        k, z, c, l : float
            Current capital, productivity, consumption, and labor
        
        Returns:
        --------
        float : Next period capital
        """
        y = self.production(k, z, l)
        return (1 - self.delta) * k + y - c
    
    def productivity_evolution(self, z, shock):
        """
        Productivity evolution: log(z') = rho * log(z) + shock
        
        Parameters:
        -----------
        z : float
            Current productivity
        shock : float
            Productivity shock
        
        Returns:
        --------
        float : Next period productivity
        """
        return np.exp(self.rho * np.log(z) + shock)
    
    # ========================================================================
    # STEADY STATE
    # ========================================================================
    
    def compute_steady_state(self):
        """
        Compute steady state and calibrate chi so that l_ss = l_ss_target
        
        Steps:
        1. Set l_ss = l_ss_target
        2. Solve for k_ss from Euler equation: 1 = beta * R
        3. Compute c_ss from resource constraint: c = y - delta*k
        4. Calibrate chi from intratemporal FOC
        """
        self.l_ss = self.l_ss_target
        
        # Step 1: Solve for k_ss from Euler equation
        # At steady state: 1 = beta * R, where R = MPK + (1-delta)
        # So: 1/beta = alpha*z*k^(alpha-1)*l^(1-alpha) + (1-delta)
        self.k_ss = ((1/self.beta - (1-self.delta)) / 
                     (self.alpha * self.z_ss * self.l_ss**(1-self.alpha)))**(1 / (self.alpha - 1))
        
        # Step 2: Compute c_ss from resource constraint
        y_ss = self.production(self.k_ss, self.z_ss, self.l_ss)
        self.c_ss = y_ss - self.delta * self.k_ss
        
        # Step 3: Calibrate chi from intratemporal FOC
        # chi * l^(1/nu) = MPL / c = (1-alpha)*z*k^alpha*l^(-alpha) / c
        mpl_ss = self.marginal_product_labor(self.k_ss, self.z_ss, self.l_ss)
        self.chi = mpl_ss / (self.c_ss * self.l_ss**(1/self.nu))
    
    # ========================================================================
    # SOLUTION METHODS (for Chebyshev projection)
    # ========================================================================
    
    def compute_euler_error_and_update(self, k, z, c, chebyshev_approx, coefficients):
        """
        Compute Euler error and updated consumption at point (k, z) given consumption c
        
        This method computes the expectation E[beta * u'(c') * R'] once and returns:
        - Euler error: E - u'(c)
        - Updated consumption: c_new = 1/E (for log utility) to satisfy Euler equation
        
        Parameters:
        -----------
        k, z, c : float
            Capital, productivity, and consumption
        chebyshev_approx : ChebyshevApproximation
            Chebyshev approximation object for consumption policy
        coefficients : array
            Chebyshev coefficients for consumption policy
        
        Returns:
        --------
        tuple : (euler_error, c_new)
            euler_error : float - Euler equation error
            c_new : float - Updated consumption to satisfy Euler equation
        """
        # Compute labor from intratemporal FOC
        l = self.labor_from_consumption(k, z, c)
        
        # Compute next period capital
        k_prime = self.capital_evolution(k, z, c, l)
        
        # Compute expectation over productivity shocks using quadrature
        E = 0.0
        for q_node, q_weight in zip(self.q_nodes, self.q_weights):
            # Transform quadrature node to shock
            shock = np.sqrt(2) * self.sigma * q_node
            z_prime = self.productivity_evolution(z, shock)
            
            # Evaluate consumption at (k', z')
            c_prime = chebyshev_approx.evaluate(k_prime, z_prime, coefficients)
            
            # Compute labor and return on capital
            l_prime = self.labor_from_consumption(k_prime, z_prime, c_prime)
            R_prime = self.return_on_capital(k_prime, z_prime, l_prime)
            
            # Accumulate expectation: beta * u'(c') * R'
            E += q_weight * self.euler_equation(k, z, c, c_prime, k_prime, z_prime, l_prime)
        
        # Normalize expectation
        E /= np.sqrt(np.pi)
        
        # Compute Euler error: E - u'(c)
        euler_error = E - self.marginal_utility_consumption(c)
        
        # Update consumption: u'(c) = E, so c_new = 1/E for log utility
        c_new = 1/E if E > 0 else c
        
        return euler_error, c_new

# ============================================================================
# MAIN CODE: SOLVE THE MODEL
# ============================================================================

print("="*80)
print("STOCHASTIC NEOCLASSICAL GROWTH MODEL WITH LABOR")
print("CHEBYSHEV PROJECTION METHOD")
print("="*80)

# ============================================================================
# STEP 1: INITIALIZE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 1: INITIALIZE MODEL")
print("="*80)

model = NeoclassicalGrowthModel(
    beta=0.99, alpha=0.33, delta=0.025, nu=1.0, 
    rho=0.95, sigma=0.02, l_ss_target=1/3
)

print(f"Parameters: beta={model.beta}, alpha={model.alpha}, delta={model.delta}")
print(f"           nu={model.nu}, rho={model.rho}, sigma={model.sigma}")
print(f"Target: l_ss = {model.l_ss_target:.3f} (chi will be calibrated)")

# ============================================================================
# STEP 2: COMPUTE STEADY STATE
# ============================================================================
print("\n" + "="*80)
print("STEP 2: COMPUTE STEADY STATE")
print("="*80)

model.compute_steady_state()
print(f"Steady state: k_ss={model.k_ss:.6f}, l_ss={model.l_ss:.6f}, "
      f"c_ss={model.c_ss:.6f}, chi={model.chi:.6f}")

# ============================================================================
# STEP 3: SET UP CHEBYSHEV APPROXIMATION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SET UP CHEBYSHEV APPROXIMATION")
print("="*80)

# Domain bounds
k_low, k_high = 0.5 * model.k_ss, 1.5 * model.k_ss
z_var = model.sigma**2 / (1 - model.rho**2)
z_low_log, z_high_log = -3 * np.sqrt(z_var), 3 * np.sqrt(z_var)

# Initialize Chebyshev approximation
cheb_approx = ChebyshevApproximation(
    n_nodes_k=10, n_nodes_z=10,
    k_low=k_low, k_high=k_high,
    z_low_log=z_low_log, z_high_log=z_high_log
)

print(f"Nodes: n_k={cheb_approx.n_k}, n_z={cheb_approx.n_z} "
      f"(total coeffs: {cheb_approx.n_coeffs})")
print(f"Domains: k∈[{k_low:.3f}, {k_high:.3f}], "
      f"log(z)∈[{z_low_log:.3f}, {z_high_log:.3f}]")
print(f"Quadrature: {model.n_q} Gauss-Hermite nodes")

# ============================================================================
# STEP 4: FIXED-POINT ITERATION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FIXED-POINT ITERATION")
print("="*80)

# Initialize consumption at steady state
c_values = np.full(cheb_approx.n_k * cheb_approx.n_z, model.c_ss)
cheb_approx.coefficients = cheb_approx.coefficients_from_values(c_values)

print(f"Initial consumption: c = c_ss = {model.c_ss:.6f} everywhere")
print("Starting iteration...")

max_iter, tol, dampening = 2000, 1e-8, 1.0

for iter in range(max_iter):
    # Compute Euler errors and update consumption at all grid points
    c_new = np.zeros(len(c_values))
    euler_errors = np.zeros(len(c_values))
    
    idx = 0
    for z in cheb_approx.z_grid:
        for k in cheb_approx.k_grid:
            c = c_values[idx]
            # Compute both error and updated consumption in one call
            euler_errors[idx], c_new[idx] = model.compute_euler_error_and_update(
                k, z, c, cheb_approx, cheb_approx.coefficients)
            idx += 1
    
    max_euler_error = np.max(np.abs(euler_errors))
    if iter % 10 == 0 or max_euler_error < tol:
        print(f"Iter {iter:4d}: max |Euler error| = {max_euler_error:.6e}")
    
    if max_euler_error < tol:
        print(f"✓ Converged after {iter} iterations!")
        break
    
    # Update consumption with damping
    c_values = (1 - dampening) * c_values + dampening * c_new
    
    # Update coefficients
    cheb_approx.coefficients = cheb_approx.coefficients_from_values(c_values)
    
    # Recompute from coefficients for consistency
    idx = 0
    for z in cheb_approx.z_grid:
        for k in cheb_approx.k_grid:
            c_values[idx] = cheb_approx.evaluate(k, z, cheb_approx.coefficients)
            idx += 1

print(f"Final max |Euler error|: {max_euler_error:.6e}")

# ============================================================================
# STEP 5: VISUALIZE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING PLOTS")
print("="*80)

# 3D plots
K_3d, Z_3d = np.meshgrid(np.linspace(k_low, k_high, 50),
                        np.exp(np.linspace(z_low_log, z_high_log, 50)))
C_3d = np.zeros_like(K_3d)
L_3d = np.zeros_like(K_3d)

for i in range(K_3d.shape[0]):
    for j in range(K_3d.shape[1]):
        C_3d[i,j] = cheb_approx.evaluate(K_3d[i,j], Z_3d[i,j], cheb_approx.coefficients)
        L_3d[i,j] = model.labor_from_consumption(K_3d[i,j], Z_3d[i,j], C_3d[i,j])

fig = plt.figure(figsize=(18, 6))
for i, (data, title, zlabel, cmap) in enumerate([
    (C_3d, 'Consumption', 'c (Consumption)', 'viridis'),
    (L_3d, 'Labor', 'l (Labor)', 'plasma'),
    ((1-model.delta)*K_3d + Z_3d*K_3d**model.alpha*L_3d**(1-model.alpha) - C_3d, 
     'Capital Accumulation', "k'", 'coolwarm')
]):
    ax = fig.add_subplot(131+i, projection='3d')
    surf = ax.plot_surface(K_3d, Z_3d, data, cmap=cmap, alpha=0.9)
    
    ax.set_xlabel('k (Capital)', fontweight='bold')
    ax.set_ylabel('z (Productivity)', fontweight='bold')
    ax.set_zlabel(zlabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.6)

os.makedirs('presentation', exist_ok=True)
plt.savefig('../presentation/stochastic_Chebyshev_labor_presentation.png', dpi=300, bbox_inches='tight')
print("✓ Saved 3D plots")
plt.close()

# 2D plots
z_levels = [np.exp(z_low_log), 1.0, np.exp(z_high_log)]
z_labels = ['small', 'medium', 'large']
k_plot = np.linspace(k_low, k_high, 200)

c_data = [[cheb_approx.evaluate(k, z, cheb_approx.coefficients) 
           for k in k_plot] for z in z_levels]
l_data = [[model.labor_from_consumption(k, z, c_data[i][j]) 
           for j, k in enumerate(k_plot)] for i, z in enumerate(z_levels)]
k_prime_data = [[model.capital_evolution(k, z, c_data[i][j], l_data[i][j])
                 for j, k in enumerate(k_plot)] for i, z in enumerate(z_levels)]

# Create figure with enhanced styling
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.patch.set_facecolor('white')

# Modern color palette
colors = ['#2E86AB', '#06A77D', '#F24236']  # Blue, Green, Red
line_styles = ['-', '-', '-']
line_widths = [3, 3, 3]

# Plot policy functions with enhanced styling
for i, (c, l, kp, z, label, color) in enumerate(zip(c_data, l_data, k_prime_data, z_levels, z_labels, colors)):
    # Consumption plot
    ax1.plot(k_plot, c, color=color, linestyle=line_styles[i], linewidth=line_widths[i], 
             label=f'z = {z:.3f} ({label})', alpha=0.9, zorder=5)
    # Labor plot
    ax2.plot(k_plot, l, color=color, linestyle=line_styles[i], linewidth=line_widths[i], 
             label=f'z = {z:.3f} ({label})', alpha=0.9, zorder=5)
    # Capital accumulation plot
    ax3.plot(k_plot, kp, color=color, linestyle=line_styles[i], linewidth=line_widths[i], 
             label=f'z = {z:.3f} ({label})', alpha=0.9, zorder=5)

# Mark steady state with enhanced styling
c_ss_plot = c_data[1][np.argmin(np.abs(k_plot - model.k_ss))]
l_ss_plot = l_data[1][np.argmin(np.abs(k_plot - model.k_ss))]
k_prime_ss_plot = k_prime_data[1][np.argmin(np.abs(k_plot - model.k_ss))]

for ax, y_val in [(ax1, c_ss_plot), (ax2, l_ss_plot), (ax3, k_prime_ss_plot)]:
    # Steady state marker with shadow effect
    ax.plot(model.k_ss, y_val, 'o', color='#1a1a1a', markersize=14, 
            markeredgecolor='white', markeredgewidth=2.5, zorder=15, 
            label='Steady state' if ax == ax1 else '')
    # Reference lines with better styling
    ax.axvline(model.k_ss, color='#666666', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    ax.axhline(y_val, color='#666666', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

# 45-degree line for capital accumulation
ax3.plot([k_low, k_high], [k_low, k_high], 'k--', linewidth=2, alpha=0.5, 
         label='k\' = k', zorder=1)

# Enhanced font sizes
fontsize_labels, fontsize_title, fontsize_legend, fontsize_ticks = 16, 18, 13, 13

# Styling for all axes
axes = [ax1, ax2, ax3]
titles = ['Consumption Policy Function', 'Labor Policy Function', 'Capital Accumulation Function']
xlabels = ['k (Capital)', 'k (Capital)', 'k (Current Capital)']
ylabels = ['c (Consumption)', 'l (Labor)', "k' (Next Period Capital)"]

for ax, title, xlabel, ylabel in zip(axes, titles, xlabels, ylabels):
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=fontsize_labels, labelpad=10)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=fontsize_labels, labelpad=10)
    ax.set_title(title, fontweight='bold', fontsize=fontsize_title, pad=15)
    
    # Enhanced grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # Enhanced tick parameters
    ax.tick_params(labelsize=fontsize_ticks, width=1.5, length=6, 
                   grid_color='gray', grid_alpha=0.3)
    
    # Enhanced legend
    legend = ax.legend(fontsize=fontsize_legend, frameon=True, fancybox=True, 
                       shadow=True, framealpha=0.95, edgecolor='gray', 
                       borderpad=0.8, labelspacing=0.7)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.5)

# Set limits for capital accumulation plot
ax3.set_xlim([k_low, k_high])
ax3.set_ylim([k_low, k_high])

# Add subtle background color to distinguish plots
for ax in axes:
    ax.set_facecolor('#fafafa')
    
# Adjust spacing
plt.tight_layout(pad=3.0)

plt.tight_layout()
plt.savefig('../presentation/stochastic_Chebyshev_labor_presentation_2d.png', dpi=300, bbox_inches='tight')
print("✓ Saved 2D plots")
plt.close()

# ============================================================================
# STEP 6: EULER ERROR PLOTS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING EULER ERROR PLOTS")
print("="*80)

# Compute Euler errors at Chebyshev nodes
euler_errors_nodes = np.zeros((cheb_approx.n_z, cheb_approx.n_k))
idx = 0
for i_z, z in enumerate(cheb_approx.z_grid):
    for i_k, k in enumerate(cheb_approx.k_grid):
        c = cheb_approx.evaluate(k, z, cheb_approx.coefficients)
        euler_error, _ = model.compute_euler_error_and_update(
            k, z, c, cheb_approx, cheb_approx.coefficients)
        euler_errors_nodes[i_z, i_k] = euler_error
        idx += 1

# Create meshgrid for nodes
K_nodes, Z_nodes = np.meshgrid(cheb_approx.k_grid, cheb_approx.z_grid)

# 3D plot of Euler errors at nodes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_nodes, Z_nodes, euler_errors_nodes, cmap='RdBu_r', 
                       alpha=0.9, linewidth=0.5, edgecolor='black')
ax.set_xlabel('k (Capital)', fontweight='bold', fontsize=12)
ax.set_ylabel('z (Productivity)', fontweight='bold', fontsize=12)
ax.set_zlabel('Euler Error', fontweight='bold', fontsize=12)
ax.set_title('Euler Errors at Chebyshev Nodes', fontweight='bold', fontsize=14)
fig.colorbar(surf, ax=ax, shrink=0.6, label='Euler Error')
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.savefig('../presentation/stochastic_Chebyshev_labor_euler_errors_3d.png', dpi=300, bbox_inches='tight')
print("✓ Saved 3D Euler error plot")
plt.close()

# 2D plot of Euler errors as function of k for different z levels
euler_error_data = []
for z in z_levels:
    errors = []
    for k in k_plot:
        c = cheb_approx.evaluate(k, z, cheb_approx.coefficients)
        euler_error, _ = model.compute_euler_error_and_update(
            k, z, c, cheb_approx, cheb_approx.coefficients)
        errors.append(euler_error)
    euler_error_data.append(errors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Euler errors as function of k (absolute values in log scale)
for i, (errors, z, label, color) in enumerate(zip(euler_error_data, z_levels, z_labels, colors)):
    # Use absolute value of Euler errors for log scale
    ax1.semilogy(k_plot, np.abs(errors), color=color, linestyle='-', linewidth=2, label=f'z = {z:.3f} ({label})', alpha=0.8)

ax1.set_xlabel('k (Capital)', fontweight='bold', fontsize=fontsize_labels)
ax1.set_ylabel('|Euler Error| (log scale)', fontweight='bold', fontsize=fontsize_labels)
ax1.set_title('Euler Errors as Function of Capital', fontweight='bold', fontsize=fontsize_title)
ax1.legend(fontsize=fontsize_legend)
ax1.tick_params(labelsize=fontsize_ticks)
ax1.grid(True, alpha=0.3)

# Plot 2: Euler errors at Chebyshev nodes (scatter/contour)
contour = ax2.contourf(K_nodes, Z_nodes, np.abs(euler_errors_nodes), 
                      levels=20, cmap='Reds', alpha=0.8)
ax2.scatter(K_nodes, Z_nodes, c=np.abs(euler_errors_nodes), 
           s=100, cmap='Reds', edgecolors='black', linewidths=1, zorder=10)
ax2.scatter([model.k_ss], [model.z_ss], c='green', s=300, marker='*', 
           edgecolors='black', linewidths=2, zorder=15, label='Steady state')
fig.colorbar(contour, ax=ax2, label='|Euler Error|')

ax2.set_xlabel('k (Capital)', fontweight='bold', fontsize=fontsize_labels)
ax2.set_ylabel('z (Productivity)', fontweight='bold', fontsize=fontsize_labels)
ax2.set_title('Euler Errors at Chebyshev Nodes', fontweight='bold', fontsize=fontsize_title)
ax2.legend(fontsize=fontsize_legend)
ax2.tick_params(labelsize=fontsize_ticks)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../presentation/stochastic_Chebyshev_labor_euler_errors_2d.png', dpi=300, bbox_inches='tight')
print("✓ Saved 2D Euler error plots")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Max Euler error: {np.max(np.abs(euler_errors)):.6e}")
print(f"Mean Euler error: {np.mean(np.abs(euler_errors)):.6e}")
print(f"\nSteady state: k_ss={model.k_ss:.6f}, l_ss={model.l_ss:.6f}, "
      f"c_ss={model.c_ss:.6f}, chi={model.chi:.6f}")
c_ss_actual = cheb_approx.evaluate(model.k_ss, model.z_ss, cheb_approx.coefficients)
l_ss_actual = model.labor_from_consumption(model.k_ss, model.z_ss, c_ss_actual)
print(f"Policy at SS: c={c_ss_actual:.6f}, l={l_ss_actual:.6f}")
print("\n✓ Done!")
