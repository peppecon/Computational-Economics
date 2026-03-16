#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis: Policy Functions at z=1 for Different Risk Aversion and Volatility
Uses a class-based approach to solve the stochastic NGM with labor
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.join(os.path.dirname(parent_dir), 'scripts'))
from functions_library import *

from scipy.optimize import root
from scipy.special import roots_hermite

class StochasticNGMLabor:
    """
    Stochastic Neoclassical Growth Model with Labor Supply
    Solves the model using Chebyshev polynomial projection method
    """
    
    def __init__(self, β=0.99, α=0.33, δ=0.025, χ=8.042775, ν=1, ρ=0.95, σ=0.007, 
                 γ=1, n_k=10, n_z=10, dampening=1.0):
        """
        Initialize the model with parameters
        
        Parameters:
        -----------
        β : float
            Discount factor
        α : float
            Capital share
        δ : float
            Depreciation rate
        χ : float
            Labor disutility parameter
        ν : float
            Frisch elasticity parameter
        ρ : float
            Persistence of productivity shock
        σ : float
            Standard deviation of productivity shock
        γ : float
            Risk aversion parameter (CRRA: u'(c) = c^(-γ))
        n_k : int
            Number of Chebyshev nodes for capital
        n_z : int
            Number of Chebyshev nodes for productivity
        dampening : float
            Damping parameter for fixed-point iteration
        """
        self.β = β
        self.α = α
        self.δ = δ
        self.χ = χ
        self.ν = ν
        self.ρ = ρ
        self.σ = σ
        self.γ = γ
        self.n_k = n_k
        self.n_z = n_z
        self.p_k = n_k
        self.p_z = n_z
        self.dampening = dampening
        
        # Will be set during solve
        self.k_grid = None
        self.z_grid = None
        self.k_low = None
        self.k_high = None
        self.z_low_log = None
        self.z_high_log = None
        self.gamma_c = None
        self.k_ss = None
        self.ℓ_ss = None
        self.c_ss = None
    
    def c_cheb(self, k, z):
        """Returns Chebyshev approximation of consumption"""
        k_cheb = Change_Variable_Tocheb(self.k_low, self.k_high, k)
        k_cheb = np.clip(k_cheb, -1.0, 1.0)
        z_cheb = Change_Variable_Tocheb(self.z_low_log, self.z_high_log, np.log(z))
        z_cheb = np.clip(z_cheb, -1.0, 1.0)
        T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), self.p_k)
        T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), self.p_z)
        kron_kz = np.kron(T_k, T_z)
        c = float(self.gamma_c @ kron_kz)
        c = max(c, 1e-10)
        return c
    
    def smooth_max(self, x, a, smoothness=100.0):
        """
        Smooth approximation of max(x, a) using log-sum-exp trick
        smooth_max(x, a) ≈ max(x, a) when smoothness is large
        """
        # For numerical stability, use log-sum-exp trick
        # max(x, a) ≈ (1/smoothness) * log(exp(smoothness*x) + exp(smoothness*a))
        # But we need to handle the case when values are large
        diff = smoothness * (x - a)
        # Use log-sum-exp trick: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
        if diff > 20:  # x >> a
            return x
        elif diff < -20:  # x << a
            return a
        else:
            return a + (1.0 / smoothness) * np.log(1.0 + np.exp(diff))
    
    def smooth_min(self, x, a, smoothness=100.0):
        """
        Smooth approximation of min(x, a) = -max(-x, -a)
        """
        return -self.smooth_max(-x, -a, smoothness)
    
    def l_from_c(self, k, z, c):
        """
        Computes labor from consumption using intratemporal FOC with smooth max/min constraint
        
        Uses smooth max/min operators to ensure labor is bounded between 0 and 1,
        while maintaining smoothness for better convergence of intratemporal errors.
        
        For γ=1 (log utility): EXACTLY as in NGM_stochastic_Chebyshev_labor.py
        For γ≠1 (CRRA utility): Uses c^(-γ) instead of 1/c
        """
        if c <= 1e-10:
            return 1e-6
        
        # Compute unconstrained optimal labor from FOC
        if self.γ == 1:
            # Log utility: EXACTLY as in original file
            # From intratemporal FOC: χ * ℓ^(1/ν) = (1-α) * z * k^α * ℓ^(-α) / c
            # Rearranging: χ * ℓ^(1/ν + α) = (1-α) * z * k^α / c
            # So: ℓ = [(1-α) * z * k^α / (χ * c)]^(ν/(1+αν))
            ℓ_unconstrained = ((1-self.α) * z * k**self.α / (self.χ * c))**(self.ν / (1 + self.α*self.ν))
        else:
            # CRRA utility: u'(c) = c^(-γ)
            # Intratemporal FOC: c^(-γ) / (χ * ℓ^(1/ν)) = (1-α) * z * k^α * ℓ^(-α)
            # Rearranging: c^(-γ) = χ * (1-α) * z * k^α * ℓ^(1/ν - α)
            # Solving for ℓ: ℓ = [c^(-γ) / (χ * (1-α) * z * k^α)]^(ν/(1 - αν))
            c_safe = max(c, 1e-6)
            c_safe = min(c_safe, 100.0)
            
            # Compute c^(-γ) in log-space for numerical stability
            log_c = np.log(c_safe)
            log_c_power = -self.γ * log_c
            log_c_power = np.clip(log_c_power, -20, 20)  # Cap to prevent overflow
            c_power = np.exp(log_c_power)  # This is c^(-γ)
            
            # ℓ = [c^(-γ) / (χ * (1-α) * z * k^α)]^(ν/(1 - αν))
            denominator = self.χ * (1-self.α) * z * k**self.α
            if denominator <= 1e-20:
                return 1e-6
            
            ℓ_unconstrained = (c_power / denominator)**(self.ν / (1 - self.α*self.ν))
        
        # Apply smooth max/min constraint
        # Use smooth operators to maintain differentiability and reduce discontinuities
        # Smoothness parameter: larger = sharper transition (closer to hard max/min)
        smoothness = 100.0  # Can be adjusted: larger = sharper, smaller = smoother
        ℓ = self.smooth_max(ℓ_unconstrained, 1e-6, smoothness)  # Lower bound
        ℓ = self.smooth_min(ℓ, 1.0, smoothness)  # Upper bound
        
        return ℓ
    
    def invert_to_gamma(self, values):
        """Inverts values at grid points to get Chebyshev coefficients"""
        n_k_grid = len(self.k_grid)
        n_z_grid = len(self.z_grid)
        n = n_k_grid * n_z_grid
        T_matrix = np.zeros((n, self.p_k * self.p_z))
        idx = 0
        for i_z in range(n_z_grid):
            z = self.z_grid[i_z]
            z_cheb = Change_Variable_Tocheb(self.z_low_log, self.z_high_log, np.log(z))
            z_cheb = np.clip(z_cheb, -1.0, 1.0)
            T_z = Chebyshev_Polynomials_Recursion_mv(np.array([z_cheb]), self.p_z)
            for i_k in range(n_k_grid):
                k = self.k_grid[i_k]
                k_cheb = Change_Variable_Tocheb(self.k_low, self.k_high, k)
                k_cheb = np.clip(k_cheb, -1.0, 1.0)
                T_k = Chebyshev_Polynomials_Recursion_mv(np.array([k_cheb]), self.p_k)
                kron_kz = np.kron(T_k, T_z)
                T_matrix[idx, :] = kron_kz.ravel()
                idx += 1
        if n == self.p_k * self.p_z:
            gamma = np.linalg.solve(T_matrix, values)
        else:
            gamma = np.linalg.lstsq(T_matrix, values, rcond=None)[0]
        return gamma
    
    def compute_euler_errors_and_update(self, c_values):
        """
        Computes Euler errors and returns updated consumption and labor values
        Uses Gauss-Hermite quadrature for expectations over productivity shocks
        EXACTLY as in NGM_stochastic_Chebyshev_labor.py, but with CRRA utility support
        """
        n_k_grid = len(self.k_grid)
        n_z_grid = len(self.z_grid)
        n = n_k_grid * n_z_grid
        c_new = np.zeros(n)
        l_new = np.zeros(n)
        euler_errors = np.zeros(n)
        intratemporal_errors = np.zeros(n)
        
        # Setup quadrature (exactly as in original)
        n_q = 5
        gh_quad = np.polynomial.hermite.hermgauss(n_q)
        q_nodes, q_weights = gh_quad
        
        idx = 0
        for i_z in range(n_z_grid):
            z = self.z_grid[i_z]
            for i_k in range(n_k_grid):
                k = self.k_grid[i_k]
                c = c_values[idx]
                # Compute labor from consumption to ensure intratemporal FOC is satisfied
                # Use the corrected l_from_c function that handles CRRA utility
                ℓ = self.l_from_c(k, z, c)
                
                # Compute next period capital from resource constraint
                k_prime = (1 - self.δ) * k + z * k**self.α * ℓ**(1-self.α) - c
                
                # Compute expectation over productivity shocks using Gauss-Hermite quadrature
                # CRRA utility: u'(c) = c^(-γ)
                # Euler equation: c^(-γ) = β * E[c'^(-γ) * R']
                E = 0.0
                
                for i_q in range(len(q_nodes)):
                    # Transform GH node to shock
                    e_prime = np.sqrt(2) * self.σ * q_nodes[i_q]
                    # Next period productivity: AR(1) in logs
                    z_prime = np.exp(self.ρ * np.log(z) + e_prime)
                    
                    # Evaluate consumption at (k_prime, z_prime), then compute labor from it
                    c_prime = self.c_cheb(k_prime, z_prime)
                    ℓ_prime = self.l_from_c(k_prime, z_prime, c_prime)
                    
                    # Return on capital
                    R_prime = self.α * z_prime * k_prime**(self.α - 1) * ℓ_prime**(1-self.α) + (1 - self.δ)
                    
                    # Compute marginal utility: u'(c') = c'^(-γ)
                    # Ensure c_prime is bounded to prevent numerical issues
                    c_prime_safe = max(c_prime, 1e-6)
                    c_prime_safe = min(c_prime_safe, 100.0)  # Also cap from above
                    
                    # Compute marginal utility in log-space for numerical stability
                    log_c_prime = np.log(c_prime_safe)
                    log_u_prime = -self.γ * log_c_prime
                    
                    # Cap marginal utility to prevent overflow (u'(c) <= exp(20) ≈ 5e8)
                    log_u_prime = np.clip(log_u_prime, -20, 20)
                    u_prime_c = np.exp(log_u_prime)
                    
                    # Accumulate expectation (EXACTLY as in original)
                    E += q_weights[i_q] * self.β * u_prime_c * R_prime
                
                # Normalize expectation (EXACTLY as in original)
                E = E / np.sqrt(np.pi)
                
                # Euler error: E - u'(c) = 0
                # CRRA utility: u'(c) = c^(-γ)
                # Euler equation: c^(-γ) = β * E[c'^(-γ) * R'] = E (after normalization)
                # So error = E - c^(-γ)
                
                # Compute current marginal utility safely
                c_safe = max(c, 1e-6)
                c_safe = min(c_safe, 100.0)  # Cap from above too
                
                # Use log-space computation for numerical stability
                log_c_safe = np.log(c_safe)
                log_u_prime_c = -self.γ * log_c_safe
                log_u_prime_c = np.clip(log_u_prime_c, -20, 20)  # Cap to prevent overflow
                u_prime_c = np.exp(log_u_prime_c)
                
                euler_error = E - u_prime_c
                euler_errors[idx] = euler_error
                
                # Update consumption using Euler equation
                # CRRA utility: u'(c) = c^(-γ)
                # Euler: c^(-γ) = E, so c = E^(-1/γ)
                if E > 0:
                    if self.γ == 1:
                        # Log utility (special case): 1/c = E, so c = 1/E
                        c_target = 1 / E
                    else:
                        # CRRA utility: c^(-γ) = E, so c = E^(-1/γ)
                        # Use log-space computation for numerical stability
                        log_E = np.log(max(E, 1e-20))  # Ensure E is positive
                        log_c_target = -log_E / self.γ
                        
                        # Clip to reasonable bounds
                        c_max = z * k**self.α * ℓ**(1-self.α) + (1-self.δ)*k
                        log_c_min = np.log(1e-6)  # Minimum consumption
                        log_c_max = np.log(c_max)  # Maximum consumption
                        log_c_target = np.clip(log_c_target, log_c_min, log_c_max)
                        
                        c_target = np.exp(log_c_target)
                    
                    # Final bounds check
                    c_target = max(c_target, 1e-6)  # Ensure minimum consumption
                    c_target = min(c_target, z * k**self.α * ℓ**(1-self.α) + (1-self.δ)*k)
                    c_new[idx] = (1 - self.dampening) * c + self.dampening * c_target
                else:
                    # If E <= 0, keep current consumption
                    c_new[idx] = c
                
                # Ensure consumption stays positive and reasonable
                c_new[idx] = max(c_new[idx], 1e-10)
                c_new[idx] = min(c_new[idx], z * k**self.α * ℓ**(1-self.α) + (1-self.δ)*k)
                
                # Update labor using intratemporal FOC
                # Labor is computed directly from consumption to satisfy FOC exactly
                # Use the corrected l_from_c function that handles CRRA utility
                l_new[idx] = self.l_from_c(k, z, c_new[idx])
                
                # Compute intratemporal error AFTER updating consumption and labor
                # Handle constraints properly: if labor is at boundary, use complementary slackness
                # For γ=1: EXACTLY as in NGM_stochastic_Chebyshev_labor.py
                # For γ≠1: Uses CRRA formula
                
                # Check if labor is at boundaries
                at_lower_bound = l_new[idx] <= 1e-6 + 1e-8
                at_upper_bound = l_new[idx] >= 1.0 - 1e-8
                
                if at_lower_bound:
                    # At lower bound: FOC becomes inequality u'(c) / |u_ℓ(ℓ)| ≤ w
                    # Error should be ≤ 0 (shadow cost of constraint)
                    if self.γ == 1:
                        lhs = 1 / c_new[idx]
                        rhs = self.χ * l_new[idx]**(1/self.ν) * (1-self.α) * z * k**self.α * l_new[idx]**(-self.α)
                        intratemporal_error = lhs - rhs  # Should be ≤ 0
                    else:
                        c_safe = max(c_new[idx], 1e-6)
                        c_safe = min(c_safe, 100.0)
                        log_c = np.log(c_safe)
                        log_u_prime = -self.γ * log_c
                        log_u_prime = np.clip(log_u_prime, -20, 20)
                        u_prime_c = np.exp(log_u_prime)
                        rhs = self.χ * (1-self.α) * z * k**self.α * l_new[idx]**(1/self.ν - self.α)
                        intratemporal_error = u_prime_c - rhs  # Should be ≤ 0
                elif at_upper_bound:
                    # At upper bound: FOC becomes inequality u'(c) / |u_ℓ(ℓ)| ≥ w
                    # Error should be ≥ 0 (shadow cost of constraint)
                    if self.γ == 1:
                        lhs = 1 / c_new[idx]
                        rhs = self.χ * l_new[idx]**(1/self.ν) * (1-self.α) * z * k**self.α * l_new[idx]**(-self.α)
                        intratemporal_error = lhs - rhs  # Should be ≥ 0
                    else:
                        c_safe = max(c_new[idx], 1e-6)
                        c_safe = min(c_safe, 100.0)
                        log_c = np.log(c_safe)
                        log_u_prime = -self.γ * log_c
                        log_u_prime = np.clip(log_u_prime, -20, 20)
                        u_prime_c = np.exp(log_u_prime)
                        rhs = self.χ * (1-self.α) * z * k**self.α * l_new[idx]**(1/self.ν - self.α)
                        intratemporal_error = u_prime_c - rhs  # Should be ≥ 0
                else:
                    # Interior solution: FOC holds with equality, error should be ≈ 0
                    if self.γ == 1:
                        # Log utility: EXACTLY as in original file
                        lhs = self.χ * l_new[idx]**(1/self.ν)
                        rhs = (1-self.α) * z * k**self.α * l_new[idx]**(-self.α) / c_new[idx]
                        intratemporal_error = lhs - rhs
                    else:
                        # CRRA utility: u'(c) = c^(-γ)
                        c_safe = max(c_new[idx], 1e-6)
                        c_safe = min(c_safe, 100.0)
                        log_c = np.log(c_safe)
                        log_u_prime = -self.γ * log_c
                        log_u_prime = np.clip(log_u_prime, -20, 20)
                        u_prime_c = np.exp(log_u_prime)
                        rhs = self.χ * (1-self.α) * z * k**self.α * l_new[idx]**(1/self.ν - self.α)
                        intratemporal_error = u_prime_c - rhs
                
                intratemporal_errors[idx] = intratemporal_error
                
                idx += 1
        
        return c_new, l_new, euler_errors, intratemporal_errors
    
    def solve(self, max_iter=2000, tol=1e-8, verbose=False):
        """
        Solve the model using fixed-point iteration
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress if True
        
        Returns:
        --------
        dict with solution information
        """
        # Calculate steady state
        z_ss = 1.0
        
        def steady_state_system(x):
            k, ℓ = x
            k = max(k, 1e-6)
            ℓ = max(ℓ, 1e-6)
            ℓ = min(ℓ, 1.0)
            c = z_ss * k**self.α * ℓ**(1-self.α) - self.δ * k
            if c <= 1e-10:
                return [1e10, 1e10]
            
            # Euler equation: 1 = β * R (independent of γ in steady state)
            R = self.α * z_ss * k**(self.α-1) * ℓ**(1-self.α) + (1 - self.δ)
            euler_err = 1 - self.β * R
            
            # Intratemporal FOC
            # For γ=1: EXACTLY as in NGM_stochastic_Chebyshev_labor.py
            # For γ≠1: Uses CRRA formula
            if self.γ == 1:
                # Log utility: EXACTLY as in original file
                # Intratemporal FOC error: χ * ℓ^(1/ν) = (1-α) * z * k^α * ℓ^(-α) / c
                lhs = self.χ * ℓ**(1/self.ν)
                rhs = (1-self.α) * z_ss * k**self.α * ℓ**(-self.α) / c
                intratemporal_err = lhs - rhs
            else:
                # CRRA utility: u'(c) = c^(-γ)
                # Intratemporal FOC: c^(-γ) / (χ * ℓ^(1/ν)) = (1-α) * z * k^α * ℓ^(-α)
                # Rearranging: c^(-γ) = χ * (1-α) * z * k^α * ℓ^(1/ν - α)
                c_safe = max(c, 1e-6)
                c_safe = min(c_safe, 100.0)
                
                log_c = np.log(c_safe)
                log_u_prime = -self.γ * log_c
                log_u_prime = np.clip(log_u_prime, -20, 20)
                lhs = np.exp(log_u_prime)  # This is c^(-γ)
                rhs = self.χ * (1-self.α) * z_ss * k**self.α * ℓ**(1/self.ν - self.α)
                intratemporal_err = lhs - rhs
            return [euler_err, intratemporal_err]
        
        k_guess = (self.β * self.α / (1 - self.β * (1 - self.δ)))**(1 / (1 - self.α))
        ℓ_guess = 0.33
        result = root(steady_state_system, [k_guess, ℓ_guess], method='hybr', options={'xtol': 1e-12})
        if result.success:
            k_ss, ℓ_ss = result.x[0], result.x[1]
        else:
            from scipy.optimize import minimize
            def obj(x):
                errs = steady_state_system(x)
                return errs[0]**2 + errs[1]**2
            result_min = minimize(obj, [k_guess, ℓ_guess], method='BFGS', bounds=[(1e-6, 100), (1e-6, 1.0)])
            k_ss, ℓ_ss = result_min.x[0], result_min.x[1]
        
        self.k_ss = k_ss
        self.ℓ_ss = ℓ_ss
        c_ss = z_ss * k_ss**self.α * ℓ_ss**(1-self.α) - self.δ * k_ss
        self.c_ss = c_ss
        
        # Set up grids
        self.k_low = 0.5 * k_ss
        self.k_high = 1.5 * k_ss
        self.z_low_log = -3 * np.sqrt(self.σ**2 / (1 - self.ρ**2))
        self.z_high_log = 3 * np.sqrt(self.σ**2 / (1 - self.ρ**2))
        
        cheb_nodes_k = Chebyshev_Nodes(self.n_k).ravel()
        cheb_nodes_z_log = Chebyshev_Nodes(self.n_z).ravel()
        
        self.k_grid = Change_Variable_Fromcheb(self.k_low, self.k_high, cheb_nodes_k)
        z_grid_log = Change_Variable_Fromcheb(self.z_low_log, self.z_high_log, cheb_nodes_z_log)
        self.z_grid = np.exp(z_grid_log)
        
        # Initialize consumption at grid points (constant at steady state) - EXACTLY as in original
        c_current = np.full(self.n_k * self.n_z, c_ss)
        
        # Initialize gamma coefficients (only for consumption) - EXACTLY as in original
        self.gamma_c = self.invert_to_gamma(c_current)
        
        # Compute labor from consumption using intratemporal FOC - EXACTLY as in original
        l_current = np.zeros(self.n_k * self.n_z)
        idx = 0
        for i_z in range(self.n_z):
            z = self.z_grid[i_z]
            for i_k in range(self.n_k):
                k = self.k_grid[i_k]
                l_current[idx] = self.l_from_c(k, z, c_current[idx])
                idx += 1
        
        # Fixed-point iteration (EXACTLY as in original)
        for iter in range(max_iter):
            # Compute Euler errors and update consumption (labor computed from consumption)
            c_new, l_new, euler_errors, intratemporal_errors = self.compute_euler_errors_and_update(c_current)
            
            # Check convergence: max Euler error (EXACTLY as in original)
            max_euler_error = np.max(np.abs(euler_errors))
            max_intratemporal_error = np.max(np.abs(intratemporal_errors))
            max_error = max(max_euler_error, max_intratemporal_error)
            
            # Print progress every 10 iterations (EXACTLY as in original)
            if verbose and (iter % 10 == 0 or max_error < tol):
                print(f"  Iteration {iter:4d}: max |Euler error| = {max_euler_error:.6e}, "
                      f"max |Intratemporal error| = {max_intratemporal_error:.6e}")
            
            # Check convergence
            if max_error < tol:
                if verbose:
                    print(f"  Converged after {iter} iterations!")
                break
            
            # Update consumption and labor with damping (EXACTLY as in original)
            # For high γ, use more aggressive damping to prevent instability
            if self.γ > 1:
                # For CRRA with γ > 1, use smaller damping to prevent numerical explosions
                # Adaptive damping: smaller when errors are large
                if max_error > 0.1:
                    damping_factor = 0.1  # Very small damping for large errors
                elif max_error > 0.01:
                    damping_factor = 0.2  # Small damping for moderate errors
                else:
                    damping_factor = min(self.dampening, 0.5)  # Cap at 0.5 for small errors
            else:
                damping_factor = self.dampening
            
            if max_error < 1.0:
                c_current = (1 - damping_factor) * c_current + damping_factor * c_new
                l_current = (1 - damping_factor) * l_current + damping_factor * l_new
            else:
                # Use even smaller damping if errors are large
                small_damping = damping_factor * 0.1
                c_current = (1 - small_damping) * c_current + small_damping * c_new
                l_current = (1 - small_damping) * l_current + small_damping * l_new
            
            # Invert consumption to get new gamma coefficients
            self.gamma_c = self.invert_to_gamma(c_current)
            
            # Recompute consumption from gamma, then compute labor from consumption
            # This ensures intratemporal FOC is always satisfied exactly (EXACTLY as in original)
            idx = 0
            for i_z in range(self.n_z):
                z = self.z_grid[i_z]
                for i_k in range(self.n_k):
                    k = self.k_grid[i_k]
                    c_current[idx] = self.c_cheb(k, z)
                    # Compute labor from consumption using intratemporal FOC
                    # This ensures the FOC is satisfied exactly
                    l_current[idx] = self.l_from_c(k, z, c_current[idx])
                    idx += 1
        
        if iter == max_iter - 1:
            if verbose:
                print(f"  Warning: Reached maximum iterations ({max_iter})")
        
        return {
            'max_euler_error': max_euler_error,
            'max_intratemporal_error': max_intratemporal_error,
            'iterations': iter + 1
        }

# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("SENSITIVITY ANALYSIS: Policy Functions at z=1")
    print("="*80)
    
    # Parameters to vary
    γ_values = [1, 4, 10]  # Risk aversion (CRRA parameter)
    σ_values = [0.003, 0.007, 0.015]  # Productivity volatility (low, medium, high)
    
    print(f"\nParameters:")
    print(f"  Risk aversion (γ): {γ_values}")
    print(f"  Volatility (σ): {σ_values}")
    print(f"  Grid size: n_k = n_z = 10")
    print("="*80)
    
    # Create output directory
    os.makedirs('../NGM_figures/stochastic/sensitivity_analysis', exist_ok=True)
    
    # Solve for each combination
    results = {}
    for γ in γ_values:
        results[γ] = {}
        for σ in σ_values:
            print(f"\nSolving for γ = {γ}, σ = {σ}...")
            model = StochasticNGMLabor(γ=γ, σ=σ, n_k=10, n_z=10, dampening=1.0)
            solution = model.solve(max_iter=2000, tol=1e-8, verbose=True)
            results[γ][σ] = model
    
    # Generate plots for each γ value
    z_ss = 1.0
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '--', '-.']
    σ_labels = [f'σ = {σ:.3f}' for σ in σ_values]
    
    for γ in γ_values:
        print(f"\nGenerating plots for γ = {γ}...")
        
        # Find common k domain
        all_k_lows = [results[γ][σ].k_low for σ in σ_values]
        all_k_highs = [results[γ][σ].k_high for σ in σ_values]
        k_low_common = min(all_k_lows)
        k_high_common = max(all_k_highs)
        k_fine = np.linspace(k_low_common, k_high_common, 200)
        
        # Create 1x3 figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Storage for policy functions
        c_policies = {σ: [] for σ in σ_values}
        k_prime_policies = {σ: [] for σ in σ_values}
        l_policies = {σ: [] for σ in σ_values}
        
        # Evaluate policy functions for each σ
        for σ in σ_values:
            model = results[γ][σ]
            
            for k_val in k_fine:
                # Skip if outside domain
                if k_val < model.k_low or k_val > model.k_high:
                    c_policies[σ].append(np.nan)
                    k_prime_policies[σ].append(np.nan)
                    l_policies[σ].append(np.nan)
                    continue
                
                # Consumption
                c_val = model.c_cheb(k_val, z_ss)
                c_policies[σ].append(c_val)
                
                # Labor
                ℓ_val = model.l_from_c(k_val, z_ss, c_val)
                l_policies[σ].append(ℓ_val)
                
                # Next period capital
                k_prime = (1 - model.δ) * k_val + z_ss * k_val**model.α * ℓ_val**(1-model.α) - c_val
                k_prime_policies[σ].append(k_prime)
        
        # Convert to arrays
        for σ in σ_values:
            c_policies[σ] = np.array(c_policies[σ])
            k_prime_policies[σ] = np.array(k_prime_policies[σ])
            l_policies[σ] = np.array(l_policies[σ])
        
        # Plot 1: Consumption
        ax1 = axes[0]
        for i, σ in enumerate(σ_values):
            valid = ~np.isnan(c_policies[σ])
            ax1.plot(k_fine[valid], c_policies[σ][valid], 
                    color=colors[i], linestyle=linestyles[i], 
                    linewidth=2.5, label=σ_labels[i])
        ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Consumption Policy Function\n(γ = {γ}, z = 1)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Capital accumulation
        ax2 = axes[1]
        for i, σ in enumerate(σ_values):
            valid = ~np.isnan(k_prime_policies[σ])
            ax2.plot(k_fine[valid], k_prime_policies[σ][valid], 
                    color=colors[i], linestyle=linestyles[i], 
                    linewidth=2.5, label=σ_labels[i])
        ax2.plot(k_fine, k_fine, 'k--', linewidth=1.5, alpha=0.5, label='45° line')
        ax2.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
        ax2.set_ylabel("k' (Next Period Capital)", fontsize=12, fontweight='bold')
        ax2.set_title(f'Capital Accumulation Policy Function\n(γ = {γ}, z = 1)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Labor
        ax3 = axes[2]
        for i, σ in enumerate(σ_values):
            valid = ~np.isnan(l_policies[σ])
            ax3.plot(k_fine[valid], l_policies[σ][valid], 
                    color=colors[i], linestyle=linestyles[i], 
                    linewidth=2.5, label=σ_labels[i])
        ax3.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('ℓ (Labor)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Labor Policy Function\n(γ = {γ}, z = 1)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'../NGM_figures/stochastic/sensitivity_analysis/policy_functions_z1_gamma{γ}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path}")
        plt.close()
    
    # ============================================================================
    # PRECAUTIONARY SAVINGS INVESTIGATION
    # ============================================================================
    print("\n" + "="*80)
    print("PRECAUTIONARY SAVINGS INVESTIGATION")
    print("="*80)
    print("\nAnalyzing how consumption at z=1 changes with productivity volatility (σ)")
    print("Precautionary savings: Higher σ → Lower consumption (more savings)")
    
    # Focus on γ = 4 for precautionary savings analysis
    γ_ps = 4
    σ_values_ps = [0.003, 0.007, 0.015, 0.03]  # Different volatility levels
    
    if γ_ps in γ_values:
        print(f"\nSolving models for γ = {γ_ps} with different σ values...")
        
        # Solve deterministic case for comparison
        print("  Solving deterministic case (σ ≈ 0)...")
        model_det = StochasticNGMLabor(γ=γ_ps, σ=0.0001, n_k=10, n_z=10, dampening=1.0)
        model_det.solve(max_iter=2000, tol=1e-8, verbose=False)
        
        # Solve stochastic cases
        models_ps = {}
        for σ in σ_values_ps:
            print(f"  Solving σ = {σ:.3f}...")
            if σ in σ_values and γ_ps in results and σ in results[γ_ps]:
                # Use already solved model if available
                models_ps[σ] = results[γ_ps][σ]
            else:
                # Solve new model
                model = StochasticNGMLabor(γ=γ_ps, σ=σ, n_k=10, n_z=10, dampening=1.0)
                model.solve(max_iter=2000, tol=1e-8, verbose=False)
                models_ps[σ] = model
        
        # Evaluate consumption at z=1 across different k values
        z_ps = 1.0  # Steady-state productivity
        k_fine_ps = np.linspace(model_det.k_low, model_det.k_high, 200)
        
        # Store consumption for each σ
        c_ps_data = {}
        c_ps_data[0.0] = []  # Deterministic
        
        for k_val in k_fine_ps:
            if k_val < model_det.k_low or k_val > model_det.k_high:
                c_ps_data[0.0].append(np.nan)
            else:
                c_det = model_det.c_cheb(k_val, z_ps)
                c_ps_data[0.0].append(c_det)
        
        for σ in σ_values_ps:
            model = models_ps[σ]
            c_ps_data[σ] = []
            for k_val in k_fine_ps:
                if k_val < model.k_low or k_val > model.k_high:
                    c_ps_data[σ].append(np.nan)
                else:
                    c_stoch = model.c_cheb(k_val, z_ps)
                    c_ps_data[σ].append(c_stoch)
        
        # Convert to arrays
        for σ_key in c_ps_data:
            c_ps_data[σ_key] = np.array(c_ps_data[σ_key])
        
        # Print summary statistics
        print("\n" + "="*80)
        print("PRECAUTIONARY SAVINGS SUMMARY: Consumption at z=1")
        print("="*80)
        
        # Compare at steady-state capital
        k_ss_ps = model_det.k_ss
        print(f"\nAt k = k_ss = {k_ss_ps:.6f}:")
        print(f"  σ       c(k_ss, z=1)    Difference from det    % Change")
        print(f"  {'-'*70}")
        
        c_det_ss = model_det.c_cheb(k_ss_ps, z_ps)
        print(f"  0.000   {c_det_ss:.6f}       0.000000           0.00%")
        
        for σ in σ_values_ps:
            model = models_ps[σ]
            c_stoch_ss = model.c_cheb(k_ss_ps, z_ps)
            diff = c_det_ss - c_stoch_ss
            pct_change = (diff / c_det_ss) * 100
            print(f"  {σ:.3f}   {c_stoch_ss:.6f}       {diff:+.6f}          {pct_change:+.4f}%")
        
        # Check if precautionary savings are present
        c_values_ss = [c_det_ss] + [models_ps[σ].c_cheb(k_ss_ps, z_ps) for σ in σ_values_ps]
        has_ps = c_det_ss > min(c_values_ss[1:])
        is_monotonic = all(c_values_ss[i] >= c_values_ss[i+1] for i in range(len(c_values_ss)-1))
        
        print(f"\nPrecautionary savings check:")
        if has_ps:
            max_ps = c_det_ss - min(c_values_ss[1:])
            print(f"  ✓ PRECAUTIONARY SAVINGS PRESENT")
            print(f"  Maximum precautionary savings: {max_ps:.6f} ({max_ps/c_det_ss*100:.2f}%)")
        else:
            print(f"  ✗ No precautionary savings detected")
        
        if is_monotonic:
            print(f"  ✓ Consumption decreases monotonically with σ")
        else:
            print(f"  ? Consumption pattern with σ is not monotonic")
        
        # Create precautionary savings plots
        print("\nGenerating precautionary savings plots...")
        
        # Plot 1: Consumption vs k for different σ values
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        colors_ps = ['black', 'blue', 'green', 'orange', 'red']
        linestyles_ps = ['--', '-', '-', '-', '-']
        σ_labels_ps = ['Deterministic (σ=0)'] + [f'σ = {σ:.3f}' for σ in σ_values_ps]
        
        ax1.plot(k_fine_ps, c_ps_data[0.0], color=colors_ps[0], linestyle=linestyles_ps[0],
                 linewidth=2.5, label=σ_labels_ps[0], alpha=0.8)
        
        for i, σ in enumerate(σ_values_ps):
            valid = ~np.isnan(c_ps_data[σ])
            ax1.plot(k_fine_ps[valid], c_ps_data[σ][valid], 
                    color=colors_ps[i+1], linestyle=linestyles_ps[i+1],
                    linewidth=2.5, label=σ_labels_ps[i+1])
        
        ax1.axvline(x=k_ss_ps, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='k_ss')
        ax1.set_xlabel('k (Capital)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Precautionary Savings: Consumption at z=1\n(γ = {γ_ps})', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path_ps1 = '../NGM_figures/stochastic/sensitivity_analysis/precautionary_savings_consumption.png'
        plt.savefig(output_path_ps1, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path_ps1}")
        plt.close()
        
        # Plot 2: Consumption vs σ at different k values
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        k_points_ps = [model_det.k_low, model_det.k_ss, model_det.k_high]
        k_labels_ps = [f'k = {k:.2f}' for k in k_points_ps]
        
        σ_plot = [0.0] + list(σ_values_ps)
        
        for k_val, k_label in zip(k_points_ps, k_labels_ps):
            c_plot = [model_det.c_cheb(k_val, z_ps)]
            for σ in σ_values_ps:
                model = models_ps[σ]
                c_val = model.c_cheb(k_val, z_ps)
                c_plot.append(c_val)
            
            ax2.plot(σ_plot, c_plot, 'o-', linewidth=2.5, markersize=8, label=k_label)
        
        ax2.set_xlabel('σ (Productivity Volatility)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('c (Consumption)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Precautionary Savings: Consumption vs Volatility at z=1\n(γ = {γ_ps})', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Add annotation
        if has_ps:
            max_ps_val = c_det_ss - min(c_values_ss[1:])
            ax2.text(0.05, 0.95, f'Max precautionary savings:\n{max_ps_val:.6f} ({max_ps_val/c_det_ss*100:.2f}%)', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path_ps2 = '../NGM_figures/stochastic/sensitivity_analysis/precautionary_savings_vs_sigma.png'
        plt.savefig(output_path_ps2, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path_ps2}")
        plt.close()
        
        # Plot 3: Percentage change in consumption vs σ
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        
        pct_changes = []
        for σ in σ_values_ps:
            model = models_ps[σ]
            c_stoch_ss = model.c_cheb(k_ss_ps, z_ps)
            pct_change = ((c_det_ss - c_stoch_ss) / c_det_ss) * 100
            pct_changes.append(pct_change)
        
        ax3.plot(σ_values_ps, pct_changes, 'o-', linewidth=2.5, markersize=10, 
                color='blue', label='% Change from deterministic')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.set_xlabel('σ (Productivity Volatility)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('% Change in Consumption', fontsize=12, fontweight='bold')
        ax3.set_title(f'Precautionary Savings: % Change in Consumption vs Volatility\n(γ = {γ_ps}, k = k_ss, z = 1)', 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path_ps3 = '../NGM_figures/stochastic/sensitivity_analysis/precautionary_savings_percentage.png'
        plt.savefig(output_path_ps3, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path_ps3}")
        plt.close()
        
        print("\n" + "="*80)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
