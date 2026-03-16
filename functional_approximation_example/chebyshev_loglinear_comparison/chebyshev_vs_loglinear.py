#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of Chebyshev Polynomial Approximation vs Log-Linear Approximation

This script compares Chebyshev polynomial approximation with log-linear 
approximations (first and second order) for various functions:
- Smooth functions: e^x, sin(x), x^2
- Non-smooth functions: max(0, x-0.5), |x-0.5|, step function

Based on teaching_figures.py structure for Chebyshev approximation.

Author: Created for functional approximation presentation
"""

import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(script_dir)
sys.path.append(parent_dir)  # Add parent directory to path for functions_library

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from functions_library import *

# Create output directory
os.makedirs('figures', exist_ok=True)

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

# =============================================================================
# PARAMETERS
# =============================================================================
n_x = 20
p_x = 20
x_min, x_max = 0, 1
x_0 = (x_min + x_max) / 2  # Expansion point for log-linear

# Fine grid for evaluation
n_fine = 200
x_fine = np.linspace(x_min, x_max, n_fine)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def derivative(func, x0, dx=1e-6, n=1):
    """Compute numerical derivative using finite differences"""
    if n == 1:
        return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
    elif n == 2:
        return (func(x0 + dx) - 2 * func(x0) + func(x0 - dx)) / (dx**2)
    else:
        raise ValueError("n must be 1 or 2")

def chebyshev_approximation(func, n_x, p_x, x_min, x_max, x_eval):
    """
    Compute Chebyshev polynomial approximation of a function.
    Based on teaching_figures.py approach.
    
    Parameters:
    -----------
    func : callable
        Function to approximate
    n_x : int
        Number of Chebyshev nodes
    p_x : int
        Polynomial order
    x_min, x_max : float
        Domain boundaries
    x_eval : array
        Points at which to evaluate the approximation
        
    Returns:
    --------
    y_approx : array
        Approximated function values at x_eval
    """
    # Generate Chebyshev nodes and polynomial matrix
    T_x, cheb_nodes_x = Tx(n_x, p_x)
    grid_x = Change_Variable_Fromcheb(x_min, x_max, cheb_nodes_x)
    
    # Evaluate function at nodes
    f = func(grid_x)
    
    # Solve for optimal coefficients (closed-form when n_x = p_x)
    gamma_star = np.linalg.solve(T_x, f)
    
    # Evaluate at new points
    T_x_new, _ = Tx_new_points(x_eval, p_x)
    y_approx = gamma_star @ T_x_new
    
    return y_approx

def loglinear_approximation(func, x_0, x_eval, order=1):
    """
    Compute log-linear approximation of a function.
    
    Parameters:
    -----------
    func : callable
        Function to approximate (must be positive)
    x_0 : float
        Expansion point
    x_eval : array
        Points at which to evaluate the approximation
    order : int
        Order of approximation (1 or 2)
        
    Returns:
    --------
    y_approx : array
        Approximated function values at x_eval
    """
    f_0 = func(x_0)
    
    if f_0 <= 0:
        # If function is not positive, return NaN
        return np.full_like(x_eval, np.nan)
    
    log_f_0 = np.log(f_0)
    
    # Compute derivatives
    f_prime_0 = derivative(func, x_0, dx=1e-6)
    elasticity_1 = f_prime_0 / f_0
    
    if order == 1:
        # First-order: log(f(x)) ≈ log(f(x_0)) + (f'(x_0)/f(x_0)) * (x - x_0)
        log_y_approx = log_f_0 + elasticity_1 * (x_eval - x_0)
    elif order == 2:
        # Second-order: add quadratic term
        f_double_prime_0 = derivative(func, x_0, dx=1e-6, n=2)
        elasticity_2 = (f_double_prime_0 / f_0) - (f_prime_0 / f_0)**2
        log_y_approx = log_f_0 + elasticity_1 * (x_eval - x_0) + 0.5 * elasticity_2 * (x_eval - x_0)**2
    else:
        raise ValueError("order must be 1 or 2")
    
    # Clip log values to avoid overflow
    log_y_approx = np.clip(log_y_approx, -50, 50)
    y_approx = np.exp(log_y_approx)
    
    # Replace any inf or nan with a large number for plotting purposes
    y_approx = np.where(np.isfinite(y_approx), y_approx, 1e10)
    
    return y_approx

# =============================================================================
# DEFINE FUNCTIONS TO APPROXIMATE
# =============================================================================

functions = {
    'exp(x)': {
        'func': lambda x: np.exp(x),
        'label': r'$f(x) = e^x$',
        'smooth': True
    },
    'sin(2πx)': {
        'func': lambda x: np.sin(2 * np.pi * x),
        'label': r'$f(x) = \sin(2\pi x)$',
        'smooth': True,
        'positive': False  # Not positive, so log-linear won't work
    },
    'x^2': {
        'func': lambda x: x**2 + 0.01,  # Add small constant to make positive
        'label': r'$f(x) = x^2 + 0.01$',
        'smooth': True
    },
    'max(0, x-0.5)': {
        'func': lambda x: np.maximum(0, x - 0.5) + 0.01,  # Add small constant
        'label': r'$f(x) = \max(0, x-0.5) + 0.01$',
        'smooth': False
    },
    '|x-0.5|': {
        'func': lambda x: np.abs(x - 0.5) + 0.01,  # Add small constant
        'label': r'$f(x) = |x-0.5| + 0.01$',
        'smooth': False
    },
    'step function': {
        'func': lambda x: 0.5 * (x >= 0.5) + 0.01,  # Step at x=0.5
        'label': r'$f(x) = \mathbf{1}_{x \geq 0.5} + 0.01$',
        'smooth': False
    }
}

# =============================================================================
# COMPUTE APPROXIMATIONS AND PLOT
# =============================================================================

print("="*60)
print("Chebyshev vs Log-Linear Approximation Comparison")
print("="*60)

for func_name, func_info in functions.items():
    func = func_info['func']
    label = func_info['label']
    is_smooth = func_info.get('smooth', True)
    is_positive = func_info.get('positive', True)
    
    print(f"\nProcessing: {func_name}")
    
    # True function
    y_true = func(x_fine)
    
    # Chebyshev approximation
    y_cheb = chebyshev_approximation(func, n_x, p_x, x_min, x_max, x_fine)
    
    # Log-linear approximations (only if function is positive)
    if is_positive:
        y_loglin_1 = loglinear_approximation(func, x_0, x_fine, order=1)
        y_loglin_2 = loglinear_approximation(func, x_0, x_fine, order=2)
    else:
        y_loglin_1 = np.full_like(x_fine, np.nan)
        y_loglin_2 = np.full_like(x_fine, np.nan)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top panel: Function values
    ax1 = axes[0]
    ax1.plot(x_fine, y_true, 'k-', linewidth=2, label='True function')
    ax1.plot(x_fine, y_cheb, 'b--', linewidth=2, label='Chebyshev')
    
    if is_positive:
        ax1.plot(x_fine, y_loglin_1, 'r:', linewidth=2, label='Log-linear (1st order)')
        ax1.plot(x_fine, y_loglin_2, 'g:', linewidth=2, label='Log-linear (2nd order)')
    
    ax1.axvline(x_0, color='gray', linestyle='--', alpha=0.5, label=f'Expansion point (x0={x_0:.2f})')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title(f'Approximation Comparison: {label}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Approximation errors
    ax2 = axes[1]
    error_cheb = np.abs(y_true - y_cheb)
    # Filter out any inf or nan values for plotting
    valid_cheb = np.isfinite(error_cheb)
    ax2.semilogy(x_fine[valid_cheb], error_cheb[valid_cheb], 'b--', linewidth=2, label='Chebyshev error')
    
    if is_positive:
        error_loglin_1 = np.abs(y_true - y_loglin_1)
        error_loglin_2 = np.abs(y_true - y_loglin_2)
        # Filter out inf/nan values
        valid_loglin_1 = np.isfinite(error_loglin_1) & (error_loglin_1 < 1e10)
        valid_loglin_2 = np.isfinite(error_loglin_2) & (error_loglin_2 < 1e10)
        if np.any(valid_loglin_1):
            ax2.semilogy(x_fine[valid_loglin_1], error_loglin_1[valid_loglin_1], 'r:', linewidth=2, label='Log-linear (1st) error')
        if np.any(valid_loglin_2):
            ax2.semilogy(x_fine[valid_loglin_2], error_loglin_2[valid_loglin_2], 'g:', linewidth=2, label='Log-linear (2nd) error')
    
    ax2.axvline(x_0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('|Error| (log scale)', fontsize=12)
    ax2.set_title('Approximation Errors', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = func_name.replace(' ', '_').replace('(', '').replace(')', '').replace('|', 'abs')
    filepath = os.path.join('figures', f'comparison_{filename}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")
    plt.close()
    
    # Print summary statistics
    print(f"  Chebyshev - Max error: {np.max(error_cheb):.2e}, Mean error: {np.mean(error_cheb):.2e}")
    if is_positive:
        print(f"  Log-linear (1st) - Max error: {np.max(error_loglin_1):.2e}, Mean error: {np.mean(error_loglin_1):.2e}")
        print(f"  Log-linear (2nd) - Max error: {np.max(error_loglin_2):.2e}, Mean error: {np.mean(error_loglin_2):.2e}")

print("\n" + "="*60)
print("All comparisons completed!")
print("="*60)
print(f"\nFigures saved in: {os.path.join(script_dir, 'figures/')}")

