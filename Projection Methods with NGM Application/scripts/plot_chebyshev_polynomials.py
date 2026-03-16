#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Chebyshev Polynomials up to 4th order
"""

import sys
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Create output directory
os.makedirs('latex/teaching', exist_ok=True)

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

# Set scientific plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.5,
    'patch.linewidth': 1.5,
})

# Define Chebyshev polynomials recursively
def chebyshev_polynomial(x, n):
    """Compute Chebyshev polynomial of order n at point x"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T_prev = np.ones_like(x)  # T_0
        T_curr = x  # T_1
        for i in range(2, n + 1):
            T_next = 2 * x * T_curr - T_prev
            T_prev = T_curr
            T_curr = T_next
        return T_curr

# Generate x values in [-1, 1]
x = np.linspace(-1, 1, 1000)

# Compute polynomials up to order 4
T0 = chebyshev_polynomial(x, 0)
T1 = chebyshev_polynomial(x, 1)
T2 = chebyshev_polynomial(x, 2)
T3 = chebyshev_polynomial(x, 3)
T4 = chebyshev_polynomial(x, 4)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each polynomial
ax.plot(x, T0, label=r'$T_0(x) = 1$', linewidth=2.5, color='#1f77b4')
ax.plot(x, T1, label=r'$T_1(x) = x$', linewidth=2.5, color='#ff7f0e')
ax.plot(x, T2, label=r'$T_2(x) = 2x^2 - 1$', linewidth=2.5, color='#2ca02c')
ax.plot(x, T3, label=r'$T_3(x) = 4x^3 - 3x$', linewidth=2.5, color='#d62728')
ax.plot(x, T4, label=r'$T_4(x) = 8x^4 - 8x^2 + 1$', linewidth=2.5, color='#9467bd')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Set labels and title
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$T_n(x)$', fontsize=14)
ax.set_title('Chebyshev Polynomials up to 4th Order', fontsize=16, pad=15)

# Set axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1.2, 1.2)

# Add legend
ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, ncol=1)

# Add horizontal and vertical lines at 0
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'latex/teaching/chebyshev_polynomials_4th_order.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")

plt.close()

