#!/usr/bin/env python3
"""
Diagnostic script to understand why higher-order Taylor expansions 
can have larger errors
"""

import numpy as np
import matplotlib.pyplot as plt

def curvy_function(x):
    """Curvy function: exp(x) * sin(2*pi*x)"""
    return np.exp(x) * np.sin(2 * np.pi * x)

def compute_derivative(func, x0, n=1, dx=1e-6):
    """Compute numerical derivative using finite differences"""
    if n == 1:
        return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
    elif n == 2:
        return (func(x0 + dx) - 2 * func(x0) + func(x0 - dx)) / (dx**2)
    elif n == 3:
        return (func(x0 + 2*dx) - 2*func(x0 + dx) + 2*func(x0 - dx) - func(x0 - 2*dx)) / (2 * dx**3)
    elif n == 4:
        return (func(x0 + 2*dx) - 4*func(x0 + dx) + 6*func(x0) - 4*func(x0 - dx) + func(x0 - 2*dx)) / (dx**4)
    elif n == 5:
        return (func(x0 + 3*dx) - 5*func(x0 + 2*dx) + 10*func(x0 + dx) - 10*func(x0 - dx) + 5*func(x0 - 2*dx) - func(x0 - 3*dx)) / (2 * dx**5)
    else:
        raise ValueError(f"Derivative order {n} not implemented")

# Test parameters
x_min, x_max = 0, 2
x_center = 1.0
x_test = np.linspace(x_min, x_max, 1000)

print("="*70)
print("DIAGNOSING TAYLOR EXPANSION ERRORS")
print("="*70)
print(f"\nFunction: exp(x) * sin(2πx)")
print(f"Interval: [{x_min}, {x_max}]")
print(f"Expansion point (center): {x_center}")
print(f"Distance from center: max = {max(abs(x_min - x_center), abs(x_max - x_center))}")
print()

# Compute derivatives at center
print("Derivatives at expansion point (x_center = 1.0):")
print("-" * 70)
for n in range(1, 6):
    df = compute_derivative(curvy_function, x_center, n=n)
    print(f"f^({n})({x_center}) = {df:15.6e}")
print()

# Show how terms grow
print("Magnitude of Taylor terms at different points:")
print("-" * 70)
print(f"{'x':>8} {'|x-x0|':>12} {'1st term':>15} {'2nd term':>15} {'3rd term':>15} {'4th term':>15} {'5th term':>15}")
print("-" * 70)

test_points = [0.0, 0.5, 1.0, 1.5, 2.0]
for x in test_points:
    dx = x - x_center
    terms = []
    for n in range(1, 6):
        df = compute_derivative(curvy_function, x_center, n=n)
        factorial = np.math.factorial(n)
        term = abs(df / factorial * dx**n)
        terms.append(term)
    print(f"{x:8.1f} {abs(dx):12.4f} {terms[0]:15.6e} {terms[1]:15.6e} {terms[2]:15.6e} {terms[3]:15.6e} {terms[4]:15.6e}")

print()
print("="*70)
print("KEY OBSERVATIONS:")
print("="*70)
print("""
1. NUMERICAL ERRORS IN HIGH-ORDER DERIVATIVES:
   - Finite difference methods become increasingly inaccurate for higher-order derivatives
   - Small rounding errors get amplified when dividing by dx^n (where dx is very small)
   - The step size dx=1e-6 may not be optimal for all derivative orders

2. DISTANCE FROM EXPANSION POINT:
   - Taylor series are LOCAL approximations around x_center
   - At x=0 and x=2, we're 1 unit away from the center
   - Higher-order terms scale as (x - x_center)^n, which can be large far from center

3. LARGE DERIVATIVES:
   - The function exp(x) * sin(2πx) has derivatives that can be very large
   - Higher-order derivatives grow rapidly, especially with the exp(x) factor
   - When multiplied by (x - x_center)^n, these terms can dominate

4. TAYLOR SERIES CONVERGENCE:
   - Even for analytic functions, Taylor series may not converge well over the entire interval
   - Oscillatory functions (like sin) can cause the series to diverge or converge slowly
   - The radius of convergence might be smaller than the interval [0, 2]

5. NUMERICAL INSTABILITY:
   - High-order finite differences involve subtracting nearly equal numbers
   - This causes loss of precision (catastrophic cancellation)
   - The error in computing f^(n)(x0) gets multiplied by (x-x0)^n, amplifying errors
""")

# Compute actual Taylor approximations
print("\n" + "="*70)
print("COMPUTING ACTUAL ERRORS:")
print("="*70)

y_true = curvy_function(x_test)

for order in [1, 2, 3, 4, 5]:
    # Build Taylor series
    y_approx = curvy_function(x_center) * np.ones_like(x_test)
    for n in range(1, order + 1):
        df = compute_derivative(curvy_function, x_center, n=n)
        factorial = np.math.factorial(n)
        y_approx += (df / factorial) * (x_test - x_center)**n
    
    error = np.abs(y_true - y_approx)
    max_error = np.max(error)
    rmse = np.sqrt(np.mean(error**2))
    
    print(f"Order {order}: Max Error = {max_error:.6e}, RMSE = {rmse:.6e}")

print()
print("Notice how errors can increase with order due to the factors above!")


