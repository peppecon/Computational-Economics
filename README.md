# Computational Economics: Projection Methods and NGM Applications

This repository contains implementations of numerical methods for solving dynamic economic models, focusing on **Projection Methods** (Chebyshev Polynomials) and their application to the **Neoclassical Growth Model (NGM)**.

## Project Overview

The project is structured to demonstrate:
1.  **Direct Implementation of Projection Methods**: Using Chebyshev polynomials to approximate value and policy functions.
2.  **Advanced Application**: Solving the **Stochastic NGM with Endogenous Labor Supply** using global solution methods.

## 1. Function Approximation & Teaching Material

The following figures illustrate the core concepts of function approximation, demonstrating the convergence properties and structural advantages of Chebyshev projection methods.

### Example 1: Convergence with Number of Points
![Convergence with Points](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/02_approximation_quality.png)

### Chebyshev Approximation: Smooth vs Non-Smooth Functions
![Smooth vs Non-Smooth](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/03_smooth_vs_nonsmooth.png)

### Chebyshev vs Taylor Polynomial: Exponential Function
![Chebyshev vs Taylor Exp](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/09a_chebyshev_vs_polynomial_exp.png)

### Approximation of a Curvy Function
Comparing the performance of global vs. local methods on functions with complex curvature.

| Chebyshev Approximation: Curvy Function | Taylor Polynomial Approximation: Curvy Function |
| :---: | :---: |
| ![Chebyshev Curvy](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/09b_chebyshev_vs_polynomial_curvy.png) | ![Taylor Curvy](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/13_taylor_expansions_only.png) |

---

## 2. Neoclassical Growth Model (NGM)

### Calibration Parameters
The model is calibrated using standard quarterly parameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| $\beta$ | 0.99 | Discount Factor |
| $\alpha$ | 0.33 | Capital Share |
| $\delta$ | 0.025 | Depreciation Rate |
| $\nu$ | 1.0 | Frisch Elasticity |
| $\rho$ | 0.95 | Persistence of Productivity Shock |
| $\sigma$ | 0.02 | Std. Dev. of Innovation |
| $L_{ss}$ | 0.33 | Target Labor Supply (Steady State) |

### Deterministic Solution Comparison
![Deterministic Comparison](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/deterministic_Chebyshev_direct_comparison.png)

### Stochastic NGM with Endogenous Labor Supply

#### Policy Functions & Results
The stochastic solution provides the policy functions for $c(k,z)$ and $l(k,z)$.

![3D Policy Functions](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/presentation/stochastic_Chebyshev_labor_presentation.png)

![Policy Functions Z1](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/stochastic/stochastic_labor_policy_functions_z1.png)

#### Accuracy & Convergence Analysis
We verify the solution using Euler equation residuals and observe the convergence as the polynomial degree increases.

![Euler Errors 3D Combined](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_combined.png)

![Convergence Study](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/stochastic/stochastic_labor_convergence_study.png)

| High Order Accuracy ($n=20$) | Lower Order Accuracy ($n=5$) |
| :---: | :---: |
| ![Euler Error n20](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n20.png) | ![Euler Error n5](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n5.png) |

---

## Repository Structure

- `Projection Methods with NGM Application/`: Core implementations and comparisons.
  - `chebyshev_loglinear_comparison/`: Codes for the comparison study.
  - `solve_NGM_model/`: Stochastic NGM with endogenous labor.

## Requirements
- Python 3.x
- NumPy, SciPy, Matplotlib
