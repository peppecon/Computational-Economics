# Computational Economics: Projection Methods and NGM Applications

This repository contains implementations of numerical methods for solving dynamic economic models, focusing on **Projection Methods** (Chebyshev Polynomials) and their application to the **Neoclassical Growth Model (NGM)**.

## Project Overview

The project is structured to demonstrate:
1.  **Direct Implementation of Projection Methods**: Using Chebyshev polynomials to approximate value and policy functions.
2.  **Advanced Application**: Solving the **Stochastic NGM with Endogenous Labor Supply** using global solution methods.

## 1. Function Approximation & Teaching Material

The following figures illustrate the core concepts of function approximation, demonstrating the convergence properties and structural advantages of Chebyshev projection methods.

### Example 1: Convergence with Number of Points
![Convergence with Points](./presentation/figures/teaching/02_approximation_quality.png)

### Chebyshev Approximation: Smooth vs Non-Smooth Functions
![Smooth vs Non-Smooth](./presentation/figures/teaching/03_smooth_vs_nonsmooth.png)

### Chebyshev vs Taylor Polynomial: Exponential Function
![Chebyshev vs Taylor Exp](./presentation/figures/teaching/09a_chebyshev_vs_polynomial_exp.png)

### Chebyshev Approximation: Curvy Function
![Chebyshev Curvy](./presentation/figures/teaching/09b_chebyshev_vs_polynomial_curvy.png)

### Taylor Polynomial Approximation: Curvy Function
![Taylor Curvy](./presentation/figures/teaching/13_taylor_expansions_only.png)

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
![Deterministic Comparison](./solve_NGM_model/NGM_figures/deterministic_Chebyshev_direct_comparison.png)

### Stochastic NGM with Endogenous Labor Supply

#### Policy Functions & Results
The stochastic solution provides the policy functions for $c(k,z)$ and $l(k,z)$.

![2D Policy Functions](./solve_NGM_model/presentation/stochastic_Chebyshev_labor_presentation_2d.png)

![Policy Functions Z1](./solve_NGM_model/NGM_figures/stochastic/stochastic_labor_policy_functions_z1.png)

#### Euler Equation Errors: Convergence and Accuracy
Measurement of numerical precision as a function of the polynomial degree ($n$). As $n$ increases, the Euler residuals vanish, demonstrating high global accuracy.

| Degree $n=3$ | Degree $n=5$ |
| :---: | :---: |
| ![Euler n3](./solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n3.png) | ![Euler n5](./solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n5.png) |
| **Degree $n=10$** | **Degree $n=20$** |
| ![Euler n10](./solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n10.png) | ![Euler n20](./solve_NGM_model/NGM_figures/stochastic/stochastic_labor_euler_errors_3d_n20.png) |

---

## Repository Structure

- `./`: Core implementations and comparisons.
  - `chebyshev_loglinear_comparison/`: Codes for the comparison study.
  - `solve_NGM_model/`: Stochastic NGM with endogenous labor.

## Requirements
- Python 3.x
- NumPy, SciPy, Matplotlib
