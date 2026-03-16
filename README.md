# Computational Economics: Projection Methods and NGM Applications

This repository contains implementations of numerical methods for solving dynamic economic models, focusing on **Projection Methods** (Chebyshev Polynomials) and their application to the **Neoclassical Growth Model (NGM)**.

## Project Overview

The project is structured to demonstrate:
1.  **Direct Implementation of Projection Methods**: Using Chebyshev polynomials to approximate value and policy functions.
2.  **Performance Comparison**: Evaluating the accuracy and efficiency of projection methods against standard **Log-Linearization**.
3.  **Advanced Application**: Solving the **Stochastic NGM with Endogenous Labor Supply** using global solution methods.

## 1. Function Approximation & Teaching Material

The following figures illustrate the core concepts of function approximation used in this project, highlighting why global projection methods (like Chebyshev) are superior to local methods (like Taylor expansions) for many economic applications.

### Smoothness and Approximation
![Smooth vs Non-Smooth](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/03_smooth_vs_nonsmooth.png)
*Comparison of approximating smooth vs. non-smooth functions.*

### Global vs. Local Approximation
Comparing Taylor expansions (local) with Chebyshev polynomials (global):

| Local (Taylor) | Global (Chebyshev) |
| :---: | :---: |
| ![Taylor Expansion Only](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/13_taylor_expansions_only.png) | ![Chebyshev Only](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/14_chebyshev_only.png) |

### Efficiency of Chebyshev Polynomials
Chebyshev polynomials are particularly efficient at approximating complex curvatures without the oscillatory behavior seen in standard high-order polynomials (Runge's phenomenon).

![Chebyshev vs Polynomial Exp](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/09a_chebyshev_vs_polynomial_exp.png)
![Chebyshev vs Polynomial Curvy](Projection%20Methods%20with%20NGM%20Application/presentation/figures/teaching/09b_chebyshev_vs_polynomial_curvy.png)

---

## 2. Deterministic NGM: Model Comparison

We compare the deterministic steady state and transition paths computed via different numerical methods.

![Deterministic Comparison](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/NGM_figures/deterministic_Chebyshev_direct_comparison.png)
*Figure: Comparison of deterministic solutions.*

---

## 3. Stochastic NGM with Endogenous Labor Supply

We extend the model to include stochastic productivity shocks and an endogenous labor-leisure choice. 

### Policy Functions (Stochastic)
The 3D policy functions capture the interaction between capital, labor supply, and uncertainty across the state space.

![NGM Policy Functions 3D](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/presentation/stochastic_Chebyshev_labor_presentation.png)

### Performance: Projection vs. Log-Linearization
![Comparison: Chebyshev vs Log-Linear](Projection%20Methods%20with%20NGM%20Application/chebyshev_loglinear_comparison/figures/comparison_expx.png)
*Figure: Global Chebyshev vs. Local Log-Linearization errors.*

### Euler Equation Errors
Low errors across the state space confirm the high accuracy of the global Chebyshev solution.

![Euler Errors 3D](Projection%20Methods%20with%20NGM%20Application/solve_NGM_model/presentation/stochastic_Chebyshev_labor_euler_errors_3d.png)

---

## Repository Structure

- `Projection Methods with NGM Application/`: Core implementations and comparisons.
  - `chebyshev_loglinear_comparison/`: Codes for the comparison study.
  - `solve_NGM_model/`: Stochastic NGM with endogenous labor.
    - `stochastic/`: Main solver scripts.
    - `presentation/`: Generated figures and visualizations.

## Requirements
- Python 3.x
- NumPy, SciPy, Matplotlib
