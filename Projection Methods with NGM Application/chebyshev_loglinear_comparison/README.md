# Chebyshev vs Log-Linear Approximation Comparison

This folder contains code and figures comparing Chebyshev polynomial approximation with log-linear approximations (first and second order) for various functions.

## Files

- `chebyshev_vs_loglinear.py` - Main script that generates comparison figures
- `figures/` - Directory containing all generated comparison figures

## Functions Tested

1. **Smooth Functions:**
   - `e^x` - Exponential function
   - `sin(2πx)` - Sinusoidal function (log-linear not applicable)
   - `x² + 0.01` - Quadratic function

2. **Non-Smooth Functions:**
   - `max(0, x-0.5) + 0.01` - Maximum function with kink
   - `|x-0.5| + 0.01` - Absolute value function
   - Step function - Discontinuous function

## Usage

Run the script from this directory:

```bash
conda run -n phd_econ python chebyshev_vs_loglinear.py
```

## Output

For each function, the script generates:
- **Top panel**: Comparison of true function, Chebyshev approximation, and log-linear approximations
- **Bottom panel**: Approximation errors (log scale)

All figures are saved in the `figures/` directory with filenames like `comparison_<function_name>.png`.

## Key Findings

- **Chebyshev polynomials** approximate well for both smooth and non-smooth functions
- **Log-linear approximations** work well for smooth functions (especially `e^x`) but fail dramatically for non-smooth functions
- **Chebyshev is more robust** across different function types

## Dependencies

- `numpy`
- `matplotlib`
- `functions_library.py` (from parent directory)

