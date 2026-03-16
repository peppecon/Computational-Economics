# Comparison: Original vs Chebyshev Code

## Parameters

### Original Code (`NGM_nolabor_QE_original.py`)
- **Inside `euler_err` function:**
  - β = 0.99
  - α = 0.33
  - δ = 0.025
  - **γ = 1** ⚠️ (This is what's actually used!)
  - ρ = 0.95
  - σ = 0.1

- **At module level (NOT used in euler_err):**
  - γ = 4 (This is misleading - not actually used!)

### Chebyshev Code (`NGM_nolabor_Chebyshev.py`)
- **At module level (used in euler_err):**
  - β = 0.99 ✓
  - α = 0.33 ✓
  - δ = 0.025 ✓
  - **γ = 1** ✓ (FIXED: was 4, now matches original)
  - ρ = 0.95 ✓
  - σ = 0.1 ✓

## Expectation Calculation

### Original:
```python
E = 0
for i_q in range(len(q_nodes)):
    e_prime = np.sqrt(2) * σ * q_nodes[i_q]
    z_prime = np.exp(ρ * np.log(z) + e_prime)
    c_prime = c_poly(k_prime, z_prime, η)
    E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))
E = E / np.sqrt(np.pi)
ssr += (E - c**(-γ))**2
```

### Chebyshev:
```python
E = 0
for i_q in range(len(q_nodes)):
    e_prime = np.sqrt(2) * σ * q_nodes[i_q]
    z_prime = np.exp(ρ * np.log(z) + e_prime)
    c_prime = c_cheb(k_prime, z_prime, gamma, ...)
    R_prime = α * z_prime * k_prime**(α-1) + (1-δ)
    E += q_weights[i_q] * β * c_prime**(-γ) * R_prime
E = E / np.sqrt(np.pi)
euler_error = E - c**(-γ)
ssr += euler_error**2
```

**Status:** ✅ Mathematically identical (just written differently)

## Loop Order

### Original:
```python
for i_k in range(len(k_grid)):  # k outer
    for i_z in range(len(z_grid)):  # z inner
```

### Chebyshev:
```python
for i_z in range(len(z_grid)):  # z outer
    for i_k in range(len(k_grid)):  # k inner
```

**Status:** ✅ Different but correct - Chebyshev needs z outer, k inner to match tensor product `np.kron(T_k, T_z)` ordering

## Grid Setup

### Original:
- Regular linspace grid: `k_grid = np.linspace(k_low, k_high, n_k)`
- Regular linspace in log space: `z_grid = np.exp(np.linspace(z_low, z_high, n_z))`
- n_k = 10, n_z = 10

### Chebyshev:
- Chebyshev nodes: `k_grid_cheb = Change_Variable_Fromcheb(k_low, k_high, cheb_nodes_k)`
- Chebyshev nodes in log space: `z_grid_cheb = np.exp(Change_Variable_Fromcheb(z_low_log, z_high_log, cheb_nodes_z_log))`
- n_k_cheb = 10, n_z_cheb = 10

**Status:** ✅ Different grid type (Chebyshev vs regular), but same number of points

## Additional Checks

### Original:
- No explicit checks for invalid k_prime

### Chebyshev:
- `if k_prime <= 0: return 1e10` (penalty for negative capital)
- (Previously had `if k_prime > 1.5 * k_high: return 1e10` but removed)

**Status:** ✅ Additional safeguard in Chebyshev (shouldn't affect valid solutions)

## Summary

**CRITICAL FIX APPLIED:** Changed γ from 4 to 1 to match the original code.

All other differences are either:
1. Necessary for Chebyshev implementation (grid type, loop order)
2. Additional safeguards (k_prime checks)
3. Cosmetic (variable naming, code organization)

The expectation calculation is mathematically identical.

