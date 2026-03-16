# Gauss-Hermite Quadrature Comparison

## Setup

### Original Code:
```python
# Line 32-35
n_q = 5  # Number of nodes and weights for the Gauss-Hermite quadrature
gh_quad = np.polynomial.hermite.hermgauss(n_q)
```

### Chebyshev Code:
```python
# Lines 43-45
# Number of nodes and weights for the Gauss-Hermite quadrature
n_q = 5
gh_quad = np.polynomial.hermite.hermgauss(n_q)
```

**Status:** ✅ **IDENTICAL** - Both use `n_q = 5` and same function

---

## Usage in euler_err Function

### Original Code:
```python
# Line 44: Unpacking
q_nodes, q_weights = quad

# Lines 67-75: Expectation calculation
for i_q in range(len(q_nodes)):
    e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # Transform GH node to shock
    z_prime = np.exp(ρ * np.log(z) + e_prime)       # Next period productivity
    c_prime = c_poly(k_prime, z_prime, η)           # Next period consumption
    
    E += q_weights[i_q] * β * c_prime**(-γ) * (α * z_prime * k_prime**(α-1) + (1-δ))

E = E / np.sqrt(np.pi)      # Normalize
ssr += (E - c**(-γ))**2     # Euler error
```

### Chebyshev Code:
```python
# Line 134: Unpacking
q_nodes, q_weights = quad

# Lines 160-179: Expectation calculation
for i_q in range(len(q_nodes)):
    e_prime = np.sqrt(2) * σ * q_nodes[i_q]         # Transform GH node to shock
    z_prime = np.exp(ρ * np.log(z) + e_prime)       # Next period productivity
    c_prime = c_cheb(k_prime, z_prime, gamma, ...)  # Next period consumption
    
    R_prime = α * z_prime * k_prime**(α-1) + (1-δ)
    E += q_weights[i_q] * β * c_prime**(-γ) * R_prime

E = E / np.sqrt(np.pi)      # Normalize
euler_error = E - c**(-γ)
ssr += euler_error**2       # Euler error
```

**Status:** ✅ **MATHEMATICALLY IDENTICAL**
- Same transformation: `e_prime = np.sqrt(2) * σ * q_nodes[i_q]`
- Same productivity evolution: `z_prime = np.exp(ρ * np.log(z) + e_prime)`
- Same expectation formula: `E += q_weights[i_q] * β * c_prime**(-γ) * R_prime`
- Same normalization: `E = E / np.sqrt(np.pi)`
- Same Euler error: `(E - c**(-γ))**2`

---

## Gauss-Hermite Quadrature Theory

The Gauss-Hermite quadrature approximates integrals of the form:
```
∫_{-∞}^{∞} f(x) * exp(-x²) dx ≈ (1/√π) * Σ w_i * f(√2 * σ * x_i)
```

For a normally distributed shock ε ~ N(0, σ²), we want to compute:
```
E[g(ε)] = ∫_{-∞}^{∞} g(ε) * (1/(σ√(2π))) * exp(-ε²/(2σ²)) dε
```

Using change of variable: ε = √2 * σ * x, we get:
```
E[g(ε)] = (1/√π) * ∫_{-∞}^{∞} g(√2 * σ * x) * exp(-x²) dx
        ≈ (1/√π) * Σ w_i * g(√2 * σ * x_i)
```

Where:
- `x_i` are the Gauss-Hermite nodes (from `hermgauss`)
- `w_i` are the Gauss-Hermite weights (from `hermgauss`)
- The factor `1/√π` comes from the normalization

---

## Verification

Both codes correctly implement:
1. ✅ **Node transformation**: `e_prime = np.sqrt(2) * σ * q_nodes[i_q]`
   - This transforms the standard GH nodes to shocks with std σ
   
2. ✅ **Productivity evolution**: `z_prime = np.exp(ρ * np.log(z) + e_prime)`
   - AR(1) process in logs: log(z') = ρ*log(z) + ε'
   - Where ε' = e_prime (the transformed shock)
   
3. ✅ **Expectation accumulation**: `E += q_weights[i_q] * β * c_prime**(-γ) * R_prime`
   - Weighted sum of the integrand evaluated at quadrature nodes
   
4. ✅ **Normalization**: `E = E / np.sqrt(np.pi)`
   - This is the `1/√π` factor from the GH quadrature formula

---

## Potential Issues to Check

### 1. **Quadrature Order (n_q = 5)**
- Both use 5 nodes, which should be sufficient for most problems
- If the function is highly nonlinear, may need more nodes

### 2. **Normalization Factor**
- Both use `1/√π` normalization
- This is correct for the standard Gauss-Hermite quadrature formula

### 3. **Shock Transformation**
- Both use `e_prime = np.sqrt(2) * σ * q_nodes[i_q]`
- This is correct: transforms standard GH nodes to N(0, σ²) shocks

### 4. **Productivity Process**
- Both use `z_prime = np.exp(ρ * np.log(z) + e_prime)`
- This correctly implements AR(1) in logs

---

## Conclusion

**The quadrature implementation is IDENTICAL between both codes.**

Both correctly:
- Set up Gauss-Hermite quadrature with 5 nodes
- Transform nodes to shocks: `e_prime = √2 * σ * q_nodes[i_q]`
- Evolve productivity: `z_prime = exp(ρ * log(z) + e_prime)`
- Accumulate expectation: `E += weights * integrand`
- Normalize: `E = E / √π`

**No issues found with the quadrature implementation.**

