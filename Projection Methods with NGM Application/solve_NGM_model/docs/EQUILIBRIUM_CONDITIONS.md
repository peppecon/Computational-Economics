# Stochastic Neoclassical Growth Model with Labor - Equilibrium Conditions and Algorithm

## Model Setup

### Utility Function
\[
U(c_t, \ell_t) = \log(c_t) - \chi \frac{\ell_t^{1+1/\nu}}{1+1/\nu}
\]

where:
- \(c_t\) is consumption
- \(\ell_t\) is labor supply
- \(\chi > 0\) is the labor disutility parameter
- \(\nu > 0\) is the Frisch elasticity parameter (inverse of labor supply elasticity)

### Production Function
\[
y_t = z_t k_t^\alpha \ell_t^{1-\alpha}
\]

where:
- \(y_t\) is output
- \(k_t\) is capital
- \(z_t\) is productivity (stochastic)
- \(\alpha \in (0,1)\) is the capital share

### Productivity Process
\[
\log(z_{t+1}) = \rho \log(z_t) + \varepsilon_{t+1}
\]

where:
- \(\rho \in (0,1)\) is the persistence parameter
- \(\varepsilon_{t+1} \sim N(0, \sigma^2)\) is an i.i.d. shock

### Resource Constraint
\[
k_{t+1} = (1-\delta) k_t + z_t k_t^\alpha \ell_t^{1-\alpha} - c_t
\]

where \(\delta \in (0,1)\) is the depreciation rate.

---

## Equilibrium Conditions

### 1. Euler Equation (Intertemporal FOC)

The Euler equation equates the marginal utility of consumption today with the expected discounted marginal utility of consumption tomorrow times the return on capital:

\[
u'(c_t) = \beta \mathbb{E}_t \left[ u'(c_{t+1}) R_{t+1} \right]
\]

For log utility (\(u(c) = \log(c)\)), we have \(u'(c) = 1/c\), so:

\[
\frac{1}{c_t} = \beta \mathbb{E}_t \left[ \frac{1}{c_{t+1}} R_{t+1} \right]
\]

where the return on capital is:

\[
R_{t+1} = \alpha z_{t+1} k_{t+1}^{\alpha-1} \ell_{t+1}^{1-\alpha} + (1-\delta)
\]

**Euler Error:**
\[
\text{EE}(k_t, z_t) = \beta \mathbb{E}_t \left[ \frac{1}{c(k_{t+1}, z_{t+1})} R_{t+1} \right] - \frac{1}{c(k_t, z_t)}
\]

### 2. Intratemporal FOC (Labor-Leisure Choice)

The intratemporal FOC equates the marginal disutility of labor with the marginal utility of consumption times the wage:

\[
\chi \ell_t^{1/\nu} = \frac{(1-\alpha) z_t k_t^\alpha \ell_t^{-\alpha}}{c_t}
\]

Rearranging:

\[
\chi \ell_t^{1/\nu + \alpha} = \frac{(1-\alpha) z_t k_t^\alpha}{c_t}
\]

Solving for labor:

\[
\ell_t = \left[ \frac{(1-\alpha) z_t k_t^\alpha}{\chi c_t} \right]^{\frac{\nu}{1+\alpha\nu}}
\]

**Intratemporal Error:**
\[
\text{IE}(k_t, z_t) = \chi \ell_t^{1/\nu} - \frac{(1-\alpha) z_t k_t^\alpha \ell_t^{-\alpha}}{c_t}
\]

Since labor is computed directly from consumption using the FOC, this error should be zero (up to numerical precision).

---

## Policy Functions

We approximate the consumption policy function using Chebyshev polynomials:

\[
c(k, z) = \sum_{i=0}^{n_k-1} \sum_{j=0}^{n_z-1} \gamma_{i,j}^c T_i(\tilde{k}) T_j(\tilde{z})
\]

where:
- \(T_i(\cdot)\) are Chebyshev polynomials
- \(\tilde{k} = \frac{2k - (k_{\text{low}} + k_{\text{high}})}{k_{\text{high}} - k_{\text{low}}}\) maps \(k \in [k_{\text{low}}, k_{\text{high}}]\) to \([-1, 1]\)
- \(\tilde{z} = \frac{2\log(z) - (z_{\text{low}}^{\log} + z_{\text{high}}^{\log})}{z_{\text{high}}^{\log} - z_{\text{low}}^{\log}}\) maps \(\log(z)\) to \([-1, 1]\)
- \(\gamma_{i,j}^c\) are the Chebyshev coefficients

Labor is computed directly from consumption:

\[
\ell(k, z) = \left[ \frac{(1-\alpha) z k^\alpha}{\chi c(k, z)} \right]^{\frac{\nu}{1+\alpha\nu}}
\]

---

## Numerical Integration (Gauss-Hermite Quadrature)

The expectation in the Euler equation is computed using Gauss-Hermite quadrature:

\[
\mathbb{E}_t \left[ \frac{1}{c(k_{t+1}, z_{t+1})} R_{t+1} \right] \approx \frac{1}{\sqrt{\pi}} \sum_{q=1}^{n_q} w_q \frac{1}{c(k_{t+1}, z_{t+1}^q)} R_{t+1}^q
\]

where:
- \(n_q\) is the number of quadrature nodes
- \(w_q\) are the Gauss-Hermite weights
- \(z_{t+1}^q = \exp(\rho \log(z_t) + \sqrt{2}\sigma \xi_q)\) with \(\xi_q\) being the GH nodes
- \(k_{t+1} = (1-\delta) k_t + z_t k_t^\alpha \ell_t^{1-\alpha} - c_t\)

---

## Algorithm: Fixed-Point Iteration with Collocation

### Step 0: Initialization

1. Set up Chebyshev nodes:
   - Capital nodes: \(k_i \in [k_{\text{low}}, k_{\text{high}}]\), \(i = 1, \ldots, n_k\)
   - Productivity nodes: \(z_j \in [z_{\text{low}}, z_{\text{high}}]\), \(j = 1, \ldots, n_z\)

2. Initialize consumption: \(c^{(0)}(k_i, z_j) = c_{ss}\) (steady-state consumption)

3. Compute initial coefficients: \(\gamma_c^{(0)}\) from \(c^{(0)}\) using least squares

### Step 1: Compute Euler Errors and Update Consumption

For each collocation point \((k_i, z_j)\):

1. **Compute labor from consumption:**
   \[
   \ell_{i,j} = \left[ \frac{(1-\alpha) z_j k_i^\alpha}{\chi c^{(n)}(k_i, z_j)} \right]^{\frac{\nu}{1+\alpha\nu}}
   \]

2. **Compute next period capital:**
   \[
   k'_{i,j} = (1-\delta) k_i + z_j k_i^\alpha \ell_{i,j}^{1-\alpha} - c^{(n)}(k_i, z_j)
   \]

3. **Compute expectation using Gauss-Hermite quadrature:**
   \[
   E_{i,j} = \frac{1}{\sqrt{\pi}} \sum_{q=1}^{n_q} w_q \frac{1}{c(k'_{i,j}, z'_{i,j,q})} R'_{i,j,q}
   \]
   where:
   - \(z'_{i,j,q} = \exp(\rho \log(z_j) + \sqrt{2}\sigma \xi_q)\)
   - \(R'_{i,j,q} = \alpha z'_{i,j,q} (k'_{i,j})^{\alpha-1} \ell'_{i,j,q}^{1-\alpha} + (1-\delta)\)
   - \(\ell'_{i,j,q} = \left[ \frac{(1-\alpha) z'_{i,j,q} (k'_{i,j})^\alpha}{\chi c(k'_{i,j}, z'_{i,j,q})} \right]^{\frac{\nu}{1+\alpha\nu}}\)

4. **Compute Euler error:**
   \[
   \text{EE}_{i,j} = E_{i,j} - \frac{1}{c^{(n)}(k_i, z_j)}
   \]

5. **Update consumption:**
   \[
   c^{\text{new}}_{i,j} = (1-\lambda) c^{(n)}_{i,j} + \lambda \frac{1}{E_{i,j}}
   \]
   where \(\lambda \in (0,1]\) is the damping parameter.

6. **Update labor:**
   \[
   \ell^{\text{new}}_{i,j} = \left[ \frac{(1-\alpha) z_j k_i^\alpha}{\chi c^{\text{new}}_{i,j}} \right]^{\frac{\nu}{1+\alpha\nu}}
   \]

### Step 2: Update Coefficients

1. **Invert consumption to get new coefficients:**
   \[
   \gamma_c^{(n+1)} = \arg\min_{\gamma} \sum_{i,j} \left[ c^{\text{new}}_{i,j} - \sum_{i',j'} \gamma_{i',j'} T_{i'}(\tilde{k}_i) T_{j'}(\tilde{z}_j) \right]^2
   \]

   This is solved using least squares:
   \[
   \gamma_c^{(n+1)} = (T^T T)^{-1} T^T c^{\text{new}}
   \]
   where \(T\) is the matrix of Chebyshev polynomials evaluated at grid points.

2. **Recompute consumption from new coefficients:**
   \[
   c^{(n+1)}(k_i, z_j) = \sum_{i',j'} \gamma_{i',j'}^{(n+1)} T_{i'}(\tilde{k}_i) T_{j'}(\tilde{z}_j)
   \]

3. **Recompute labor from new consumption:**
   \[
   \ell^{(n+1)}(k_i, z_j) = \left[ \frac{(1-\alpha) z_j k_i^\alpha}{\chi c^{(n+1)}(k_i, z_j)} \right]^{\frac{\nu}{1+\alpha\nu}}
   \]

### Step 3: Check Convergence

Compute:
\[
\text{max error} = \max_{i,j} |\text{EE}_{i,j}|
\]

If \(\text{max error} < \text{tolerance}\), stop. Otherwise, set \(n = n+1\) and go to Step 1.

---

## Key Features

1. **Collocation Method**: The Euler equation is enforced exactly at the Chebyshev nodes.

2. **Intratemporal FOC**: Labor is computed directly from consumption, ensuring the intratemporal FOC is satisfied exactly (up to numerical precision).

3. **Clipping**: When evaluating the policy function, if \(k'\) or \(z'\) fall outside the Chebyshev domain, they are clipped to \([-1, 1]\):
   \[
   \tilde{k}' = \max(-1, \min(1, \tilde{k}'))
   \]
   \[
   \tilde{z}' = \max(-1, \min(1, \tilde{z}'))
   \]

4. **Damping**: The update uses damping to ensure stability:
   \[
   c^{\text{new}} = (1-\lambda) c^{\text{old}} + \lambda c^{\text{target}}
   \]
   where \(\lambda\) is typically 1.0 but can be reduced if errors are large.

---

## Summary

The algorithm solves for the consumption policy function \(c(k, z)\) that satisfies:
- **Euler equation**: Intertemporal optimality condition
- **Intratemporal FOC**: Labor-leisure optimality (satisfied exactly by construction)
- **Resource constraint**: Feasibility condition

The solution is found iteratively by:
1. Evaluating the Euler equation at collocation points
2. Updating consumption to reduce Euler errors
3. Updating Chebyshev coefficients to match the new consumption values
4. Repeating until convergence




