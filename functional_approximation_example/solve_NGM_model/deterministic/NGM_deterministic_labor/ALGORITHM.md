# Mathematical Algorithm: Deterministic NGM with Labor Supply

## Model Setup

**Utility Function:**
\[
U(c, \ell) = \log(c) - \chi \frac{\ell^{1+\frac{1}{\nu}}}{1+\frac{1}{\nu}}
\]

**Production Function:**
\[
y = k^{\alpha} \ell^{1-\alpha}
\]

**Resource Constraint:**
\[
k' = (1-\delta)k + y - c
\]

## Equilibrium Conditions

### 1. Euler Equation (Intertemporal FOC)
\[
1 = \beta \frac{c}{c'} R'
\]
where \( R' = \alpha \frac{y'}{k'} + (1-\delta) = \alpha k'^{\alpha-1} \ell'^{1-\alpha} + (1-\delta) \)

### 2. Intratemporal FOC (Labor Supply)
\[
\chi \ell^{\frac{1}{\nu}} = (1-\alpha) \frac{y}{\ell} \frac{1}{c} = (1-\alpha) \frac{k^{\alpha} \ell^{-\alpha}}{c}
\]

Solving for labor:
\[
\ell(k, c) = \left[\frac{(1-\alpha) k^{\alpha}}{\chi c}\right]^{\frac{\nu}{1+\alpha\nu}}
\]

## Chebyshev Polynomial Approximation

### Approximation Form
We approximate the consumption policy function directly:
\[
c(k) \approx \sum_{i=0}^{p_k-1} \gamma_i T_i(\tilde{k})
\]
where:
- \( T_i(\tilde{k}) \) are Chebyshev polynomials of order \( i \)
- \( \tilde{k} = \frac{2k - (k_{low} + k_{high})}{k_{high} - k_{low}} \in [-1, 1] \) (transformed to Chebyshev domain)
- \( \gamma = [\gamma_0, \gamma_1, \ldots, \gamma_{p_k-1}] \) are coefficients to be determined

### Chebyshev Nodes
We use \( n_k \) Chebyshev nodes (typically \( n_k = p_k \)):
\[
\tilde{k}_j = \cos\left(\frac{(2j+1)\pi}{2n_k}\right), \quad j = 0, 1, \ldots, n_k-1
\]

Mapped to economic domain:
\[
k_j = \frac{k_{high} - k_{low}}{2} \tilde{k}_j + \frac{k_{high} + k_{low}}{2}
\]

## Algorithm: Fixed-Point Iteration with Collocation

### Step 0: Steady-State Calculation
Solve the system simultaneously:
\[
\begin{cases}
1 = \beta R_{ss} \quad \text{(Euler)} \\
\chi \ell_{ss}^{\frac{1}{\nu}} = \frac{(1-\alpha) k_{ss}^{\alpha} \ell_{ss}^{-\alpha}}{c_{ss}} \quad \text{(Intratemporal)} \\
c_{ss} = k_{ss}^{\alpha} \ell_{ss}^{1-\alpha} - \delta k_{ss} \quad \text{(Resource)}
\end{cases}
\]

### Step 1: Initialization
- Set \( c^{(0)}(k_j) = c_{ss} \) for all nodes \( j = 1, \ldots, n_k \)
- Compute initial coefficients \( \gamma^{(0)} \) by solving:
  \[
  \mathbf{T} \gamma^{(0)} = \mathbf{c}^{(0)}
  \]
  where \( \mathbf{T}_{ij} = T_i(\tilde{k}_j) \) and \( \mathbf{c}^{(0)} = [c^{(0)}(k_1), \ldots, c^{(0)}(k_{n_k})]^T \)

### Step 1: For Each Iteration \( t = 0, 1, 2, \ldots \)

#### 1.1: For Each Collocation Node \( k_j \), \( j = 1, \ldots, n_k \)

**Given:** Current consumption \( c^{(t)}(k_j) \) and coefficients \( \gamma^{(t)} \)

**Step 1.1.1:** Compute labor from intratemporal FOC
\[
\ell_j = \left[\frac{(1-\alpha) k_j^{\alpha}}{\chi c^{(t)}(k_j)}\right]^{\frac{\nu}{1+\alpha\nu}}
\]

**Step 1.1.2:** Compute output
\[
y_j = k_j^{\alpha} \ell_j^{1-\alpha}
\]

**Step 1.1.3:** Compute next-period capital
\[
k_j' = (1-\delta) k_j + y_j - c^{(t)}(k_j)
\]

**Step 1.1.4:** Evaluate consumption at \( k_j' \) using Chebyshev approximation
\[
c_j' = \sum_{i=0}^{p_k-1} \gamma_i^{(t)} T_i(\tilde{k}_j')
\]
where \( \tilde{k}_j' \) is the transformed value of \( k_j' \)

**Step 1.1.5:** Compute next-period labor
\[
\ell_j' = \left[\frac{(1-\alpha) (k_j')^{\alpha}}{\chi c_j'}\right]^{\frac{\nu}{1+\alpha\nu}}
\]

**Step 1.1.6:** Compute next-period return
\[
R_j' = \alpha (k_j')^{\alpha-1} (\ell_j')^{1-\alpha} + (1-\delta)
\]

**Step 1.1.7:** Compute Euler error
\[
EE_j^{(t)} = 1 - \beta \frac{c^{(t)}(k_j)}{c_j'} R_j'
\]

**Step 1.1.8:** Update consumption using Euler equation
From Euler: \( 1 = \beta \frac{c}{c'} R' \), so:
\[
c_j^{target} = \frac{c_j'}{\beta R_j'}
\]

Apply bounds:
\[
c_j^{new} = \max\left(\varepsilon, \min\left(c_j^{target}, y_j + (1-\delta)k_j\right)\right)
\]
where \( \varepsilon = 10^{-10} \) is a small positive number.

#### 1.2: Update Consumption at All Nodes

**Step 1.2.1:** Apply adaptive dampening
\[
c_j^{(t+1)} = (1 - \lambda^{(t)}) c_j^{(t)} + \lambda^{(t)} c_j^{new}
\]
where \( \lambda^{(t)} \) is the adaptive dampening parameter:
- Starts at \( \lambda^{(0)} = 1.0 \) (full update)
- Reduces if errors increase: \( \lambda^{(t+1)} = 0.9 \lambda^{(t)} \) if error increases by >10% for 3 consecutive iterations
- Increases if errors decrease: \( \lambda^{(t+1)} = 1.05 \lambda^{(t)} \) (capped at 1.0)

**Step 1.2.2:** Invert to get new coefficients
Solve:
\[
\mathbf{T} \gamma^{(t+1)} = \mathbf{c}^{(t+1)}
\]
where \( \mathbf{c}^{(t+1)} = [c^{(t+1)}(k_1), \ldots, c^{(t+1)}(k_{n_k})]^T \)

**Step 1.2.3:** Recompute consumption from coefficients (for consistency)
\[
c_j^{(t+1)} = \sum_{i=0}^{p_k-1} \gamma_i^{(t+1)} T_i(\tilde{k}_j)
\]

### Step 2: Check Convergence

Compute maximum Euler error:
\[
\max_j |EE_j^{(t+1)}|
\]

**Convergence criterion:**
\[
\max_j |EE_j^{(t+1)}| < \varepsilon_{tol}
\]
where \( \varepsilon_{tol} = 10^{-8} \)

If converged, stop. Otherwise, return to Step 1.

## Summary

The algorithm uses:
1. **Collocation method**: Enforces Euler equation at Chebyshev nodes
2. **Direct approximation**: \( c(k) = \gamma^T \mathbf{T}(\tilde{k}) \) (no log-exp transformation)
3. **Labor computed from consumption**: Uses intratemporal FOC analytically
4. **Fixed-point iteration**: Updates consumption to satisfy Euler equation
5. **Adaptive dampening**: Starts with full updates (\( \lambda = 1 \)), reduces only if needed

The method ensures:
- Intratemporal FOC satisfied exactly (labor computed analytically)
- Euler equation satisfied at collocation nodes (up to tolerance)
- Global approximation: can evaluate \( c(k) \) at any \( k \in [k_{low}, k_{high}] \)






