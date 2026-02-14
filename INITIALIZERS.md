# INITIALIZERS.md -- Mathematical Reference

All initializers are registered in `src/rp_study/models/initializers.py` via the `@register_initializer("name")` decorator and called via `initialize_layer(layer, strategy="name", **kwargs)`.

Notation: `d = fan_in` (input dimension), `W` is `(fan_out, fan_in)`.

---

## Baseline Initializers

### 1. `he` -- Standard He/Kaiming

**Formula:**
```
W_ij ~ N(0, sqrt(2/d))
```

**Motivation:** Preserves the variance of activations through ReLU layers (He et al., 2015). Since ReLU zeros out ~50% of values, the factor 2 compensates for this halving.

**Properties:**
- Gradient gain ~1.0 per layer (healthy flow)
- Severe geometric collapse in deep networks
- ~50% activation zeros, moderate neuron death

### 2. `xavier` -- Xavier/Glorot

**Formula:**
```
W_ij ~ N(0, sqrt(2 / (fan_in + fan_out)))
```

**Motivation:** Preserves variance for linear/sigmoid/tanh activations (not ReLU). Included as a reference baseline.

### 3. `uniform_he` -- Uniform with He Variance

**Formula:**
```
W_ij ~ U(-a, a),  where a = sqrt(6/d)
```
so that `Var(W) = a^2/3 = 2/d`.

**Motivation:** Same variance as He but with uniform distribution. Matches the original notebook's uniform initialization.

### 4. `orthogonal` -- Generic Orthogonal

**Formula:**
```
W = Q * gain,  where Q is from QR decomposition of a random Gaussian matrix
```
Default `gain=1.0`.

**Motivation:** Orthogonal weight matrices preserve norms exactly in the forward pass (before nonlinearity). Configurable gain parameter.

---

## Row-Centered Family (Advisor's Line of Investigation)

### 5. `row_centered_he` -- Row-Centered He

**Formula:**
```
W_ij^(He) ~ N(0, sqrt(2/d))
W_ij = W_ij^(He) - (1/d) * sum_k W_ik^(He)
```
Each row sums to zero: `sum_j W_ij = 0`.

**Motivation:** Advisor's proposal to prevent geometric collapse. When row sums are zero, the neuron output is invariant to constant shifts, which prevents the DC drift that accelerates angle contraction under the arc-cosine kernel.

**Properties:**
- Prevents geometric collapse (empirically validated)
- Creates a "gradient trap": the zero-sum constraint propagates through backprop
- Variance reduced by factor `(1 - 1/d)` per layer
- Nearly zero dead neurons (rows summing to zero guarantee both positive and negative pre-activations)

### 6. `row_centered_he_var_adj` -- Row-Centered + Variance Adjustment

**Formula:**
```
Step 1: W^(He) ~ N(0, sqrt(2/d))
Step 2: W_i = W_i^(He) - mean(W_i^(He))           (center each row)
Step 3: W_i = W_i * sqrt(2/d) / std(W_i)            (rescale to He variance)
```

**Motivation:** Row centering reduces per-element variance by factor `(1 - 1/d)`. Over many layers this compounds. This variant restores the target He variance after centering.

**Properties:**
- Same geometric benefits as `row_centered_he`
- Correct variance per element
- Gradient trap persists (zero-sum constraint is not affected by rescaling)

### 7. `row_centered_final` -- Centered with Tuned Factor 1.65

**Formula:**
```
W_ij ~ N(0, sqrt(1.65/d))
W_i = W_i - mean(W_i)                               (center)
W_i = W_i * sqrt(1.65/d) / std(W_i)                 (rescale)
```

**Motivation:** Data-driven tuning. With row-centering, the active fraction is ~60% (not 50% as in standard He). To achieve gain=1.0 per layer, we need `factor = 1/active_fraction = 1/0.6 ~= 1.65` instead of the standard He factor of 2.0.

**Properties:**
- Better gradient flow than `row_centered_he_var_adj` (tuned for gain=1.0)
- Still has the gradient trap from zero-sum constraint

---

## Experimental Initializers (Alternatives to Address the Gradient Trap)

### 8. `partial_centered_he` -- Soft Centering

**Formula:**
```
W_ij^(He) ~ N(0, sqrt(2/d))
W_ij = W_ij^(He) - alpha * mean_j(W_ij^(He))        (partial centering)
W_i = W_i * sqrt(2/d) / std(W_i)                     (rescale to He variance)
```
Default `alpha=0.5`.

**Motivation:** Instead of forcing rows to sum to exactly zero (hard constraint), use a "soft" centering with `alpha < 1`. This provides partial geometric benefit without the full gradient trap.

**Properties:**
- `alpha=0`: reduces to He (no centering)
- `alpha=1`: reduces to full row centering (full trap)
- `alpha=0.5`: a compromise -- partial geometry preservation, partial gradient trap
- Trade-off is tunable

### 9. `orthogonal_he` -- Orthogonal with He Scaling

**Formula:**
```
W = Q * sqrt(2),  where Q from QR decomposition
```

**Motivation:** Orthogonal rows are linearly independent (prevents collapse) but do NOT have the zero-sum constraint (no gradient trap). The gain `sqrt(2)` matches the He scaling for ReLU's ~50% active fraction.

**Properties:**
- Good geometric preservation (orthogonal rows)
- No gradient trap (rows don't sum to zero)
- May show mild gradient explosion (gain slightly > 1.0 with ReLU)

### 10. `orthogonal_tuned` -- Orthogonal with Tuned Gain

**Formula:**
```
W = Q * sqrt(1.65)
```

**Motivation:** If the active fraction with structured initialization is ~60% rather than ~50%, then `gain = sqrt(2)` overcompensates. Using `sqrt(1.65)` accounts for the higher survival rate.

**Properties:**
- More stable than `orthogonal_he` for deep networks
- Still no gradient trap

### 11. `centered_with_dc_he` -- Centered + DC Component

**Formula:**
```
Step 1: W^(He) ~ N(0, sqrt(2/d))
Step 2: W_i = W_i^(He) - mean(W_i^(He))             (center)
Step 3: W_i = W_i * sqrt(2/d) / std(W_i)             (rescale)
Step 4: W_ij = W_ij + delta * sqrt(2/d)              (add DC)
```
Default `dc_scale (delta) = 0.1`.

**Motivation:** After row-centering, add a small constant to every element. This breaks the zero-sum constraint (`sum_j W_ij = d * delta * sqrt(2/d) != 0`) while preserving most of the centering benefit.

**Properties:**
- Rows are approximately centered (small residual row sum)
- Gradient trap partially broken
- Trade-off controlled by `dc_scale`

### 12. `kernel_preserving` -- Optimization-Based

**Formula:**
```
min_W  sum_{i,j} ( <ReLU(W x_i), ReLU(W x_j)> / d_out  -  <x_i, x_j> )^2
      + lambda * sum_i ( ||ReLU(W x_i)||^2 / d_out  -  1 )^2
```
where `{x_i}` are random unit vectors (data-independent proxy).

Warm-started from He initialization. Optimized with Adam (lr=0.01, 200 steps).

**Motivation:** Directly minimize the kernel distortion caused by the ReLU nonlinearity. Instead of structural constraints (centering, orthogonality), find the weight matrix that best preserves inner products.

**Properties:**
- Best geometry preservation for shallow networks (1-10 layers)
- Expensive to compute (~200 optimizer steps per layer)
- May become numerically unstable at 20+ layers (each layer optimized independently, doesn't account for cumulative drift)
- Data-independent: optimizes over random unit vectors, not actual data

---

## Utility Initializer

### 13. `custom_variance` -- Custom Variance Scale

**Formula:**
```
W_ij ~ N(mean, sqrt(variance))
```

**Motivation:** Allows experimenting with different variance scales (e.g., `2/d`, `2.5/d`, `3/d`, `4/d`) and means. Useful for probing the sensitivity of gradient flow to the variance parameter.

---

## Quick Reference Table

| Name | Row Sum = 0? | Variance Adjusted? | Gradient Trap? | Best For |
|------|:---:|:---:|:---:|------|
| `he` | No | N/A (standard) | No | Gradient baseline |
| `row_centered_he` | Yes | No (reduced) | Yes | Geometry baseline |
| `row_centered_he_var_adj` | Yes | Yes | Yes | Geometry + correct variance |
| `row_centered_final` | Yes | Yes (tuned 1.65) | Yes | Best row-centered variant |
| `partial_centered_he` | Partial | Yes | Partial | Trade-off experiments |
| `orthogonal_he` | No | N/A (orthogonal) | No | Geometry without trap |
| `orthogonal_tuned` | No | N/A (tuned 1.65) | No | Stable orthogonal |
| `centered_with_dc_he` | Nearly | Yes | Partially | Breaking the trap |
| `kernel_preserving` | No | Optimized | No | Best geometry (shallow) |
