# CONTEXT.md -- Thesis Research Context

## Topic

**Initialization strategies for deep ReLU networks, studied through the lens of random projections and the arc-cosine kernel.**

The core question: how should we initialize weight matrices so that deep networks (with ReLU activations) preserve the geometry of input data while maintaining healthy gradient flow?

## The Problem

When data passes through multiple layers of `(Linear + ReLU)`, three failure modes emerge:

### 1. Geometric Collapse
The ReLU arc-cosine kernel `K(alpha) = (sin(alpha) + (pi - alpha) * cos(alpha)) / (2*pi)` contracts angles between vectors. After L layers, the angle between any two data points shrinks toward zero. Eventually all representations look the same -- the network cannot distinguish inputs.

### 2. Gradient Vanishing / Explosion
With standard He initialization (`Var(W) = 2/d`), the expected squared norm of activations is preserved through each layer. However, the gradient gain per layer depends on more subtle factors (active fraction, weight correlations). Small deviations from gain=1.0 compound exponentially over L layers.

### 3. Dead Neurons (Neuron Death)
A neuron that receives only negative pre-activations for all training samples outputs zero forever. Its gradient is also zero, so it can never recover. This "neuron death" phenomenon worsens with depth and with certain initialization strategies.

## The Advisor's Proposal: Row-Centered He

**Idea**: After sampling weights from He distribution, subtract the row mean:
```
W[i,:] = W_he[i,:] - mean(W_he[i,:])
```
This forces each row to sum to zero: `sum_j W[i,j] = 0`.

**Why it helps geometry**: When the row sum is zero, the output of a neuron is invariant to constant shifts in the input. This prevents the "DC drift" that accelerates geometric collapse. Empirically, row-centered networks maintain data geometry dramatically better through many layers.

**Why it hurts gradients**: The zero-sum constraint on rows creates a "gradient trap." During backpropagation, the structural constraint `sum_j W[i,j] = 0` induces a corresponding constraint on gradient updates, causing systematic gradient decay.

Additionally, row centering reduces the per-element variance by factor `(1 - 1/d)`. Over L layers, this compounds to `(1 - 1/d)^L`, which for `d=784, L=100` gives approximately `0.88` -- an 12% reduction even before considering the gradient trap.

## Current State of Experiments

### What We've Tried
We have 15 registered initialization strategies (see [INITIALIZERS.md](INITIALIZERS.md)). The key ones and their findings:

| Initializer | Geometry | Gradients | Notes |
|------------|----------|-----------|-------|
| `he` (baseline) | Collapses with depth | Healthy (gain ~1.0) | Standard, well-understood |
| `row_centered_he` | Preserved | Vanishing (trap) | Advisor's proposal |
| `row_centered_he_var_adj` | Preserved | Still vanishing | Variance fix not sufficient |
| `partial_centered_he` | Partially preserved | Better than full centering | Trade-off controlled by alpha |
| `orthogonal_he` | Good preservation | Healthy | No zero-sum constraint |
| `orthogonal_tuned` | Good preservation | Tuned gain ~1.0 | Factor 1.65 instead of 2.0 |
| `centered_with_dc_he` | Mostly preserved | Partially breaks trap | DC component breaks zero-sum |
| `kernel_preserving` | Best (optimized) | Unknown | Expensive; unstable at 20+ layers |
| `row_centered_final` | Preserved | Tuned for gain=1.0 | Magic factor 1.65 from active fraction |

### Key Trade-off
There is a fundamental tension: **geometric preservation vs. gradient health**. Full row-centering gives the best geometry but the worst gradients. Standard He gives the best gradients but the worst geometry. All alternatives sit somewhere on this Pareto frontier.

### Analysis Tools
We evaluate initializers on two axes:
1. **Geometry**: Pass MNIST/Fashion-MNIST through multi-layer RP+ReLU, then PCA to 2D. Check if class structure is preserved.
2. **Gradient flow**: Build a deep network with the initialization, do one forward+backward pass, measure per-layer gradient row norms, zero proportions, dead neurons.

Both are automated in `notebooks/05_initializer_dashboard.ipynb`.

## Current Work in Progress

The student is working closely with the advisor to **derive the backpropagation equations from scratch** for the row-centered case. The goal is to understand exactly *why* the gradients behave as they do -- specifically:
- How does the zero-sum constraint on W propagate through backprop?
- What is the exact per-layer gradient decay factor for row-centered networks?
- Can we find a modified centering that preserves the geometric benefit without the gradient penalty?

Once the math is clear, we expect to design new initialization strategies based on theoretical understanding rather than trial and error.

## Next Steps

1. Complete the backprop derivation for row-centered W
2. Identify the exact mechanism of the gradient trap
3. Design a principled initialization that achieves both geometry and gradient objectives
4. Validate with the dashboard notebook
