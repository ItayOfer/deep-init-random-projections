# Meeting Prep: Product-Balanced Initializers

**Date:** April 13, 2026
**Notebook:** `notebooks/09_depth_geometry_comparison.ipynb`
**Figures:** `reports/latex/product_balanced_report/figures/`

---

## How to use this document

Open the figures directory (or the executed notebook) side by side with this file. Each section tells you which figure to show and what to say.

---

## Part 1: The Product-Balanced Initializer (V1)

### 1.1 Setup and notation

> **Show:** nothing yet, verbal.

We have a deep ReLU network with constant width $d$ and $L$ hidden layers. Each weight matrix $W^{(\ell)} \in \mathbb{R}^{d \times d}$ is row-centered: $\sum_j W_{ij}^{(\ell)} = 0$ for every row $i$.

We write the per-element variance as $\text{Var}(W_{ij}) = v/d$ and define a variance factor:

$$s = \sqrt{v/2}$$

When $s = 1$ we recover He variance ($v = 2$, i.e., $\text{Var}(W_{ij}) = 2/d$).

**Key point for the advisor:** the derivation below assumes constant width $d$, but the final formula $\text{Var}(W_{ij}^{(\ell)}) = 2\sqrt{\pi/(\pi-1)}/d_{\ell-1}$ applies to any architecture — each layer just uses its own fan-in $d_{\ell-1}$. The variance *factor* $v^* \approx 2.422$ is universal, independent of width or architecture.

### 1.2 Forward gain derivation

> **Show:** `forward_gain_multi_depth.png` — point to V1 (purple) being flat at ~0.909.

Row centering makes each neuron compute:

$$z_i = \sum_j W_{ij} a_j = \sum_j W_{ij}(a_j - \bar{a})$$

because $\sum_j W_{ij} = 0$. The neuron only sees the deviations $a_j - \bar{a}$, not the mean $\bar{a}$.

Post-ReLU activations $a = \text{ReLU}(z)$ with $z \sim \mathcal{N}(0, \sigma^2)$ satisfy:

$$\mathbb{E}[a] = \sigma/\sqrt{2\pi}, \quad \mathbb{E}[a^2] = \sigma^2/2, \quad \text{Var}(a) = \sigma^2(\pi - 1)/(2\pi)$$

The fraction of energy in the deviations (what row-centered weights access):

$$\frac{\text{Var}(a)}{\mathbb{E}[a^2]} = \frac{\pi - 1}{\pi} \approx 0.682$$

This is a universal ReLU property — independent of $\sigma$ or $d$. We define $r = \sqrt{(\pi-1)/\pi} \approx 0.826$.

Since row-centered weights see $\text{Var}(a)$ instead of $\mathbb{E}[a^2]$:

$$g_{\text{fwd}} = r \cdot s$$

The factor $r$ is structural — no variance adjustment can remove it.

### 1.3 Backward gain derivation

> **Show:** `backward_gain_multi_depth.png` — point to V1 (purple) being flat at ~1.10.

The backward pass uses $W^\top$, which has columns summing to zero. By the same logic, $W^\top$ is blind to the mean of its input $\delta^{(\ell+1)}$. But error signals have near-zero mean ($\text{Var}(\delta)/\mathbb{E}[\delta^2] \approx 1$), so there's almost no DC component to lose:

$$g_{\text{bwd}} = s$$

### 1.4 The coupling and the product

> **Show:** `gain_product_multi_depth.png` (top row, zoomed) — V1 (purple) flat at exactly 1.0.

Dividing: $g_{\text{fwd}}/g_{\text{bwd}} = r \approx 0.826$, always, regardless of $s$. No single scalar variance makes both gains equal to 1.

**The product-balanced solution:** set $g_{\text{fwd}} \cdot g_{\text{bwd}} = 1$:

$$r \cdot s^2 = 1 \implies s^* = \left(\frac{\pi}{\pi - 1}\right)^{1/4} \approx 1.1006$$

$$v^* = 2(s^*)^2 = 2\sqrt{\frac{\pi}{\pi-1}} \approx 2.4224$$

$$\boxed{\text{Var}(W_{ij}) = \frac{2\sqrt{\pi/(\pi-1)}}{d} \approx \frac{2.422}{d}}$$

Resulting gains:
- $g_{\text{fwd}}^* = r \cdot s^* = ((\pi-1)/\pi)^{1/4} \approx 0.909$
- $g_{\text{bwd}}^* = s^* = (\pi/(\pi-1))^{1/4} \approx 1.101$
- $g_{\text{fwd}}^* \cdot g_{\text{bwd}}^* = 1$ (exact)

The initialization procedure:
1. Set $\sigma = \sqrt{2\sqrt{\pi/(\pi-1)} / d}$
2. Sample $W_{ij} \sim \mathcal{N}(0, \sigma^2)$
3. Row-center: $W_{ij} \leftarrow W_{ij} - \frac{1}{d}\sum_k W_{ik}$
4. Rescale each row: $w_i \leftarrow w_i \cdot \sigma / \text{std}(w_i)$
5. Set bias to zero

### 1.5 Why uniform variance still gives non-uniform gradients

> **Show:** `gradient_norms_linear_L50.png` — look at V1 panel (3rd from top): max=20298, min=10.7, ratio=1902x.

From BP4: $\|\partial\mathcal{C}/\partial W^{(\ell)}\| \propto \text{rms}(A^{(\ell-1)}) \cdot \text{rms}(\Delta^{(\ell)})$

The forward chain: $\text{rms}(A^{(\ell-1)}) \propto (r \cdot s^*)^{\ell-1}$

The backward chain: $\text{rms}(\Delta^{(\ell)}) \propto (s^*)^{L-\ell}$

Since $\prod s_k$ is constant across $\ell$ (uniform variance), the gradient simplifies to:

$$\left\|\frac{\partial\mathcal{C}}{\partial W^{(\ell)}}\right\| \propto \frac{r^{\ell-1}}{s_\ell} = \frac{r^{\ell-1}}{s^*}$$

This is a monotonically decreasing function of $\ell$ (since $r < 1$). Layer 1 gets $r^0 = 1$, layer 50 gets $r^{49} \approx 1/14{,}465$. That's where the ~1902x ratio comes from.

**In plain words:** even though the per-layer gain product is exactly 1.0, the gains are NOT exactly 1.0 individually. The forward gain is 0.909 — just 9% below 1. But $0.909^{50} = 0.008$. A 9% per-layer loss becomes a 125x cumulative decay over 50 layers.

> **Show:** `gradient_norms_log.png` for the same data on log scale — V1 (purple) is a straight line going down, which is exactly $r^{\ell-1}$ on log scale.

### 1.6 Weight magnitudes for V1

> **Show:** `weight_std_per_layer.png` — V1 panel (3rd from top): flat line at 0.0556, ratio=1.0x.

V1's advantage: the weights are perfectly uniform across layers. Every layer gets the same std. There are no "huge" or "tiny" layers. The entire cost is paid in gradient non-uniformity.

---

## Part 2: What about V2? (Preview, for discussion)

V2 adds a per-layer schedule $s_\ell = s^* \cdot r^{\eta(\ell - (L+1)/2)}$ that redistributes variance across layers to flatten the gradient profile.

> **Show:** `gradient_norms_linear_L50.png` — compare V1 (1902x) to V2 (36x).

V2 reduces the gradient ratio from 1902x to 36x at L=50. But look at the price:

> **Show:** `weight_std_per_layer.png` — V2 panel (bottom): layer 1 std = 0.5811, layer 50 std = 0.0053, ratio = 109x.

Layer 1's weights are **109x larger in std** (11,900x in variance) than layer 50's. This is what I mentioned in the email about "massive weight values."

The Pareto theorem (Section 7 of the report) proves this trade-off is inescapable:

$$G(\eta) \cdot V(\eta) = r^{-(L-1)}$$

At L=50 the budget is 14,465x. You can split it however you want between gradient ratio $G$ and weight variance ratio $V$, but the product is fixed.

---

## Part 3: Geometry — the real problem

> **Show:** `knn_multi_depth.png`

All row-centered variants (V1, V2, fwd-balanced) produce **identical** k-NN curves. By 20 layers, k-NN = 0.11 (chance level). He stays at 0.64.

The variance doesn't matter. The constraint $\sum_j W_{ij} = 0$ makes each neuron blind to the mean activation (DC component), destroying class structure. This is a structural property of row centering, not a variance problem.

---

## Part 4: Where this leaves us

**V1 gives us:**
- A clean, provable result: $v^* = 2\sqrt{\pi/(\pi-1)}/d$ is the unique product-balanced variance
- The coupling ratio $g_{\text{fwd}}/g_{\text{bwd}} = r$ is structural and cannot be removed
- The product = 1 property minimizes the dynamic range of both activations and error signals simultaneously

**V1 does NOT give us:**
- Uniform gradients (1902x ratio at L=50)
- Geometry preservation (identical collapse to all other RC variants)

**V2 gives us:**
- Better gradient uniformity (36x vs 1902x)
- But at the cost of 109x weight std ratio
- And the Pareto theorem proves this is the best possible trade-off

**The conclusion:** Row centering is a dead end for deep networks. Both the gradient barrier (Pareto budget) and the geometry barrier (DC-blindness) are structural consequences of $\sum_j W_{ij} = 0$. No variance schedule can escape either.

**Proposed next step:** Run CNN/FNN + BatchNorm comparison on GPU cluster. Does BN's dynamic centering (which adapts to data per-batch) avoid the barriers that static weight centering cannot escape?
