# Gradient Diagnostics Analysis Report

## Notebook 06: Forward-Backward Gain Asymmetry

**Date**: February 2026
**Architecture**: 784 → [784] × 50 → 1 (50 hidden layers, constant width)
**Dataset**: Fashion-MNIST (2000 samples for full analysis, 1 for single-sample test)
**Initializers tested**:
- `he` — Standard He/Kaiming (baseline)
- `row_centered_he_var_adj` — Row-centered with variance adjustment
- `row_centered_forward_balanced` — Row-centered with variance tuned for forward gain = 1

---

## 1. RMS: Why Divide by √(n · d_l)?

The activation matrix at layer l is A^l ∈ ℝ^{n × d_l}, where n = number of samples and d_l = width of layer l. The Frobenius norm is:

$$\|A^l\|_F = \sqrt{\sum_{i=1}^{n} \sum_{k=1}^{d_l} (a^l_{ik})^2}$$

This quantity depends on **both** n and d_l. If you double the width, the Frobenius norm grows by √2 even if the per-neuron signal hasn't changed at all. Same for doubling the batch size.

What we actually want is the **typical signal level per neuron per sample** — a quantity that is comparable across layers of different widths and across experiments with different batch sizes. The RMS achieves this:

$$\text{rms}(A^l) = \frac{\|A^l\|_F}{\sqrt{n \cdot d_l}} = \sqrt{\frac{1}{n \cdot d_l} \sum_{i=1}^{n} \sum_{k=1}^{d_l} (a^l_{ik})^2} = \sqrt{\mathbb{E}_{i,k}[(a^l_{ik})^2]}$$

So the RMS is simply the **root of the average squared activation entry**. It is the "typical magnitude of a single activation value."

### Why this matters for us

Our network has constant width (784 everywhere), so in our specific case, using the Frobenius norm directly would work for comparing across layers. But the RMS is the principled choice because:

- If we ever change widths between layers, the gains would still be meaningful.
- The RMS has a clean probabilistic interpretation: it equals √(E[a²]), which connects directly to the He initialization theory (He et al. derived that Var(a^l) = Var(a^{l-1}) when Var(W) = 2/d).

### General case with varying widths

Suppose layer l-1 has width d_{l-1} and layer l has width d_l. Then:

$$\text{gain}_{\text{fwd}}^l = \frac{\text{rms}(A^l)}{\text{rms}(A^{l-1})} = \frac{\sqrt{\mathbb{E}[(a^l)^2]}}{\sqrt{\mathbb{E}[(a^{l-1})^2]}}$$

The n cancels (same samples), and the d's are absorbed into the averaging. The gain is purely a statement about "does the typical activation magnitude change between layers," independent of how wide each layer is.

### Code reference

In `_collect_signal_stats` (gradient_analysis.py, line 360):
```python
def rms(tensor: torch.Tensor) -> float:
    return torch.norm(tensor).item() / (tensor.numel() ** 0.5)
```

`torch.norm(tensor)` computes the Frobenius norm ||A||_F. `tensor.numel()` returns n · d_l (total number of elements). So we get ||A||_F / √(n · d_l) = rms(A).

---

## 2. Gain: Intuition and Why We Want ≈ 1

### Why "gain"?

This is borrowed from signal processing / electrical engineering. A gain is the ratio of output signal amplitude to input signal amplitude. Each layer is an "amplifier": it takes a signal (activations from the previous layer) and produces a new signal. The gain tells you the amplification factor.

### Why ≈ 1?

Consider what happens over L layers if the per-layer gain is g:

$$\text{rms}(a^L) = g^L \cdot \text{rms}(a^0)$$

| Per-layer gain g | L = 50 layers | Effect |
|:---:|:---:|:---|
| 1.01 | 1.01^50 ≈ 1.64 | Mild growth — acceptable |
| 1.1 | 1.1^50 ≈ 117 | Explosion |
| 0.83 | 0.83^50 ≈ 10^{-4} | Vanishing |
| 0.99 | 0.99^50 ≈ 0.61 | Mild decay — borderline |

The same applies backward for error signals. If the backward gain is consistently < 1, gradients at early layers are exponentially smaller than at late layers — the network can't learn in the early layers. If > 1, gradients explode.

So gain ≈ 1 is the **critical point** between exponential explosion and exponential vanishing. It is the only value that keeps signals at a stable magnitude across arbitrary depth.

Note: gain exactly = 1 isn't strictly required for training — what matters is that the *product* of all gains doesn't blow up or collapse. But per-layer gain ≈ 1 is the way to ensure this uniformly.

---

## 3. The Centering Ratio: What It Is and Why We Care

### Definition of ratio_i

For sample i, the activations from the previous layer are a vector a^{l-1}_i ∈ ℝ^d. We compute:

$$\text{ratio}_i = \frac{\text{Var}_k(a^{l-1}_{ik})}{\mathbb{E}_k[(a^{l-1}_{ik})^2]}$$

where the expectation and variance are taken **across the feature dimension** (across the d neurons), not across samples.

Using the identity Var(X) = E[X²] - (E[X])²:

$$\text{ratio}_i = \frac{\mathbb{E}_k[(a_{ik})^2] - (\mathbb{E}_k[a_{ik}])^2}{\mathbb{E}_k[(a_{ik})^2]} = 1 - \frac{(\bar{a}_i)^2}{\overline{a_i^2}}$$

where ā_i = (1/d) Σ_k a_{ik} is the mean activation for sample i.

The centering ratio is then the average over samples: centering_ratio = mean_i(ratio_i).

### Why does this matter?

When the weight matrix W has zero-sum rows (Σ_j W_{lj} = 0), the pre-activation for neuron l is:

$$z_l = \sum_j W_{lj} a_j = \sum_j W_{lj}(a_j - \bar{a})$$

The last equality holds because Σ_j W_{lj} = 0 means Σ_j W_{lj} · ā = 0. So the neuron only "sees" the **centered** version of its input. Instead of operating on a_j with E[a_j²], the effective input has second moment:

$$\mathbb{E}[(a_j - \bar{a})^2] = \text{Var}(a_j) = \mathbb{E}[a_j^2] - (\bar{a})^2$$

The ratio of the "effective" second moment (what row-centered weights see) to the "full" second moment (what standard He weights see) is exactly:

$$\frac{\text{Var}(a)}{\mathbb{E}[a^2]} = 1 - \frac{(\bar{a})^2}{\mathbb{E}[a^2]}$$

**This is the centering ratio.** It tells you: *what fraction of the input's energy is "visible" to a row-centered weight vector?*

- If the activations have mean zero (ā = 0), then Var(a) = E[a²] and the ratio = 1 — centering doesn't lose anything.
- But post-ReLU activations are **non-negative** (half-Gaussian), so they have a **positive mean**. A significant chunk of their energy is in the DC component (the mean), and row-centered weights are blind to it. The ratio < 1 tells you exactly how much energy is lost.

### Code reference

In `_collect_signal_stats` (gradient_analysis.py, lines 397–413):
```python
per_sample_mean_sq = (a_prev_detached ** 2).mean(dim=1)  # E_k[a²] per sample
per_sample_mean = a_prev_detached.mean(dim=1)            # E_k[a] per sample
per_sample_ratio = 1.0 - (per_sample_mean ** 2) / safe_mean_sq
centering_ratio = per_sample_ratio.mean().item()
```

---

## 4. Why (π-1)/π ≈ 0.6817 for Post-ReLU Activations

### 4.1 Half-Gaussian moments

After ReLU, the activations follow a half-Gaussian distribution. If the pre-activation z ~ N(0, σ²), then a = ReLU(z) = max(0, z).

For a half-Gaussian random variable a:

$$\mathbb{E}[a] = \frac{\sigma}{\sqrt{2\pi}}, \qquad \mathbb{E}[a^2] = \frac{\sigma^2}{2}$$

The first is the mean of the positive half of N(0, σ²). The second comes from: half the time a = 0, half the time a² = z² where z ~ |N(0, σ²)|, and E[z²] = σ², so E[a²] = (1/2) · σ².

Now:

$$\text{Var}(a) = \mathbb{E}[a^2] - (\mathbb{E}[a])^2 = \frac{\sigma^2}{2} - \frac{\sigma^2}{2\pi} = \frac{\sigma^2}{2}\left(1 - \frac{1}{\pi}\right) = \frac{\sigma^2(\pi - 1)}{2\pi}$$

Therefore:

$$\frac{\text{Var}(a)}{\mathbb{E}[a^2]} = \frac{\sigma^2(\pi-1)/(2\pi)}{\sigma^2/2} = \frac{\pi - 1}{\pi} \approx 0.6817$$

### 4.2 Decomposing the signal: DC component vs. informative variance

A post-ReLU activation vector a = (a₁, a₂, ..., a_d) can be decomposed into two parts:

$$a_j = \underbrace{\bar{a}}_{\text{mean (DC)}} + \underbrace{(a_j - \bar{a})}_{\text{deviation (AC)}}$$

where ā = (1/d) Σⱼ aⱼ is the mean activation across neurons.

The total energy of the signal is E[a²]. This energy splits into:
- **DC energy**: (E[a])² = σ²/(2π) — this is a constant offset, the same for every input sample. It carries no information about what the input was.
- **AC energy**: Var(a) = σ²(π-1)/(2π) — this is the part that varies between inputs. It carries the actual information.

The ratio: AC energy / total energy = (π-1)/π ≈ 68.2%. The remaining 31.8% is DC — a fixed positive shift present in every post-ReLU vector, regardless of what input produced it.

### 4.3 Why row-centered weights are blind to DC

A row-centered weight row has Σⱼ wⱼ = 0. When it computes its output (note we can write aⱼ = E[a] + (aⱼ - E[a]):):

$$\text{output} = \sum_j w_j \cdot a_j = \sum_j w_j \cdot \bar{a} + \sum_j w_j \cdot (a_j - \bar{a}) = \bar{a} \cdot \underbrace{\sum_j w_j}_{= 0} + \sum_j w_j (a_j - \bar{a})$$

The first term vanishes. The row-centered neuron responds only to deviations from the mean — the AC part. It is **mathematically incapable** of using the DC part.

Standard He weights (Σⱼ wⱼ ≠ 0) use the full signal: both DC and AC components contribute. The effective input energy is E[a²].

Row-centered weights use only the AC part. The effective input energy is Var(a) = (π-1)/π · E[a²].

### 4.4 From energy loss to forward gain: why √ and not just the ratio

Now the key step. The **forward gain** at a layer is defined as:

$$\text{gain}_{\text{fwd}}^l = \frac{\text{rms}(a^l)}{\text{rms}(a^{l-1})}$$

where rms(a) = √(E[a²]) is a **root-mean-square** (an amplitude, not an energy).

**For standard He** (no centering): He initialization is calibrated so that E[(a^l)²] = E[(a^{l-1})²]. The output energy equals the input energy. Therefore:

$$\text{gain}_{\text{fwd}}^{\text{He}} = \frac{\sqrt{E[(a^l)^2]}}{\sqrt{E[(a^{l-1})^2]}} = 1.0$$

**For row-centered He** (var-adj): The weights have the same variance as He (that's what var-adj does). So the layer "tries" to produce the same output energy as He. BUT: the layer can only access Var(a^{l-1}) instead of E[(a^{l-1})²] as its effective input energy. Everything else is the same (same weight variance, same ReLU, same architecture).

Think of it this way. A standard He layer performs the operation:

$$E[(a^l)^2] = \underbrace{\text{Var}(W) \cdot d}_{\text{weight energy}} \cdot \underbrace{\frac{1}{2}}_{\text{ReLU survival}} \cdot \underbrace{E[(a^{l-1})^2]}_{\text{input energy}}$$

He sets Var(W) = 2/d so that the prefactor equals 1: (2/d)·d·(1/2) = 1. This gives E[(a^l)²] = E[(a^{l-1})²], i.e. gain = 1.

For a row-centered layer, the only thing that changes is the input energy term. Instead of the full E[(a^{l-1})²], the layer sees only Var(a^{l-1}):

$$E[(a^l)^2] = \underbrace{\text{Var}(W) \cdot d \cdot \frac{1}{2}}_{= 1 \text{ (same as He)}} \cdot \underbrace{\text{Var}(a^{l-1})}_{\text{reduced input}} = \text{Var}(a^{l-1})$$

Now, what is Var(a^{l-1})? At the previous layer, the same thing happened: the layer could only use the AC part of its input. So the recurrence is:

$$E[(a^l)^2] = \text{Var}(a^{l-1}) = \frac{\pi-1}{\pi} \cdot E[(a^{l-1})^2]$$

This is a multiplicative recurrence in the **energy** E[a²]:

$$E[(a^l)^2] = \frac{\pi-1}{\pi} \cdot E[(a^{l-1})^2]$$

Each layer's output energy is (π-1)/π ≈ 0.682 times the previous layer's output energy.

The forward gain is the ratio of **amplitudes** (RMS values), not energies:

$$\text{gain}_{\text{fwd}} = \frac{\text{rms}(a^l)}{\text{rms}(a^{l-1})} = \frac{\sqrt{E[(a^l)^2]}}{\sqrt{E[(a^{l-1})^2]}} = \sqrt{\frac{E[(a^l)^2]}{E[(a^{l-1})^2]}} = \sqrt{\frac{\pi-1}{\pi}} \approx 0.826$$

**The square root appears because forward gain is an amplitude ratio, while (π-1)/π is an energy ratio.** This is the same reason that in physics/engineering, if you halve the power of a signal, the amplitude drops by √(1/2) ≈ 0.707, not by 1/2. Power ∝ amplitude².

### 4.5 Compounding over depth

Over L layers, each layer multiplies the amplitude by 0.826:

$$\text{rms}(a^L) = 0.826^L \cdot \text{rms}(a^0)$$

| Depth L | Total forward attenuation 0.826^L |
|:---:|:---:|
| 10 | 0.826¹⁰ ≈ 0.147 (7× decay) |
| 20 | 0.826²⁰ ≈ 0.022 (46× decay) |
| 50 | 0.826⁵⁰ ≈ 7 × 10⁻⁵ (14,000× decay) |

This is the "catastrophic forward signal decay." By layer 50, the activations are 14,000 times smaller than at the input. From BP4, the gradient at layer l is proportional to the activation magnitude at that layer, so later layers get vanishingly small gradients.

### 4.6 Summary: the three quantities and how they relate

| Quantity | Value | What it is |
|:---|:---:|:---|
| **(π-1)/π ≈ 0.682** | Energy ratio | Fraction of post-ReLU signal energy that is informative variance (vs. DC offset). This is what the centering ratio plot measures. |
| **√((π-1)/π) ≈ 0.826** | Amplitude ratio | Per-layer forward RMS gain for row-centered weights. This is the square root of the energy ratio because gain = amplitude ratio = √(energy ratio). |
| **0.826^L** | Accumulated amplitude | Total forward signal decay over L layers. Exponential in depth. |

The variance sweep confirms this: the measured F/B gain ratio is 0.827, matching √((π-1)/π) = √0.682, NOT (π-1)/π itself.
- The variance sweep table shows F/B ratio ≈ 0.827, which matches √((π-1)/π), NOT (π-1)/π

---

## 5. Plot Analysis

### 5a. Single vs Multi-Sample Gradient Profiles

**What we're comparing**: The mean gradient row norm at each layer, computed with n = 2000 samples versus n = 1 sample.

**Log-scale correlation**: We compute `np.corrcoef(log(norms_2000), log(norms_1))`. This checks whether the **shape** of the gradient profile (the relative growth/decay pattern across layers) is the same in both cases. Log-scale because the norms span many orders of magnitude, and we care about the multiplicative pattern (exponential growth/decay), not the absolute scale. The correlation is between:
- The vector of log(mean_row_norm) at each layer for 2000 samples
- The vector of log(mean_row_norm) at each layer for 1 sample

If correlation ≈ 1, the same exponential growth/decay pattern appears in both cases → the phenomenon is structural, not statistical.

#### a1 — He (baseline): correlation = 0.86 ("different pattern")

- **Left** (2000 samples): Smooth monotonic increase from ~400 at L1 to ~3500 at L50. Mild gradient growth — about one order of magnitude over 50 layers.
- **Right** (1 sample): Noisier, y-axis is ~3–7 (much smaller absolute values). The upward trend is still present but obscured by noise.
- The correlation is lower (0.86) because **a single sample introduces substantial randomness in He's forward pass** (no centering to stabilize the signal). But the overall trend is still upward in both cases.
- **Y-axis magnitudes**: 2000 samples gives norms ~10² to ~10³. Single sample gives ~10⁰. This makes sense: gradient ∝ Σ_samples, so the ratio is roughly the number of samples (plus dimensional factors).

#### a2 — Row-Centered (var adj): correlation = 0.98 ("same pattern")

- Both show a clean, monotonically **decreasing** profile — gradients vanish from L1 to L50.
- 2000 samples: norms drop from ~200 at L1 to ~0.1 at L50 (about 3 orders of magnitude decay).
- 1 sample: norms drop from ~5 at L1 to ~3×10⁻⁴ at L50 (about 4 orders of magnitude).
- The pattern is nearly identical in shape.

**This confirms the advisor's hypothesis**: the gradient decay is structural (in σ'(z^l) and the weight structure), not an artifact of sample aggregation.

#### a3 — Row-Centered (fwd balanced): correlation = 0.98 ("same pattern")

- Same clean decay as a2, but with **much larger magnitudes**: 2000 samples start at ~3×10⁶, single sample at ~5×10⁴.
- The larger magnitudes come from the higher variance used in fwd_balanced (designed to make forward gain = 1 → backward gain > 1 → backward explosion → large early-layer gradients).

**Key takeaway**: The row-centered gradient decay is NOT a statistical artifact. It appears identically for a single data point. This pins the cause on the weight structure itself (the zero-sum constraint).

---

### 5b. Forward-Backward Gain Decomposition

#### b1 — He (baseline): Signal Norm Decomposition

Three panels: Activation RMS | Error Signal RMS | Gradient Row Norms

- **Activation RMS** (left): Fluctuates around 0.38–0.48. Slightly noisy but roughly flat — this is what He initialization is designed to do. The forward signal is preserved.
- **Error signal RMS** (center): Fluctuates around 0.31–0.37. Roughly flat. The backward signal is also preserved.
- **Gradient row norms** (right): Monotonically increasing from ~400 to ~3500. Since BP4 gives ∂C/∂W^l = δ^l · (a^{l-1})^T, and both δ and a are roughly flat, the mild growth (~1 order of magnitude) suggests a small systematic gain slightly > 1 compounding over 50 layers.

#### b2 — He (baseline): Per-Layer Gains

- **Forward gain**: Fluctuates around 1.0, ranging from 0.90 to 1.11. Mean is approximately 1.0. This is the classic He result.
- **Backward gain**: Also fluctuates around 1.0, range 0.96 to 1.06. Slightly tighter distribution.
- Both are **noisy but centered on 1.0** — exactly what we expect.

#### Row-Centered (var adj): Signal Norm Decomposition

(Not labeled in the saved plots, but included in the notebook output between b2 and b4.)

- **Activation RMS**: Gradually decays, consistent with forward gain < 1 compounding.
- **Error signal RMS**: Approximately flat (backward gain ≈ 1).
- **Gradient row norms**: Decreasing — the forward signal decay dominates.

#### b4 — Row-Centered (var adj): Per-Layer Gains

- **Forward gain**: Starts around 0.75–0.83 in early layers, then stabilizes at approximately **0.825** across all remaining layers. Systematically below 1.0.
- **Backward gain**: Stabilizes around **0.995–1.0**. Almost exactly 1.0!

This matches the theory precisely. The forward gain for row-centered with He variance is:

$$\text{gain}_{\text{fwd}} = \sqrt{\text{Var}(W) \cdot d \cdot \text{survival} \cdot \text{centering\_ratio}} = \sqrt{\frac{2}{d} \cdot d \cdot \frac{1}{2} \cdot \frac{\pi-1}{\pi}} = \sqrt{\frac{\pi-1}{\pi}} \approx 0.826$$

**The observed 0.825 matches this prediction almost perfectly.**

#### b5 — Row-Centered (fwd balanced): Signal Norm Decomposition

- **Activation RMS**: Quickly stabilizes around 0.39–0.40. Very flat — the forward signal IS preserved (design goal achieved).
- **Error signal RMS**: Grows from ~1 to ~5000 (log scale!) — **backward explosion**. By boosting variance to fix forward gain = 1, the backward gain has been pushed above 1, causing exponential backward growth.
- **Gradient row norms**: Decreasing from ~10⁶ to ~10³. Even though forward is stable, the backward explosion dominates: early layers get huge error signals → large gradients, late layers get small error signals → small gradients.

#### b6 — Row-Centered (fwd balanced): Per-Layer Gains

- **Forward gain**: Stabilizes around **1.0** after initial transient — success!
- **Backward gain**: Stabilizes around **1.21** — systematically above 1. Over 50 layers: 1.21^50 ≈ 1.3 × 10⁴.
- The ratio fwd/bwd ≈ 1.0/1.21 ≈ 0.826 — the same centering ratio as before.

---

### 5c. Forward and Backward Gain Comparison (All Three Initializers)

#### c1 — Forward Gain Comparison

- **He (blue)**: Noisy, oscillating around 1.0. Good.
- **Row-Centered var adj (orange)**: Flat at ≈ 0.825. Systematically below 1.0.
- **Row-Centered fwd balanced (green)**: Flat at ≈ 1.0. Fixed the forward gain.

Clear visual: RC var adj has a ~17.5% forward gain deficit per layer.

#### c2 — Backward Gain Comparison

- **He (blue)**: Noisy around 1.0. Good.
- **Row-Centered var adj (orange)**: Very close to 1.0 — backward is fine!
- **Row-Centered fwd balanced (green)**: Flat at ≈ **1.21**. This is the price of fixing forward: backward gain is now > 1.

**This pair of plots is the clearest visual evidence of the fundamental asymmetry**: you can fix forward OR backward, but not both simultaneously with a simple variance scaling.

---

### 5d. Centering Ratio

- **All three initializers** show the centering ratio at approximately **0.68**, matching the theoretical (π-1)/π dashed line almost perfectly.
- This makes sense: the centering ratio is a property of the **activations** (post-ReLU half-Gaussian), NOT the weights. All three initializers produce the same activation distribution (half-Gaussian with ~50% survival), so the ratio is the same.
- The slight deviation in early layers (especially L1, where it starts around 0.59–0.60) is because the first layer's input is Fashion-MNIST data, not a half-Gaussian. The input has a different distribution (sparse, with many zeros from black pixels). By L3-L4, the activations converge to the half-Gaussian steady state.

**Key insight**: The centering ratio is universal — it doesn't depend on the initializer, it depends on the activation function. For ReLU, it is always (π-1)/π. This means **any** row-centered scheme with ReLU will face this 0.68 factor.

---

### 5e. ReLU Survival Analysis

- All three initializers show survival rate ≈ **50% ± 2%** across all layers.
- He is slightly noisier (ranging 0.46–0.55), RC variants are tighter (0.49–0.52).
- No significant difference — row centering does NOT change the ReLU survival rate.
- This is important because it means the centering ratio issue is NOT about changing the fraction of active neurons. It is purely about the **mean of the surviving activations** (the DC component that row-centered weights cannot see).

---

### 5f. Variance Sweep

#### What "RC var" means

These are row-centered initializations with different variance factors. "RC var=2.0/d" means Var(W_{ij}) = 2.0/d (standard He level after centering). "RC var=3.0/d" means 50% more variance, etc.

In the code (notebook cell 14), each sweep configuration:
1. Samples W ~ N(0, √(factor/d))
2. Row-centers: W_i ← W_i - mean(W_i)
3. Rescales each row to have std = √(factor/d)

#### f1 — Forward Gain vs Variance Factor

Each curve is flat across layers (good — the gain is uniform):
- RC var=1.5/d: forward gain ≈ 0.71
- RC var=2.0/d: forward gain ≈ 0.83
- RC var=2.5/d: forward gain ≈ 0.92
- RC var=3.0/d: forward gain ≈ 1.01

Forward gain scales as √(factor · (π-1)/(2π)) (accounting for centering ratio and survival ≈ 0.5).

#### f2 — Backward Gain vs Variance Factor

- RC var=1.5/d: backward gain ≈ 0.87
- RC var=2.0/d: backward gain ≈ 1.00
- RC var=2.5/d: backward gain ≈ 1.12
- RC var=3.0/d: backward gain ≈ 1.22

#### The crucial observation

The ratio forward/backward is **constant at ≈ 0.827** across ALL variance factors:

| Factor | Median Fwd | Median Bwd | Ratio F/B |
|:---|:---:|:---:|:---:|
| RC var=1.5/d | 0.7152 | 0.8647 | 0.8270 |
| RC var=2.0/d | 0.8258 | 0.9985 | 0.8270 |
| RC var=2.5/d | 0.9233 | 1.1164 | 0.8270 |
| RC var=3.0/d | 1.0114 | 1.2230 | 0.8269 |

This is the empirical confirmation that:

$$\frac{\text{gain}_{\text{fwd}}}{\text{gain}_{\text{bwd}}} = \sqrt{\frac{\pi-1}{\pi}} \approx 0.826$$

The centering ratio (π-1)/π ≈ 0.682 applies to the **variance**, but gains are **amplitude** (RMS) ratios, so the gain ratio is √0.682 ≈ 0.826.

**Note**: The notebook prints "Theoretical ratio (π-1)/π = 0.6817" but the observed F/B gain ratio is 0.827. These are consistent — one is a variance ratio, the other is an amplitude ratio. This distinction is worth clarifying in any presentation.

**Bottom line**: No single scalar variance can make forward = backward = 1 simultaneously. At var=2.0/d, backward=1 but forward=0.83. At var=3.0/d, forward=1 but backward=1.22. The mismatch is structural.

---

### 5g. Alpha Sweep for Partial Centering

#### What alpha does

W = W_He - α · mean(W_He, dim=1), then rescale to restore He variance.
- α = 0: pure He (no centering)
- α = 1: full centering (rows sum to zero)

#### g1 — Geometry (PCA after 20 layers)

- **α = 0.0**: Points are collapsed into a tight cone — the characteristic geometric collapse of He.
- **α = 0.2**: Still quite collapsed, slightly more spread.
- **α = 0.4**: Noticeably better separation, fan-shaped.
- **α = 0.6 through α = 1.0**: Good geometric preservation, with class structure visible. Diminishing returns past α = 0.6.

#### g2 — Forward Gain vs Alpha

- **α = 0.0** (He): Noisy around 1.0. Good forward gain.
- **α = 0.2**: ~0.89–0.96, some noise.
- **α = 0.4**: ~0.85–0.93.
- **α = 0.6**: ~0.83–0.88.
- **α = 0.8 and α = 1.0**: Both converge to ≈ 0.83, nearly identical.

The forward gain penalty saturates around α ≈ 0.6–0.8.

#### g3 — Backward Gain vs Alpha

- **All alpha values** show backward gain clustered around 1.0 (range 0.94–1.06).
- There is no significant trend with alpha! The backward gain is essentially unaffected by the centering parameter.
- This is significant: the backward pass is robust to partial centering because the transposed weights don't interact with the DC component in the same way.

#### g4 — Gradient Row Norms (log scale)

- **α = 0.0**: **Increasing** curve (mild gradient growth, ~1 decade over 50 layers).
- **α = 0.2**: Nearly **flat** — very uniform gradients across all layers!
- **α = 0.4**: Mild **decrease** (about 1 decade).
- **α = 0.6, 0.8, 1.0**: Progressively steeper decay (2–4 decades).

#### Pareto Frontier Table

| Alpha | Med Fwd | Med Bwd | Grad Spread | Geometry |
|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.996 | 0.997 | 6.6x | 0.955 |
| 0.2 | 0.937 | 0.996 | 2.4x | 0.905 |
| 0.4 | 0.887 | 0.999 | 36.8x | 0.835 |
| 0.6 | 0.853 | 0.998 | 294.7x | 0.822 |
| 0.8 | 0.832 | 0.999 | 1219.3x | 0.808 |
| 1.0 | 0.826 | 0.999 | 1908.2x | 0.818 |

#### Grad Spread

**Gradient spread** = max(mean_row_norm) / min(mean_row_norm) across hidden layers. It measures how **non-uniform** the gradient distribution is across the network.
- Spread = 1x means perfectly uniform: every layer gets the same gradient magnitude.
- Large spread means some layers get much bigger gradients than others, which is bad for training — the optimizer can't find a single learning rate that works for all layers.

Code reference (notebook cell 21):
```python
spread = max(norms) / min(norms)
```

#### Geometry Metric

The geometry column reports Var(PC₁) / (Var(PC₁) + Var(PC₂)) from the PCA projection after 20 layers of random projection + ReLU.

- A value close to **1.0** means all variance is in the first principal component → the data has **collapsed** into essentially 1D — bad geometry.
- A value closer to **0.5** means the variance is spread across both PCs → good 2D structure preserved.

So **lower is better** for this metric. α = 0.0 has geometry = 0.955 (very collapsed), while α = 1.0 has geometry ≈ 0.818 (better, more 2D structure).

Code reference (notebook cell 21):
```python
geom_quality = np.var(X_vis[:, 0]) / (np.var(X_vis[:, 0]) + np.var(X_vis[:, 1]) + 1e-10)
```

**Caveat**: This metric is crude — it measures elongation, not class separability. A better metric might be k-NN accuracy in the projected space or the silhouette score. The actual PCA scatter plots (g1) should be used for visual judgment.

#### Key insight from the alpha sweep

**α = 0.2 is an interesting sweet spot**: gradient spread is only 2.4x (the most uniform of all), forward gain is 0.937 (a mild 6% deficit per layer). However, over 50 layers: 0.937^50 ≈ 0.04, so the forward signal still decays by ~25x. This is vastly better than α = 1.0 (spread = 1908x), but still not perfect.

The geometry at α = 0.2 (0.905) is still quite collapsed, so the geometry improvement over pure He is modest at this alpha value.

---

### 5h. Layer-Balanced η Sweep

#### Motivation: a different approach to the gradient problem

The alpha sweep (Section 5g) showed that partial centering is a Pareto trade-off: you can't get good geometry AND good gradients by tuning a single α. The layer-balanced approach takes a completely different strategy:

**Keep full row centering (α = 1) to get the best geometry, but accept the forward decay and compensate for it by using different weight variances at different layers.**

The key insight comes from BP4. The gradient at layer l is:

$$\|\nabla W^l\| \propto \|a^{l-1}\| \cdot \|\delta^l\|$$

With row centering, ||a^{l-1}|| shrinks exponentially (forward decay), while ||δ^l|| is roughly constant (backward gain ≈ 1). So the gradient is big at early layers (where activations haven't decayed yet) and tiny at late layers.

**What if we could make the gradients come out uniform despite the forward decay?**

#### The mathematical derivation

Let each layer l have its own weight standard deviation s_l. Then:

**Forward chain** (activation magnitude at layer l-1):

$$\|a^{l-1}\| \propto \prod_{k=1}^{l-1} (r \cdot s_k)$$

where r = √((π-1)/π) ≈ 0.826 is the structural centering loss per layer, and s_k is the weight std at layer k. The forward amplitude per layer is r · s_k: the structural loss times the weight scale.

**Backward chain** (error signal at layer l):

$$\|\delta^l\| \propto \prod_{k=l+1}^{L} s_k$$

The backward gain at layer k is approximately s_k (the centering doesn't affect backward — Section 2).

**Gradient at layer l** (multiply forward × backward):

$$\|\nabla W^l\| \propto \prod_{k=1}^{l-1}(r \cdot s_k) \cdot \prod_{k=l+1}^{L} s_k = r^{l-1} \cdot \frac{\prod_{k=1}^{L} s_k}{s_l}$$

The product ∏s_k is a constant (independent of l). So the gradient depends on l only through:

$$\|\nabla W^l\| \propto \frac{r^{l-1}}{s_l}$$

**For uniform gradients** we need r^{l-1}/s_l = constant, which gives:

$$s_l \propto r^{l-1}$$

Early layers (l small) → s_l large (boosted weights). Late layers (l large) → s_l small (reduced weights).

**Centering at the middle**: We parameterize as:

$$s_l = r^{\eta \cdot (l - (L+1)/2)}$$

This centers the scaling at the middle layer (s_{middle} = 1), so early layers get s > 1 and late layers get s < 1. The parameter η controls the strength:

- **η = 0**: s_l = 1 for all l → equivalent to row_centered_he_var_adj (no correction)
- **η = 1**: full correction → gradients should be perfectly uniform (in theory)
- **η = 0.5**: half correction → a compromise

**Base variance**: On top of the per-layer s_l, we use the var-adj base (He variance × π/(π-1)) so that the geometric mean of forward gains across layers is approximately 1.

#### What the results show

| Config | Grad Ratio | Grad CV | Med Fwd Gain | Med Bwd Gain |
|:---|:---:|:---:|:---:|:---:|
| He (baseline) | 1.93× | 0.182 | 0.979 | 1.022 |
| η = 0.00 (= var-adj) | 11.41× | 0.926 | 0.999 | 1.213 |
| η = 0.25 | 6.52× | 0.603 | 1.045 | 1.212 |
| **η = 0.50** | **4.32×** | **0.521** | 1.096 | 1.212 |
| η = 0.75 | 6.67× | 0.764 | 1.150 | 1.212 |
| η = 1.00 | 12.93× | 1.075 | 1.206 | 1.212 |

**Observations**:

1. **η = 0 vs var-adj**: η = 0 gives gradient ratio 11.41×, confirming the baseline forward-decay problem. (This is less than the 1908× from the alpha sweep because the var-adj base already has boosted variance — the backward gain is 1.21 instead of 1.0, which partially compensates. More on this below.)

2. **η = 0.5 is the sweet spot**: Gradient ratio drops to 4.32× and CV to 0.521 — a ~2.6× improvement over η = 0. But it's still not as good as He (1.93×).

3. **η = 1.0 is WORSE than η = 0**: Gradient ratio is 12.93×, even higher than the uncorrected version! This is surprising if the theory predicts perfect uniformity at η = 1.

4. **Median forward gain increases with η**: From 0.999 (η=0) to 1.206 (η=1). The boosted early-layer weights push the median forward gain above 1.

5. **Backward gain is constant at 1.21 regardless of η**: This confirms the backward gain depends on the average weight scale (which is the var-adj base), not on how variance is distributed across layers.

#### Why η = 1 doesn't achieve perfect uniformity (the backward gain problem)

The derivation above assumed backward gain per layer = s_l. But the actual backward gain has TWO components:
- The weight scale s_l (which varies across layers)
- The var-adj base factor √(π/(π-1)) ≈ 1.21

The var-adj base means the **average** backward gain is 1.21, not 1.0. Over 20 layers, the total backward amplification is ~1.21^20 ≈ 55×. This creates its own gradient non-uniformity (early layers get amplified backward signals).

At η = 1, we've fully compensated the forward decay, but the backward explosion from the var-adj base is still there — and at η = 1, we've also made the backward gain **non-uniform** across layers (huge at early layers, tiny at late layers). These two effects combine to make η = 1 worse than η = 0.

At η = 0.5, we partially compensate the forward decay AND the non-uniform backward gain from the layer-dependent s_l partially cancels the backward explosion. It's a compromise that works better than either extreme.

#### The backward gain plot

The backward gain plot shows the η = 1 curve starting at ~7 for early layers and dropping to ~0.2 for late layers. This is because:
- Early layers have s_l > 1 (boosted weights) → backward gain = 1.21 × s_l ≫ 1
- Late layers have s_l < 1 (reduced weights) → backward gain = 1.21 × s_l < 1

At η = 0.5, the backward gain ranges from ~2.9 (L1) to ~0.5 (L19), which is less extreme.

#### Geometry

All η values have identical geometry because they all use full row centering (α = 1). The per-layer variance scaling doesn't affect the centering constraint. The geometry metrics (k-NN, distance correlation, effective dimension) are scale-invariant, so redistributing variance across layers doesn't change them.

#### Bottom line

The layer-balanced approach is a **partial success**:
- It preserves full row-centered geometry (best available)
- It reduces gradient non-uniformity by ~2.6× at η = 0.5
- But it can't fully equalize gradients because the var-adj base creates backward gain > 1, and the per-layer scaling that fixes forward decay also destabilizes backward gain
- The optimal η (≈ 0.5) is a compromise between forward compensation and backward stability

---

## Summary of Key Findings

### 1. The gradient decay is structural, not statistical
Single-sample experiments reproduce the same decay pattern as 2000-sample experiments (log-scale correlation > 0.97 for row-centered initializers). The cause is in the weight structure, not in sample aggregation.

### 2. Forward-backward gain asymmetry
Row centering creates a systematic asymmetry:
- **Forward gain** is multiplied by √((π-1)/π) ≈ 0.826 at every layer (because row-centered weights are blind to the DC component of post-ReLU activations)
- **Backward gain** is unaffected (the transposed weights interact with error signals that have near-zero mean)

### 3. The centering ratio is universal
Var(a)/E[a²] ≈ (π-1)/π ≈ 0.682 for all initializers — it is a property of ReLU activations, not of the weights. What differs is whether this ratio affects the forward gain (it does for row-centered weights, but not for standard He).

### 4. No scalar variance fix exists
The F/B gain ratio is constant at √((π-1)/π) ≈ 0.826 regardless of variance factor. You can tune variance to fix forward gain = 1 (at var ≈ 3/d), but then backward gain ≈ 1.21, causing exponential backward explosion.

### 5. ReLU survival is unaffected
All initializers show ~50% survival rate. The issue is not about dead neurons — it is about the positive mean of surviving activations.

### 6. Partial centering Pareto frontier
The alpha sweep reveals a trade-off between gradient uniformity and geometric preservation:
- α = 0 (He): best gradients (6.6x spread), worst geometry (0.955)
- α = 0.2: best gradient uniformity (2.4x spread!), mild geometry improvement
- α ≥ 0.6: good geometry (≈ 0.82), but severe gradient non-uniformity (>300x spread)

### 7. Proper geometry metrics overturn the row-centering narrative (Notebook 07)

Using k-NN accuracy (which measures class separability, not just spread), row_centered_var_adj drops to CHANCE LEVEL (0.110) by 20 layers. Standard He stays at 0.639, orthogonal_he at 0.662. The PCA scatter plots were misleading: row centering spreads data out (high eff_dim) but destroys class structure. The forward signal decay (0.826^L) randomizes the activations at depth.

---

## Notebook 07: Geometry Comparison with Proper Metrics

### 7a. Geometry Preservation vs Depth

**Setup**: Fashion-MNIST (2000 samples), depths [5, 10, 15, 20], five initializers.

**New metrics** (replacing the broken PCA variance ratio):
- **k-NN accuracy (k=5)**: classify projected data with k-nearest neighbors. Chance = 0.10.
- **Pairwise distance correlation (Spearman)**: rank correlation between original and projected pairwise distances.
- **Effective dimensionality**: exp(entropy(eigenvalue distribution)). High = spread out, Low = collapsed.

#### Results

**k-NN Accuracy** (higher = better geometry):

| Initializer | 5L | 10L | 15L | 20L |
|:---|:---:|:---:|:---:|:---:|
| he | 0.736 | 0.710 | 0.673 | **0.639** |
| orthogonal_he | **0.755** | **0.724** | **0.696** | **0.662** |
| orthogonal_tuned | 0.755 | 0.724 | 0.696 | 0.662 |
| row_centered_he_var_adj | 0.717 | 0.482 | 0.186 | 0.110 |
| kernel_preserving | 0.737 | 0.628 | 0.540 | 0.100 |

**Distance Correlation** (higher = better):

| Initializer | 5L | 10L | 15L | 20L |
|:---|:---:|:---:|:---:|:---:|
| he | 0.843 | 0.767 | 0.690 | 0.654 |
| orthogonal_he | **0.875** | 0.728 | 0.670 | 0.600 |
| row_centered_he_var_adj | 0.743 | 0.476 | 0.363 | 0.349 |
| kernel_preserving | 0.722 | 0.598 | 0.524 | NaN |

**Effective Dimensionality** (interpretation varies — see below):

| Initializer | 5L | 10L | 15L | 20L |
|:---|:---:|:---:|:---:|:---:|
| he | 19.7 | 13.0 | 7.5 | 5.8 |
| orthogonal_he | 26.6 | 17.5 | 12.2 | 7.3 |
| row_centered_he_var_adj | 136.2 | 289.7 | 372.7 | 388.3 |
| kernel_preserving | 28.0 | 16.3 | 1.0 | 1.0 |

#### The critical finding: row centering destroys class structure

Row-centered var-adj drops from 0.717 (5L) to 0.110 (20L) — **chance level**. Meanwhile He degrades gracefully from 0.736 to 0.639, losing only ~13% of its class separability over 20 layers.

This directly contradicts the PCA scatter plot evidence from notebook 05, where row centering appeared to "preserve geometry" better than He. The PCA plots were measuring the wrong thing.

#### Why the PCA metric was misleading

The effective dimensionality reveals the mechanism:
- **Row-centered**: eff_dim = 388 at 20L (data spreads across ~388 dimensions)
- **He**: eff_dim = 5.8 at 20L (data compressed into ~6 dimensions)

The PCA scatter plots showed row-centered data "spreading out" in 2D, which looked like preserved structure. But spreading out ≠ preserving structure.

What actually happens to row-centered data:
1. Forward signal decays by 0.826^L per layer (Section 4 analysis)
2. By 20 layers: 0.826^20 ≈ 0.022, so activations are ~46× smaller than input
3. At this scale, the signal is dominated by random fluctuations from weight initialization
4. The data becomes **uniformly dispersed noise** — high dimensional (high eff_dim) but with no class information (k-NN = chance)

What happens to He data:
1. Forward gain ≈ 1.0, so activations maintain their magnitude
2. The ReLU kernel contracts angles: D(α) = (1/π)[sin α - α cos α] > 0
3. This compresses data into fewer dimensions (low eff_dim)
4. But the **class structure is preserved within those dimensions** — relative ordering of inter-class vs intra-class distances is maintained

**The eff_dim metric alone is misleading**: high eff_dim looks good in isolation, but combined with low k-NN it means "randomized noise spread across many dimensions."

#### Orthogonal results

- **orthogonal_he ≡ orthogonal_tuned** at all depths: with square matrices (784×784) and the same seed, QR gives the same orthogonal matrix. Scaling (He gain vs tuned gain) is irrelevant for scale-invariant metrics (k-NN, Spearman distance correlation). Eff_dim is also identical because the eigenvalue distribution is scale-invariant.

- **Orthogonal is slightly better than He**: 0.662 vs 0.639 at 20L (~3.6% relative improvement). Theory predicts the same expected kernel K(α), so same systematic collapse rate. The marginal improvement comes from reduced variance (orthogonal = deterministic singular values, vs He = random). Less noise in the forward pass means the stochastic component of collapse is slower. But the systematic D(α) > 0 contraction is the same.

- **Both decay gradually**: Neither prevents collapse — they just collapse slowly enough that 64-66% of class accuracy survives 20 layers.

#### kernel_preserving failure

- 5L: comparable to He (0.737 vs 0.736)
- 10L: already worse (0.628 vs 0.710)
- 15L: eff_dim = 1.0, k-NN = 0.540 — partial collapse
- 20L: k-NN = 0.100 (chance), distance correlation = NaN (numerical overflow)

The NaN occurs because kernel_preserving optimizes each layer independently (200 Adam steps). By 15-20 layers, the accumulated weight matrices produce output magnitudes that overflow float64. The distance correlation code detects this: the pairwise projected distances contain Inf/NaN values, causing Spearman correlation to return NaN (constant input warning from scipy).

The optimizer also doesn't account for cumulative kernel drift — each layer is locally optimal but globally the composition diverges.

#### PCA scatter plots at 10 and 20 layers — visual confirmation

**At 10 layers:**
- **He** (kNN=0.710): Cone-shaped scatter with clear color clustering — classes overlap but are still separable by a k-NN classifier. The angular contraction has compressed the data but not yet destroyed the class boundaries.
- **orthogonal_he / orthogonal_tuned** (kNN=0.724): Very similar cone shape to He, slightly more spread. Consistent with same kernel K(α) but less noise.
- **row_centered_var_adj** (kNN=0.482): Visually the scatter looks more "spread out" and less cone-shaped — colors appear more mixed. At first glance one might think the wider spread is good, but the kNN of 0.482 tells the truth: almost half the class information is already gone.
- **kernel_preserving** (kNN=0.626): Shows a distinct bimodal structure (two lobes). The optimizer found a local minimum that splits the data, but this structure doesn't map to the original 10-class geometry.

**At 20 layers:**
- **He** (kNN=0.639): Still a recognizable cone with color clusters, though more compressed. The 7% kNN drop from 10L to 20L is modest.
- **orthogonal_he / orthogonal_tuned** (kNN=0.662): Slightly wider cone than He. Still shows clear color grouping.
- **row_centered_var_adj** (kNN=0.110 = chance): The scatter looks like a uniform cloud — colors are completely mixed. This is what "high eff_dim + no class structure" looks like: the data fills a high-dimensional ball uniformly.
- **kernel_preserving** (kNN=0.100 = chance): Collapsed into a dense blob with a few outliers. The numerical overflow has made the data meaningless.

**Key visual takeaway**: The row-centered scatter at 20L is the most "spread out" of all five — and yet has the worst kNN. This is the clearest visual demonstration of why the PCA variance ratio was a misleading geometry metric. Spread ≠ structure.

---

### 7b. Biased ReLU Kernel Analysis

#### Motivation

From Section 7a, the real enemy is the **ReLU kernel distortion**: for standard He weights, the expected normalized output similarity after one layer is:

$$\frac{K(\alpha)}{K(0)} > \cos(\alpha) \quad \text{for all } \alpha \in (0, \pi)$$

This means ReLU systematically makes every pair of vectors look more similar than they are. Over L layers, this compounds and all angles contract toward 0 — geometric collapse.

Row centering tried to fix this by constraining the weights, but that killed the forward signal. **Can we instead modify the ReLU itself?**

The idea: use ReLU(z + b) with a **negative bias** b < 0. This raises the activation threshold:
- Standard ReLU (b=0): neuron fires when z > 0 (50% of the time)
- Biased ReLU (b < 0): neuron fires when z > -b > 0 (less often)

The intuition for why this might help geometry: with b = 0, weakly-positive activations (small z > 0) fire for almost every input — they contribute to the DC component that makes all vectors look similar. With b < 0, only strongly-positive activations survive. These carry more directional information (they indicate that the input had a large component along that neuron's weight vector, not just a small positive fluctuation).

Crucially, **the weights remain unconstrained** — no zero-sum constraint, so no gradient trap from the weight structure itself.

#### What K_β(α) is

The biased ReLU kernel is defined as:

$$K_\beta(\alpha) = E[\text{ReLU}(u + \beta) \cdot \text{ReLU}(v + \beta)]$$

where (u, v) ~ N(0, [[1, ρ], [ρ, 1]]) with ρ = cos(α), and β = b/σ is the normalized bias (bias divided by weight standard deviation).

This is the expected output inner product between two inputs separated by angle α, when we use biased ReLU. At β = 0, this reduces to the standard arc-cosine kernel K(α) from the thesis proposal.

We compute K_β via Monte Carlo (1M samples) because there is no known closed-form for β ≠ 0.

**Verification**: At β = 0, the MC estimate matches the analytical K(α) to within 1.3% relative error — confirming the MC is accurate enough to trust for β ≠ 0.

#### The three metrics

For each β value, we compute:

1. **K_β(0)** — the expected output norm² for identical inputs (α=0). This measures overall signal strength. As β becomes more negative, fewer neurons fire, so K_β(0) decreases.

2. **Integrated distortion ∫D_β(α)² dα** — where D_β(α) = K_β(α)/K_β(0) - cos(α) is the deviation of the normalized kernel from the identity cos(α). This is the **objective**: lower = better geometry preservation across all angles. We integrate over α ∈ [0, π] to get a single number.

3. **Survival rate** P(u + β > 0) = Φ(β) — the fraction of neurons that fire. At β = 0: 50%. At β = -1: 15.9%. Low survival means very sparse activations.

#### The biased ReLU kernel plots

**Left panel — Raw kernel K_β(α):**
Each curve is K_β(α) for a different β. The dashed line cos(α)/2 is the target (what K would be if one layer preserved angles perfectly, with K(0) = 1/2). As β becomes more negative, the curves shrink toward zero because fewer neurons contribute. The shape is the interesting part, which we see after normalization.

**Middle panel — Normalized kernel K_β(α)/K_β(0):**
This is the output cosine similarity. The dashed line cos(α) is the ideal (identity mapping of angles). At β = 0 (blue), the normalized kernel sits above cos(α) everywhere — this is the systematic distortion. As β decreases:
- For **small angles** (α < 1): the curves get closer to cos(α) — improvement
- For **large angles** (α > 2): the curves drop below cos(α) — overcorrection (vectors that were nearly orthogonal now appear negatively correlated in the output)

No value of β makes the curve match cos(α) everywhere.

**Right panel — Distortion D_β(α) = K_β(α)/K_β(0) - cos(α):**
Direct visualization of the error. D = 0 would be perfect. At β = 0 (blue): positive distortion everywhere, peaking at α ≈ 2.3. As β decreases: the positive peak shrinks but a negative dip appears at large α. The distortion changes shape but never goes to zero.

#### The optimal bias results

**Left panel — Integrated distortion vs β:**
The U-shaped curve shows ∫D_β(α)² dα as a function of β. The minimum is at β* ≈ -0.93 with distortion = 0.795, compared to 0.912 at β = 0.

**Reduction factor: only 1.15×** — a 13% improvement. Disappointing.

**Right panel — Survival rate vs β:**
At β* = -0.93, only 17.6% of neurons fire. This extreme sparsity is the cost of the modest distortion reduction.

#### Variance compensation

With only 17.6% of neurons firing, the output signal is much weaker. To maintain the same forward signal strength as standard He:

| β | E[a²] | σ²_adj factor |
|:---:|:---:|:---:|
| 0.00 | 0.4988 | 1.0 (He baseline) |
| -0.93 | 0.0872 | 5.72 |

The weight variance must be multiplied by 5.72 to compensate: σ² = 5.72 × (2/d). This means much larger weights, which would push the backward gain above 1 — potentially creating its own gradient instability.

#### Assessment: negative bias is a dead end

The biased ReLU approach is theoretically elegant but practically disappointing:

1. **Marginal improvement**: Only 13% reduction in integrated distortion — nowhere near enough to prevent collapse over 20+ layers
2. **Extreme sparsity**: 82% of neurons are dead at β*, requiring huge variance compensation
3. **Doesn't eliminate distortion**: Just reshapes it (reduces at small angles, overcorrects at large angles)
4. **Fundamental limit**: No pointwise nonlinearity (applied independently per neuron) can fully preserve the input kernel. The distortion D(α) is intrinsic to any nonlinear map from 2D → 1D, and a bias shift can only change D's shape, not make it zero everywhere.

The improvement is so small because the kernel distortion comes from the **geometry of the bivariate Gaussian projected through a nonlinearity**, not from the threshold location. Shifting the threshold changes which part of the joint distribution contributes, but the fundamental structure of the integral remains.

---

### 7c. Analyzing Kernel-Preserving Optimizer Solutions

#### Motivation

The `kernel_preserving` initializer (Section 7a) directly optimizes W to minimize kernel distortion:

$$\min_W \sum_{i,j} \left(\frac{\langle \text{ReLU}(Wx_i), \text{ReLU}(Wx_j)\rangle}{d_{out}} - \langle x_i, x_j \rangle\right)^2 + \lambda \sum_i \left(\frac{\|\text{ReLU}(Wx_i)\|^2}{d_{out}} - 1\right)^2$$

Starting from He initialization, it runs 200 Adam steps on 500 random unit vectors. At 5 layers, it achieves k-NN = 0.737 (comparable to He). The question: **what structure does the optimizer discover?** If there's a consistent pattern, we could design a closed-form initializer matching it — avoiding the 200-step optimization and potentially finding a composable version that works at depth.

Specifically, we want to know:
- Does the optimizer learn row centering (row sums → 0)? If so, this validates the centering approach.
- Does the optimizer learn a negative bias (pre-activations shifted negative)? If so, this validates the biased ReLU approach.
- Or does it find something completely different?

We compare the structure of KP-optimized weight matrices to standard He matrices.

#### Plot 1: Row Sums (∑_j w_{ij} per output neuron)

- **He**: row sums cluster tightly around 0, with mean |row sum| ≈ 1.2. This is expected: each row has 784 i.i.d. N(0, 2/784) entries, so the row sum has distribution N(0, 784 × 2/784) = N(0, 2), giving E[|row sum|] ≈ √(2 × 2/π) ≈ 1.1.
- **KP**: row sums are **10× larger** (mean |row sum| ≈ 12.2), spread widely in both positive and negative directions.

**Key finding: the optimizer does NOT learn row centering.** It moves AWAY from zero row sums. If centering were the right strategy for kernel preservation, the optimizer should have converged toward row sums ≈ 0. Instead, it discovers a completely different approach with large, non-zero row sums.

This is consistent with Section 7a: row centering preserves spread (high eff_dim) but destroys class structure (k-NN → chance). The optimizer, targeting actual kernel preservation (inner product matching), avoids centering.

#### Plot 2: Row Norms (||w_i||₂ per output neuron)

- **He**: row norms ≈ 1.3, tightly concentrated. The row norm for He has a chi distribution: ||w_i|| = √(∑ w²_ij) with 784 entries of variance 2/784, giving E[||w||] ≈ √(2) ≈ 1.41, consistent with the observed ~1.3.
- **KP**: row norms ≈ 15 — **11.5× larger than He**.

The optimizer inflates weight magnitudes dramatically. Combined with the norm preservation constraint E[||ReLU(Wx)||²/d_out] ≈ 1, this means the ReLU must be selectively killing most of the output — the large weights produce large pre-activations, but the careful structure ensures that the surviving activations preserve the input kernel.

This massive scale inflation is why KP becomes numerically unstable at depth: 15^L grows catastrophically fast.

#### Plot 3: Weight Element Distribution

- **He**: Gaussian bell curve (as initialized), centered at 0.
- **KP**: nearly **uniform** (flat histogram). The Q-Q plot against a normal reference shows heavy tails — the distribution is bounded (no extreme outliers) but fills its range uniformly instead of concentrating at zero.

This is a radical departure from any standard initialization scheme. The optimizer found that uniformly-distributed weights better preserve the kernel than Gaussian weights. Mathematically: when each weight w_ij is drawn from a uniform-like distribution instead of Gaussian, the pre-activation z_k = ∑ w_kj x_j has a different distribution than the Gaussian predicted by CLT (the finite-sample effects matter when the weights have bounded support). This changes how ReLU filters the pre-activations.

#### Plot 4: Pre-Activation Distribution and DC Component

- **He**: pre-activations z = Wx follow a Gaussian bell (Central Limit Theorem: z_k = ∑_{j=1}^{784} w_kj x_j is a sum of 784 random terms).
- **KP**: pre-activations are nearly **uniform/flat** — much wider and flatter than Gaussian.

The DC component analysis:
- **He**: mean pre-activation ≈ 0.03 (near zero, as expected for zero-mean Gaussian weights on centered data)
- **KP**: mean pre-activation ≈ 0.39 — a **13× increase**

This is the **opposite** of what the negative bias approach (Section 7b) tried to do. The biased ReLU pushed the threshold UP (killing weakly-positive activations); KP pushes the mean activation UP (more neurons fire, and they fire with larger values). The optimizer found that preserving the kernel requires AMPLIFYING the DC component, not suppressing it.

Why? Because ReLU's kernel distortion D(α) > 0 makes outputs too similar. One way to compensate is to make the output representation higher-dimensional in an effective sense — large, varied pre-activations create more distinct post-ReLU patterns per input, counteracting the similarity inflation.

#### Plot 5: Singular Values of W

- **He**: flat SVD spectrum at ~2.5 (nearly isotropic). For an m×n Gaussian random matrix with variance σ² = 2/n, the Marchenko-Pastur law predicts singular values concentrated in a narrow band, consistent with the flat profile.
- **KP**: decaying SVD spectrum from ~41 (top SV) to ~20 (bottom), with a higher condition number (~2× vs ~1.1× for He).

The optimizer creates an **anisotropic** weight matrix — it amplifies certain input directions much more than others. This is the mechanism for selective ReLU survival:
1. Input components along the top singular vectors get amplified into large pre-activations → survive ReLU with high probability
2. Input components along bottom singular vectors get smaller pre-activations → more likely killed by ReLU
3. The resulting post-ReLU representation preferentially preserves directions that carry the most kernel information

This targeted amplification is what lets KP achieve kernel preservation despite ReLU's inherent distortion — at the cost of massive overall weight inflation.

#### Row-Mean Analysis Table

| Metric | He | KP | KP/He Ratio |
|:---|:---:|:---:|:---:|
| mean(w_ij) | 0.0001 | ~0.001 | ~10× |
| std(w_ij) | 0.050 | 0.019 | 0.38× |
| mean(\|row_sum\|) | 1.20 | 12.2 | 10.2× |
| mean(row_norm) | 1.30 | 15.0 | 11.5× |

The per-element std is actually SMALLER for KP (0.019 vs 0.050), which seems contradictory with the larger row norms. The resolution: KP weights are uniformly spread (not concentrated near zero), so the individual elements are smaller on average but they are all non-negligible (no near-zero entries), making the sum of squares per row much larger.

#### Summary: What the Optimizer Teaches Us

The kernel-preserving optimizer finds a solution that is:
1. **NOT row-centered** — row sums are 10× LARGER than He (rules out centering)
2. **NOT negatively biased** — DC component is 13× LARGER than He (opposite of biased ReLU)
3. **Uniform, not Gaussian** — weight distribution is flat, not bell-shaped
4. **Anisotropic** — decaying SVD spectrum selectively amplifies certain directions
5. **Massively inflated** — 11.5× larger row norms than He

This structure is fundamentally **non-composable**: it works for a single layer because the anisotropy is carefully tuned to the input distribution (random unit vectors). But after one KP layer transforms the data, the output distribution is no longer unit vectors — the next KP layer's optimization assumptions are violated. This is why KP collapses by 15-20 layers.

**Implication for the thesis**: The optimizer confirms that kernel preservation requires a radically different weight structure than any standard initialization. But the structure it finds (massive inflation + anisotropy) is inherently per-layer and data-dependent — there is no simple closed-form generalization. This reinforces the conclusion from Section 7b that the ReLU kernel distortion cannot be eliminated by a single-layer weight structure.

---

### 7d. Empirical Angle Map — Two Different Fixed Points

#### Motivation

The theoretical kernel K(α) gives the angle map α → α' = arccos(2K(α)) for He initialization (notebook 04). But different initializers create different weight structures — do they produce different angle maps? By generating pairs of unit vectors at controlled angles and passing them through one initialized layer + ReLU, we measure the empirical angle map directly.

#### Single-layer result: ALL initializers are identical

| Initializer | Mean Ratio α'/α | Min Ratio |
|:---|:---:|:---:|
| he | 0.7751 | 0.5081 |
| orthogonal_he | 0.7754 | 0.5081 |
| row_centered_he_var_adj | 0.7751 | 0.5081 |
| kernel_preserving | 0.7772 | 0.5081 |

All four initializers produce **identical** single-layer angle maps. He empirical matches theory to within 0.14°.

**Why**: The experiment uses random unit vectors (zero DC component). Row centering (Σw_j = 0) is invisible when input has no DC. The pre-activation is the same bivariate Gaussian for all initializers (by CLT), so the ReLU angle distortion is identical.

**Key insight**: The single-layer angle contraction is purely a property of **ReLU**, not of the weight structure.

#### Multi-layer result: Two DIFFERENT fixed points

After layer 1, activations are post-ReLU (non-negative, positive DC). Now row centering matters.

| Init | α₀ = 30° → 20L | α₀ = 60° → 20L | α₀ = 90° → 20L | α₀ = 120° → 20L |
|:---|:---:|:---:|:---:|:---:|
| **he** | 13.8° | 17.4° | 18.6° | 19.0° |
| **orthogonal_he** | 12.5° | 15.6° | 16.8° | 17.3° |
| **row_centered_var_adj** | **70.9°** | **71.3°** | **71.3°** | **71.7°** |

**He/orthogonal**: All starting angles converge toward **0°** — classical geometric collapse where all representations align.

**Row-centered**: All starting angles converge to **~71°** (1.24 rad). Whether input was 30° or 120°, the output converges to the same fixed point.

#### Why 71° is the row-centered fixed point

At the fixed point, all pairwise angles are identical (~71°) — all points become **equidistant** on the hypersphere. This happens because:

1. Row centering strips the DC component from post-ReLU activations at each layer
2. The remaining AC component (directional information) decays by 0.826/layer
3. After enough layers, only random fluctuations remain
4. Random non-negative vectors in high dimension have pairwise angles concentrated around ~71° (determined by the geometry of the positive orthant)

This explains **all** the Section 7a results simultaneously:
- k-NN = chance: all points equidistant → no class structure
- High eff_dim: data fills many dimensions (unlike He's low-dimensional cone)
- PCA scatter "looks good": spread = equidistant, not structured

#### He vs row-centered: opposite failure modes

| Property | He | Row-centered |
|:---|:---|:---|
| Fixed point | 0° (identity) | 71° (equidistance) |
| Failure mode | All vectors align | All vectors become equidistant |
| What survives | Relative ordering (class structure) | Spread (dimensionality) |
| k-NN at 20L | 0.639 (degrades slowly) | 0.110 (chance) |
| eff_dim at 20L | 5.8 (collapsed) | 388 (high) |

Both are geometric collapse, but they destroy **different** aspects:
- He preserves **ordering** (which vectors are closer) but loses **scale** (all angles shrink)
- Row-centered preserves **scale** (angles stay ~71°) but loses **ordering** (all angles become identical)

For classification, ordering matters more than scale — explaining why He outperforms row-centered despite lower dimensionality.

---

## Leads for Further Investigation

1. **The √((π-1)/π) vs (π-1)/π distinction**: The variance sweep prints the theoretical ratio as (π-1)/π = 0.6817, but the observed F/B gain ratio is 0.827 ≈ √0.6817. The centering ratio is a **variance** ratio, but gains are **amplitude** (RMS) ratios. Worth clarifying in any presentation.

2. **Why is the backward gain unaffected by centering?** The error signal propagation involves (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l). The transpose of a row-centered matrix is a column-centered matrix. But the "input" to the backward multiplication is δ^{l+1}, which can have both positive and negative values (not half-Gaussian like activations). So the centering ratio for the backward signal would be close to 1.0 because E[δ] ≈ 0. This is worth formalizing mathematically.

3. **Is the geometry metric meaningful?** The Var(PC₁)/(Var(PC₁)+Var(PC₂)) metric doesn't capture class separability — it just measures elongation. Better alternatives: k-NN accuracy in projected space, silhouette score, or inter-class vs intra-class distance ratios.

4. **Can we decouple forward and backward gains?** The variance sweep shows they're locked in a fixed ratio. What about using **different** variance for each layer? Or different α for different layers (more centering in early layers for geometry, less in later layers for gradients)? Or using an activation function with E[a] = 0 (like shifted ReLU or GELU)?

5. **α = 0.2 as a practical choice**: It gives the most uniform gradients, but geometry at 20 layers is still quite collapsed. Could we combine α = 0.2 with other structural tricks (orthogonal weights, DC component) to improve geometry without hurting gradient uniformity?