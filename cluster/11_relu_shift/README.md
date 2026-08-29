# 11 — post-ReLU DC removal: the activation-space dual of row-centering

**Question.** ReLU output is non-negative, so `E[a] = rms(a)/√π ≈ 0.564·rms(a) > 0`. That positive DC component — shared by every sample — is the engine of the arc-cosine kernel's `ρ → 1` collapse. Subtract it *after* the ReLU, on top of **unconstrained He weights**:

```
a = relu(Wx) − c · rms(relu(Wx))
```

and ask whether the three requirements — (i) no geometric collapse, (ii) stable gradients, (iii) preserved class content — are met better than by either `he` or `row_centered_he`. Brief: [`docs/plans_handoffs/briefs/2026-08-15_campaign11-relu-shift-dc-removal.md`](../../docs/plans_handoffs/briefs/2026-08-15_campaign11-relu-shift-dc-removal.md).

> **Retraction note (2026-08-15, evening).** An earlier revision of this README claimed a training win over He at 30L ("+13.2 pp on cifar10; every shift arm beats He; the first accuracy win in the project"). Those numbers are `eval_train_accuracy` at the **final epoch** only. On **held-out test** with robust estimators the win vanishes: `c = 0.10` is +6.6 pp at the final epoch but **+2.1 pp** on mean-of-last-5 and **+0.2 pp** at best-epoch, and the other arms go negative — He's test curve oscillates 0.38–0.47 and its final epoch lands on the minimum. Corrected analysis: [`docs/reports/2026-08-15_campaign10_followup_synthesis.md`](../../docs/reports/2026-08-15_campaign10_followup_synthesis.md) §4.3. The passages below have been corrected; §8's table is genuine data but reports final-epoch train accuracy — read it alongside the synthesis's robust tables.

**Headline.** Two closed-form results at initialization, and — after the training runs landed on 2026-08-15 and the retraction above — an empirical picture that is **level with He at 30L: not worse, not established better**.

1. **Exact DC removal costs exactly `r = √((π−1)/π) ≈ 0.8256` of forward gain per layer — the *same* constant row-centering pays.** This is not an analogy; it is the duality expressed as a number, and it is a theorem (§The closed form). The brief's hope that the shift "decays the forward pass far less than row-centering" is **false**: the 60L measurement that suggested it (0.908 vs 0.828) was reading the gain in a regime where the shift had already stopped removing the DC.
2. `G(c) < 1` for every `c ∈ (0, 2/√π)`, so DC removal always costs forward gain. **Oracle correction (2026-08-15):** an earlier draft of this section concluded from that "the three requirements are not jointly satisfiable by DC removal — the only shift with unit forward gain is no shift at all." That overreaches. Boosting the He weights by `1/G(c)` — the exact analogue of what `row_centered_forward_balanced` does for row-centering — restores `g_fwd ≈ 1` and drives `g_bwd` to `1.2114 ≈ 1/r`, with the ratio `g_fwd/g_bwd` invariant under the boost (0.8862 unboosted vs 0.8861 boosted at `c = 1/√π`) — the signature of a **lock**, not an impossibility. The cost is transferable, and the correct, stronger claim is a **unification**: the gain-coupling lock is a property of DC removal *itself*, not of row-centering, and both routes — weight-space and activation-space — pay the identical constant `r`.
3. **At 30 layers, DC removal is level with He in real training — not worse, not established better** (§8, corrected 2026-08-15). The apparent final-epoch gap does not survive held-out, robust reading: `c = 0.10` is **+0.2 pp** test at best-epoch, the other arms negative (retraction note above; synthesis §4.3). The project's best result against He remains campaign 09's *same bar, six epochs sooner*.

This campaign was originally written up as a negative result, briefly relabeled a positive one, and settled — after the retraction — as: **no established win anywhere, no loss at 30L.** The 100-layer end-to-end comparison is unmeasured (8 of 10 arms aborted under the plain recipe — synthesis §4.3), and the frozen-readout ranking has He first at both depths — see [campaign 12](../12_frozen_readout/README.md).

**Builds on.** Campaign [09](../09_rcfwd_rescale/README.md) (three-requirements frame, conditioning-vs-content 2×2), campaign [10](../10_rc_frozen_ends/README.md) (100L row-centered nets are not trainable through frozen windows; forward-scale death at `~1e-8` is a real failure mode, so a candidate whose 100L forward RMS is `6e-8` is a predicted failure, not a surprise), and W2's dying-neurons proof (`P[dead] → ½`), which this campaign independently confirms empirically.

---

## The closed form

Let `u = relu(z)` with `z` zero-mean Gaussian, `R² = E[u²]`. He initialization gives `E[u²]` exactly equal to the previous layer's mean square, which is why `he` has per-layer forward gain 1. Write

```
A(c) = c² − 2c/√π
```

**Forward gain.** Using the half-Gaussian moments `E[u] = R/√π` and `E[u²] = R²` (`gradient_diagnostics_analysis.md` §4):

```
E[(u − cR)²] = R² − 2cR·E[u] + c²R² = R²(1 − 2c/√π + c²) = R²(1 + A(c))
```

so the per-layer forward gain is

> **G(c) = √(1 − 2c/√π + c²)**

`G` is minimised at `c = 1/√π`, where `G = √(1 − 1/π) = √((π−1)/π) = r ≈ 0.82565`. Since exact DC removal *requires* `c = 1/√π`, **removing the ReLU's DC necessarily multiplies the forward RMS by `r` per layer** — the identical constant row-centering pays. And `G(c) < 1` for all `c ∈ (0, 2/√π ≈ 1.128)`, so no nonzero DC removal is forward-scale-free.

**Cosine recursion.** With `g(ρ) = (1/π)[√(1−ρ²) + ρ(π − arccos ρ)]` the arc-cosine kernel ratio (`g(0) = 1/π`, `g(1) = 1`, `g'(ρ) = 1 − arccos(ρ)/π`), the post-shift pairwise cosine obeys

> **ρ_{ℓ+1} = (g(ρ_ℓ) + A) / (1 + A)**

`ρ = 1` remains a fixed point but its multiplier becomes `1/(1+A) > 1`, i.e. **repelling** — this is exactly why the shift attacks collapse. The attracting fixed point `ρ*` solves `Φ(ρ*) = −A` with `Φ(ρ) = (g(ρ) − ρ)/(1 − ρ)`, decreasing on `[0,1)` from `Φ(0) = 1/π`. Because `max_c |A| = 1/π`, attained *exactly* at `c = 1/√π`, the theory predicts `ρ* = 0` there and `ρ* > 0` on both sides — a U-shaped curve centred on the theoretically exact constant.

Both quantities are emitted per candidate by `scripts/relu_shift_geometry_screen.py` (`analytic_gain_G`, `analytic_cosine_fixed_point`) so every measurement below is stated next to its prediction.

## The duality, verified

`W(a − s·𝟙) = Wa − s·(W𝟙)`, so on a row-centered weight (`W𝟙 = 0`) the shift is exactly a no-op. `scripts/relu_shift_duality_check.py` → `reports/results/relu_shift_duality_check.json` (20L, width 500, fmnist, N=256):

| init | relative output diff | relative loss diff | max relative **grad** diff |
|---|---|---|---|
| `row_centered_he`, c=0.25…1.0 | 2.07e-5 – 3.47e-5 | **0.0** (c ≥ 0.5642) | 0.267 – 1.066 |
| `he`, c=0.25…1.0 | **0.845 – 1.076** | 0.200 – 0.216 | 1.12 – 1.17 |

The residual `2e-5` is float32 round-off: the premise itself only holds to `max|Σ_j W_ij| = 1.4e-6` (field `row_centering_premise`). The identity is a **forward** identity only — weight gradients differ even under row centering, by the rank-one term `s·δᵀ𝟙`, because `grad_W = δᵀa_prev` and `a_prev` *is* shifted.

## What ran (local, at initialization)

| Script | Output | Covers |
|---|---|---|
| `scripts/relu_shift_noop_check.py` | `relu_shift_noop_verification.json` | the `src/` gate |
| `scripts/relu_shift_geometry_screen.py` | `relu_shift_geometry_screen_{30,60,100}L_{fmnist,cifar10}.json` + two `_persample` controls | forward + geometry + content, 16 candidates × 3 depths × 2 datasets |
| `scripts/relu_shift_funnel_fwd_bwd.py` | `relu_shift_funnel_fwd_bwd.json` | backward, 100L, both datasets, both fork arms |
| `scripts/relu_shift_duality_check.py` | `relu_shift_duality_check.json` | the duality identity |
| `scripts/relu_shift_local_pretriage.py` | `relushift_local_pretriage_30L_{fmnist,cifar10}.json` | GO/NO-GO: does each arm descend at all |

`c` grid: `0.1 … 1.0` step `0.1`, plus `0.25`, `0.65`, `0.75` and `1/√π = 0.5642`. Width 500, seed 42, N = 512 probe samples. **`dataset_dead_fraction` is reported at N = 512 throughout** — it is not scale-free (41.2% at N=256 vs 40.0% at N=512 for the same 60L He net).

### The `src/` change and its no-op gate

`ClassifierConfig.relu_shift: Optional[float] = None` and `relu_shift_detach: bool = True`, applied in `DeepFCClassifier.forward` after `torch.relu(x)` and before the `_GradRescale` hook. `relu_shift=None` inserts **no tensor op at all**.

`reports/results/relu_shift_noop_verification.json`, verdict **PASS**, against `main` (`e7d7a33`):

- **Part A** (100L, width 500): output, loss, all **6** gradient tensors and all **202** parameter tensors after one SGD step are `torch.equal` to the reference revision. Control with `relu_shift=0.25` **does** change the output, so "identical" is not vacuous.
- **Part B** (campaign-10 config `rcfrozen_first3_smoke_fmnist_100L`, full 60k-sample epoch): every field of the logged history record matches bitwise.
- Cross-device caveat, stated because it bounds the claim: the committed campaign-10 JSONs are CUDA, this is CPU. `eval_train_loss` agrees at the reported 6 dp (`2.302585`); `eval_train_accuracy` is `0.10143` (CPU) vs `0.10187` (committed CUDA) because that network's logits are `O(1e-6)` and the argmax over 10 near-degenerate logits is decided by float noise. OLD-vs-NEW on the same machine — the actual no-op claim — is bitwise.

## Findings

### 1. He's dying neurons, confirmed and extended (W2's `→½`)

`dataset_dead_fraction` at the final layer, N = 512 (`relu_shift_geometry_screen_*.json`):

| depth | `he` fmnist | `he` cifar10 | every shift `c ≤ 0.75` |
|---|---|---|---|
| 30L | 0.342 | 0.302 | **0.000** |
| 60L | 0.400 | — | 0.000 – 0.014 |
| 100L | **0.476** | **0.466** | 0.000 – 0.100 |

He climbs monotonically toward ½ with depth, exactly as the proof predicts. **DC removal eliminates dying neurons outright** — vindicating the mechanism behind the user's original intuition.

### 2. Lead 1 resolved — the non-monotonicity in `c` is a *batch-statistic* artifact, not a property of DC removal

`rms(a)` is one scalar over the whole `(batch × units)` tensor, so the subtracted quantity is an **absolute** constant, not one proportional to each sample's own scale. A sample at relative scale `t = rms_s/rms_batch` is therefore effectively shifted with coefficient `c/t` — it sits at a *different point* of the U-shaped `A(c)` curve. And absolute subtraction **amplifies** relative norm spread (a fixed subtraction costs a smaller sample proportionally more), so the spread compounds with depth.

The diagnostic is `norm_heterogeneity_kappa = mean_s(rms_s)/rms_global` (1.0 = homogeneous). At 100L/fmnist it tracks the cosine exactly:

| `c` | 0.10 | 0.25 | 0.5642 | 0.70 | **0.75** | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|
| `kappa` @100 | 0.324 | 0.325 | 0.552 | 0.676 | **0.998** | 0.999 | 1.000 |
| `cos` @100 | 0.697 | 0.927 | 0.974 | 0.953 | **0.196** | 0.306 | 0.843 |
| `ρ*(c)` predicted | 0.884 | 0.493 | **0.000** | 0.100 | 0.185 | 0.292 | 0.820 |

Above a threshold `c` the constant term `c·r` dominates every sample's RMS, norms homogenise (`kappa → 0.998`), and the theory becomes accurate. Below it, heterogeneity runs away and the measurement inverts relative to the prediction.

**The control settles it.** Re-running with the RMS computed **per sample** (`--shift-scope per_sample`, a diagnostic only — not a training candidate) removes the absolute-subtraction effect. At 100L/fmnist, measured vs predicted:

| `c` | 0.40 | 0.50 | **0.5642** | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| cos, per-sample | 0.168 | 0.029 | **0.0037** | 0.010 | 0.039 | 0.099 | 0.188 | 0.306 | 0.844 |
| `ρ*(c)` predicted | 0.145 | 0.023 | **0.000** | 0.007 | 0.040 | 0.100 | 0.185 | 0.292 | 0.820 |
| cos, global (same run) | 0.954 | 0.964 | 0.974 | 0.972 | 0.963 | 0.953 | 0.196 | 0.306 | 0.843 |

With a per-sample RMS the theory is **exact to ~0.01 across the whole grid**, monotonicity in `c` is restored, and the optimum snaps back to exactly `c = 1/√π`, giving **mean pairwise cosine 0.0037 at 100 layers** — the lowest ever measured in this project. So:

> **What governs the non-monotonicity is the competition between DC-cancellation quality (maximised at `c = 1/√π`) and norm-homogenisation (monotone increasing in `c`). With a batch-global RMS the second effect fails below `c ≈ 0.75`, displacing the empirical optimum away from the theoretical constant. The displacement is depth- and dataset-dependent** — best `c` on fmnist is 0.65 (30L, cos 0.089) → 0.70 (60L, 0.137) → 0.75 (100L, 0.196); on cifar10 0.75 (30L) → 0.80 (60L and 100L).

This is a **live recommendation**, flagged for the oracle rather than acted on: a per-sample RMS is one line of code and appears to make the family behave as designed. It was not in the brief's scope (which fixed the batch-global form to match the committed screen), so it is not in the training grid.

### 3. Lead 2 resolved — `√r` is a coincidence of reading at L = 60; the real constant is `r`

The brief flagged that the implied per-layer gain at `c = 1/√π` is 0.9083, within 4e-4 of `√r = 0.90866`. **Derived, and it does not hold.** The analytic gain is `G(1/√π) = r = 0.82565` — not `√r`. The measured 0.9083 is the geometric *mean* of a gain that drifts upward with depth as heterogeneity breaks the DC cancellation (`implied_forward_gain_at_depth`, `relu_shift_geometry_screen_100L_*.json`):

| read at | L=10 | L=20 | L=30 | L=60 | L=100 |
|---|---|---|---|---|---|
| `c = 1/√π`, fmnist | 0.8381 | 0.8438 | 0.8580 | **0.9083** | 0.9404 |
| `c = 1/√π`, cifar10 | 0.8422 | 0.8593 | 0.8921 | 0.9403 | 0.9608 |
| `c = 0.75`, fmnist | 0.8552 | 0.8459 | 0.8464 | 0.8463 | 0.8468 |

It is not a constant, so it cannot equal a constant. It merely crosses `√r` somewhere near L = 60. The closed form is confirmed instead by `c = 0.75`, which stays in the homogeneous regime on fmnist and whose implied gain is **depth-independent and equal to `G(0.75) = 0.8463` to four decimals** at every read depth. That is a direct verification of `G(c)`, and it is the constant that belongs in the thesis.

### 4. Backward pass: flat error, catastrophic gradient funnel

`relu_shift_funnel_fwd_bwd.json`, 100L, width 500, batch 256, at init:

| candidate | fwd RMS @L100 | bwd δ ratio | **grad row-norm ratio** | cos @L100 |
|---|---|---|---|---|
| `he` | 1.29e+00 | 1.52 | 2.51 | 0.9993 |
| `row_centered_he` | 5.23e-09 | 1.27 | 1.5e+08 | 0.3200 |
| shift `c=0.25` | 1.39e-02 | 1.17 | 154 | 0.9129 |
| shift `c=1/√π` | 2.04e-03 | 1.13 | 856 | 0.9704 |
| shift `c=0.70` | 2.40e-08 | 1.09 | 4.48e+07 | 0.1185 |
| shift `c=0.75` | 5.88e-08 | 1.07 | 1.84e+07 | 0.1900 |

The back-propagated **error** `δ` is the flattest of any family this project has measured (ratio 1.07–1.17 over 100 layers, versus He's 1.52). But `grad ≈ a ⊗ δ`, so the `G(c)^L` forward decay lands directly on the weight gradient: the good-geometry candidates sit at `1e7`–`5e7` gradient funnels and forward RMS `~6e-8` — **the exact regime campaign 10 showed underflows to float32 zero and freezes training**. Requirements (i) and (ii) are in direct, quantified conflict.

### 5. The trade-off curve, and where the shift genuinely wins

At **30 layers** the picture is much better than at 100 (`relu_shift_geometry_screen_30L_*.json`, `distance_correlation_probe_layers["30"]`, `cosine_knn_accuracy[-1]`, chance = 0.10):

| candidate | fmnist dist-corr | fmnist 10NN | cifar10 dist-corr | cifar10 10NN | dead | fwd RMS |
|---|---|---|---|---|---|---|
| input | 1.0 | 0.746 | 1.0 | 0.254 | — | — |
| `he` | 0.396 | **0.381** | 0.388 | 0.123 | 0.342 / 0.302 | 7.8e-1 |
| `row_centered_he` | 0.225 | 0.107 | 0.422 | 0.106 | 0.000 | 3.8e-3 |
| shift `c=0.10` | 0.414 | 0.283 | 0.619 | **0.143** | **0.000** | 2.0e-1 |
| shift `c=0.20` | **0.426** | 0.143 | **0.691** | 0.113 | **0.000** | 8.1e-2 |
| shift `c=0.25` | 0.423 | 0.178 | 0.660 | 0.117 | **0.000** | 5.8e-2 |
| shift `c=0.65` | 0.083 | 0.117 | 0.229 | 0.107 | **0.000** | 5.2e-3 |

`c = 0.1–0.25` **beats both baselines on distance-correlation-to-input on both datasets** (0.426 vs He's 0.396 on fmnist; 0.691 vs 0.388 on cifar10 — a 1.8× improvement), with **zero** dead units against He's 30–34%, at a forward scale two orders of magnitude healthier than row-centering's. On cifar10 `c = 0.1` also beats He on k-NN content (0.143 vs 0.123). This is the first candidate in the project to beat He on the input-geometry metric.

The honest counterweight, and the brief's own caveat: **He's advantage at depth may be a norm artifact.** He retains a large shared DC, which makes pairwise distances track `‖a_i‖` and hence `‖x_i‖`. The shift removes exactly that. A norm-controlled variant is **not** run here and remains the gap before "the shift preserves input geometry better than He" is written down as a result.

At **100 layers** nothing survives: every arm including `he` is at or near chance k-NN (fmnist: He 0.190, best shift 0.170, `row_centered_he` 0.061; cifar10: He 0.115, best shift 0.117, rc 0.098), and the arms with good geometry have forward RMS `~1e-7`.

### 6. The detached / differentiable fork — recommendation: **detached**

`rms(a)` carries gradient, so the shift's Jacobian is `I − (c/(N·rms))·𝟙aᵀ`, a rank-one term that couples every unit **and every sample in the batch**. Measured directly (`fork_relative_grad_diff` = `‖g_diff − g_detach‖/‖g_detach‖` per hidden layer, 100L):

| `c` | fmnist max / median | cifar10 max / median | forward identical |
|---|---|---|---|
| 0.25 | 0.119 / 0.077 | 0.476 / 0.296 | True |
| 1/√π | **0.708** / 0.316 | 0.164 / 0.105 | True |
| 0.70 | 0.061 / 0.032 | 0.424 / 0.133 | True |
| 0.75 | 0.062 / 0.032 | 0.055 / 0.035 | True |

**Status after the 20-epoch smoke (2026-08-15): this recommendation is REOPENED.** The differentiable arm beat detached on *both* datasets — 0.9391 vs 0.9289 (fmnist) and 0.6011 vs 0.5931 (cifar10) — reproducing the pre-triage gap that this section dismissed as "not evidence". By the section's own stated terms ("if the 20-epoch smoke reproduces the gap, the recommendation should be revisited... the principled arguments are about *cleanliness of the comparison*, and would not outrank a reproducible training advantage"), the differentiable variant now has the better empirical case and the detached default should be re-decided rather than assumed. The original argument, unchanged, follows.

So the fork is **not** numerically negligible — it moves the per-layer weight gradient by a median 3–32% and up to 71%. **Recommended detached at the time** (`relu_shift_detach=True`, now the config default), for three reasons:

1. **It is the exact dual.** Row-centering is a pure weight-side constraint with no gradient modification of its own; its activation-space dual should be a pure forward-side intervention. The differentiable arm adds a coupling with no counterpart in row-centering, which would confound the very comparison the campaign exists to make.
2. **It is the clean ablation.** Detached, the backward pass is bit-identical to plain He backprop, so any training difference is attributable *purely* to the forward/geometry change.
3. **The coupling has no principled scaling.** Its size is set by `Σ_k g_k` over the whole batch, giving the erratic 3–71% spread above rather than a controlled per-layer factor — and it couples samples within a batch, BatchNorm-style, on top of a forward pass that is *already* batch-dependent.

**The counter-evidence, stated plainly:** the 2-epoch local pre-triage marginally *favours* the differentiable arm — `c=0.25` reaches eval-train 0.8260 differentiable vs 0.7986 detached (fmnist) and 0.2497 vs 0.2392 (cifar10) (`relushift_local_pretriage_30L_*.json`, `arms.c025_diff` / `arms.c025`, `eval_train_accuracy[-1]`). That is one seed, two epochs, and a 1.1–2.7 pp gap on a recipe whose epoch-to-epoch movement is larger than that — it is not evidence, and it is exactly why the differentiable arm is kept as a **cluster control** (`*_diff` labels, `c=0.25` at 30L on both datasets) rather than the recommendation being treated as settled. If the 20-epoch smoke reproduces the gap, the recommendation should be revisited: the principled arguments above are about *cleanliness of the comparison*, and would not outrank a reproducible training advantage.

**Batch-dependence watch item.** `rms` is a batch statistic with **no running-statistics mechanism**, so unlike BatchNorm the eval-time forward uses the eval batch's own RMS. The runner asserts `batch_size == eval_batch_size` (both 256) so train and eval stay like-for-like; a partial final batch still computes a slightly different shift. This is a genuine architectural wart of the batch-global form and a second argument for the per-sample variant in §2.

### 7. Local pre-triage: which arms descend at all

`relushift_local_pretriage_30L_{fmnist,cifar10}.json` — the exact cluster recipe (NoBN, width 500, SGD lr 1e-2 mom 0 wd 0, bs 256, seed 42, no clipping), 2 epochs, CPU. **Not a result** — 2 epochs cannot separate "slow" from "stuck", and these are CPU numbers. It is a GO/NO-GO so 18 SLURM jobs are not queued behind a dead arm. `eval_train_accuracy` at epoch 1 → epoch 2:

| arm | fmnist | cifar10 | final grad range (fmnist) |
|---|---|---|---|
| `he` | 0.750 → **0.830** | 0.243 → **0.329** | 8.7e-1 – 3.1e+0 |
| `row_centered_he` | 0.123 → 0.346 | 0.113 → **0.107** (stuck) | 1.4e+0 – 3.9e+0 |
| `c=0.10` | 0.696 → 0.807 | **0.273** → 0.295 | 1.0e+0 – 2.9e+0 |
| `c=0.25` | 0.701 → 0.799 | 0.159 → 0.239 | 7.8e-1 – 4.0e+0 |
| `c=0.25` differentiable | 0.683 → 0.826 | 0.174 → 0.250 | 9.4e-1 – 4.3e+0 |
| `c=0.70` | 0.118 → **0.139** | 0.102 → **0.103** | 2.4e-2 – 2.9e+0 |

Three things worth carrying into the smoke triage. **(a)** The small-`c` arms train comparably to He and far better than `row_centered_he`, which is flat at chance on cifar10 — so DC removal on He weights is *not* inheriting row-centering's trainability failure at 30L. **(b)** At epoch 1 on cifar10, `c = 0.10` (0.273) is ahead of `he` (0.243). **(c)** `c = 0.70` — the best-geometry arm — looks **stuck at chance on both datasets** at 2 epochs, with its minimum layer gradient two orders of magnitude below the others. This was written up as "the clearest single piece of evidence for the campaign's negative conclusion." **That reading was wrong, and the 20-epoch smoke refutes it** (§8): `c = 0.70` reaches 0.9291 / 0.5394 final-epoch train — comparable to He, not dead. It is a slow starter (0.1254 at epoch 1 on fmnist), not a dead arm, and 2 CPU epochs cannot tell those apart — which is exactly the limitation this section states about itself. The lesson generalises: a 2-epoch pre-triage is a GO/NO-GO on *crashes*, not evidence about *learning*.

### 8. Training results (2026-08-15, corrected) — level with He at 30L under robust statistics

The 12 30-layer smokes ran on the cluster (20 epochs, NoBN, width 500, SGD lr 1e-2, bs 256, seed 42, no clipping).
`eval_train_accuracy`, epoch 1 → epoch 20, from `reports/results/relushift_*_smoke_*_30L*.json`:

| arm | fmnist | cifar10 | mean | vs `he` |
|---|---|---|---|---|
| **`c = 0.10`** | 0.6666 → 0.9246 | 0.2637 → **0.6374** | **0.7810** | **+1.4 / +13.2 pp** |
| `c = 0.25` (differentiable) | 0.6877 → **0.9391** | 0.1817 → 0.6011 | 0.7701 | +2.8 / +9.6 pp |
| `c = 0.25` | 0.6701 → 0.9289 | 0.1656 → 0.5931 | 0.7610 | +1.8 / +8.8 pp |
| `c = 0.70` | 0.1254 → 0.9291 | 0.1032 → 0.5394 | 0.7342 | +1.8 / +3.4 pp |
| `row_centered_he` | 0.1364 → 0.9020 | 0.1145 → 0.5480 | 0.7250 | −0.9 / +4.3 pp |
| `he` | 0.7313 → 0.9107 | 0.2726 → 0.5052 | 0.7080 | baseline |

**Correction (retraction note at top).** The table is final-epoch **train** accuracy, and an earlier revision read it as
"every shift arm beats He — the first accuracy win in the project." Retracted: on held-out test with robust estimators
the gap collapses (`c = 0.10`: +6.6 pp final-epoch → +2.1 mean-of-last-5 → **+0.2 best-epoch**; the other arms go
negative), because He's test curve oscillates 0.38–0.47 and lands on its worst value at epoch 20 (synthesis §4.3).
What survives: `c = 0.10` is *not worse* than He at 30L, and the within-family ordering below is real. The project's
best result against He remains campaign 09's rcfwd reaching the *same* pass bar six epochs sooner (ep74 vs ep80).

Three things the table settles:

**The optimum is monotone in `c`, and it is small.** `0.10 > 0.25 > 0.70` by mean. Not the theoretically exact
`1/√π ≈ 0.564`, and emphatically **not** the geometry-optimal `c ≈ 0.70–0.75` that §5 identified. Mild DC removal wins;
aggressive DC removal is the worst shift arm. Whatever the shift is buying, it is not "better cosine geometry".

**The init-time geometry screen is anti-correlated with training outcome, at least across this family.** §5 ranked
`c = 0.65–0.75` best on mean pairwise cosine and worst on probe content; training ranks `c = 0.70` *last* among the
shifts. Any future screening that gates cluster time on probe or cosine metrics — including the α-family screen parked
as W3 — needs re-basing on a trained-readout metric (see [campaign 12](../12_frozen_readout/README.md)).

**`row_centered_he` is fine at 30L**, reaching 0.9020 / 0.5480 from a chance-level start. Its failures in campaigns
06/07/09 are depth failures, not a property of row-centering as such.

**What this does not show.** These are 20-epoch smokes on the *train* set at 30 layers — a rate comparison, not a
ceiling, and silent about generalization. The 18 audits are still ungated. And the ranking **inverts at 100 layers**:
under campaign 12's frozen-readout protocol every shift arm loses to He. Depth and protocol are confounded between the
two campaigns; the 30L frozen-readout arms (`frozenro_*_30L`) were queued to separate them.

## Reproduce

```bash
# local, minutes on CPU -- the whole init-time story
python scripts/relu_shift_noop_check.py                      # the src gate; must print PASS
python scripts/relu_shift_duality_check.py
for d in 30 60 100; do
  for ds in fashion_mnist cifar10; do
    tag=$( [ "$ds" = fashion_mnist ] && echo "${d}L_fmnist" || echo "${d}L_cifar10" )
    python scripts/relu_shift_geometry_screen.py --depth $d --dataset $ds --tag $tag
  done
done
python scripts/relu_shift_geometry_screen.py --depth 100 --shift-scope per_sample \
       --tag 100L_fmnist_persample                            # the mechanism control
python scripts/relu_shift_funnel_fwd_bwd.py --depth 100
python scripts/relu_shift_local_pretriage.py --epochs 2 --depth 30 --dataset fashion_mnist
```

```bash
# cluster -- 18 smoke jobs; gate the 18 audits on smoke triage
cd ~/thesis   # after sync; clear __pycache__ first (config.py gained two new fields)
for f in cluster/11_relu_shift/relushift_*_smoke_*.sub; do sbatch "$f"; done
```

Pull back with `bash cluster/pull_results.sh 'relushift_*_smoke_*' 11_relu_shift`. Logs end with `SUMMARY <label> | PASS/fail | ...`. Pass criterion is the advisor's campaign-10-onward rule: `eval_train_accuracy ≥ 0.99`, loss condition dropped, both logged.

## Evidence & gaps

- **Init-time evidence is complete**: 8 screen JSONs (3 depths × 2 datasets + 2 per-sample controls), the funnel, the duality check, the no-op verification, and 2 pre-triage JSONs — all in `reports/results/`, all reproducible by the commands above.
- **Training results landed 2026-08-15** (§8): the 12 30L smokes ran; `reports/results/relushift_*_smoke_*_30L*.json`. The six **100L** smokes were deliberately **not** submitted — campaign 10 showed end-to-end training at 100L fails for *every* initialization (rcfwd: 0.1746 with all layers trainable vs 0.8335 with three), so a 100L end-to-end arm would measure that, not DC removal. The 100L question is answered by [campaign 12](../12_frozen_readout/README.md)'s frozen-readout protocol instead. The 18 audits remain ungated.
- **Gap — the norm-control experiment is not run.** "The shift preserves input geometry better than He at 30L" (§5) is stated on distance correlation and k-NN. The brief's caveat that He may be surviving on *norm* rather than *angular* information is not yet tested, and until it is, §5 is a measurement, not an interpretation.
- **Open recommendation, deliberately not acted on: the per-sample RMS variant** (§2). It makes the closed-form theory exact, restores monotonicity in `c`, puts the optimum back at `c = 1/√π`, reaches cosine 0.0037 at 100 layers, and removes the batch-dependence wart — but it was outside the brief's settled scope, so it is flagged for the oracle rather than added to the grid. It does not resolve the requirement-(ii) conflict: `G(1/√π) = r < 1` regardless of how the RMS is computed, so the `r^L` forward decay survives.
- The `c` grid is 0.05-resolution near the interesting region but 0.1 elsewhere; the 30L fmnist optimum (`c = 0.65`, cos 0.089) is bracketed but not resolved to better than ±0.05.
- 60L cifar10 and the 60L per-sample control were run for completeness; only 30L/100L feed the training grid.
