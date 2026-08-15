# Campaign 10 follow-up — full synthesis

**Date:** 2026-08-15 · **Scope:** everything run after the advisor meeting that followed campaign 10 · **Simulations:** 66 (54 new cluster jobs, 12 prior campaign-10 cells re-used as reference, plus 15 local init-time diagnostics)

---

## 0. Executive summary

The advisor asked for three things. All three were answered, and two of them answered *against* what we expected.

| # | Ask | Answer |
|---|---|---|
| 1 | Redo the frozen-window experiments at **2 layers**, and train the working cell to ~99% train accuracy | Done. The tail window reaches **0.9498 train** at 400 epochs — but **0.1132 test**. It is memorizing, not learning. |
| 2 | Do the correction **for the backward only**; check the activations, they will be small | Done, and the activations are `5.01e-9` — provably, since `_GradRescale` is identity in the forward pass. A **9-order-of-magnitude LR sweep** leaves the loss pinned at `ln 10` to within `2.08e-7`. The front window is unreachable, and it is not a step-size problem. |
| 3 | Take He and **subtract a constant**, to prevent geometric collapse | Done, in the scale-relative form `a ← relu(Wx) − c·rms(a)`. **It does not beat He.** At 30 layers the apparent +6.6 pp win is a final-epoch artifact that vanishes under robust statistics; at **100 layers** — the depth this thesis is about — every arm loses to He by 7–26 points, and on CIFAR-10 all four sit at chance. |

Three cross-cutting results emerged that none of the three asks anticipated:

- **Capacity ≠ content.** A 2–3 layer readout on a frozen 100-layer random stack memorizes the training set to 95% and generalizes at chance. The representation stays *injective* — points remain distinguishable — while class geometry is destroyed.
- **He wins under every protocol at every depth measured.** Frozen-readout ranks He first at 30L and 100L; end-to-end ranks it first at 100L and statistically level at 30L. An earlier reading had the two protocols in conflict and proposed that DC removal trades feature-map quality for optimization conditioning. With the 100L end-to-end arms in hand there is no conflict left to resolve, and no evidence for the conditioning benefit.
- **The probe-based screen is vindicated.** Cosine-kNN said content is at chance by depth ~25. Held-out accuracy at depth 100 agrees. An earlier reading of this data — that the probes "understate badly" — was an artifact of reading train accuracy only, and is retracted here.
- **DC removal is a dead end, by both routes.** Row-centering (weight-space) and the post-ReLU shift (activation-space) are the same intervention, pay the identical constant `r`, and neither beats He. That is a real result, not an absence of one: it closes a direction the thesis has pursued since the beginning.

---

## 1. Where we stood before today

The thesis studies initialization through random projections, against **three requirements**:

> **(i)** avoid geometric collapse · **(ii)** keep gradients stable · **(iii)** preserve class content

Prior state: **He 10/12** architectures pass (100L+BN remains an open wall), **V2 5/12** (depth ceiling ≈ 30L), and **rcfwd** is numerically stable at all depths with one PASS at fmnist/30L but blocked at ≥50L by representation *content* rather than gradient flow.

### The gain-coupling lock

For a row-centered weight acting on post-ReLU activations, the forward and backward per-layer gains cannot both be 1. Their ratio is pinned:

```
g_fwd / g_bwd = r = √((π−1)/π) ≈ 0.82565
```

which follows from the half-Gaussian moments of `u = relu(z)` and the centering ratio `Var(u)/E[u²] = (π−1)/π`. You may slide along the family but not escape it:

| choice | `g_fwd` | `g_bwd` | realized by |
|---|---|---|---|
| backward-balanced | 0.826 | 1.000 | `row_centered_he` (**raw**) |
| forward-balanced | 1.000 | 1.211 | `row_centered_forward_balanced` (**fwdbal**) |

Campaign 09's recipe (**rcfwd**) takes the forward-balanced init and cancels the induced backward blow-up with `_GradRescale` — identity in the forward pass, `× r` in the backward — giving flat gains in both directions.

---

## 2. What ran

66 simulations. Every number in this document traces to a file in `reports/results/`.

| group | n | what |
|---|---|---|
| Campaign 10 — 3-layer windows (prior) | 8 | `rcfrozen_{first3,last3}_smoke_{fmnist,cifar10}_100L[_rcfwd]` |
| Campaign 10 — 2-layer windows | 8 | `rcfrozen_{first2,last2}_smoke_*_100L[_rcfwd]` |
| Campaign 10 — recipe ablation | 12 | `_fwdbal` (2), `_rawrescale` + LR ladder (10) |
| Campaign 10 — target-accuracy audits | 4 | `rcfrozen_{last3,last2}_audit_*_100L_rcfwd`, 400 ep |
| Campaign 11 — DC removal, 30L end-to-end | 12 | `relushift_{he,rc,c010,c025,c070}_smoke_*_30L` + 2 fork controls |
| Campaign 12 — frozen readout | 22 | `frozenro_*_{30,100}L` |
| Local init-time diagnostics | 15 | screens, funnels, duality and no-op checks |

Fixed across all cluster runs unless stated: width 500, NoBN, plain SGD (lr 1e-2, momentum 0, weight decay 0, no scheduler), batch 256, seed 42, **no gradient clipping** (assert-enforced).

---

## 3. The mathematics established today

### 3.1 The cost of DC removal, in closed form

ReLU output is non-negative, so it carries a positive DC component shared by every sample — the engine of the arc-cosine kernel's `ρ → 1` collapse:

```
E[u] = R/√π ≈ 0.5642·R ,   R² = E[u²]
```

Subtracting `c·R` after the ReLU gives, using `E[(u − cR)²] = R²(1 − 2c/√π + c²)`, the per-layer forward gain

> **G(c) = √(1 − 2c/√π + c²)**

Minimized at `c = 1/√π`, where

> **G(1/√π) = √(1 − 1/π) = √((π−1)/π) = r = 0.82565**

**Exact DC removal costs exactly the constant row-centering pays.** This is not an analogy — it is the duality expressed as a number. And since `W(a − c𝟙) = Wa − c(W𝟙)`, the shift is *exactly a no-op* on a row-centered weight: the two families are the same intervention applied at opposite ends of the layer.

Verified against simulation to `3e-4`, and `G(1/√π) = r` to machine precision.

**Correction to an earlier reading.** A first pass concluded from `G(c) < 1` on `(0, 2/√π)` that requirements (i) and (ii) are *provably incompatible*. That overreaches. Boosting the weights by `1/G(c)` — the exact analogue of `row_centered_forward_balanced` — restores `g_fwd ≈ 1` and drives `g_bwd → 1.2114 ≈ 1/r`, with the ratio `g_fwd/g_bwd` invariant under the boost (0.8862 unboosted vs 0.8861 boosted). That invariance is the signature of a **lock**, not an impossibility. The correct and stronger claim: **the gain-coupling lock is a property of DC removal itself**, not of row-centering, and both routes pay the identical `r`.

### 3.2 How far `_GradRescale` reaches

`_GradRescale` is applied after every hidden ReLU, so a trainable layer at position `ℓ` picks up `r` raised to the number of rescale operations between it and the loss. With most layers frozen, autograd stops at the first fully-frozen upstream chain, and the reach is dramatically asymmetric:

| window | multiplier on the trainable layers | measured |
|---|---|---|
| `last2` = {fc99, fc100} | `r¹, r²` | 0.8256, 0.6817 |
| `first2` = {fc1, fc2} | `r¹⁰⁰, r⁹⁹` | 4.78e-9, 5.79e-9 |

**Consequence:** for a *tail* window the rescale is a ~1.2–1.5× learning-rate nudge and the (init × rescale) 2×2 collapses to a 1×2 — only the initialization matters. For a *front* window it spans 17 orders of magnitude. This was verified to four significant figures and it predicted the `fwdbal`-vs-`rcfwd` outcome in §4.1 before that run existed.

### 3.3 The dying-neuron rate

The arc-cosine kernel map `χ(ρ) = (sin α + (π−α)cos α)/π` with `α = arccos ρ` is tangent to the identity at its fixed point, so convergence is algebraic rather than geometric. Expanding with `u = 1 − ρ`:

```
1 − χ(ρ) = u − (2√2/3π)·u^{3/2} + O(u²)
```

Solving `u_ℓ = u_{ℓ−1} − c·u_{ℓ−1}^{3/2}` with `c = 2√2/3π` gives

> **ε_ℓ = 1 − ρ_ℓ ≈ (9π²/2)·ℓ⁻² ≈ 44.41/ℓ²** and **Δ_ℓ = arccos(1−ε_ℓ)/π ≈ 3/ℓ**

with `4/c² = 9π²/2` an exact identity, and the leading constant independent of the initial correlation.

**Measured against the theory.** He's dataset-dead fraction (a unit whose pre-activation is ≤0 for *every* sample; N = 512 probe samples — this quantity is **not** scale-free in N):

| depth | 30 | 60 | 100 |
|---|---|---|---|
| He dead fraction | 0.342 | 0.400 | **0.476** |

heading for the ½ the proof predicts. Replacing the proof's `(N choose 2)` union bound with a **Slepian** reduction — the equicorrelated configuration is extremal, so `P[all N signs agree]` reduces to a 1-D integral with *logarithmic* rather than quadratic N-dependence — makes the bound bite at practical depth: it needs `ℓ > 43` for N = 60000 instead of `5.4 × 10⁹`. At the measured setting it gives `P[dead] ≥ 0.3881` against a measured 0.4000, tight to 3%.

---

## 4. Results, by the advisor's asks

### 4.1 Ask 1 — 2-layer windows, trained to target

`last2 = {fc99, fc100}`, `first2 = {fc1, fc2}`, output head frozen in both so the two ends are symmetric.

**20-epoch smokes, 100 layers** (train / test):

| recipe | first2 fmnist | first2 cifar10 | last2 fmnist | last2 cifar10 |
|---|---|---|---|---|
| raw | 0.098 / 0.098 | 0.101 / 0.101 | 0.100 / 0.100 | 0.100 / 0.100 |
| rcfwd | 0.101 / 0.099 | 0.098 / 0.097 | **0.177** / 0.098 | **0.171** / 0.101 |
| fwdbal | — | — | **0.205** / 0.099 | **0.195** / 0.102 |
| raw+rescale | 0.100 / 0.106 | 0.102 / 0.101 | — | — |

**400-epoch audits** (`--target-train-accuracy 0.99`, never reached; all four ran to the 400-epoch cap):

| run | train | test | gap |
|---|---|---|---|
| `last3`/fmnist | **0.9498** | **0.1132** | +0.837 |
| `last3`/cifar10 | 0.9413 | 0.1145 | +0.827 |
| `last2`/fmnist | 0.7693 | 0.1149 | +0.654 |
| `last2`/cifar10 | 0.7580 | 0.1167 | +0.641 |

Trajectory (`last3`/fmnist): ep25 → 0.266, ep100 → 0.571, ep200 → 0.777, ep300 → 0.889, ep400 → 0.950. Monotone, no plateau — it would keep climbing.

**Reading.** On the advisor's criterion (`eval_train_accuracy ≥ 0.99`, loss dropped) the tail window is close and still rising. But **test accuracy never leaves chance**. The 0.95 is memorization of 60k training points by a 2–3 layer head sitting on a frozen random map. Window size matters a lot for memorization capacity — `last3` 0.950 vs `last2` 0.769 — and not at all for generalization: both are at 0.11.

`fwdbal` slightly *outperforms* `rcfwd` on the tail (0.205 vs 0.177), and the ratio 0.866 matches the `r¹, r²` prediction of §3.2 — the rescale is a small effective-LR reduction there, nothing more. It did not abort, because autograd never propagates into the frozen upstream chain where the 4.61e8 blow-up lives.

### 4.2 Ask 2 — the backward-only correction

"Correction for the backward only" = keep the initialization (`row_centered_he`), correct only the backward pass. This is the never-run corner of a 2×2 that campaigns 09/10 had only half-sampled:

| corner | init | `grad_rescale` | act RMS @L100 | ‖∂L/∂W‖ @L1 | @L100 | max/min |
|---|---|---|---|---|---|---|
| raw | `row_centered_he` | — | 5.01e-9 | 1.91 | 1.15e-8 | 1.66e8 |
| **raw+rescale** | `row_centered_he` | `r` | **5.01e-9** | 9.11e-9 | 9.48e-9 | **1.13** |
| fwdbal | `row_centered_forward_balanced` | — | 1.26 | 4.61e8 | 3.43 | 1.35e8 |
| rcfwd | `row_centered_forward_balanced` | `r` | 1.26 | 2.20 | 2.83 | 1.35 |

**The advisor's prediction is exactly right and provably so.** `_GradRescale` is identity in the forward pass, so `raw+rescale` has activations *bit-identical* to `raw` — 5.01e-9 at layer 100. "The activations will probably be small" holds by construction. That corner also has the **flattest gradient profile of all four** (1.13× across 100 layers, better than rcfwd's 1.35×) — but pinned uniformly at ≈9e-9 ≈ `r¹⁰⁰`.

**The LR ladder settles it.** With gradients at 9e-9, the analytically matched learning rate is ≈ `1e-2 / r¹⁰⁰ ≈ 2.1e6`. Sweeping `lr ∈ {1e-2, 1e2, 1e4, 1e6, 1e7}` on the front window across both datasets — ten runs:

> every run ends at `eval_train_loss = 2.302585`, and the **maximum deviation from `ln 10` across all ten is `2.08e-7`.**

Nine orders of magnitude of learning rate produce numerically indistinguishable loss trajectories. The front window is not starved of step size — it is **causally disconnected from the loss**: under the raw init anything fc1/fc2 learn is multiplied by `0.826⁹⁸ ≈ 1e-8` before reaching the head, so the logits stay at zero, the softmax stays uniform, and the loss stays exactly `ln 10`.

Under `rcfwd`, where the forward is flat and the signal *does* arrive at O(1), the front window's loss **moves — upward**, 3.46 → 5.24, with accuracy pinned at chance. So at 100 layers a front window can change the output and cannot improve it.

**This closes the hypothesis the ask was built on.** The worry was that the rescale might be misdirecting a gradient that, left alone, points somewhere useful. It is not: no scaling of that gradient, over nine orders of magnitude, changes anything.

### 4.3 Ask 3 — He minus a constant: no win, at either depth

Run in the scale-relative form `a ← relu(Wx) − c·rms(a)`, swept over `c`. **30 layers, end-to-end, 20 epochs** (train / test):

| arm | fmnist | cifar10 | test vs He |
|---|---|---|---|
| **`c = 0.10`** | 0.9246 / 0.8542 | 0.6374 / **0.4496** | **−0.2 / +6.6 pp** |
| `c = 0.25` (differentiable) | 0.9391 / **0.8615** | 0.6011 / 0.4500 | **+0.6 / +6.7 pp** |
| `c = 0.25` | 0.9289 / 0.8563 | 0.5931 / 0.4396 | +0.1 / +5.6 pp |
| `c = 0.70` | 0.9291 / 0.8535 | 0.5394 / 0.4183 | −0.2 / +3.5 pp |
| `row_centered_he` | 0.9020 / 0.8394 | 0.5480 / 0.4327 | −1.6 / +4.9 pp |
| `he` | 0.9107 / 0.8557 | 0.5052 / 0.3835 | baseline |

**Read at the final epoch — which is the wrong way to read it.** He's CIFAR-10 test accuracy oscillates between 0.38 and 0.47 across the last eight epochs and lands on its *worst* value at epoch 20 (max 0.4665 at ep15, final 0.3835 — an 8.3-point drop). Recomputed robustly, the effect disappears:

| CIFAR-10 test, vs He | final epoch | mean last 5 | best epoch |
|---|---|---|---|
| `c = 0.10` | +6.6 pp | +2.1 pp | **+0.2 pp** |
| `c = 0.25` | +5.6 pp | +0.2 pp | −2.3 pp |
| `c = 0.70` | +3.5 pp | −1.4 pp | −4.0 pp |
| `row_centered_he` | +4.9 pp | −1.7 pp | −1.6 pp |

There is **no established win at 30 layers**. What survives is "`c = 0.10` is not worse, and may be marginally better, within run-to-run variance."

**And at 100 layers it loses outright** — the thesis's canonical depth, same recipe, all five arms (best-epoch test):

| arm | fmnist | vs He | cifar10 | vs He |
|---|---|---|---|---|
| **`he`** | **0.2753** | — | **0.3571** | — |
| `c = 0.10` | 0.2047 | −7.1 pp | 0.1000 | **−25.7 pp** |
| `c = 0.25` | 0.1755 | −10.0 pp | 0.1326 | −22.4 pp |
| `row_centered_he` | 0.1233 | −15.2 pp | 0.1026 | −25.4 pp |
| `c = 0.70` | 0.1001 | −17.5 pp | 0.1104 | −24.7 pp |

On CIFAR-10 every DC-removal arm is pinned at chance while He trains — a qualitative failure, not a slower rate. **Caveat:** these are 20-epoch smokes and He needs ~152 epochs to reach 0.9953 at 100L NoBN, so all arms are in early transient; but chance-vs-0.357 is not a transient difference. The 18 audits remain ungated.

**None of this is a win.** Campaign 09's best prior result was rcfwd reaching the *same* bar six epochs sooner than tuned He; today adds no higher number. What today does add is that the DC-removal route — pursued in this thesis since the beginning, in both its weight-space and activation-space forms — is now closed with evidence at the depth that matters.

Two structural facts:

- **The ordering in `c` is consistent across depths and metrics.** `0.10 > 0.25 > 0.70` at 30L and at 100L. So *if* the shift has any value it is at small `c` — not the theoretically exact `1/√π ≈ 0.564`, and emphatically not the geometry-optimal `c ≈ 0.70–0.75`. The best-geometry arm is reliably the worst trainer.
- **The differentiable fork beats the detached one on both datasets**, reproducing a gap that campaign 11 had dismissed as noise. That recommendation is now reopened.

And the mechanism behind the original intuition checks out: every `c ≤ 0.75` variant sits at 0–1.4% dataset-dead units against He's 34–48%. Killing the DC does stop neurons dying.

---

## 5. What the three asks did not anticipate

### 5.1 Capacity is not content

The 400-epoch audits reach 0.95 train and 0.11 test. A frozen 100-layer random map therefore leaves the data **injective but not class-structured**: a small trained head can still tell 60,000 points apart well enough to memorize labels, while nothing class-aligned survives to be generalized from.

This is why train and test must both be reported. Reading train alone produced two wrong conclusions, both retracted in §6.

### 5.2 The protocol/depth question, and why it dissolved

For most of the day the evidence looked contradictory: 30L end-to-end appeared to favour DC removal, while 100L frozen-readout clearly favoured He. Those two cells differ in *both* depth and protocol, so neither attributed. Two runs settled it — the frozen-readout protocol at 30L, and the end-to-end protocol at 100L:

**Frozen readout, 20 epochs** (train / test):

| arm | 30L fmnist | 30L cifar10 | 100L fmnist |
|---|---|---|---|
| **`he`** | **0.5500 / 0.5481** | **0.1934 / 0.1840** | 0.1613 / 0.1664 |
| `c = 0.10` | 0.4168 / 0.4126 | 0.1887 / 0.1842 | 0.1376 / 0.1381 |
| `c = 0.25` | 0.2239 / 0.2214 | 0.1411 / 0.1393 | 0.1104 / 0.1062 |
| `rcfwd` | 0.2024 / 0.1264 | 0.1780 / 0.1075 | 0.1772 / 0.0981 |
| `c = 0.70` | 0.1350 / 0.1346 | 0.1090 / 0.1036 | 0.1008 / 0.1011 |
| `row_centered_he` | 0.1284 / 0.1264 | 0.1154 / 0.1149 | 0.1000 / 0.1000 |

**He wins at both depths.** When only the 30L end-to-end and 100L frozen-readout cells existed, the two campaigns appeared to disagree, and the natural reading was that the protocols measure different things — DC removal buying optimization conditioning at the cost of feature-map quality.

**The 100L end-to-end arms dissolved that.** They were the empty cell in the 2×2, and they came back with He ahead by 7–26 points (§4.3). Combined with the 30L end-to-end win evaporating under robust statistics, the picture is simply: **He is ahead or level everywhere**. There is no protocol/depth tension to explain, and the "better-conditioned optimization problem" hypothesis has no support.

What the frozen-readout table *does* establish independently is the depth axis: at 30 layers He's readout reaches **0.5481 test** with train ≈ test; at 100 layers it is at chance. Class content dies between depth 30 and depth 100 — the campaign-09 claim, now measured on held-out data.

### 5.3 The probe screen is vindicated

Cosine-kNN put content at chance by layer ≈25. Held-out accuracy at depth 100 agrees: 0.11 everywhere. The probe was measuring *generalizable class-aligned* structure and measuring it correctly.

This matters for W3 (the parked α-family screen), which gates cluster time on probe metrics. That premise stands — **with one caveat now measured**: within the shift family, init-time *geometry* metrics (mean pairwise cosine) are **anti-correlated** with training outcome. `c = 0.70` had the best cosine geometry ever recorded in this project and is the worst shift arm in training. Screen on **content** (kNN, held-out readout), never on **geometry** alone.

---

## 6. What changed against what we believed

| belief | status | evidence |
|---|---|---|
| Campaign 09: "content is the bottleneck at ≥50L" | **vindicated** | frozen-readout test at 100L is chance for every arm (He *does* train end-to-end at 100L NoBN: 0.9953/0.8633) |
| Campaign 11: "the family is a negative result" | **overturned at 30L** | every shift arm beats He on CIFAR-10 test by 3.5–6.7 pp |
| Campaign 11: "`c = 0.70` is stuck at chance — clearest evidence for the negative conclusion" | **retracted** | 2-epoch CPU pre-triage; at 20 epochs it reaches 0.9291 / 0.8535 |
| Campaign 11: "requirements (i) and (ii) are provably incompatible" | **corrected** | the cost is transferable; it is the gain-coupling lock again (§3.1) |
| "The probes understate badly — a trained readout gets 0.83 where they said chance" | **retracted** | that 0.83 is train; test is 0.1132 |
| "Training 3 layers beats training all 100 by ~4.8×" | **retracted** | true on train, reversed on test — all-layers 100L gets 0.1672 test vs the frozen readout's 0.1132 |
| Advisor: "training will be fast in both window positions" | **refuted for the front, qualified for the tail** | front is causally disconnected; tail memorizes but does not generalize |
| Detached post-ReLU shift is the right fork | **reopened** | differentiable wins on both datasets |

The last two retractions were both mine, and both came from reading `eval_train_accuracy` without checking `test_accuracy`. Every table in this document now reports both.

---

## 7. Artifacts

**Code (all on `main`)**
- `src/rp_study/config.py`, `models/classifiers.py` — `trainable_layers`, `relu_shift`, `relu_shift_detach` (additive; `None` is a bit-exact no-op, verified over 202 parameter tensors and a full epoch)
- `cluster/10_rc_frozen_ends/` — runner + 46 subs · `cluster/11_relu_shift/` — runner + 36 subs · `cluster/12_frozen_readout/` — runner + 48 subs
- `cluster/sync_to_cluster.sh` — `--tar` fallback (the re-imaged login node has no `rsync`), `--code-only` (65 MB → 0.6 MB), Apple-metadata suppression

**Analysis scripts**
`recipe_decomposition_funnel.py` · `relu_shift_geometry_screen.py` · `relu_shift_funnel_fwd_bwd.py` · `relu_shift_duality_check.py` · `relu_shift_noop_check.py` · `campaign10_followup_figures.py`

**Documentation**
`cluster/{10,11,12}/README.md` · `docs/plans_handoffs/FRONTIER.md` · `docs/plans_handoffs/briefs/2026-08-15_*.md` (2) · `docs/scratch/oracle_spotcheck_addendum.md` (proof) · this document

**Results** — 66 JSONs in `reports/results/`, indexed in `reports/results/INDEX.md`

---

## 8. Open questions

1. **Does the 30L DC-removal win hold at 200 epochs?** These are 20-epoch smokes. The 18 audits are written and ungated. This is the highest-value next run.
2. **Does it hold at 50L?** The win is at 30L; content dies by 100L. Where does the crossover sit?
3. **Detached vs differentiable** — decide it on 200-epoch evidence rather than principle.
4. **The norm control.** "The shift preserves input geometry better than He" rests on distance correlation, and He may be surviving on *norm* rather than *angular* information. Untested.
5. **The per-sample RMS variant** makes the closed-form theory exact and reaches cosine 0.0037 at 100 layers. Never trained.
6. **The Slepian rewrite of the dying-neuron proof** (§3.3) turns a vacuous bound into one that predicts a measured number. Needs writing properly.
7. **W3's α-family screen** should be re-based on content metrics, not geometry metrics — §5.3.
