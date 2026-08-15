# Brief — campaign 11: post-ReLU DC removal (2026-08-15)

**Onboarding chain (read in order before starting):** `README.md` → `docs/RESEARCH_LOG.md` → `docs/plans_handoffs/FRONTIER.md` → this brief → `CLAUDE.md` (conventions). Task-specific deep dives: `cluster/09_rcfwd_rescale/README.md` (the three-requirements frame and the conditioning-vs-content 2×2), `INITIALIZERS.md` (the row-centered family), `docs/reports/gradient_diagnostics_analysis.md` §4 (half-Gaussian moments and the centering ratio).

## Goal

Test a new route to the same target the row-centered family has been chasing: remove the ReLU's DC component **in activation space** (`a = relu(Wx) − c·rms(relu(Wx))`) on top of unconstrained He weights, and measure whether it satisfies the three requirements — (i) no geometric collapse, (ii) stable gradients, (iii) preserved class content — better than either He or `row_centered_he`.

## Context

**Why now.** Advisor meeting 2026-08-15. The user's original (pre-research) intuition was that geometric collapse happens because ReLU discards data, and that shifting the representation into the positive orthant would fix it. The advisor's version: take plain He and subtract a constant, then look at forward, backward, and geometry. The user confirmed the **scale-relative** form (`c · rms(a)`, swept over `c`, including the theoretically exact value) as the variant to run; the fixed-constant, batch-mean, and weight-space variants were considered and set aside.

**The mechanism, stated precisely.** ReLU output is non-negative, so `E[a] = rms(a)/√π ≈ 0.564·rms(a) > 0`. That positive DC component — shared by every sample — is the engine of the arc-cosine kernel's `ρ → 1` collapse. Subtracting it after the ReLU attacks the mechanism head-on. `c = 1/√π` removes exactly `E[a]`.

**This is the activation-space dual of row-centering, not a new family.** The next layer computes `W(a − c𝟙) = Wa − c(W𝟙)`. If `W` is row-centered then `W𝟙 = 0` and the subtraction is **exactly a no-op**. So row-centering and post-ReLU subtraction are the same idea applied at two different points: kill the DC on the way *in* (weights) versus on the way *out* (activations). The difference that makes it worth running: the shift leaves the weights at full He, so it does not pay row-centering's `(1−1/d)` variance penalty or its hard rank-one row constraint.

**Prior art — already measured, do not re-derive.** `scripts/relu_shift_geometry_screen.py` → `reports/results/relu_shift_geometry_screen.json` screens the family at initialization (60L, width 500, Fashion-MNIST, 512 samples, seed 42). Input baseline: mean pairwise cosine 0.295, cosine-10NN 0.746.

| candidate | mean cos @L60 | 10NN @L60 | dataset-dead @L60 | act RMS @L60 | implied fwd gain | dist-corr @L60 |
|---|---|---|---|---|---|---|
| `he` | 0.9929 | 0.268 | **40.0%** | 8.4e-1 | 0.997 | 0.307 |
| `row_centered_he` | 0.3220 | 0.082 | 0.0% | 1.2e-5 | 0.828 | 0.194 |
| shift `c=0.25` | 0.7689 | 0.129 | 0.2% | 2.0e-2 | **0.937** | **0.278** |
| shift `c=0.5642` (=1/√π) | 0.9025 | 0.143 | 1.4% | 3.1e-3 | 0.908 | 0.083 |
| shift `c=0.75` | **0.2089** | 0.084 | 0.0% | 4.5e-5 | 0.846 | −0.004 |
| shift `c=1.0` | 0.8438 | 0.096 | 9.4% | 1.3e-2 | 0.930 | 0.015 |

Five things this table already establishes — build on them, do not repeat them:

1. **He really does kill ~40% of its neurons by layer 60** (dataset-dead = pre-activation ≤ 0 for every sample in the probe set), heading toward the ½ that the dying-neurons proof (W2) predicts. Note the dead fraction is monotone *decreasing* in probe-set size (41.2% at 256 samples, 40.0% at 512) — always report N alongside it.
2. **DC removal does eliminate dead neurons**, vindicating the mechanism behind the user's original intuition: every `c ≤ 0.75` variant sits at 0–1.4% dead versus He's 40%.
3. **The effect is strongly non-monotone in `c`, and `c = 1/√π` is not the best point.** The theoretically exact constant collapses to 0.90 by L60 while `c = 0.75` stays at 0.21. Whatever the right theory is, it is not "subtract exactly `E[a]`" — that is the campaign's most interesting open question.
4. **Requirements (i) and (iii) are in tension, again.** `c=0.75` achieves the best geometry of any candidate ever measured in this project (0.209, *below* the input's 0.295) but its content is at chance (10NN 0.084) and its distance correlation to the input is **−0.004** — it avoids collapse by destroying the signal. `c=0.25` is the only candidate that improves on He's dead-neuron count while nearly matching He's distance correlation (0.278 vs 0.307).
5. **The shift decays the forward pass far less than row-centering does** for comparable DC removal: implied per-layer forward gain 0.937 (`c=0.25`) / 0.908 (`c=1/√π`) versus row-centering's 0.828. The `c=1/√π` value 0.908 sits within 4e-4 of `√r = √0.8256 = 0.9086`; if that is exact rather than coincidence it is a derivable "the shift removes half the gain-coupling exponent" result and belongs in the thesis, not just the campaign README.

**The honest prior.** Every DC-removal variant screened so far loses class content faster than plain He does (10NN at L60: He 0.268 vs 0.082–0.143 for the whole family). That is campaign 09's conditioning-versus-content trade-off showing up on a third axis. Expect it; the campaign's job is to find out whether some `c` escapes it, and to document the trade-off curve if none does. **A well-documented negative result here is a successful campaign** — it would be the sharpest statement yet that the three requirements may not be jointly satisfiable by DC removal at all, which is a thesis-chapter-level claim.

**A caveat on the metrics.** He's higher distance correlation and k-NN at depth may be surviving on *norm* information rather than *angular* information, precisely because of the DC it retains (a large shared component makes pairwise distances track `‖a_i‖`, which tracks `‖x_i‖`). Any claim of the form "He preserves geometry best" must be checked against a norm-controlled variant before it is written down.

## Deliverables

1. **Registry + forward-pass support** (this is the only `src/` change; keep it minimal and additive):
   - `ClassifierConfig.relu_shift: Optional[float] = None` — the coefficient `c`. `None` = no shift, so no existing experiment changes behavior.
   - In `DeepFCClassifier.forward`, after `torch.relu(x)` and **before** the existing `_GradRescale` hook: if `relu_shift is not None`, `x = x - relu_shift * x.pow(2).mean().sqrt()`. Document that this is a *differentiable* operation (the rms term carries gradient) and state whether you keep it differentiable or detach it — **run both**; it is a real fork and the answer is not obvious. Detached = a pure per-layer bias; differentiable = an extra global coupling across the batch.
   - Extend `notebooks/05_initializer_dashboard.ipynb`'s `INIT_STRATEGIES` per `CLAUDE.md`'s "when adding a new initializer" rule if you end up registering this as a named initializer rather than a forward-pass flag; if it stays a forward flag, say so in `INITIALIZERS.md` instead so the reference doc stays complete.
2. **Extended screen** — rerun `scripts/relu_shift_geometry_screen.py` over a finer `c` grid (at least `0.1 … 0.9` in steps of 0.1, plus `1/√π`), at depths {30, 60, 100} and on both datasets, to locate the operating point and check whether the best `c` is depth-dependent. Commit the JSONs. **Screen before training** — this is the W3 discipline and it is cheap.
3. **Backward/geometry diagnostics at init** for the two or three surviving candidates: a per-layer gradient funnel in the style of `scripts/he_funnel_fwd_bwd.py` / `scripts/rcfwd_gradrescale_funnel.py`. The advisor asked for forward, backward, **and** geometry on every simulation from now on; the screen covers forward + geometry, this covers backward.
4. **Training runs**: `cluster/11_relu_shift/` with runner + subs. Smoke (20 ep) on the surviving `c` values × {fmnist, cifar10} × {30L, 100L}, NoBN, width 500, plain SGD — matching campaign 10's minimal recipe so the comparison is like-for-like. Gate 200-epoch audits on smoke triage. Baselines `he` and `row_centered_he` at the same settings must be in the grid, not cited from older campaigns with different recipes.
5. **`cluster/11_relu_shift/README.md`** in the standard shape (Question / Builds on / What ran / Findings with numbers ← JSONs / Reproduce / Evidence & gaps), plus rows in `cluster/README.md`, `reports/results/INDEX.md`, `scripts/README.md`, and FRONTIER.

## Constraints

- **Branch: `work/relu-shift`** — this touches `src/rp_study/config.py` and `src/rp_study/models/classifiers.py`, which campaign 10 also touched. Rebase on `main` (currently at the campaign-10 merge) before starting.
- Pass criterion: `eval_train_accuracy ≥ 0.99`, loss condition dropped (see the campaign-10 brief for the scan of what this changes historically). Log both.
- Standing rules: seed 42, width 500, NoBN unless stated, plain SGD (momentum 0, wd 0, scheduler none), bs 256, `normalize_inputs=True`, **no gradient clipping**.
- Report `dataset_dead_fraction` **with the probe-set size**, always — it is not a scale-free quantity.
- Do **not** re-run the fixed-constant (`a − 1.0`), batch-mean, or weight-space (`W − c`) variants. They were screened and set aside with the user: subtracting a fixed 1.0 leaves 43% dead units versus He's 41% (it overshoots `E[a]≈0.56` and replaces a positive shared DC with a negative one, which drives cosine back toward 1 just as hard), and `W − c` adds a rank-one DC to the weights — the opposite of row-centering — which on non-negative inputs pushes every pre-activation *down* and should make dying neurons worse, not better. If a result makes one of these worth revisiting, raise it with the oracle rather than adding it silently.
- Budget: the screen is local and cheap (minutes on CPU) — exhaust it before requesting cluster time. Cluster grid should be smoke-first, and no larger than campaign 10's.

## Definition of done

- [ ] `relu_shift=None` verified to be a bit-exact no-op: one existing campaign-10 label reproduces its committed JSON's first-epoch metrics on the new code.
- [ ] Screen JSONs committed for the finer `c` grid at ≥2 depths and both datasets; the chosen operating point(s) justified from those numbers.
- [ ] Both the detached and differentiable shift variants measured, with a stated recommendation.
- [ ] Forward, backward, and geometry diagnostics exist for every candidate carried into training.
- [ ] Training JSONs pulled + committed; README written with every number traced to a file; `cluster/README.md`, `INDEX.md`, `scripts/README.md`, FRONTIER and this brief's Outcome updated.
- [ ] The `√r` observation (item 5 above) either derived analytically or explicitly recorded as an open numerical coincidence.
- [ ] Verification note: which numbers were re-read from which JSON, by field name.

## Outcome  *(filled by the worker at the end)*

