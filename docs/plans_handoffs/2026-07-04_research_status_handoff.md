# Handoff — Simulation Program Summary & Status of the Last Training

**Written:** 2026-07-04. **Purpose:** hand this project to a fresh Claude Code session that will
**organize the findings**. This document (a) inventories the whole simulation program and its
artifacts, and (b) states precisely where the *last training* left off so the next session can
resume without re-deriving context.

**Grounding note:** Sections marked ✅*verified* were checked directly against files on disk
(JSON contents, `git status`, `ls -lat`) while writing this. Sections marked 📓*from project
memory/docs* are drawn from `SUMMARY.md`, `post_meeting_followup_plan.md`, and the dated memory
files; they were point-in-time observations and should be spot-checked before being asserted as
final thesis claims.

---

## PART B (read first) — Status of the LAST training we were trying to do

### TL;DR
The last active work (**2026-05-25**) was **not** a full training campaign — it was a
**mechanism-hunting / diagnostic day** aimed at the two remaining open problems (see Part A §5).
It ended by formulating a **new idea (`rcfwd` gradient-rescale)**, validating it *at
initialization* (figures show it works), writing the runner + SLURM scripts — but **the actual
training runs for that idea were never launched.** That is the cliff-edge. ✅*verified*

### What May 25 actually did (four threads, all diagnostic)

1. **η-sweep for the V2 initializer** (training-free). Swept η ∈ [0,1] (36 values) × depth
   {30,50,100} × {cifar10, fmnist}, measuring gradient ratio, weight-std ratio, peak activation
   RMS, theoretical G/V. Output: `reports/results/eta_sweep_research.json` +
   `eta_star_recommended.json` + 5 `eta_*` figures.
   - **Recommended η\*:** fmnist {30L:0.6, 50L:0.8, 100L:0.36}, cifar10 {30L:0.85, 50L:0.9, 100L:0.36}.
   - **But the sweep confirms a wall:** at L=100 even the best η still gives `peak_act_rms = Infinity`
     and `grad_ratio ≈ 2×10⁵` — i.e. η-tuning alone cannot save V2 at depth 100. ✅*verified*

2. **V2 + NoBN + plain SGD at tiny LR** (`run_v2_nobn_sgd.py`). 20-epoch smoke runs, η=0.36 for 100L.
   - At **lr=1e-2**: overflows immediately — `history` length 0, `abort` (files are ~2 KB). ✅*verified*
   - At **lr=1e-6**: stays numerically finite for 20 epochs but **frozen at chance**
     (cifar10/100L: eval_train_acc 0.0987→0.0996, loss flat at 2.3026). Conclusion: low LR keeps
     V2-NoBN-100L *alive* but it does not learn. ✅*verified*
   - Figures: `v2_nobn_bulge_by_depth*.png`, `v2_nobn_smoke_curves_lr1e{2,6}.png`, `v2_nobn_*_init_profile.png`.

3. **He + 100L/BN + plain SGD at ultra-low LR** (`run_he_sgd_lowlr{,2}_smoke.py`). Attacking the
   L=100/BN wall with lr as low as **1e-9**, bn_momentum=0.01. Output:
   `he_sgd_lowlr2_smoke_{cifar10,fmnist}_100L_bn.json` + `he_lowlr2_100L_grad_heatmap.png`.
   (Same story: survives numerically, does not cross the bar.) ✅*verified files exist*

4. **The new idea → `rcfwd` gradient-rescale** (the latest artifact, `run_rcfwd_gradrescale.py`,
   timestamped 16:28 — last thing touched). ✅*verified*
   - **Mechanism (custom backprop — this is the "edited back-propagation"):** a bespoke autograd op
     **`_GradRescale(torch.autograd.Function)`** at `src/rp_study/models/classifiers.py:15`. Its
     `forward` is the **identity** (`return x`); its `backward` returns **`grad_out * r`** — i.e. it
     touches ONLY the backward pass. It is spliced in **after every hidden ReLU** inside
     `DeepFCClassifier.forward` (`classifiers.py:102-103`, `x = _GradRescale.apply(x, grad_rescale)`),
     gated by the new `ClassifierConfig.grad_rescale` field. ⚠️ Note: it is a `torch.autograd.Function`
     (a custom differentiable op), **not** an `nn.Module` subclass — the network `DeepFCClassifier`
     stays standard; only the inter-layer gradient op is custom.
   - **Why:** use `row_centered_forward_balanced` init (keeps forward flat, g_fwd=1) but its backward
     gain g_bwd = 1/r ≈ 1.21 compounds to ~1e8 over 100 layers. The per-layer factor r ≈ 0.826 in
     backward cancels this exactly, so the cumulative `(1/r)^depth` → ~1. Mathematically equivalent
     to a **closed-form, non-adaptive geometric per-layer learning rate `r^(L−l)`**. NoBN, width 500,
     plain SGD (mom=0, wd=0), fixed LR, **no gradient clipping** (assert-enforced).
   - **Validated at init (figures, 15:0x–15:58):** `rcfwd_gradrescale_funnel100_100L.png`,
     `rcfwd_funnel100_fwd_bwd_100L_seed123.png`, `rcfwd_fwd_bwd_gain_vs_magnitude.png` — the docstring
     reports it flattens rms(δ) from 1e8 → ~1.2× and the gradient ratio from 5e7 → ~6× (He-like).
   - **Open question the runner poses verbatim:** *"does it TRAIN?"* — untested.

### Exactly where it stopped / next action
- **`reports/results/rcfwd_*` — DOES NOT EXIST.** No rcfwd training has run. ✅*verified*
- Prepared and ready to launch: `cluster/run_rcfwd_gradrescale.py` + 12 sub files
  (`cluster/rcfwd_rescale_{smoke,audit}_{cifar10,fmnist}_{30,50,100}L.sub`), 6 architectures ×
  {smoke 20ep, audit 200ep}. ✅*verified*
- **The single next action** is: sync to cluster, run the 6 `rcfwd_rescale_smoke_*` jobs, and
  answer "does it train?" — then triage to 200-ep audits.
- **Cluster state is unknown** — no local `.out` logs newer than May 23; check `squeue -u $CLUSTER_USER`
  before assuming nothing is running. ✅*verified (newest local .out is May 23)*

### Uncommitted code that supports the last work (⚠️ review before organizing)
`git status` shows **modified, uncommitted**: `src/rp_study/config.py`,
`src/rp_study/experiments/supervised_training.py`, `src/rp_study/models/classifiers.py`, plus
`post_meeting_followup_plan.md`. All the May 25 `cluster/*.py`, `cluster/*.sub`, results JSONs and
figures are **untracked** (`??`). The GradRescale mechanism for `rcfwd` almost certainly lives in
these modified `src/` files — **do not lose them.** (`reports/` is intentionally kept out of git
per user preference.) ✅*verified*

---

## PART A — The full simulation program

### 1. Research question 📓
How to initialize deep ReLU networks so they **preserve input geometry without killing gradient
flow**. Framed via random projections and the ReLU **arc-cosine kernel**
`K(α) = (sin α + (π−α)cos α)/(2π)`. Three coupled, Pareto-tensioned failure modes at depth:
geometric (angular) collapse, gradient vanishing/explosion, dead neurons.

The advisor's line of inquiry is **row-centered He** (subtract each row's mean so rows sum to
zero): it fixes geometry (kills DC drift) but creates a **gradient trap** — a structural forward
gain `r = √((π−1)/π) ≈ 0.826/layer` and a locked ratio `g_fwd/g_bwd = r` that no single scalar
variance can balance. The thesis engineers initializers that try to escape this trade-off.

### 2. Initializer family (registry) ✅*verified — 19 registered*
Single source of truth: `src/rp_study/models/initializers.py`. Registered names include:
`he`, `xavier`, `uniform_he`, `orthogonal`, `orthogonal_he`, `orthogonal_tuned`,
`row_centered_he`, `row_centered_he_var_adj`, `partial_centered_he`, `centered_with_dc_he`,
`custom_variance`, `kernel_preserving`, `angle_preserving`, `row_centered_final`,
`row_centered_layer_balanced`, `row_centered_layer_balanced_he_base`,
`row_centered_forward_balanced`, `row_centered_product_balanced`,
`row_centered_layer_balanced_product_base`.

- **"V1"** = `row_centered_product_balanced` (base variance 2.422/d so g_fwd·g_bwd=1).
- **"V2"** = `row_centered_layer_balanced_product_base` — V1 base **plus** per-layer scaling
  `s_l = s* · r^{η·(l−(L+1)/2)}`, η default 0.5. This is the thesis's "recommended recipe."

### 3. Three evaluation axes 📓
1. **Geometry probe** (training-free): `multi_layer_rp_with_init(X, L, init)` pushes samples
   through L×(Linear+ReLU); measure k-NN accuracy / pairwise angles / PCA. Now supports a
   `use_batch_norm` flag (added for the BN-geometry experiment).
2. **Gradient probe** (one forward+backward): per-layer grad L2 norms, dead-neuron count, ratios.
3. **Supervised training** (the real test): `run_supervised_experiment(...)` full loop with
   schedulers {none, cosine, step, onecycle, plateau}, per-epoch diagnostics, checkpoint/resume.

### 4. The 12-architecture FC audit (the core experiment) 📓
Plain fully-connected (MLP) nets, **He init, width 500, ReLU**, grid = **{BN, NoBN} × {30, 50,
100 L} × {CIFAR-10, Fashion-MNIST}**. BN vs NoBN is the **comparison axis** (not a tuning knob) —
the point is to show BN's behavior in deep FNNs. **Pass criterion:** `eval_train_accuracy ≥ 0.995`
AND `eval_train_loss ≤ 0.10` (eval-mode, full train set — chosen because train-mode metrics mask
BN running-stat failures).

### 5. Campaign chronology & headline results 📓 (spot-check against JSONs)

| Campaign | Date | Result |
|---|---|---|
| Original per-architecture HP sweep (He) | ~May 2 | 8/12 pass |
| Post-meeting recoveries (100L/NoBN) | May 22 | → **10/12** (both 100L/NoBN rescued by *plain SGD* + plateau) |
| Recovery3 for 100L/BN pair (Adam+warmup+clip) | May 23 | **Both FAIL** — peak 21–47% then drift back to 13–31% |
| **V2 audit** — the *balanced row-centered* init (`row_centered_layer_balanced_product_base`, η=0.5) | May 23 | **5/12 pass** (see table below) |
| May 25 mechanism-hunting (η-sweep, low-LR, rcfwd) | May 25 | diagnostic only; rcfwd idea prepared, **not run** |

**V2 (balanced row-centered) per-architecture verdict** ✅*verified against the `row_centered_*` JSONs on disk* —
we ran this init extensively and **most architectures could not be trained to the bar**:

| Depth | cifar10 / NoBN | cifar10 / BN | fmnist / NoBN | fmnist / BN |
|---|---|---|---|---|
| **30L** | ✅ PASS (SGD) | ✅ PASS (Adam & SGD) | ✅ PASS (SGD) | ✅ PASS (Adam & SGD) |
| **50L** | ❌ diverged ep45, peak 0.28 | ❌ stuck 0.28 (Adam) | ❌ diverged, peak 0.54 | ✅ **PASS 0.999 (SGD only)** |
| **100L** | ❌ overflow at init (ep0) | ❌ stuck 0.10 | ❌ overflow at init (ep0) | ❌ peak 0.23, no cross |

**5/12 PASS — all four 30L, plus the single deep unlock `fmnist/50L/BN` under plain SGD.** Every 50L-NoBN
case diverged, and every 100L case failed (NoBN overflows in float32 *before the first gradient step*;
BN survives numerically but sits at/near chance). This is the empirical basis for "V2 has a depth ceiling
≈ L=30" and motivated the May 25 rescue attempts (η-sweep, low-LR) and finally the `rcfwd` custom-backprop idea.

**Key mechanism findings (the actual thesis content):**
- **Adam pathology at 100L/NoBN:** tiny-but-nonzero gradients make Adam's `v̂` underflow in float32,
  so the update becomes ≈ `lr·m̂/√ε` (huge, noisy) → forward-pass death. Plain SGD (step ∝ g)
  survives. **Optimizer choice was the binary on/off — not the architecture.**
- **V2 "double-preconditioning":** V2's per-layer weight schedule *is* a preconditioner, so it
  composes with plain SGD's uniform step but **fights Adam's** adaptive step. Direct evidence:
  `fmnist/50L/BN` went from stuck-at-33% (Adam) to **99.95% (SGD)** — the 5th V2 pass.
- **V2 depth ceiling ≈ L=30** at η=0.5: layer-1 std blows up by r^{−η(L−1)/2} (~126× He's at L=100)
  → float32 overflow around layer ~11 before any gradient. η-sweep (May 25) confirms no η rescues L=100.
- **Geometry surprise (k-NN):** He+NoBN preserves k-NN best across depth (0.639 @20L); **adding BN
  crashes k-NN to chance (0.123).** RC family at chance under this Euclidean-k-NN protocol
  (they differ only by a global scalar — documented caveat).

**Open problems (both unsolved):**
1. **V2 depth ceiling** at L≥50 (NoBN overflow / divergence; BN stuck under Adam).
2. **L=100/BN joint wall** — fails under *both* He (recovery3) and V2 (Adam stuck, SGD reached only
   ~23% on fmnist). A depth+BN+optimizer wall, not initializer-specific. Candidate attacks logged:
   LARS/per-layer-LR, depth-dependent η ("V2-capped"), or the new **rcfwd gradient-rescale** (Part B).

### 6. Artifact inventory ✅*verified on disk*

**Results JSONs** (`reports/results/`) — one file per architecture per campaign, e.g.
`fnn_he_targeted_best12*.json` (He audit), `recovery{,2,3}_*`, `row_centered_{smoke,audit,smoke2,smoke3,smoke4,audit4}_*`,
`eta_sweep_research.json` + `eta_star_recommended.json`, `v2_nobn_sgd{,_lr1e6}_smoke_*`,
`he_sgd_lowlr2_smoke_*_100L_bn`. (No `rcfwd_*` — see Part B.)

**Figures** (`reports/figures/`, ~40 PNGs, NOT in git): `final_*` (audit story), `final_v2_*` (V2),
`eta_*` (η-sweep), `v2_nobn_*` + `he_funnel*` + `rc_funnel*` + `rcfwd_*` (May 25 gradient-flow diagnostics).

**Runners** (`cluster/run_*.py`): `run_supervised_sweep.py`, `run_final.py`,
`run_plain_sgd_recovery{,2}.py`, `run_adam_recovery3.py`, `run_row_centered_audit{,_round2,3,4}.py`,
`run_v2_nobn_sgd.py`, `run_he_sgd_lowlr{,2}_smoke.py`, **`run_rcfwd_gradrescale.py`** (newest).

**Notebooks:** `09_meeting_comparison` (V1-vs-V2 + geometry k-NN), `11_eta_sweep_analysis`,
`13_final_results` (meeting deliverable, Part 4 = V2 story §19.1–19.8).

**Key docs at root:** `SUMMARY.md`, `post_meeting_followup_plan.md` (the definitive plan + §8 open
follow-ups + §7 definition-of-done), `INITIALIZERS.md` (math for all inits), `CONTEXT.md`, `CLAUDE.md`.

**Memory files:** `project_state_may{22,23}_2026.md`, `feedback_no_grad_clipping.md`
(⚠️ **gradient clipping is forbidden for V2/rcfwd** — theoretical purity; enforced by assert in the runner).

### 7. Conventions the next session must respect 📓
- `nn.Linear` weight is `(fan_out, fan_in)`; row-centering subtracts mean along `dim=1`.
- Reset the seed before each initializer when comparing.
- One SLURM job per architecture, one JSON per job (label matches filename — no cross-contamination).
- `#SBATCH --exclude=dgx01,dgx04`; pyxis container `${HOME}/nvidia_pt.sqsh`; stdout `%x-%j.out`.
- Use `bash cluster/sync_to_cluster.sh` for uploads; clear `__pycache__` on the cluster after
  `config.py`/`experiments/*.py` edits.
- **No gradient clipping** for V2/rcfwd work.

---

## Suggested first moves for the organizing session
1. Read `post_meeting_followup_plan.md` §7–8 (definition-of-done + open follow-ups) and
   `SUMMARY.md` "Current state / V2 audit" — they are the authoritative narrative.
2. Decide the disposition of the **uncommitted `src/` changes** (the GradRescale hook) — review,
   then commit or stash deliberately.
3. Decide whether to **launch the prepared `rcfwd` smoke runs** (answers a live thesis question) or
   to freeze the experiment set and just organize existing findings.
4. Consolidate the scoreboard once: He = 10/12, V2 = 5/12, L=100/BN = joint open problem.
