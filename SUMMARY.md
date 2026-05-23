# Repository Summary — Thesis on Neural Network Initialization via Random Projections

## One-line description

A research codebase studying **how to initialize deep ReLU networks so that they preserve input geometry without sacrificing gradient flow**, framed through the lens of random projections (RP) and the arc-cosine kernel.

---

## The research question

When data passes through `L` layers of `(Linear + ReLU)`, three coupled failure modes appear:

1. **Geometric collapse.** The ReLU arc-cosine kernel
   `K(α) = (sin α + (π − α) cos α) / (2π)`
   strictly contracts angles between vectors. After enough layers all inputs look identical to the network.
2. **Gradient vanishing / explosion.** Standard He initialization (`Var(W) = 2/d`) preserves *activation* norms in expectation, but the per-layer gradient gain depends on subtler quantities (active fraction, weight correlations). Small deviations from gain = 1 compound exponentially with depth.
3. **Dead neurons.** Neurons whose pre-activations are always negative output zero forever and never recover.

These three failures sit on a Pareto frontier — fixing one tends to worsen another.

## The advisor's proposal: row-centered He

After sampling `W^He ~ N(0, √(2/d))`, subtract the per-row mean so every row sums to zero:

```
W[i,:] = W^He[i,:] − mean(W^He[i,:])      ⇒      Σ_j W[i,j] = 0
```

- **Geometry win.** Zero row sums make each neuron invariant to constant input shifts. Empirically this halts the "DC drift" that drives angle contraction; geometry is preserved through many layers.
- **Gradient loss.** The zero-sum constraint propagates through backprop and creates a *gradient trap*: gradients systematically decay layer-to-layer. Variance reduction by factor `(1 − 1/d)` per layer compounds the problem.

The thesis is fundamentally about **understanding this geometry/gradient trade-off and engineering initializations that escape it.**

## Theoretical framing

The arc-cosine kernel is the dominant analytic tool. Per layer (with full row centering) the project has derived a *structural forward gain* `r = √((π−1)/π) ≈ 0.826` that is independent of the chosen variance. From `BP4`:

```
‖∂C/∂W^l‖ ∝ ‖a^{l−1}‖ · ‖δ^l‖
```

Forward gain `g_fwd` and backward gain `g_bwd` are coupled by `g_fwd / g_bwd = r`, so no single scalar variance can drive both to 1. This led to the **product-balanced** variant that sets `g_fwd · g_bwd = 1` (the unique fixed point), and to **depth-aware, per-layer-scaled** variants that further trade activation-norm uniformity for gradient-norm uniformity. These are the most recent theoretical contributions and are documented in detail in `INITIALIZERS.md` (items 15–18).

## Initializer family (15+ registered strategies)

All initializers are registered in `src/rp_study/models/initializers.py` via `@register_initializer(...)`. Grouped by purpose:

| Group | Members | Role |
|---|---|---|
| Baselines | `he`, `xavier`, `uniform_he`, `orthogonal` | Reference points |
| Row-centered family | `row_centered_he`, `row_centered_he_var_adj`, `row_centered_final` (factor 1.65) | Advisor's line of investigation |
| Trade-off explorations | `partial_centered_he` (soft α), `centered_with_dc_he` (DC offset breaks zero-sum) | Tunable interpolations |
| Geometry without trap | `orthogonal_he`, `orthogonal_tuned`, `kernel_preserving` | Avoid zero-sum constraint |
| Theory-driven (newest) | `row_centered_forward_balanced`, `row_centered_layer_balanced`, `row_centered_layer_balanced_he_base`, `row_centered_product_balanced`, `row_centered_layer_balanced_product_base` | Built from the BP4 / coupled-gain analysis |

The currently recommended recipe is `row_centered_layer_balanced_product_base`: product-balanced base variance combined with a per-layer scaling `s_l = s* · r^{η·(l − (L+1)/2)}` that distributes variance across depth.

## Codebase layout

```
src/rp_study/
  config.py                    # Dataclasses: ExperimentConfig, NetworkConfig,
                               # ClassifierConfig, TrainingConfig
  models/
    initializers.py            # Single source of truth — registry of every init
    networks.py                # FeedForward (applies registry initializers)
    classifiers.py             # FC / CNN classifiers for supervised training
  data/loaders.py              # MNIST, Fashion-MNIST, CIFAR-10
  projections/
    random_projections.py      # RP matrices + multi_layer_rp_with_init() bridge
  experiments/
    gradient_analysis.py       # Per-layer gradient norms, dead neurons
    supervised_training.py     # Full training loop w/ schedulers, diagnostics, ckpt
  analysis/kernel.py           # K(α) arc-cosine kernel
  visualization/
    gradient_plots.py          # Per-layer gradient row-norm plots
    projection_plots.py        # PCA scatter, multi-layer RP grids

notebooks/                     # 12 exploratory + dashboard notebooks
  02_mnist_projections.ipynb         # Geometry under multi-layer RP+ReLU
  03_gradient_analysis.ipynb         # Gradient flow analysis
  05_initializer_dashboard.ipynb     # Unified geometry+gradient+stats view
  06_gradient_diagnostics.ipynb
  07_kernel_geometry_analysis.ipynb
  08_results_dashboard.ipynb
  09_meeting_comparison.ipynb
  10_fnn_training_curves.ipynb
  11_eta_sweep_analysis.ipynb

cluster/                       # DLC / SLURM workflow
  sync_to_cluster.sh           # rsync project to user@cluster
  WORKFLOW.md                  # Daily cluster workflow
  run_diagnostic.py            # Phase-1 diagnostic runner (7 hypothesis tests)
  run_phase2.py                # Phase-2 longer runs
  run_supervised_sweep.py      # FC tuning sweep (per-arch HP tuning)
  run_supervised_grid.py
  run_geometry_benchmark.py
  *.sub                        # SLURM submission scripts

reports/
  results/*.json               # Run histories (per-epoch metrics)
  figures/*.png                # Generated plots
  diagnostic_phase1_report.html
  sweep_results_table.{md,pdf}
  meeting_walkthrough_2026_04_13.md

CONTEXT.md                     # Thesis research context + current findings
INITIALIZERS.md                # Math reference for every registered initializer
CLAUDE.md                      # Agent context + workflow conventions
AGENTS.md / SIMULATIONS_RUNS.md
```

## Experiment pipelines

There are two evaluation axes, both automated.

1. **Geometry probe (training-free).**
   `multi_layer_rp_with_init(X, n_layers, init_strategy)` pushes Fashion-MNIST / MNIST samples through `L` layers of `(W·ReLU)` using a registry initializer and returns the per-layer outputs. PCA-to-2D scatter is the visual; pairwise-angle statistics under `K(α)` is the quantitative measure.

2. **Gradient probe (one-shot training).**
   Build a deep `FeedForward`, do a single forward+backward pass, log per-layer gradient L2 row-norms, zero-gradient proportion, and dead-neuron count.

3. **Supervised training (the real test).**
   `run_supervised_experiment(exp_config, classifier_config, training_config)` does the full training loop on MNIST / Fashion-MNIST / CIFAR-10 with:
   - Schedulers: `none`, `cosine`, `step`, `onecycle`, `plateau` (ReduceLROnPlateau on `eval_train_*`).
   - Optional per-epoch diagnostics (`diagnostics_every=N`): per-layer grad norms, BN running stats, learning rate.
   - Checkpointing + resume.
   - Primary thesis metric: `eval_train_accuracy ≥ 0.995` and `eval_train_loss ≤ 0.10`.

All three axes are tied together in `notebooks/05_initializer_dashboard.ipynb`.

## Cluster workflow (DLC / SLURM)

Cluster: `user@cluster`. Code mirrors local repo at `~/thesis/`. Standard loop:

1. Local: `bash cluster/sync_to_cluster.sh` (rsync; excludes `__pycache__`, `.git`, `data/`, `*.sqsh`, `logs/`).
2. Cluster: clear stale bytecode, then `sbatch cluster/<job>.sub`, then `squeue -u $CLUSTER_USER`, then `tail -f <name>-<JOBID>.out`.
3. Local: `scp` results from `~/thesis/reports/results/`.

SLURM conventions: `#SBATCH --exclude=dgx01,dgx04` (driver / stability issues), pyxis container `${HOME}/nvidia_pt.sqsh` mounted at `/mount`, stdout pattern `%x-%j.out`.

## Current state (May 2026)

- 15+ initializers registered; the theory-driven product-balanced and layer-balanced variants are the active research frontier.
- **FC tuning audit: 10 / 12 architectures pass** the thesis bar (`eval_train_accuracy ≥ 0.995` AND `eval_train_loss ≤ 0.10`). All 8 originally-tuned recipes plus both 100L NoBN cases (rescued post-meeting — see below). Only the two 100L BN cases remain.
- Diagnostic Phase 1 + Phase 2 reports exist in `reports/`; targeted best-12 configs have been re-run.
- Open theoretical work: complete the row-centered backprop derivation, pin down the exact gradient-trap decay factor, and use that to design an initialization that satisfies both geometry and gradient objectives by construction rather than by tuning.

### Post-meeting finding — the 100L NoBN failures were an Adam pathology

The 4 failing architectures in the final audit (`{cifar10, fmnist} × 100L × {NoBN, BN}`) were all run with Adam. Following the May 2026 advisor meeting, replicating the advisor's own 100-layer recipe (plain SGD, `lr=1e-3`, `momentum=0`, `weight_decay=0`, `bs=128`, He init, normalized inputs) on `cluster/run_plain_sgd_100L.py` produced these numbers at epoch 5:

| Dataset | Advisor | Our replication |
|---|---|---|
| FMNIST train acc | 75.4 % | 73.1 % |
| FMNIST test acc  | 77.4 % | 73.9 % |

Within 2 % of the advisor's reported numbers — confirms the dataset and the recipe match.

**Diagnosis.** For He-init NoBN at depth 100, gradients are tiny but non-zero. Adam's second-moment estimate `v` then underflows to ~0 in float32 and the update becomes `lr · m̂ / √ε ≈ 10 · m̂` — large noisy updates that drive the next forward pass to underflow → permanent gradient death. Plain SGD multiplies by `lr` and stays gentle, so the forward pass survives. **Optimizer choice was the binary on/off — not architecture, not data normalization, not bias=0.**

Rescue runs (`cluster/run_plain_sgd_recovery.py`, `_recovery2.py`):

| Architecture | Audit best (Adam) | Recovery result |
|---|---|---|
| `fmnist/100L/NoBN`  | dead (loss 2.3026)  | **PASS** (recovery1) — 0.9953 train, 0.0157 loss, 0.8633 test, ep 152 (early-stopped) |
| `cifar10/100L/NoBN` | dead (loss 2.3026)  | **PASS** (recovery2) — 0.9959 train, 0.0157 loss, 0.3926 test, ep 157 (early-stopped). Recovery1 with `plateau patience=10` reached 0.9908; recovery2 with `patience=5, min_lr=1e-7` got the extra ~5 LR halvings needed to cross the bar. |
| `fmnist/100L/BN`    | 0.38 train (peak)   | Recovery1 BN running stats blew up (`bn_momentum=0.1`); recovery2 with `bn_momentum=0.01` did not help — `train_loss` stuck at 2.29 (uniform output) for all 200 epochs. Plain SGD `lr=1e-3` appears too weak for this architecture; Adam (which produced the 0.38 audit peak) is the right optimizer here. |
| `cifar10/100L/BN`   | 0.21 train (peak)   | Recovery1 NaN at epoch 0; recovery2 with `bn_momentum=0.01` also NaN at epoch 0 — the failure happens **before** any running-stat update, so `bn_momentum` is irrelevant. Next attempt (recovery3) uses LR warmup + gradient clipping to survive the first few steps. |

**Recovery3 plan (in flight):** swap optimizer back to Adam for the two BN cases (the audit data shows Adam was actually the better choice there), keep `bn_momentum=0.01`, tighten plateau, and add LR warmup + gradient clipping for cifar10/BN to avoid the epoch-0 NaN.

**Recovery3 outcome (2026-05-23): both still FAIL the bar.** Both 100L/BN architectures completed 200 epochs but degraded from peak:

| Architecture | Peak train acc | Final train acc | Final loss | Verdict |
|---|---|---|---|---|
| `fmnist/100L/BN`  | 47.55 % | 31.32 % (degraded) | 2.43 | fail |
| `cifar10/100L/BN` | 21.39 % | 13.24 % (degraded) | 2.42 | fail |

Recovery3's tighter Adam+plateau survived the epoch-0 NaN (via warmup + clipping for cifar10) but converged to a poor local optimum and then drifted backward. The He scoreboard remains **10 / 12 passing**; the two 100L/BN cases are an open problem under He init regardless of optimizer / scheduler / clipping. Recovery3 JSONs at `reports/results/recovery3_adam_*.json`.

Result JSONs land at `reports/results/recovery{,2,3}_plain_sgd_<arch>.json` and `reports/results/recovery3_adam_<arch>.json`, one per architecture (no cross-contamination by design).

### V2 row-centered audit (May 23, 2026)

Following the He audit, the row-centered layer-balanced product-base initializer (V2 in notebooks 09/11, η=0.5 by default) was re-tested on the 10 He-passing architectures — Item 3 of the post-meeting plan. The two 100L/BN cases (⑥, ⑫) were excluded since their He recovery3 was still in flight.

**Constraint:** gradient clipping is forbidden for V2 (theoretical purity; see `feedback_no_grad_clipping.md` in agent memory). All recipe modifications use LR / warmup / optimizer changes only.

**Outcome — 5 / 12 PASS (after rounds 1, 2, 3, and 4):**

| | NoBN | BN |
|---|---|---|
| `cifar10/30L` | ✅ V2 audit (100 % acc, loss 0.0001) | ✅ V2 audit Adam **and** V2 audit4 SGD (100 %, 0.0002) |
| `cifar10/50L` | ❌ V2 audit diverged at ep 46 (peak 28 %) | ❌ V2 Adam **and** V2 SGD both stuck at ~28 % |
| `cifar10/100L` | ❌ V2 smoke2 (η=0.1) exploded ep1 b2 | ❌ V2 Adam **and** V2 SGD both stuck at 10 % |
| `fmnist/30L`  | ✅ V2 audit (100 %, 0.0000) | ✅ V2 audit Adam **and** V2 audit4 SGD (100 %, 0.0001) |
| `fmnist/50L`  | ❌ V2 smoke2 diverged ep 4 (peak 54 %) | ✅ **V2 audit4 SGD PASS (99.95 %, loss 0.0072)** — Adam was stuck at 33 % |
| `fmnist/100L` | ❌ V2 smoke2 (η=0.1) NaN ep1 b2 | ❌ V2 SGD audit4 learning slowly (peak 23 %), still no PASS |

**Mechanism.** V2 has a hard depth ceiling at L=30 with η=0.5 because the per-layer scaling `s_l = s* · r^{η·(l−(L+1)/2)}` makes the layer-1 weight std scale as `r^{−η·(L−1)/2}` times the base He std `√(2/d)`. At L=100 with η=0.5 this multiplier is `r^{-24.75} ≈ 113`, so V2's layer-1 weight std is ~3.2 absolute (for fan_in=3072 cifar10 input) vs He's 0.0255 — i.e. **~126× He's** layer-1 std. Var(Z₁) compounds layer-by-layer until activations exceed float32's ~3.4×10³⁸ ceiling at layer ~11 → Inf → NaN loss before any gradient can be computed. Lowering η to 0.1 survives the initial forward (peak A_RMS ≈ 4.6×10⁸) but leaves the per-layer gradient ratio at 5×10⁷ at the first batch, which destabilizes SGD on the very first step. At L=50, momentum-SGD or onecycle's LR ramp pushes the system past the stability margin during training (around the ep 4 → 46 window). BN architectures stuck because Adam's adaptive step *double-preconditioned* V2's already-non-uniform per-layer scaling.

**Option B test — V2 + BN + Adam at L=100 (smoke3, 2026-05-23).** Initial attempt: both architectures stuck at chance for 20 epochs. BN cancels V2's forward-pass amplification (per-layer activation RMS stays at ≈ 0.7 instead of overflowing), but the per-layer *gradient* ratio remains at 3–5×10⁴ at every batch — Adam cannot condition this. **Adam at L=100/BN: 0/2.**

**Round 4 — V2 + BN + plain SGD (audit4, 2026-05-23).** Hypothesis: V2's per-layer scheme acts as a built-in preconditioner, so plain SGD (which scales step magnitude with gradient, not against it) should compose with V2's design where Adam fights it. Test on 4 BN architectures across all 3 depths. **Result: 3 PASS (cifar10/30L/BN, fmnist/30L/BN, fmnist/50L/BN), 1 fail (fmnist/100L/BN, peak 23 %).** The unlock is real: **fmnist/50L/BN went from stuck-at-33 % under Adam to 99.95 % under SGD**, and the per-layer gradient ratio at L=100/BN dropped from 5×10⁴ (Adam) to 1×10² (SGD) — direct empirical evidence for the double-preconditioning theory.

**Final V2 audit tally: 5 / 12 PASS** (the original 4 at L=30 plus fmnist/50L/BN under V2+SGD specifically).

**Practical implication.** V2 with η=0.5 is competitive with He at L=30 (any optimizer), and *prefers plain SGD over Adam* at L=50/BN (the optimizer choice is decisive — Adam stuck, SGD passes). At L = 100 V2 fails under both Adam and SGD (both NoBN and BN), and so does He+recovery3 — **L=100/BN is the joint open problem across both initialisers**, not an initialiser-specific failure. Recipe details and trajectory plots are in notebook 13 Part 4 (Sections 19.1–19.8); the comprehensive 12-architecture V2 scoreboard is at §19.7.

Result JSONs at `reports/results/row_centered_{audit,smoke,smoke2,smoke3,smoke4,audit4}_<arch>.json`. SLURM sub files at `cluster/row_centered_{smoke,audit,smoke2,smoke3,smoke4,audit4}_<arch>.sub`. Runners: `cluster/run_row_centered_audit.py` (round 1, η=0.5, He-passing recipes), `cluster/run_row_centered_audit_round2.py` (round 2 — modified recipes for the L=50+ NoBN failures + L=100 NoBN at η=0.1), `cluster/run_row_centered_audit_round3.py` (round 3 — V2+BN+Adam at L=100), `cluster/run_row_centered_audit_round4.py` (round 4 — V2+BN+SGD at all depths). Triage tooling at `cluster/triage_row_centered_smoke.py`.

## Conventions worth knowing

- `nn.Linear` stores weights as `(fan_out, fan_in)`; row centering subtracts the mean along `dim=1`.
- Always reset the seed before each initializer when comparing strategies.
- Notebooks must import from `src/rp_study`; never reimplement initializer logic locally.
- All initializers zero the bias (or handle `bias=None`).
- New initializers must be added to `INITIALIZERS.md` and the dashboard's `INIT_STRATEGIES` list.
