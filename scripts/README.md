# Analysis & Figure Scripts

Standalone helpers that generate figures into `reports/figures/` or produce derived JSONs in `reports/results/`. Run from the repo root (`python scripts/<name>.py`). Grouped by topic:

## η sweep (feeds `cluster/07_v2_eta_nobn/`)

| Script | Output |
|---|---|
| `eta_sweep_research.py` | Sweeps V2's η per architecture, measuring per-layer gradient ratios → `reports/results/eta_sweep_research.json` |
| `eta_sweep_pick.py` | Selects the gradient-ratio-minimizing η\* per architecture → `reports/results/eta_star_recommended.json`, `reports/figures/eta_sweep/eta_pick_minima.png` |

## Gain funnels (forward/backward signal propagation at initialization)

| Script | Shows |
|---|---|
| `he_funnel_fwd_bwd.py` / `he_funnel_fwd_bwd_30L.py` | He: activation and error-signal RMS per layer at 100L / 30L |
| `he_funnel_gradient.py` | He: per-layer gradient norms |
| `rc_funnel100_fwd_bwd.py` / `rc_funnel100_fwd_bwd_30L.py` | Row-centered: the 0.826^L forward decay and backward blowup |
| `rcfwd_funnel100_fwd_bwd.py` | Forward-balanced row-centering: flat forward, 1.21^L backward blowup |
| `rcfwd_gradrescale_funnel.py` | rcfwd with `grad_rescale=r`: both directions flattened — the initialization-time validation of campaign 09 |

Figures land in `reports/figures/gain_funnels/` and `reports/figures/rcfwd_rescale/`.

## Learning-speed analysis

| Script | Shows |
|---|---|
| `depth_learning_speed.py` | Early learning speed vs depth (acc @ ep20, epochs-to-50%) for He / V2 / rcfwd from existing result JSONs → `reports/figures/rcfwd_rescale/learning_speed_vs_depth.png` |
| `rcfwd_campaign_summary.py` | The campaign-09 verdict in one 4-panel figure (audit curves, LR ladder, rcfwd-vs-tuned-He at 30L) → `rcfwd_campaign_summary.png` (copy embedded from `docs/figures/`) |
| `content_probe_linear.py` | Scale-invariant content probes (cosine k-NN + linear probe) at the trained depths for He vs rcfwd init → `content_probe_linear.json` |
| `content_profile_per_layer.py` | Linear-probe accuracy vs layer index at init, width 500 — where content dies per init family → `content_profile_per_layer.{json,png}` |

## Recipe decomposition & initialization screens

| Script | Shows |
|---|---|
| `recipe_decomposition_funnel.py` | The rcfwd recipe split into its two interventions (init change vs `_GradRescale`): per-layer activation RMS and parameter-gradient norm for all four corners of the 2×2 → `recipe_decomposition_funnel.json`, `reports/figures/rc_frozen_ends/` |
| `relu_shift_geometry_screen.py` | Init-time screen of the post-ReLU DC-removal family (`a = relu(Wx) − c·rms(a)`) on all three requirements — mean pairwise cosine, cosine k-NN content, dataset-dead fraction, activation RMS, distance correlation vs the input. Also emits the closed-form predictions (`analytic_gain_G`, `analytic_cosine_fixed_point`) next to the measurement, plus the two diagnostics that explain where they diverge (`norm_heterogeneity_kappa`, `shared_dc_energy_fraction`) and the depth-resolved `implied_forward_gain_at_depth`. `--shift-scope per_sample` is the mechanism control; `--depth/--dataset/--tag` sweep → `relu_shift_geometry_screen[_<tag>].json`, `reports/figures/relu_shift/` |
| `relu_shift_noop_check.py` | **The campaign-11 gate.** Proves `ClassifierConfig.relu_shift=None` is a bit-exact no-op: builds the model from a git reference revision and from the working tree and requires `torch.equal` on output, loss, all gradients and all parameters after an SGD step (Part A), then replays a full campaign-10 epoch under both revisions and requires the history record to match bitwise (Part B). A `relu_shift=0.25` control must change the output, so "identical" cannot be vacuous → `relu_shift_noop_verification.json` |
| `relu_shift_funnel_fwd_bwd.py` | Forward + backward + geometry funnel for the DC-removal family through the real `DeepFCClassifier`, and the **detached-vs-differentiable fork** measured directly as `‖g_diff − g_detach‖/‖g_detach‖` per layer → `relu_shift_funnel_fwd_bwd.json`, `reports/figures/relu_shift/relu_shift_funnel_<dataset>.png` |
| `relu_shift_duality_check.py` | Numerically verifies the row-centering ↔ post-ReLU-shift duality: on `row_centered_he` (`W𝟙 = 0`) the shift is an exact **forward** no-op (relative loss diff 0.0), on `he` it is O(1), and weight gradients differ under both — the identity is forward-only → `relu_shift_duality_check.json` |
| `relu_shift_local_pretriage.py` | GO/NO-GO before the cluster grid: the exact campaign-11 recipe run for a couple of real epochs per arm on CPU, to catch dead or diverging arms before 18 SLURM jobs queue behind them → `relushift_local_pretriage_<depth>L_<dataset>.json` |

## Dying-neurons proof

| Script | Shows |
|---|---|
| `dying_neurons_depth_figure.py` | Measured dataset-dead fraction vs depth (He, from the depth-tagged `relu_shift_geometry_screen_*L_*.json` screens) against the theorem's universal Slepian lower bound (arc-cosine map iterated from ρ=0 for L−1 steps, dataset-free), the per-dataset mean-anchored heuristic curves (dashed), and the 1/2 asymptote → `reports/figures/dying_neurons/dead_fraction_vs_depth.png` |

## Depth / gradient diagnostics

| Script | Shows |
|---|---|
| `plot_100L_gradient.py` / `plot_50L_gradient.py` | Per-layer gradient norms from run diagnostics at depth 100 / 50 |
| `plot_bulge_by_depth.py` | V2 NoBN activation "bulge" (mid-network RMS peak) vs depth |
| `bulge_under_lr1e6.py` | The same bulge measured under LR=1e-6 (campaign 07's lr1e6 probes) |

Figures land in `reports/figures/v2_eta_nobn/`.

## Frozen-layer entry-point diagnostics (feeds `cluster/10_rc_frozen_ends/`)

| Script | Shows |
|---|---|
| `rc_frozen_ends_plots.py` | Campaign 10's two failure modes under the raw recipe, in one 4-panel figure: flat loss/accuracy in all 4 smoke cells; `last3` trainable-layer gradients underflowing to exact float32 zero vs. `first3` trainable-layer gradients staying healthy but causally inert → `reports/figures/rc_frozen_ends/rcfrozen_mechanisms.png` (copy embedded from `docs/figures/`) |
| `rc_frozen_ends_rcfwd_plots.py` | The corrected-recipe follow-up: mechanisms figure (`last3` now learning, `first3`'s loss worsening past ln(10)) + the direct raw-vs-rcfwd accuracy comparison that resolves the H1-vs-H2 question → `rcfrozen_rcfwd_mechanisms.png`, `rcfrozen_recipe_comparison.png` |
