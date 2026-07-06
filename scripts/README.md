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

## Depth / gradient diagnostics

| Script | Shows |
|---|---|
| `plot_100L_gradient.py` / `plot_50L_gradient.py` | Per-layer gradient norms from run diagnostics at depth 100 / 50 |
| `plot_bulge_by_depth.py` | V2 NoBN activation "bulge" (mid-network RMS peak) vs depth |
| `bulge_under_lr1e6.py` | The same bulge measured under LR=1e-6 (campaign 07's lr1e6 probes) |

Figures land in `reports/figures/v2_eta_nobn/`.
