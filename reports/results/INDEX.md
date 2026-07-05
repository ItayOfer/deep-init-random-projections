# Results Index

Every JSON here is the output of one cluster job (filename stem = SLURM job name = `--experiment` label). Files are kept **flat** because notebooks (`08_results_dashboard`, `13_final_results`) and `scripts/` helpers load them by these exact paths. This index maps each campaign to its files, producing runner, and headline outcome. Full narrative: `docs/RESEARCH_LOG.md`.

| Campaign | Files | Runner | Date | Headline outcome |
|---|---|---|---|---|
| Geometry / angle benchmarks | `geometry_smoke.json`, `geometry_benchmark_cifar10.json`, `angle_map.json`, `angle_smoke.json` | `cluster/01_geometry/` | Apr 3 | Arc-cosine-kernel geometry metrics: He collapses angles; row-centered preserves spread (later revised: spread ≠ class structure) |
| He tuning sweep | `fnn_he_bn_training.json`, `fnn_he_bn_evaltrain_training.json`, `fnn_he_targeted_best12.json`, `fnn_he_targeted_best12_configs.json`, `supervised_smoke.json` | `cluster/02_he_tuning/` | Apr 3 – May 3 | Original 12-architecture He sweep: **5/12 PASS** |
| Diagnostic phases | `diagnostic_phase{1,2,3,5}.json` (phase 4 produced no JSON) | `cluster/03_he_diagnostics/` | May 14–16 | Per-architecture recipe selection (LR schedules, BN momentum, plateau) |
| Final He audit | `final_audit.json`, `final_audit_continuation.json`, **`final_audit_merged.json`** (canonical) | `cluster/04_he_final_audit/` | May 16–18 | Unified 200-epoch audit: **8/12 PASS** |
| SGD recovery | `plain_sgd_100L_nobn_w512_{mnist,fashion_mnist,cifar10}.json`, `recovery_plain_sgd_*.json` (4), `recovery2_plain_sgd_*.json` (3), `recovery3_adam_*.json` (2) | `cluster/05_sgd_recovery/` | May 22–23 | Plain SGD rescues both 100L/NoBN → **He 10/12 PASS**; both 100L/BN remain open (recovery3 peaked 21–47% then drifted) |
| V2 row-centered audit | `row_centered_smoke_*.json` (10), `row_centered_audit_*.json` (10), `row_centered_smoke2_*.json` (4), `row_centered_smoke3_*.json` (2), `row_centered_smoke4_*.json` (6), `row_centered_audit4_*.json` (4) | `cluster/06_v2_row_centered/` | May 22–24 | **V2 5/12 PASS** (all four 30L + fmnist/50L/BN under plain SGD — the Adam→SGD switch took it from 33% to 99.95%); depth ceiling ≈ L=30 |
| V2 η\* NoBN | `v2_nobn_sgd_smoke_*.json` (6), `v2_nobn_sgd_lr1e6_smoke_*.json` (6) | `cluster/07_v2_eta_nobn/` | May 25 | Per-arch η\* does not lift the depth ceiling; L=100 still overflows/diverges. *(audit runs produced no JSONs — smoke did not justify them)* |
| η sweep (local) | `eta_sweep_research.json`, `eta_star_recommended.json` | `scripts/eta_sweep_research.py`, `scripts/eta_sweep_pick.py` | May 25 | Gradient-ratio-minimizing η per architecture (input to campaign 07) |
| He low-LR probe | `he_sgd_lowlr2_smoke_{cifar10,fmnist}_100L_bn.json` | `cluster/08_he_lowlr_probe/` | May 25 | He+SGD at ultra-low LR on 100L/BN: numerically stable but frozen at chance *(round-1 lowlr JSONs not retained)* |
| rcfwd grad-rescale | `rcfwd_rescale_smoke_*.json` (6) | `cluster/09_rcfwd_rescale/` | May 25 + Jul 4 (identical reruns) | **First stable 100L NoBN row-centered training** — all 6 smokes complete 20 ep, no NaN, grad ratios ≤18.5×; learning monotone but slow (fmnist/30L 0.78, rest 0.12–0.17); audits pending |

## Pass criterion

`eval_train_accuracy ≥ 0.995` **and** `eval_train_loss ≤ 0.10` (full train set, `model.eval()` mode). The 12-architecture matrix: {30, 50, 100}L × {CIFAR-10, Fashion-MNIST} × {BN, NoBN}.

## Known gaps

- `diagnostic_phase4.json` — phase 4 ran but its JSON was not retained.
- `v2_nobn_sgd_audit_*.json` — audit subs exist but were not run (smoke results did not justify them).
- `he_sgd_lowlr_smoke_*` round 1 — superseded by `lowlr2` within the same day.
- `rcfwd_rescale_audit_*` — the 200-epoch audits have not run yet (smoke results above justify promoting all six).
