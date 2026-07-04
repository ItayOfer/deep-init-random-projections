# reports/

Generated experimental evidence. Two kinds:

## `results/` — committed

One JSON per cluster job (filename stem = SLURM job name = `--experiment` label), containing the full per-epoch history (`eval_train_accuracy`, `eval_train_loss`, test metrics, learning rate, per-layer gradient norms, BN stats) plus config snapshots and abort/status fields. **These files are the ground truth behind every number in the thesis** — each campaign README's findings trace to them, and `INDEX.md` maps campaign → files → runner → headline outcome.

Kept **flat** (not in subdirectories) because notebooks 08/13 and `scripts/` helpers load them by these exact paths.

## `figures/` — local-only (gitignored)

All generated plots, organized by campaign: `diagnostic_phases/`, `he_tuning/`, `final_audit/`, `sgd_recovery/`, `gain_funnels/`, `v2_eta_nobn/`, `eta_sweep/`, `he_lowlr_probe/`, `rcfwd_rescale/`. Regenerate via `scripts/` (see `scripts/README.md`) or the notebooks.

A curated ~15-figure subset **is** committed at `docs/figures/` so the campaign READMEs render on GitHub.

`checkpoints/` (model checkpoints) exist cluster-side only and are gitignored.
