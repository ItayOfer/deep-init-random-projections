# Cluster scripts (DLC / SLURM)

Daily workflow lives in `WORKFLOW.md`. Campaigns are organized in **chronologically numbered subdirectories** — each holds its Python runner(s) plus the SLURM `.sub` files that submit them. The numbering follows the research timeline (see `docs/RESEARCH_LOG.md` for the full narrative).

## Layout convention

- **Campaign dir** `cluster/<NN>_<campaign>/` — one research campaign, in chronological order.
- **Runner** `run_<campaign>.py` — Python entry point. Hardcodes the per-architecture recipes for that campaign. Accepts `--experiment <label>` to select one architecture.
- **Sub file** `<label>.sub` — SLURM submission script. One job runs one architecture. Paths inside are **relative to the repo root** (`python -u cluster/<NN>_<campaign>/run_X.py`), so submit from `~/thesis` on the cluster. Output JSON lands at `reports/results/<label>.json` (filename stem = SLURM job name, no cross-contamination).
- **Output log** `<label>-<JOBID>.out` — lands wherever you ran `sbatch` (repo root on the cluster); locally archived under `logs/slurm/<NN>_<campaign>/`.

## Campaigns

| # | Directory | Dates | Question | Outcome |
|---|---|---|---|---|
| 01 | `01_geometry/` | Apr 3 | Geometry (arc-cosine kernel) + angle-map benchmarks: He vs orthogonal vs row-centered variants | `geometry_*.json`, `angle_*.json` |
| 02 | `02_he_tuning/` | Apr 11 – May 3 | Initial 12-architecture He grid sweep with per-arch HP tuning | **5/12 PASS** |
| 03 | `03_he_diagnostics/` | May 14–16 | Hypothesis-driven diagnostic phases 1–5 (LR schedules, BN momentum, gradient death) | Recipes for the audit (note: phase 4 produced no JSON) |
| 04 | `04_he_final_audit/` | May 16–18 | The unified audit: all 12 architectures, best-known recipes, 200 epochs | **8/12 PASS** (`final_audit_merged.json`) |
| 05 | `05_sgd_recovery/` | May 22–23 | Plain-SGD replication + recovery rounds 1–3 for the four failing 100L architectures | **10/12 PASS**; both 100L/BN remain open |
| 06 | `06_v2_row_centered/` | May 22–24 | V2 (`row_centered_layer_balanced_product_base`, η=0.5) smoke + audit rounds 1–4 | **5/12 PASS** (all 30L + fmnist/50L/BN under SGD); depth ceiling ≈ L=30 |
| 07 | `07_v2_eta_nobn/` | May 25 | V2 NoBN with per-architecture η\* (gradient-ratio minimizing) + lr1e-6 probes | Confirmed V2 depth ceiling; no η rescues L=100 |
| 08 | `08_he_lowlr_probe/` | May 25 | He + plain SGD at ultra-low LR on 100L/BN (mechanism probe) | Survives numerically, frozen at chance |
| 09 | `09_rcfwd_rescale/` | May 25 | rcfwd: `row_centered_forward_balanced` init + per-layer backward gradient rescale (`grad_rescale=r`) | **PREPARED, NEVER LAUNCHED** — no results yet. This is the next campaign to run. |

### Campaign details

**01_geometry** — `run_geometry_benchmark.py`, `run_angle_map.py`; subs: `geometry_benchmark`, `geometry_product_balanced`, `angle_map`.

**02_he_tuning** — `run_supervised_sweep.py` (`fnn_he_tuning_*.sub`), `run_supervised_grid.py` (`fnn_he_fashion_cifar_*`, `supervised_comparison`, `supervised_product_balanced` — the early grid runner, also used for the V1 product-balanced supervised comparison), `run_supervised_from_configs.py` (`fnn_he_tuned_rerun`, `fnn_he_targeted_best12` — re-runs from saved best configs).

**03_he_diagnostics** — `run_diagnostic.py` (phase 1) + `run_phase{2..5}.py`, subs `fnn_he_diagnostic_phase{1..5}.sub`. Hypothesis-driven short experiments; phase reports in `docs/reports/diagnostic_phase{1,2,3}_report.html`.

**04_he_final_audit** — `run_final.py`; subs `fnn_he_final.sub` + `fnn_he_final_continuation.sub`. Results merged into `final_audit_merged.json`.

**05_sgd_recovery** —
| Runner | Subs | Result |
|---|---|---|
| `run_plain_sgd_100L.py` | `plain_sgd_100L_nobn_w512_{mnist,fashion_mnist,cifar10}.sub` | Advisor's plain-SGD recipe replication |
| `run_plain_sgd_recovery.py` | `recovery_plain_sgd_*.sub` (4) | Round 1: fmnist/100L/NoBN **PASS** |
| `run_plain_sgd_recovery2.py` | `recovery2_plain_sgd_*.sub` (3) | Round 2 (tighter plateau, `bn_momentum=0.01`): cifar10/100L/NoBN **PASS** |
| `run_adam_recovery3.py` | `recovery3_adam_*.sub` (2) | Round 3 (Adam+plateau+warmup+clip): both 100L/BN peaked then drifted — **no PASS** |

**06_v2_row_centered** — all rounds use `row_centered_layer_balanced_product_base` (η=0.5) and **forbid gradient clipping** (enforced by an `assert` in each runner's `main()`).
| Runner | Smoke subs | Audit subs | Result |
|---|---|---|---|
| `run_row_centered_audit.py` | `row_centered_smoke_*` (10) | `row_centered_audit_*` (10) | Round 1: **4 PASS at L=30** |
| `run_row_centered_audit_round2.py` | `row_centered_smoke2_*` (4) | — (nothing promising at smoke) | Round 2 (η=0.1 at L≥50 NoBN): no new PASS |
| `run_row_centered_audit_round3.py` | `row_centered_smoke3_*` (2) | — (both stuck at chance) | Round 3 (V2+BN+Adam at L=100): no PASS |
| `run_row_centered_audit_round4.py` | `row_centered_smoke4_*` (6) | `row_centered_audit4_*` (4) | Round 4 (V2+BN+plain SGD): **+1 PASS fmnist/50L/BN (99.95%)** — the double-preconditioning finding |

`triage_row_centered_smoke.py` classifies smoke JSONs (PASS / LEARNING / STUCK / DEAD / ABORTED) and suggests follow-ups.

**07_v2_eta_nobn** — `run_v2_nobn_sgd.py`; subs `v2_nobn_sgd_{smoke,audit}_*` (12) + `v2_nobn_sgd_lr1e6_smoke_*` (6). Per-architecture η\* from `scripts/eta_sweep_pick.py` (`eta_star_recommended.json`).

**08_he_lowlr_probe** — `run_he_sgd_lowlr_smoke.py`, `run_he_sgd_lowlr2_smoke.py`; 4 subs on {cifar10,fmnist}×100L/BN.

**09_rcfwd_rescale** — `run_rcfwd_gradrescale.py`; 12 subs ({smoke,audit} × {cifar10,fmnist} × {30,50,100}L, NoBN, plain SGD). Uses the `grad_rescale` config field + `_GradRescale` autograd op in `src/rp_study/models/classifiers.py`. Validated at initialization only (`reports/figures/rcfwd_rescale/`); **training runs never submitted**.

## Infrastructure (this directory's root)

| File | Purpose |
|---|---|
| `sync_to_cluster.sh` | Rsync the project to `user@cluster:~/thesis/` — always use this, never scp individual files |
| `setup_container.sh` | Rebuild the pyxis container when dependencies change |
| `test_gpu.py` + `test_job.sub` | Cluster/GPU sanity check |
| `WORKFLOW.md` | Daily sync → submit → tail → pull loop |
| `DLC - User Manual.pdf`, `DLC - Pre-slurm development tutorial.pdf` | Cluster documentation |

## Shared SLURM conventions

All sub files use:

```
#SBATCH -p dlc
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx01,dgx04   # driver/stability issues
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out      # job-name + job-id stdout
```

pyxis container `${HOME}/nvidia_pt.sqsh`, mounted at `/mount`. See `WORKFLOW.md` for the full daily loop.

## Daily loop reminder

```bash
# 1. Local
bash cluster/sync_to_cluster.sh

# 2. Cluster
ssh user@cluster
find ~/thesis/src -name "__pycache__" -exec rm -rf {} +
find ~/thesis/cluster -name "__pycache__" -exec rm -rf {} +
cd ~/thesis
sbatch cluster/<NN>_<campaign>/<job>.sub
squeue -u $CLUSTER_USER -o "%.18i %.40j %.8T %.10M %R"
tail -f <jobname>-<JOBID>.out   # live log

# 3. Local
HOST=user@cluster
scp "${HOST}:~/thesis/reports/results/<label>.json" reports/results/
scp "${HOST}:~/thesis/<label>-*.out" logs/slurm/<NN>_<campaign>/
```
