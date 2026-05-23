# Cluster scripts (DLC / SLURM)

Daily workflow lives in `WORKFLOW.md`. This README organises the now-sizeable collection of runners and sub files by experiment campaign.

## Layout convention

- **Runner** `cluster/run_<campaign>.py` — Python entry point. Hardcodes the per-architecture recipes for that campaign. Accepts `--experiment <label>` to select one architecture.
- **Sub file** `cluster/<label>.sub` — SLURM submission script. One job runs one architecture. Output JSON lands at `reports/results/<label>.json` (the filename stem matches the SLURM job name exactly, no cross-contamination).
- **Output log** `<label>-<JOBID>.out` — at the project root after the job finishes.

## Campaigns

### Original sweep (5 / 12 He passing)

| Runner | Sub files | Purpose |
|---|---|---|
| `run_supervised_sweep.py` | `fnn_he_tuning_*.sub` | Initial 12-architecture grid sweep with per-arch HP tuning. |
| `run_supervised_grid.py` | (legacy) | Earlier grid runner. |
| `run_supervised_from_configs.py` | `fnn_he_tuned_rerun.sub`, `fnn_he_targeted_best12.sub` | Re-runs from saved best configs. |

### Audit phase (8 / 12 He passing — phases 1, 2, 3, 5)

| Runner | Sub files | Purpose |
|---|---|---|
| `run_diagnostic.py` | `fnn_he_diagnostic_phase{1..5}.sub` | Hypothesis-driven short experiments per phase. |
| `run_phase{2,3,4,5}.py` | (paired with phase sub files) | Longer-run confirmation. |
| `run_final.py` | `fnn_he_final.sub`, `fnn_he_final_continuation.sub` | The audit: all 12 architectures with best-known recipes, 200 epochs. |

### Post-meeting recovery (10 / 12 He passing — rounds 1, 2, 3)

| Runner | Sub files | Purpose |
|---|---|---|
| `run_plain_sgd_100L.py` | `plain_sgd_100L_nobn_w512_{mnist,fashion_mnist,cifar10}.sub` | Replication of the advisor's plain-SGD recipe. |
| `run_plain_sgd_recovery.py` | `recovery_plain_sgd_*.sub` (4 files) | Round 1 — extend plain SGD to all 4 failing 100L architectures. fmnist/100L/NoBN PASS. |
| `run_plain_sgd_recovery2.py` | `recovery2_plain_sgd_*.sub` (3 files) | Round 2 — tighter plateau + `bn_momentum=0.01`. cifar10/100L/NoBN PASS. |
| `run_adam_recovery3.py` | `recovery3_adam_*.sub` (2 files) | Round 3 — Adam + plateau + bnm=0.01 + warmup + clip. Both 100L/BN cases survived ep-0 NaN but peaked then drifted — neither PASS. |

### V2 audit (5 / 12 V2 passing — rounds 1, 2, 3, 4)

All four V2 rounds use `row_centered_layer_balanced_product_base` (η=0.5) and **forbid gradient clipping** (an `assert` in each runner's `main()` enforces this — see `feedback_no_grad_clipping.md` in agent memory for rationale).

| Runner | Smoke subs | Audit subs | Result |
|---|---|---|---|
| `run_row_centered_audit.py` | `row_centered_smoke_*.sub` (10 files) | `row_centered_audit_*.sub` (10 files) | Round 1, η=0.5, He-passing recipes. **4 PASSes at L=30.** |
| `run_row_centered_audit_round2.py` | `row_centered_smoke2_*.sub` (4 files) | (audit2 not built — nothing was promising at smoke) | Round 2, modified recipes for L=50+ NoBN and L=100 NoBN at η=0.1. **No new PASS.** |
| `run_row_centered_audit_round3.py` | `row_centered_smoke3_*.sub` (2 files) | (audit3 not built — both stuck at chance) | Round 3, V2 + BN + Adam at L=100. **No PASS.** |
| `run_row_centered_audit_round4.py` | `row_centered_smoke4_*.sub` (6 files) | `row_centered_audit4_*.sub` (4 files) | Round 4, V2 + BN + plain SGD. **+1 PASS: fmnist/50L/BN at 99.95 %** (was stuck at 33 % under Adam). |

### Triage / analysis helpers

| Script | Purpose |
|---|---|
| `triage_row_centered_smoke.py` | Loads `row_centered_smoke_*.json`, classifies each architecture by training dynamics (PASS / LEARNING / STUCK / DEAD / ABORTED), suggests a follow-up action. |

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

pyxis container `${HOME}/nvidia_pt.sqsh`, mounted at `/mount`. See `WORKFLOW.md` for the full daily sync → submit → tail → pull loop.

## Daily loop reminder

```bash
# 1. Local
bash cluster/sync_to_cluster.sh

# 2. Cluster
ssh user@cluster
find ~/thesis/src -name "__pycache__" -exec rm -rf {} +
find ~/thesis/cluster -name "__pycache__" -exec rm -rf {} +
cd ~/thesis
sbatch cluster/<job>.sub
squeue -u $CLUSTER_USER -o "%.18i %.40j %.8T %.10M %R"
tail -f <jobname>-<JOBID>.out   # live log

# 3. Local
HOST=user@cluster
scp "${HOST}:~/thesis/reports/results/<label>.json" reports/results/
scp "${HOST}:~/thesis/<label>-*.out" .
```
