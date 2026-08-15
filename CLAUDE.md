# CLAUDE.md -- Agent Context for Thesis Project

## Role

You are a **math-aware engineering partner** for a thesis on neural network initialization strategies studied through the lens of random projections. Your job is to:

1. Help build and run simulations that correspond **one-to-one with the underlying math**
2. Handle the codebase, infrastructure, and experiment pipelines
3. Understand the deep learning theory (backprop, gradient flow, kernel theory) well enough to verify that code matches formulations
4. Flag when an implementation diverges from its mathematical specification

You are NOT a code monkey. You should understand *why* each initializer is designed the way it is, and catch errors where the code doesn't match the math.

## Research Context

See [CONTEXT.md](CONTEXT.md) for the thesis problem statement.
See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the chronological narrative and current state.
See [INITIALIZERS.md](INITIALIZERS.md) for mathematical definitions of every initialization strategy.
See **[docs/plans_handoffs/FRONTIER.md](docs/plans_handoffs/FRONTIER.md)** for what is open and in flight RIGHT NOW — read it before starting any task.

## The Agentic Cycle (how tasks are spun up and closed)

Every substantive task runs the same loop — defined in full in `FRONTIER.md` §"The agentic cycle":
**brief → spin up → isolate → work → report back → oracle verifies & closes.**

- Task briefs live in `docs/plans_handoffs/briefs/` (template there). A brief + the onboarding chain must be sufficient for a fresh agent with no conversation history.
- Additive work (new campaign dir, new results, new notes) → `main`. Anything touching shared files (`src/`, existing docs/notebooks) → branch `work/<slug>`.
- On finishing: fill the brief's Outcome section, update the task's README and the FRONTIER row, commit. Every claimed number must trace to a file in `reports/results/`.

### Starting a new campaign (checklist)

1. `mkdir cluster/<NN>_<slug>/` (next number; chronological). Copy `cluster/09_rcfwd_rescale/run_rcfwd_gradrescale.py` as the skeleton — it carries the required idioms: `ROOT = Path(__file__).resolve().parents[2]`, `sys.path.insert` for `src` **and** `cluster/03_he_diagnostics` (shared `_result_to_payload`/`print_diagnostic_summary`), the `ARCH` dict, `--experiment/--seed/--device/--output/--lr` argparse, abort-on-explosion, and the no-clipping assert.
2. One `.sub` per job, copied from a 09 sub: **job name = experiment label = JSON filename stem**; body runs `python -u cluster/<NN>_<slug>/run_X.py --experiment <label> ... --output reports/results/<label>.json`; keep the SBATCH block (dlc partition, `--exclude=dgx04`, `%x-%j.out`).
3. Smoke first (20 ep, `diagnostics_every=1`), gate audits (200 ep) on smoke triage.
4. Write the campaign `README.md` from the standard shape: Question / Builds on / What ran / Findings (numbers ← JSONs) / Reproduce / Evidence & gaps. Add the campaign row in `cluster/README.md` and, when results land, in `reports/results/INDEX.md`.
5. Loop: `bash cluster/sync_to_cluster.sh` → (cluster) clear `__pycache__` → `sbatch` → (local) `bash cluster/pull_results.sh '<label-glob>' <NN>_<slug>` → commit JSONs + docs → update FRONTIER.

### Working on proofs / the manuscript

The LaTeX manuscript is `thesis/main.tex` (chapters in `thesis/chapters/`; theorem/lemma envs and macros `\E \Var \Cov \relu \diag` defined in the preamble — reuse them, don't redefine). Derivations often mature in `docs/reports/*.md` first, then get formalized into a chapter. Key existing results to cite rather than re-prove: half-Gaussian moments + centering ratio (ch3 / `gradient_diagnostics_analysis.md` §4), the gain-coupling lock, the product-balanced variance (INITIALIZERS.md items 14–18). Draft proofs not ready for the public repo stay in `docs/scratch/proofs/` (gitignored).

## Codebase Layout

```
src/rp_study/
  config.py                  # Dataclasses: ExperimentConfig, NetworkConfig, ClassifierConfig, TrainingConfig
  models/
    initializers.py          # Registry: @register_initializer("name") -- SINGLE SOURCE OF TRUTH
    networks.py              # FeedForward class (uses initialize_layer from registry)
    classifiers.py           # FC/CNN classifiers (incl. _GradRescale backward hook for rcfwd)
  data/loaders.py            # MNIST / Fashion-MNIST / CIFAR-10
  projections/
    random_projections.py    # RP matrices, multi_layer_rp_with_init() bridge
  experiments/
    gradient_analysis.py     # GradientExperiment, compare_initializations()
    supervised_training.py   # run_supervised_experiment() — full training loop, schedulers, diagnostics, checkpoints
  analysis/kernel.py         # K(alpha) arc-cosine kernel
  visualization/             # gradient_plots, training_plots, projection_plots

cluster/                     # SLURM campaigns, chronologically numbered — see cluster/README.md
  01_geometry/ ... 09_rcfwd_rescale/   # each dir: runner .py + .sub files for one campaign
  sync_to_cluster.sh         # rsync the project to the cluster (USE THIS — see workflow below)
  pull_results.sh            # fetch result JSONs + SLURM logs back by label glob
  cluster.env(.example)      # cluster identity (user/host); cluster.env is gitignored
  WORKFLOW.md                # Daily-workflow notes for the cluster

scripts/                     # standalone figure/analysis helpers — see scripts/README.md

notebooks/                   # analysis notebooks 01-13 — see notebooks/README.md
  05_initializer_dashboard.ipynb  # unified one-stop-shop for new initializers
  13_final_results.ipynb          # latest full audit narrative

reports/
  results/*.json             # run histories (full EpochMetrics per epoch), COMMITTED + INDEX.md
  figures/<campaign>/*.png   # generated matplotlib outputs (local only, gitignored)

docs/
  RESEARCH_LOG.md            # master chronology
  milestones/                # date-stamped briefings (public-facing names)
  reports/                   # diagnostic phase reports, final audit report
  plans_handoffs/            # follow-up plans, status handoffs
  scratch/                   # session working notes (gitignored)

thesis/                      # LaTeX manuscript + supporting notes
logs/slurm/<campaign>/       # SLURM .out logs (local only, gitignored)
```

## Key Conventions

- **Initializer registry**: All initializers live in `src/rp_study/models/initializers.py`. Never duplicate initializer logic in notebooks.
- **Weight layout**: `nn.Linear` stores weight as `(fan_out, fan_in)`. Row centering = subtract mean along `dim=1` (each output neuron's weights sum to zero).
- **Bridge for geometry experiments**: Use `multi_layer_rp_with_init(X, n_layers, init_strategy)` from `rp_study.projections` to apply multi-layer RP+ReLU using registry initializers. This accepts numpy, returns numpy.
- **Configs**: Use dataclasses from `config.py`. Seeds must be reset between strategies for fair comparison.
- **Notebooks**: Configuration parameters at the top. Imports from `src/rp_study`. No local re-implementations of things that exist in the package.
- **Bias**: All initializers set bias to zero (or handle `bias=None`).
- **No gradient clipping in V2/rcfwd experiments** — fix instability via LR/warmup/optimizer choice instead (runner `main()`s assert this).
- **Public-facing repo**: file names must stay professional and neutral (date-stamped briefings, no personal references).

## When Adding a New Initializer

1. Add `@register_initializer("name")` function in `src/rp_study/models/initializers.py`
2. Include a docstring with: mathematical definition, motivation, known properties
3. Document in `INITIALIZERS.md` (formula, motivation, properties)
4. Add to `INIT_STRATEGIES` list in `notebooks/05_initializer_dashboard.ipynb` and run to see geometry + gradient + statistics

## Common Pitfalls

- Row centering reduces variance by factor `(1 - 1/d)`. Always consider whether variance adjustment is needed.
- The kernel-preserving initializer is slow (~200 optimizer steps per layer). Warn before running with many layers.
- When comparing initializers, always reset the seed before each strategy.
- PCA on degenerate data (all points collapsed) produces a meaningless 2D scatter -- check norms before plotting.
- Cluster runners live one directory below `cluster/`; they anchor the repo root via `Path(__file__).resolve().parents[2]` and put `cluster/03_he_diagnostics` on `sys.path` for the shared `_result_to_payload` / `print_diagnostic_summary` helpers. Keep both if you add a runner.

## Cluster Workflow

The cluster is SLURM-managed (pyxis containers); the user/host identity lives in `cluster/cluster.env` (gitignored — create it once with `cp cluster/cluster.env.example cluster/cluster.env` and edit). Code lives at `~/thesis/` on the cluster, mirroring this repo. **Always use the sync script for uploads** — never scp individual files (zsh on macOS keeps breaking long scp lines at the newline between source and destination, which produces silent failures).

### Standard sync-and-submit loop

**1. Local Mac terminal — push code:**
```bash
bash cluster/sync_to_cluster.sh
```
This rsyncs the project root to `~/thesis/` on the cluster, excluding `__pycache__`, `.git`, `data/`, `*.sqsh`, `logs/`, `docs/scratch/`. One password prompt covers everything.

**2. Cluster terminal — clear stale bytecode and submit:**
```bash
source cluster/cluster.env && ssh "$CLUSTER_USER@$CLUSTER_HOST"    # if not already in
find ~/thesis/src ~/thesis/cluster -name "__pycache__" -exec rm -rf {} +
cd ~/thesis && sbatch cluster/<NN>_<campaign>/<job>.sub
squeue -u "$CLUSTER_USER"
tail -f <jobname>-<JOBID>.out   # live log
```
Clearing `__pycache__` matters: stale `.pyc` files have caused `AttributeError` on freshly added dataclass fields more than once. Always clear after a `config.py` or `experiments/*.py` change.

**3. Local Mac terminal — pull results back when done:**
```bash
source cluster/cluster.env; HOST="$CLUSTER_USER@$CLUSTER_HOST"
scp "${HOST}:~/thesis/reports/results/<file>.json" reports/results/
scp "${HOST}:~/thesis/<jobname>-*.out" logs/slurm/<NN>_<campaign>/
```
Quote the remote glob — `*` must expand on the cluster, not in local zsh.

### When pasting commands fails

If a long command (especially `scp src/... user@host:target`) wraps in the terminal and gets split at the newline, zsh treats it as two commands and the upload fails silently with `no such file or directory: user@...`. Workarounds:
- Prefer `cluster/sync_to_cluster.sh` for any multi-file upload.
- For one-off uploads, put the whole `scp` on a single physical line, or use a shell variable: `source cluster/cluster.env; HOST="$CLUSTER_USER@$CLUSTER_HOST"; scp foo.py "$HOST:~/thesis/foo.py"`.

### SLURM script conventions used here

- `#SBATCH --exclude=dgx04` — `dgx04`'s CUDA driver is too old. Keep this exclude.
- `dgx01` was excluded for past issues; the Aug 2026 DLC hardware/OS upgrade re-imaged it (along with `dgx02`) and the exclude was lifted after a clean `cluster/test_job.sub` run on `dgx02` (sibling node, same upgrade) confirmed the new stack works — see [cluster/cluster.env.dlc2](cluster/cluster.env.dlc2) for the upgraded-login-node test env. If dgx01/dgx02 jobs start misbehaving again, re-add `dgx01` to the exclude list here and across `cluster/*/*.sub`.
- Container image: `${HOME}/nvidia_pt.sqsh` (pyxis), mounted at `/mount`.
- Stdout pattern: `%x-%j.out` (job-name + ID) — easy to find with a glob.
- Paths inside `.sub` files are **relative to the repo root** (`python -u cluster/<NN>_<campaign>/run_X.py`, `--output reports/results/<label>.json`); submit from `~/thesis`.
- Output JSON goes to `reports/results/<label>.json` (flat — notebooks and scripts read these exact paths); checkpoints to `reports/checkpoints/<label>/`.

## Supervised Training Pipeline

For initialization comparison experiments on real training (not just geometry/gradient analysis):

- Entry point: `run_supervised_experiment(exp_config, classifier_config, training_config)` in `src/rp_study/experiments/supervised_training.py`.
- Schedulers supported via `TrainingConfig.scheduler`: `"none"`, `"cosine"`, `"step"`, `"onecycle"`, `"plateau"` (ReduceLROnPlateau, monitors `eval_train_loss` or `eval_train_accuracy`).
- Diagnostics: set `diagnostics_every=N` to log per-layer gradient L2 norms + BN running stats every N epochs. Set `log_grad_per_layer=True` for the full per-layer gradient vector. The training loop also records `learning_rate` every epoch automatically.
- `ClassifierConfig.grad_rescale=r` inserts the `_GradRescale` op (identity forward, gradient × r in backward) after each hidden ReLU — the rcfwd mechanism.
- Checkpoints: set `checkpoint_dir` + `checkpoint_every`; resume with `resume_checkpoint`.
- Primary thesis metric: `eval_train_accuracy` (full train set in `model.eval()` mode). Pass criterion:
  - **Campaigns 01–10 (historical record):** `eval_train_accuracy ≥ 0.995` **AND** `eval_train_loss ≤ 0.10`. The headline counts cited throughout the thesis (He 10/12, V2 5/12) are on this criterion — do not retroactively relabel them.
  - **Campaign 10-onward (advisor, 2026-08-15):** `eval_train_accuracy ≥ 0.99`; the loss condition is **dropped**. Justified by campaign 10, where a cell sat at chance accuracy while its loss climbed past `ln 10` — the two metrics decouple in the frozen-window regime. A scan of every committed JSON found exactly two runs that flip under the new rule (both in `fnn_he_bn_evaltrain_training.json`), and both flip on the accuracy threshold, not the dropped loss condition.
  - Always log **both** metrics regardless of which criterion gates the run.

### When adding a new scheduler

1. Extend `TrainingConfig.scheduler` Literal in `config.py`.
2. Add the construction branch in `_build_scheduler` (`supervised_training.py`).
3. If it needs a per-step metric (like plateau does), extend the per-epoch `scheduler.step(...)` routing in the training loop — there's already a `scheduler_needs_metric` flag pattern in place.
4. Smoke-test locally on CPU for 2 epochs before syncing to the cluster.
