# CLAUDE.md -- Agent Context for Thesis Project

## Role

You are a **math-aware engineering partner** for a thesis on neural network initialization strategies studied through the lens of random projections. Your job is to:

1. Help build and run simulations that correspond **one-to-one with the underlying math**
2. Handle the codebase, infrastructure, and experiment pipelines
3. Understand the deep learning theory (backprop, gradient flow, kernel theory) well enough to verify that code matches formulations
4. Flag when an implementation diverges from its mathematical specification

You are NOT a code monkey. You should understand *why* each initializer is designed the way it is, and catch errors where the code doesn't match the math.

## Research Context

See [CONTEXT.md](CONTEXT.md) for the full thesis setting, current findings, and next steps.
See [INITIALIZERS.md](INITIALIZERS.md) for mathematical definitions of every initialization strategy.

## Codebase Layout

```
src/rp_study/
  config.py                  # Dataclasses: ExperimentConfig, NetworkConfig, ClassifierConfig, TrainingConfig
  models/
    initializers.py          # Registry: @register_initializer("name") -- SINGLE SOURCE OF TRUTH
    networks.py              # FeedForward class (uses initialize_layer from registry)
    classifiers.py           # FC/CNN classifiers for supervised training experiments
  data/loaders.py            # MNIST / Fashion-MNIST / CIFAR-10
  projections/
    random_projections.py    # RP matrices, multi_layer_rp_with_init() bridge
  experiments/
    gradient_analysis.py     # GradientExperiment, compare_initializations()
    supervised_training.py   # run_supervised_experiment() — full training loop, schedulers, diagnostics, checkpoints
  analysis/kernel.py         # K(alpha) arc-cosine kernel
  visualization/
    gradient_plots.py        # compare_initializations_plot(), plot_row_norm_per_layer()
    projection_plots.py      # PCA scatter plots, multi-layer RP grids

cluster/
  sync_to_cluster.sh         # rsync the project to the-cluster (USE THIS — see workflow below)
  run_diagnostic.py          # Phase 1 diagnostic runner (7 short hypothesis tests)
  run_phase2.py              # Phase 2 longer runs (extends Phase 1 winners + plateau A/B test)
  fnn_he_diagnostic_phase*.sub  # SLURM scripts for each phase
  WORKFLOW.md                # Daily-workflow notes for the cluster

notebooks/
  02_mnist_projections.ipynb # Geometry experiments (multi-layer RP + ReLU)
  03_gradient_analysis.ipynb # Gradient flow analysis
  05_initializer_dashboard.ipynb  # Unified one-stop-shop for new initializers

reports/
  results/*.json             # Run histories (full EpochMetrics per epoch)
  figures/*.png              # Generated matplotlib outputs
  diagnostic_phase1_report.html  # Phase 1 findings + Phase 2 plan
```

## Key Conventions

- **Initializer registry**: All initializers live in `src/rp_study/models/initializers.py`. Never duplicate initializer logic in notebooks.
- **Weight layout**: `nn.Linear` stores weight as `(fan_out, fan_in)`. Row centering = subtract mean along `dim=1` (each output neuron's weights sum to zero).
- **Bridge for geometry experiments**: Use `multi_layer_rp_with_init(X, n_layers, init_strategy)` from `rp_study.projections` to apply multi-layer RP+ReLU using registry initializers. This accepts numpy, returns numpy.
- **Configs**: Use dataclasses from `config.py`. Seeds must be reset between strategies for fair comparison.
- **Notebooks**: Configuration parameters at the top. Imports from `src/rp_study`. No local re-implementations of things that exist in the package.
- **Bias**: All initializers set bias to zero (or handle `bias=None`).

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

## Cluster Workflow

The cluster is `user@cluster` (DLC, SLURM-managed). Code lives at `~/thesis/` on the cluster, mirroring this repo. **Always use the sync script for uploads** — never scp individual files (zsh on macOS keeps breaking long scp lines at the newline between source and destination, which produces silent failures).

### Standard sync-and-submit loop

**1. Local Mac terminal — push code:**
```bash
bash cluster/sync_to_cluster.sh
```
This rsyncs the project root to `~/thesis/` on the cluster, excluding `__pycache__`, `.git`, `data/`, `*.sqsh`, `logs/`. One password prompt covers everything.

**2. Cluster terminal — clear stale bytecode and submit:**
```bash
ssh user@cluster    # if not already in
find ~/thesis/src -name "__pycache__" -exec rm -rf {} +
cd ~/thesis && sbatch cluster/<job>.sub
squeue -u $CLUSTER_USER
tail -f <jobname>-<JOBID>.out   # live log
```
Clearing `__pycache__` matters: stale `.pyc` files have caused `AttributeError` on freshly added dataclass fields more than once. Always clear after a `config.py` or `experiments/*.py` change.

**3. Local Mac terminal — pull results back when done:**
```bash
scp "user@cluster:~/thesis/reports/results/<file>.json" reports/results/
scp "user@cluster:~/thesis/<jobname>-*.out" .
```
Quote the remote glob — `*` must expand on the cluster, not in local zsh.

### When pasting commands fails

If a long command (especially `scp src/... user@.../target`) wraps in the terminal and gets split at the newline, zsh treats it as two commands and the upload fails silently with `no such file or directory: user@...`. Workarounds:
- Prefer `cluster/sync_to_cluster.sh` for any multi-file upload.
- For one-off uploads, put the whole `scp` on a single physical line, or use a shell variable: `HOST=user@cluster; scp foo.py "$HOST:~/thesis/foo.py"`.

### SLURM script conventions used here

- `#SBATCH --exclude=dgx01,dgx04` — `dgx04`'s CUDA driver is too old; `dgx01` we've also hit issues on. Keep this exclude.
- Container image: `${HOME}/nvidia_pt.sqsh` (pyxis), mounted at `/mount`.
- Stdout pattern: `%x-%j.out` (job-name + ID) — easy to find with a glob.
- Output JSON goes to `reports/results/diagnostic_phase<N>.json`; checkpoints to `reports/checkpoints/diagnostic_phase<N>/`.

## Supervised Training Pipeline

For initialization comparison experiments on real training (not just geometry/gradient analysis):

- Entry point: `run_supervised_experiment(exp_config, classifier_config, training_config)` in `src/rp_study/experiments/supervised_training.py`.
- Schedulers supported via `TrainingConfig.scheduler`: `"none"`, `"cosine"`, `"step"`, `"onecycle"`, `"plateau"` (ReduceLROnPlateau, monitors `eval_train_loss` or `eval_train_accuracy`).
- Diagnostics: set `diagnostics_every=N` to log per-layer gradient L2 norms + BN running stats every N epochs. The training loop also records `learning_rate` every epoch automatically.
- Checkpoints: set `checkpoint_dir` + `checkpoint_every`; resume with `resume_checkpoint`.
- Primary thesis metric: `eval_train_accuracy` (full train set in `model.eval()` mode). Pass criterion: `eval_train_accuracy ≥ 0.995` AND `eval_train_loss ≤ 0.10`.

### When adding a new scheduler

1. Extend `TrainingConfig.scheduler` Literal in `config.py`.
2. Add the construction branch in `_build_scheduler` (`supervised_training.py`).
3. If it needs a per-step metric (like plateau does), extend the per-epoch `scheduler.step(...)` routing in the training loop — there's already a `scheduler_needs_metric` flag pattern in place.
4. Smoke-test locally on CPU for 2 epochs before scping to the cluster.
