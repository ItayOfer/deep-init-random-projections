# Neural Network Initialization through the Lens of Random Projections

M.Sc. thesis codebase studying how weight initialization shapes what deep ReLU networks can represent and learn — combining **geometry** (arc-cosine kernel / random-projection analysis), **gradient flow** (forward/backward gain analysis), and **supervised training audits** on a 12-architecture benchmark.

## The research in one paragraph

Standard He initialization trains reliably but geometrically collapses inputs as depth grows (angles shrink toward 0 under the arc-cosine kernel map). Row-centering the weights (each output neuron's weights sum to zero) prevents this collapse by making the layer blind to the DC component — but it structurally locks the forward gain to `√((π−1)/π) ≈ 0.826` per layer, creating a gradient trap that **no scalar variance can fix** (the forward/backward gain ratio is invariant). This work formalizes that conflict, designs initializers that fix the *product* of gains (V1) and redistribute variance across depth (V2), and audits everything on real training: {30, 50, 100} layers × {CIFAR-10, Fashion-MNIST} × {BatchNorm, none}.

## Headline results

| Finding | Evidence |
|---|---|
| Tuned He passes **10/12** architectures; the two failures at 100L+BN are an optimizer-recipe wall, not an initialization limit | `reports/results/final_audit_merged.json`, recovery runs |
| Adam's adaptive scaling *causes* the 100L/NoBN failures (second-moment underflow); plain SGD rescues them | recovery rounds 1–2 |
| V2 (layer-balanced row-centered) passes **5/12**, all at depth ≤ 30–50; it has a hard depth ceiling from compounding per-layer variance scaling | V2 audit rounds 1–4 |
| V2 composes with SGD but fights Adam ("double preconditioning"): fmnist/50L/BN goes 33% → 99.95% by switching optimizer | round 4 audit |
| Row-centering spreads data without preserving class structure — k-NN accuracy falls to chance at depth even as effective dimension grows ("spread ≠ structure") | notebook 09, `docs/reports/gradient_diagnostics_analysis.md` §7 |
| **Open**: 100L+BN fails for every initializer × optimizer recipe tried | `docs/plans_handoffs/2026-07-04_research_status_handoff.md` |
| **Next**: `rcfwd` — forward-balanced row-centering + closed-form per-layer backward gradient rescale; validated at initialization, training not yet run | `cluster/09_rcfwd_rescale/` |

## Repository map

```
├── README.md                  # you are here
├── CONTEXT.md                 # research problem statement
├── INITIALIZERS.md            # mathematical reference for all initializers
├── SUMMARY.md                 # detailed findings summary
├── CLAUDE.md / AGENTS.md      # agent/tooling context
├── src/rp_study/              # the Python package (models, initializer registry,
│                              #   training loop, data loaders, analysis, plotting)
├── notebooks/                 # analysis notebooks 01-13 (see notebooks/README.md)
├── cluster/                   # SLURM campaigns 01-09, chronologically numbered
│                              #   (see cluster/README.md for the campaign index)
├── scripts/                   # standalone figure/analysis helpers (see scripts/README.md)
├── reports/
│   ├── results/               # all experiment result JSONs (committed) + INDEX.md
│   └── figures/               # generated plots by campaign (local only)
├── docs/
│   ├── RESEARCH_LOG.md        # chronological narrative of the whole project
│   ├── milestones/            # date-stamped briefings and walkthroughs
│   ├── reports/               # diagnostic phase reports, final audit report
│   └── plans_handoffs/        # follow-up plans and status handoffs
├── thesis/                    # LaTeX manuscript + supporting notes
├── logs/slurm/                # SLURM job logs by campaign (local only)
└── data/                      # dataset cache (local only, auto-downloaded)
```

**Start here:** `docs/RESEARCH_LOG.md` for the story → `reports/results/INDEX.md` for the evidence → `notebooks/13_final_results.ipynb` for the latest full analysis.

## Quick start

```bash
pip install -e .          # or: pip install -r requirements.txt
```

```python
from rp_study.config import ExperimentConfig, ClassifierConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment

# Train a 30-layer FC net on Fashion-MNIST with a registry initializer
exp = ExperimentConfig(seed=42)
clf = ClassifierConfig(architecture="fc", depth=30, init_strategy="he", use_batch_norm=True)
train = TrainingConfig(num_epochs=20, learning_rate=1e-3, optimizer="adam")
result = run_supervised_experiment(exp, clf, train)
```

Geometry / gradient analysis without training:

```python
from rp_study.projections import multi_layer_rp_with_init
X_deep = multi_layer_rp_with_init(X, n_layers=20, init_strategy="row_centered_he")

from rp_study.experiments import compare_initializations
results = compare_initializations(layer_sizes=[784, 512, 256, 1],
                                  init_strategies=["he", "row_centered_he"],
                                  num_samples=1000)
```

## Initializer registry

All initialization strategies live in `src/rp_study/models/initializers.py` behind a single `@register_initializer("name")` registry — 19 strategies including `he`, `row_centered_he`, `row_centered_he_var_adj`, `partial_centered_he`, `orthogonal_he`, `kernel_preserving`, `row_centered_product_balanced` (V1), `row_centered_layer_balanced_product_base` (V2), and `row_centered_forward_balanced` (used by rcfwd). Mathematical definitions, motivations, and known properties for every strategy: [INITIALIZERS.md](INITIALIZERS.md).

```python
from rp_study.models.initializers import register_initializer

@register_initializer("my_init")
def my_init(layer, **kwargs):
    with torch.no_grad():
        layer.weight.normal_(0, 0.01)
        if layer.bias is not None:
            layer.bias.zero_()
```

## The 12-architecture audit

Pass criterion: `eval_train_accuracy ≥ 0.995` **and** `eval_train_loss ≤ 0.10` on the full training set in eval mode. Matrix: depth {30, 50, 100} × dataset {CIFAR-10, Fashion-MNIST} × {BN, NoBN}, FC width 500. Experiments run on a SLURM cluster; every campaign under `cluster/` is re-runnable (`cluster/README.md` documents the daily sync → submit → pull loop).

## Requirements

Python ≥ 3.8, PyTorch ≥ 1.9, NumPy, Matplotlib, scikit-learn (see `requirements.txt`).
