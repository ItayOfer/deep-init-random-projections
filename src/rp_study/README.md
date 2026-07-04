# rp_study — the thesis Python package

Everything the experiments run on. The cluster campaigns (`cluster/01..09`), the notebooks, and the `scripts/` helpers all import from here; nothing re-implements what this package provides. Design rule: **code corresponds one-to-one with the math** — every initializer's docstring states its formula and known properties (full derivations: `INITIALIZERS.md`).

## Role in the research

| Module | What it contributes to the story |
|---|---|
| `models/initializers.py` | **The single source of truth**: 19 strategies behind `@register_initializer("name")` — `he`, the row-centered family (`row_centered_he`, `_var_adj`, `partial_centered_he`), the theory-derived fixes (`row_centered_product_balanced` = V1, `row_centered_layer_balanced_product_base` = V2, `row_centered_forward_balanced` = rcfwd's init), `orthogonal_he`, `kernel_preserving`, … |
| `models/classifiers.py` | `DeepFCClassifier` (+CNN) used by every training campaign — including `_GradRescale`, the custom autograd op (identity forward, gradient × r backward) behind campaign 09; enabled via `ClassifierConfig.grad_rescale`. |
| `experiments/supervised_training.py` | `run_supervised_experiment()` — the training loop every audit runs: schedulers (`none/cosine/step/onecycle/plateau`), LR warmup, per-layer gradient diagnostics (`diagnostics_every`, `log_grad_per_layer`), BN stats logging, abort-on-explosion, checkpoints/resume. Its `EpochMetrics` history is the JSON schema of everything in `reports/results/`. |
| `experiments/gradient_analysis.py` | `GradientExperiment`, `compare_initializations()` — the gradient-flow measurements behind the forward/backward gain asymmetry finding. |
| `experiments/geometry_benchmark.py` + `analysis/` | k-NN / distance-correlation / effective-dim geometry metrics (campaign 01), the arc-cosine kernel `K(α)` (`analysis/kernel.py`), and the empirical angle map (`analysis/angle_map.py`). |
| `projections/random_projections.py` | `multi_layer_rp_with_init(X, n_layers, init_strategy)` — the numpy bridge that applies multi-layer RP+ReLU with any registry initializer (geometry notebooks). |
| `data/loaders.py` | MNIST / Fashion-MNIST / CIFAR-10 (auto-download to `data/`); `data/shapes.py` for synthetic 2-D geometry. |
| `config.py` | The dataclasses every entry point shares: `ExperimentConfig` (seed/device), `ClassifierConfig` (arch/depth/init/BN/`grad_rescale`), `TrainingConfig` (optimizer/LR/scheduler/warmup/targets). |
| `visualization/` | `gradient_plots`, `training_plots`, `projection_plots`. |

## Key idioms

```python
# Train one architecture the way the audits do
from rp_study.config import ExperimentConfig, ClassifierConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment

exp = ExperimentConfig(seed=42)
clf = ClassifierConfig(architecture="fc", depth=30, init_strategy="row_centered_layer_balanced_product_base",
                       init_kwargs={"eta": 0.5}, use_batch_norm=True, fc_hidden_dim=500)
train = TrainingConfig(num_epochs=200, optimizer="sgd", learning_rate=1e-2, scheduler="plateau",
                       target_train_accuracy=0.995, target_metric="eval_train_accuracy")
result = run_supervised_experiment(exp, clf, train)

# rcfwd: cancel the backward gain outside the weights
clf = ClassifierConfig(architecture="fc", depth=100, init_strategy="row_centered_forward_balanced",
                       grad_rescale=0.8256, use_batch_norm=False)

# Geometry: multi-layer RP + ReLU with any registry initializer
from rp_study.projections import multi_layer_rp_with_init
X20 = multi_layer_rp_with_init(X, n_layers=20, init_strategy="row_centered_he")

# Register a new initializer (then document in INITIALIZERS.md + add to notebook 05)
from rp_study.models.initializers import register_initializer

@register_initializer("my_init")
def my_init(layer, **kwargs):
    with torch.no_grad():
        layer.weight.normal_(0.0, 0.01)
        if layer.bias is not None:
            layer.bias.zero_()
```

## Conventions that matter for correctness

- `nn.Linear` weight is `(fan_out, fan_in)`; **row centering subtracts the mean along `dim=1`** (each output neuron's incoming weights sum to zero). Row centering shrinks variance by `(1 − 1/d)` — decide explicitly whether to re-adjust.
- All initializers zero the bias (or handle `bias=None`).
- Reset the seed between strategies when comparing initializers.
- The thesis pass criterion is computed on the **full train set in `model.eval()` mode**: `eval_train_accuracy ≥ 0.995` and `eval_train_loss ≤ 0.10`.
- No gradient clipping in V2/rcfwd experiments — instability is information; fix it via LR/warmup/optimizer.
