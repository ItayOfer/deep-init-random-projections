#!/usr/bin/env python3
"""Advisor's experiment: He init, 100 layers, width tapered 500 -> 10
(~5 neurons/layer), measured at INITIALIZATION (no training). Plot the mean
per-layer gradient row norm across layers, compared against the constant-width
500 He baseline.

Hypothesis under test: He controls the forward pass, but the backward pass
explodes because we multiply many (square, 500x500) matrices. Shrinking the
width with depth makes the backward matrices rectangular/smaller -- does that
tame the gradient profile?

Same measurement framework as the existing gradient-row-norm figures
(GradientExperiment: one forward+backward MSE pass, get_mean_row_norms),
in float64 so nothing overflows in the measurement.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

L = 100
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
TAPER = [int(round(w)) for w in np.linspace(500, 10, L)]   # 500 -> 10 over 100 layers
CONST = [500] * L
print("taper widths (first 5, last 5):", TAPER[:5], "...", TAPER[-5:])


def grad_row_norms(dataset, hidden_widths):
    ls = [IDIM[dataset]] + hidden_widths + [1]
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls, init_strategy="he")
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    return np.array(list(res.get_mean_row_norms().values())[:-1])   # drop output layer


fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for col, dataset in enumerate(["fashion_mnist", "cifar10"]):
    gc_const = grad_row_norms(dataset, CONST)
    gt = grad_row_norms(dataset, TAPER)
    for row, (g, name, widths) in enumerate(
            [(gc_const, "He constant width 500", CONST),
             (gt, "He tapered width 500->10", TAPER)]):
        ax = axes[row][col]
        ax.plot(range(len(g)), g, "o-", ms=2.5, lw=1.3,
                color=("tab:blue" if row == 0 else "tab:green"))
        ax.fill_between(range(len(g)), g, alpha=0.15,
                        color=("tab:blue" if row == 0 else "tab:green"))
        ax.set_title(f"{dataset} | {name}\nmax={g.max():.3g}, min={g.min():.3g}, "
                     f"ratio={g.max()/g.min():.1f}x", loc="left", fontsize=10)
        ax.set_ylabel("mean grad row norm"); ax.grid(alpha=0.3)
        ax.set_xlabel("hidden layer index (0=input side, 99=output side)")
        print(f"{dataset:>14s} {name:<26s}: max={g.max():.3e} min={g.min():.3e} "
              f"ratio={g.max()/g.min():.1f}x")
fig.suptitle("He init, 100L, at initialization: gradient row norm across layers "
             "-- constant width vs tapered 500->10", fontsize=13, y=1.01)
fig.tight_layout()
out = "reports/figures/gain_funnels/he_funnel_gradient_100L.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)