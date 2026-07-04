#!/usr/bin/env python3
"""He init, 100L, at initialization: decompose the gradient into forward and
backward parts, per layer. Three lines per panel:
  * rms(A^l)      -- forward activation RMS  (forward signal)
  * rms(delta^l)  -- backward error-signal RMS (backward signal)
  * grad row norm -- mean per-layer weight-gradient row norm (~ rms(A)*rms(delta))

Compares constant width 500 vs a tapered width, for fmnist and cifar10.
Two funnels are produced:
  * 500 -> 10  over 100 layers (linspace)            -> he_funnel_fwd_bwd_100L.png
  * 500 -> 100, decreasing 4/layer, floored at 100   -> he_funnel100_fwd_bwd_100L.png

float64 measurement, same GradientExperiment framework, no training.
All ratios printed with 2 decimals (a 1.35x ratio must NOT round to '1x').
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
CONST = [500] * L
FUNNELS = {
    "he_funnel_fwd_bwd_100L.png": (
        [int(round(w)) for w in np.linspace(500, 10, L)],
        "tapered 500->10"),
    "he_funnel100_fwd_bwd_100L.png": (
        [max(100, 500 - 4 * k) for k in range(L)],
        "funnel 500->100 (-4/layer, floor 100)"),
}


def decompose(dataset, hidden_widths):
    ls = [IDIM[dataset]] + hidden_widths + [1]
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls, init_strategy="he")
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    a = np.array(list(res.get_activation_rms().values()))
    d = np.array(list(res.get_error_signal_rms().values()))
    g = np.array(list(res.get_mean_row_norms().values())[:-1])
    return a, d, g


def ratio(x):
    return x.max() / x.min()


for outname, (taper, taper_label) in FUNNELS.items():
    print(f"\n=== {outname}: {taper_label} (widths {taper[0]},{taper[1]},...,{taper[-2]},{taper[-1]}) ===")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for col, dataset in enumerate(["fashion_mnist", "cifar10"]):
        for row, (widths, name) in enumerate(
                [(CONST, "constant width 500"), (taper, taper_label)]):
            a, d, g = decompose(dataset, widths)
            ax = axes[row][col]
            ax.plot(range(len(a)), a, "-", color="tab:blue", lw=1.5, label="rms(A) forward activation")
            ax.plot(range(len(d)), d, "-", color="tab:red", lw=1.5, label="rms(delta) backward error")
            ax.plot(range(len(g)), g, "-", color="black", lw=1.3, label="grad row norm (~A*delta)")
            ax.set_yscale("log")
            ax.set_title(f"{dataset} | He {name}\n"
                         f"rms(A) ratio {ratio(a):.2f}x | rms(delta) ratio {ratio(d):.2f}x | "
                         f"grad ratio {ratio(g):.2f}x", loc="left", fontsize=10)
            ax.set_xlabel("hidden layer index (0=input side, 99=output side)")
            ax.set_ylabel("RMS / norm (log)")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
            print(f"  {dataset:>14s} {name:<38s}: rms(A) ratio {ratio(a):.2f}x  "
                  f"rms(delta) ratio {ratio(d):.2f}x  grad ratio {ratio(g):.2f}x")
    fig.suptitle(f"He 100L at init: forward rms(A) vs backward rms(delta) vs grad row norm "
                 f"-- constant vs {taper_label}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"reports/figures/{outname}", dpi=130, bbox_inches="tight")
    print(f"  saved reports/figures/{outname}")
