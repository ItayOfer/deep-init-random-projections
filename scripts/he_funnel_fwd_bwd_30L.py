#!/usr/bin/env python3
"""He init, 30L, at initialization: forward/backward decomposition per layer.
Three lines per panel: rms(A) forward, rms(delta) backward, grad row norm.
Constant width 500 vs funnel 500 -> 10 (linspace over 30 layers).
float64, GradientExperiment framework, no training. Ratios printed .2f.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

L = 30
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
CONST = [500] * L
TAPER = [int(round(w)) for w in np.linspace(500, 100, L)]   # 500 -> 100 over 30 layers


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


print(f"taper widths: {TAPER[0]},{TAPER[1]},...,{TAPER[-2]},{TAPER[-1]}")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for col, dataset in enumerate(["fashion_mnist", "cifar10"]):
    for row, (widths, name) in enumerate(
            [(CONST, "constant width 500"), (TAPER, "funnel 500->100")]):
        a, d, g = decompose(dataset, widths)
        ax = axes[row][col]
        ax.plot(range(len(a)), a, "-", color="tab:blue", lw=1.5, label="rms(A) forward activation")
        ax.plot(range(len(d)), d, "-", color="tab:red", lw=1.5, label="rms(delta) backward error")
        ax.plot(range(len(g)), g, "-", color="black", lw=1.3, label="grad row norm (~A*delta)")
        ax.set_yscale("log")
        ax.set_title(f"{dataset} | He {name} (L=30)\n"
                     f"rms(A) ratio {ratio(a):.2f}x | rms(delta) ratio {ratio(d):.2f}x | "
                     f"grad ratio {ratio(g):.2f}x", loc="left", fontsize=10)
        ax.set_xlabel("hidden layer index (0=input side, 29=output side)")
        ax.set_ylabel("RMS / norm (log)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
        print(f"  {dataset:>14s} {name:<20s}: rms(A) ratio {ratio(a):.2f}x  "
              f"rms(delta) ratio {ratio(d):.2f}x  grad ratio {ratio(g):.2f}x")
fig.suptitle("He 30L at init: forward rms(A) vs backward rms(delta) vs grad row norm "
             "-- constant vs funnel 500->100", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("reports/figures/he_funnel_fwd_bwd_30L.png", dpi=130, bbox_inches="tight")
print("saved reports/figures/he_funnel_fwd_bwd_30L.png")
