#!/usr/bin/env python3
"""row_centered_forward_balanced (INITIALIZERS.md section 14), 100L, at init:
forward/backward decomposition per layer. Three lines per panel:
rms(A) forward, rms(delta) backward, grad row norm.
Constant width 500 vs funnel 500 -> 100 (-4/layer, floored at 100).

row_centered_forward_balanced: row-centered weights with variance scaled
(~2.934/d) so the forward gain g_fwd = r*s ~ 1 (compensates the row-centering
r decay). Consequence: backward gain g_bwd = s > 1 (~1.21), so the backward
chain amplifies with depth. Expectation: forward ~flat, backward grows.

float64, GradientExperiment framework, no training. Ratios .2f.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

INIT = "row_centered_forward_balanced"
L = 100
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
CONST = [500] * L
TAPER = [max(100, 500 - 4 * k) for k in range(L)]   # 500 -> 104, floor 100


def decompose(dataset, hidden_widths):
    ls = [IDIM[dataset]] + hidden_widths + [1]
    ec = ExperimentConfig(seed=SEED, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls, init_strategy=INIT)
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    a = np.array(list(res.get_activation_rms().values()))
    d = np.array(list(res.get_error_signal_rms().values()))
    g = np.array(list(res.get_mean_row_norms().values())[:-1])
    return a, d, g


def ratio(x):
    return x.max() / max(x.min(), 1e-300)


print(f"init={INIT}  taper widths: {TAPER[0]},{TAPER[1]},...,{TAPER[-2]},{TAPER[-1]}")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for col, dataset in enumerate(["fashion_mnist", "cifar10"]):
    for row, (widths, name) in enumerate(
            [(CONST, "constant width 500"), (TAPER, "funnel 500->100 (-4/layer, floor 100)")]):
        a, d, g = decompose(dataset, widths)
        ax = axes[row][col]
        ax.plot(range(len(a)), a, "-", color="tab:blue", lw=1.5, label="rms(A) forward activation")
        ax.plot(range(len(d)), d, "-", color="tab:red", lw=1.5, label="rms(delta) backward error")
        ax.plot(range(len(g)), g, "-", color="black", lw=1.3, label="grad row norm (~A*delta)")
        ax.set_yscale("log")
        ax.set_title(f"{dataset} | row_centered_forward_balanced {name}\n"
                     f"rms(A) ratio {ratio(a):.2f}x | rms(delta) ratio {ratio(d):.2f}x | "
                     f"grad ratio {ratio(g):.2f}x", loc="left", fontsize=10)
        ax.set_xlabel("hidden layer index (0=input side, 99=output side)")
        ax.set_ylabel("RMS / norm (log)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
        print(f"  {dataset:>14s} {name:<40s}: rms(A) ratio {ratio(a):.2f}x  "
              f"rms(delta) ratio {ratio(d):.2f}x  grad ratio {ratio(g):.2f}x")
fig.suptitle(f"ROW-CENTERED FORWARD-BALANCED, 100L at init (seed={SEED}): forward rms(A) vs backward "
             "rms(delta) vs grad row norm -- constant vs funnel 500->100", fontsize=13, y=1.01)
fig.tight_layout()
suffix = "" if SEED == 42 else f"_seed{SEED}"
out = f"reports/figures/rcfwd_rescale/rcfwd_funnel100_fwd_bwd_100L{suffix}.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
