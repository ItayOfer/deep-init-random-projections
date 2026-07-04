#!/usr/bin/env python3
"""Init-profile plot for the L=50 V2 NoBN runs that failed (same format as
v2_nobn_100L_init_profile.png): forward activation bulge + per-layer gradient
row norm, computed in float64 so the true bulge shape is visible.

Each dataset uses its own ratio-minimising eta: fmnist 0.80, cifar10 0.90.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

L, WIDTH = 50, 500
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
ETA = {"fashion_mnist": 0.80, "cifar10": 0.90}


def profile(dataset):
    ls = [IDIM[dataset]] + [WIDTH] * L + [1]
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls,
                       init_strategy="row_centered_layer_balanced_product_base",
                       init_kwargs={"eta": ETA[dataset]})
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    g = np.array(list(res.get_mean_row_norms().values())[:-1])
    a = np.array(list(res.get_activation_rms().values())[:-1])
    return g, a


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ds, c in [("fashion_mnist", "tab:purple"), ("cifar10", "tab:red")]:
    g, a = profile(ds)
    lbl = f"{ds} (eta={ETA[ds]})"
    axes[0].plot(range(len(a)), a, "o-", ms=3, color=c, label=lbl)
    axes[1].plot(range(len(g)), g, "o-", ms=3, color=c, label=lbl)
    print(f"{ds} eta={ETA[ds]}: peak act RMS = {a.max():.3e} at layer {int(a.argmax())} | "
          f"grad ratio {g.max()/g.min():.1e}x")

axes[0].axhline(3.4e38, ls=":", color="black")
axes[0].text(1, 3.4e38, " float32 overflow ceiling", fontsize=8, va="bottom")
axes[0].set_yscale("log"); axes[0].set_xlabel("hidden layer index")
axes[0].set_ylabel("forward activation RMS (float64)")
axes[0].set_title("Forward activation bulge at init (V2 NoBN, L=50)\n"
                  "stays under float32 ceiling -> survives init, but diverges/stalls under SGD")
axes[0].legend(); axes[0].grid(alpha=0.3, which="both")

axes[1].set_yscale("log"); axes[1].set_xlabel("hidden layer index")
axes[1].set_ylabel("gradient row norm (float64)")
axes[1].set_title("Per-layer gradient row norm at init (L=50)")
axes[1].legend(); axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
out = "reports/figures/v2_nobn_50L_init_profile.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
