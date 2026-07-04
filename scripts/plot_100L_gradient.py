#!/usr/bin/env python3
"""Plot why the V2 NoBN L=100 (eta=0.36) runs NaN'd: the forward activation
bulge crosses the float32 ceiling at init, and the per-layer gradient profile.

The actual runs aborted on batch ~8 of epoch 1 (no completed epoch -> empty
history), so there is no per-epoch gradient data. The informative picture is
the INITIALISATION profile, computed here in float64 so the true bulge shape
is visible past where float32 overflows.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

ETA, L, WIDTH = 0.36, 100, 500
IDIM = {"fashion_mnist": 784, "cifar10": 3072}


def profile(dataset):
    ls = [IDIM[dataset]] + [WIDTH] * L + [1]
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls,
                       init_strategy="row_centered_layer_balanced_product_base",
                       init_kwargs={"eta": ETA})
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    g = list(res.get_mean_row_norms().values())[:-1]
    a = list(res.get_activation_rms().values())[:-1]
    return np.array(g), np.array(a)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ds, c in [("fashion_mnist", "tab:purple"), ("cifar10", "tab:red")]:
    g, a = profile(ds)
    axes[0].plot(range(len(a)), a, "o-", ms=2.5, color=c, label=ds)
    axes[1].plot(range(len(g)), g, "o-", ms=2.5, color=c, label=ds)
    print(f"{ds}: peak act RMS = {a.max():.3e} at layer {int(a.argmax())} | "
          f"grad row norm [{g.min():.2e}, {g.max():.2e}] ratio {g.max()/g.min():.1e}x")

axes[0].axhline(3.4e38, ls=":", color="black")
axes[0].text(2, 3.4e38, " float32 overflow ceiling", fontsize=8, va="bottom")
axes[0].set_yscale("log"); axes[0].set_xlabel("hidden layer index")
axes[0].set_ylabel("forward activation RMS (float64)")
axes[0].set_title("Forward activation bulge at init (V2 NoBN, L=100, eta=0.36)\n"
                  "peaks mid-network above the float32 ceiling -> run NaNs there")
axes[0].legend(); axes[0].grid(alpha=0.3, which="both")

axes[1].set_yscale("log"); axes[1].set_xlabel("hidden layer index")
axes[1].set_ylabel("gradient row norm (float64)")
axes[1].set_title("Per-layer gradient row norm at init (L=100, eta=0.36)")
axes[1].legend(); axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
out = "reports/figures/v2_eta_nobn/v2_nobn_100L_init_profile.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
