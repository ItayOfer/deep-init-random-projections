#!/usr/bin/env python3
"""Activation-bulge parabola at L=30, 50, 100 (each at its ratio-minimising
eta), overlaid per dataset with the float32 overflow ceiling. Shows why depth
kills V2 NoBN: the bulge peak climbs with depth and nearly touches the ceiling
at L=100, so a few SGD steps tip it over -> NaN.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

WIDTH = 500
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
# (depth, eta) per dataset -- the ratio-minimising etas we train with
CONFIGS = {
    "fashion_mnist": [(30, 0.60), (50, 0.80), (100, 0.36)],
    "cifar10":       [(30, 0.85), (50, 0.90), (100, 0.36)],
}
COLORS = {30: "tab:green", 50: "tab:orange", 100: "tab:red"}


def act_rms(dataset, L, eta):
    ls = [IDIM[dataset]] + [WIDTH] * L + [1]
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    gc = GradientExperimentConfig(num_samples=512, dataset=dataset)
    nc = NetworkConfig(layer_sizes=ls,
                       init_strategy="row_centered_layer_balanced_product_base",
                       init_kwargs={"eta": eta})
    exp = GradientExperiment(ec, nc, gc)
    exp.net = exp.net.double(); exp.inputs = exp.inputs.double(); exp.targets = exp.targets.double()
    res = exp.run()
    return np.array(list(res.get_activation_rms().values())[:-1])


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, ds in zip(axes, ["fashion_mnist", "cifar10"]):
    for L, eta in CONFIGS[ds]:
        a = act_rms(ds, L, eta)
        x = np.linspace(0, 1, len(a))           # fractional depth so curves overlay
        ax.plot(x, a, "o-", ms=2.5, color=COLORS[L], label=f"L={L} (eta={eta}), peak 1e{np.log10(a.max()):.0f}")
        print(f"{ds} L={L} eta={eta}: peak act RMS = {a.max():.2e}")
    ax.axhline(3.4e38, ls=":", color="black", lw=1.5)
    ax.text(0.02, 3.4e38, " float32 overflow ceiling (3.4e38)", fontsize=8, va="bottom")
    ax.set_yscale("log"); ax.set_ylim(1e-1, 1e42)
    ax.set_xlabel("fractional depth (0 = input side, 1 = output side)")
    ax.set_title(f"{ds}: activation bulge by depth (V2 NoBN, width 500)")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("forward activation RMS at init (float64)")
fig.suptitle("Why depth kills V2 NoBN: the activation bulge peak climbs with depth and "
             "nears the float32 ceiling at L=100", y=1.02, fontsize=13)
fig.tight_layout()
out = "reports/figures/v2_eta_nobn/v2_nobn_bulge_by_depth.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
