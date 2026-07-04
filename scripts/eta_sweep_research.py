#!/usr/bin/env python3
"""Research: find the eta that minimizes the V2 per-layer gradient ratio,
per architecture (depth), for the NoBN training family.

Method
------
For each (dataset, depth), sweep eta over a fine grid. For each eta:
  * build V2 (row_centered_layer_balanced_product_base) NoBN net, width 500
  * run one forward+backward pass (GradientExperiment, MSE/1-output proxy,
    same framework notebook 11 uses)
  * measure the empirical gradient ratio = max/min of the per-layer
    gradient row norms over the HIDDEN layers (output excluded)
  * record numerical health: is the loss finite? are all grad row norms
    finite? what is the peak per-layer activation RMS (overflow proxy)?

The theoretical gradient ratio G(eta) = r^{-(1-eta)(L-1)} is monotone in
eta (minimised at eta=1). But without BN, raising eta inflates the early-
layer weight std as r^{-eta(L-1)/2}; at deep L the forward activations
overflow float32 and the measured gradients become Inf/NaN. So the
*usable* optimum is the largest eta whose forward pass stays finite, and
that ceiling falls with depth. We report eta* = argmin(empirical ratio)
restricted to numerically-healthy eta.

Output
------
  reports/results/eta_sweep_research.json   -- full sweep data
  reports/figures/eta_sweep/eta_sweep_ratio_curves.png -- G(eta) per depth/dataset
  reports/figures/eta_sweep/eta_sweep_perlayer_<dataset>.png -- per-layer grad row
        norms at eta* for each depth (the notebook-11-style figure)
"""

import sys
import json
import math
import ssl
from pathlib import Path

# macOS python sometimes lacks CA certs for torchvision downloads; data is
# normally cached, but guard the download path just in case.
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments.gradient_analysis import GradientExperiment

SEED = 42
DATA_DIR = str(ROOT / "data")
NUM_SAMPLES = 2000
WIDTH = 500                     # matches every prior V2 / He audit
DEPTHS = [30, 50, 100]
DATASETS = ["fashion_mnist", "cifar10"]
INPUT_DIMS = {"mnist": 784, "fashion_mnist": 784, "cifar10": 3072}
INIT = "row_centered_layer_balanced_product_base"

r = math.sqrt((math.pi - 1.0) / math.pi)        # ~0.826
s_star = (math.pi / (math.pi - 1.0)) ** 0.25     # ~1.1006

# Fine grid; dense at the low end because the deep nets overflow early.
ETAS = [round(x, 3) for x in np.concatenate([
    np.arange(0.00, 0.50, 0.02),
    np.arange(0.50, 1.001, 0.05),
])]


def run_one(dataset, depth, eta):
    """One V2 NoBN gradient-analysis pass. Returns a dict of measurements."""
    input_dim = INPUT_DIMS[dataset]
    layer_sizes = [input_dim] + [WIDTH] * depth + [1]

    exp_config = ExperimentConfig(seed=SEED, data_dir=DATA_DIR)
    exp_config.setup_seeds()
    grad_config = GradientExperimentConfig(num_samples=NUM_SAMPLES, dataset=dataset)
    net_config = NetworkConfig(
        layer_sizes=layer_sizes, init_strategy=INIT, init_kwargs={"eta": eta},
    )
    exp = GradientExperiment(exp_config, net_config, grad_config)
    result = exp.run()

    norms = result.get_mean_row_norms()
    hidden_vals = list(norms.values())[:-1]          # exclude output layer
    finite = bool(np.all(np.isfinite(hidden_vals))) and math.isfinite(result.loss_value)

    if finite and min(hidden_vals) > 0:
        grad_ratio = max(hidden_vals) / min(hidden_vals)
    else:
        grad_ratio = float("inf")

    # forward overflow proxy: peak per-layer activation RMS
    act_rms = list(result.get_activation_rms().values())
    peak_act_rms = max(act_rms) if act_rms and np.all(np.isfinite(act_rms)) else float("inf")

    weight_stds = [l.weight.std().item() for l in exp.net.layers[:-1]]
    wt_ratio = (max(weight_stds) / min(weight_stds)) if min(weight_stds) > 0 else float("inf")

    return {
        "eta": eta,
        "finite": finite,
        "grad_ratio": grad_ratio,
        "weight_std_ratio": wt_ratio,
        "peak_act_rms": peak_act_rms,
        "loss": result.loss_value,
        "hidden_norms": [float(v) for v in hidden_vals],
        "theo_G": r ** (-(1 - eta) * (depth - 1)),
        "theo_V": r ** (-eta * (depth - 1)),
    }


def main():
    all_data = {}
    best = {}   # (dataset, depth) -> best entry

    for dataset in DATASETS:
        for depth in DEPTHS:
            key = f"{dataset}_{depth}L"
            print(f"\n{'='*70}\n{key}  (V2 NoBN, width={WIDTH}, {NUM_SAMPLES} samples)\n{'='*70}")
            print(f"{'eta':>5s} {'finite':>7s} {'emp_G':>12s} {'theo_G':>12s} "
                  f"{'wt_ratio':>10s} {'peak_actRMS':>12s}")
            entries = []
            for eta in ETAS:
                try:
                    e = run_one(dataset, depth, eta)
                except Exception as exc:                     # numerical blowups
                    e = {"eta": eta, "finite": False, "grad_ratio": float("inf"),
                         "weight_std_ratio": float("inf"), "peak_act_rms": float("inf"),
                         "loss": float("nan"), "hidden_norms": [],
                         "theo_G": r ** (-(1 - eta) * (depth - 1)),
                         "theo_V": r ** (-eta * (depth - 1)), "error": str(exc)}
                entries.append(e)
                print(f"{e['eta']:>5.2f} {str(e['finite']):>7s} "
                      f"{e['grad_ratio']:>12.2e} {e['theo_G']:>12.2e} "
                      f"{e['weight_std_ratio']:>10.2e} {e['peak_act_rms']:>12.2e}")

            healthy = [e for e in entries if e["finite"] and math.isfinite(e["grad_ratio"])]
            if healthy:
                b = min(healthy, key=lambda e: e["grad_ratio"])
            else:
                b = min(entries, key=lambda e: e["eta"])   # fallback
            best[key] = b
            all_data[key] = entries
            print(f"--> BEST eta* = {b['eta']:.2f}  emp_G = {b['grad_ratio']:.2f}x  "
                  f"wt_ratio = {b['weight_std_ratio']:.1f}x  "
                  f"(healthy eta range: {[e['eta'] for e in healthy][:1]}..{[e['eta'] for e in healthy][-1:]} )")

    # ---- save raw data ----
    out_json = ROOT / "reports" / "results" / "eta_sweep_research.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"config": {"width": WIDTH, "num_samples": NUM_SAMPLES, "seed": SEED,
                    "etas": ETAS, "depths": DEPTHS, "datasets": DATASETS},
         "sweep": all_data,
         "best": {k: {kk: vv for kk, vv in v.items() if kk != "hidden_norms"}
                  for k, v in best.items()}},
        indent=2))
    print(f"\nSaved sweep data -> {out_json}")

    # ---- summary table ----
    print(f"\n\n{'#'*70}\nETA* SUMMARY (gradient-ratio-minimising eta per architecture)\n{'#'*70}")
    print(f"{'architecture':>18s} {'eta*':>6s} {'emp_G':>10s} {'wt_std_ratio':>14s} "
          f"{'peak_actRMS':>12s}")
    for key in [f"{d}_{L}L" for d in DATASETS for L in DEPTHS]:
        b = best[key]
        print(f"{key:>18s} {b['eta']:>6.2f} {b['grad_ratio']:>9.1f}x "
              f"{b['weight_std_ratio']:>13.1f}x {b['peak_act_rms']:>12.2e}")

    # ---- plot 1: G(eta) curves per depth, one panel per dataset ----
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5))
    if len(DATASETS) == 1:
        axes = [axes]
    colors = {30: "tab:green", 50: "tab:orange", 100: "tab:red"}
    for ax, dataset in zip(axes, DATASETS):
        for depth in DEPTHS:
            ent = all_data[f"{dataset}_{depth}L"]
            xs = [e["eta"] for e in ent if e["finite"] and math.isfinite(e["grad_ratio"])]
            ys = [e["grad_ratio"] for e in ent if e["finite"] and math.isfinite(e["grad_ratio"])]
            ax.plot(xs, ys, "o-", color=colors[depth], ms=4, label=f"L={depth}")
            b = best[f"{dataset}_{depth}L"]
            ax.scatter([b["eta"]], [b["grad_ratio"]], s=130, facecolors="none",
                       edgecolors=colors[depth], linewidths=2, zorder=5)
            ax.annotate(f"eta*={b['eta']:.2f}", (b["eta"], b["grad_ratio"]),
                        textcoords="offset points", xytext=(4, 8),
                        fontsize=9, color=colors[depth])
        ax.set_yscale("log")
        ax.set_xlabel("eta")
        ax.set_ylabel("empirical gradient ratio (max/min hidden row norm)")
        ax.set_title(f"{dataset}  (V2 NoBN, width 500)\ncircles = ratio-minimising eta "
                     f"(curves stop where forward overflows)")
        ax.axhline(1.0, ls=":", color="gray", alpha=0.6)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out1 = ROOT / "reports" / "figures" / "eta_sweep_ratio_curves.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out1}")

    # ---- plot 2: per-layer grad row norms at eta* (notebook-11 style) ----
    for dataset in DATASETS:
        fig, axes = plt.subplots(len(DEPTHS), 1, figsize=(13, 3.3 * len(DEPTHS)))
        for ax, depth in zip(axes, DEPTHS):
            b = best[f"{dataset}_{depth}L"]
            vals = b["hidden_norms"]
            ax.plot(range(len(vals)), vals, "o-", ms=3, color=colors[depth], lw=1.4)
            ax.fill_between(range(len(vals)), vals, alpha=0.15, color=colors[depth])
            ax.set_yscale("log")
            ratio = (max(vals) / min(vals)) if vals and min(vals) > 0 else float("inf")
            ax.set_title(f"{dataset} L={depth}  eta*={b['eta']:.2f}  |  "
                         f"max={max(vals):.1f}, min={min(vals):.2g}, ratio={ratio:.1f}x",
                         loc="left", fontsize=11)
            ax.set_ylabel("grad row norm")
            ax.grid(alpha=0.3, which="both")
        axes[-1].set_xlabel("hidden layer index")
        fig.suptitle(f"Per-layer gradient row norms at ratio-minimising eta* "
                     f"({dataset}, V2 NoBN, width 500)", y=1.005, fontsize=13)
        fig.tight_layout()
        out2 = ROOT / "reports" / "figures" / f"eta_sweep_perlayer_{dataset}.png"
        fig.savefig(out2, dpi=130, bbox_inches="tight")
        print(f"Saved -> {out2}")


if __name__ == "__main__":
    main()
