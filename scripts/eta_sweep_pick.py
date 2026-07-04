#!/usr/bin/env python3
"""Post-process eta_sweep_research.json: derive eta* per architecture under
a *forward-safety-aware* criterion, and plot the gradient-ratio / overflow
tradeoff.

Bug this fixes: the sweep's inline "best" picked argmin(grad_ratio) among
configs whose gradient row norms were finite -- but at high eta the forward
activations overflow (peak_act_rms = inf) while the gradient row norms can
still come back finite-but-garbage. Such an eta is not trainable. Here we
require the forward pass to be numerically healthy.

We report eta* under two criteria:
  * min-ratio (finite fwd): argmin grad_ratio among eta whose peak_act_rms
    is finite. Aggressive -- can sit right at the overflow cliff.
  * safe: argmin grad_ratio among eta whose peak_act_rms < SAFE_THRESHOLD.
    Conservative -- leaves headroom for 200-epoch NoBN training drift.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SAFE_THRESHOLD = 1e6     # init peak activation RMS we consider training-safe
DATASETS = ["fashion_mnist", "cifar10"]
DEPTHS = [30, 50, 100]
COLORS = {30: "tab:green", 50: "tab:orange", 100: "tab:red"}


def main():
    data = json.loads((ROOT / "reports/results/eta_sweep_research.json").read_text())
    sweep = data["sweep"]

    rows = []
    for dataset in DATASETS:
        for depth in DEPTHS:
            ent = sweep[f"{dataset}_{depth}L"]
            fin = [e for e in ent
                   if e["finite"] and math.isfinite(e["grad_ratio"])
                   and math.isfinite(e["peak_act_rms"])]
            safe = [e for e in fin if e["peak_act_rms"] < SAFE_THRESHOLD]

            b_min = min(fin, key=lambda e: e["grad_ratio"]) if fin else None
            b_safe = min(safe, key=lambda e: e["grad_ratio"]) if safe else None
            rows.append((dataset, depth, b_min, b_safe,
                         max((e["eta"] for e in fin), default=float("nan")),
                         max((e["eta"] for e in safe), default=float("nan"))))

    # ---- summary table ----
    print(f"{'='*108}")
    print(f"ETA* per architecture  (SAFE_THRESHOLD on init peak activation RMS = {SAFE_THRESHOLD:.0e})")
    print(f"{'='*108}")
    print(f"{'architecture':>16s} | {'MIN-RATIO (finite fwd)':^34s} | {'SAFE (peak_act < thr)':^34s}")
    print(f"{'':>16s} | {'eta*':>6s} {'grad_ratio':>11s} {'peak_actRMS':>13s} | "
          f"{'eta*':>6s} {'grad_ratio':>11s} {'peak_actRMS':>13s}")
    print("-" * 108)
    for dataset, depth, b_min, b_safe, eta_fin_max, eta_safe_max in rows:
        key = f"{dataset}_{depth}L"
        mn = (f"{b_min['eta']:>6.2f} {b_min['grad_ratio']:>10.1f}x {b_min['peak_act_rms']:>13.2e}"
              if b_min else f"{'--':>6s} {'--':>11s} {'--':>13s}")
        sf = (f"{b_safe['eta']:>6.2f} {b_safe['grad_ratio']:>10.1f}x {b_safe['peak_act_rms']:>13.2e}"
              if b_safe else f"{'--':>6s} {'NONE':>11s} {'--':>13s}")
        print(f"{key:>16s} | {mn} | {sf}")
    print("-" * 108)
    print("Note: 'SAFE = NONE' means even eta=0 overflows the safe threshold at that depth"
          " (deep NoBN is fundamentally overflow-prone).")

    # ---- recommendation (what we'll train with) ----
    print(f"\n{'#'*70}\nRECOMMENDED eta* FOR TRAINING (safe if available, else min-ratio)\n{'#'*70}")
    recommend = {}
    for dataset, depth, b_min, b_safe, *_ in rows:
        key = f"{dataset}_{depth}L"
        chosen = b_safe if b_safe else b_min
        recommend[key] = chosen["eta"]
        tag = "safe" if b_safe else "min-ratio (no safe eta exists)"
        print(f"  {key:>16s}: eta* = {chosen['eta']:.2f}  "
              f"(grad_ratio {chosen['grad_ratio']:.1f}x, peak_actRMS {chosen['peak_act_rms']:.1e})  [{tag}]")
    (ROOT / "reports/results/eta_star_recommended.json").write_text(
        json.dumps(recommend, indent=2))
    print(f"\nSaved recommended eta* -> reports/results/eta_star_recommended.json")

    # ---- plot: grad_ratio vs eta + overflow region, per dataset ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, dataset in zip(axes, DATASETS):
        for depth in DEPTHS:
            ent = sweep[f"{dataset}_{depth}L"]
            fin = [e for e in ent
                   if e["finite"] and math.isfinite(e["grad_ratio"])
                   and math.isfinite(e["peak_act_rms"])]
            xs = [e["eta"] for e in fin]
            ys = [e["grad_ratio"] for e in fin]
            ax.plot(xs, ys, "o-", color=COLORS[depth], ms=4, label=f"L={depth}")
            # safe optimum marker
            safe = [e for e in fin if e["peak_act_rms"] < SAFE_THRESHOLD]
            if safe:
                b = min(safe, key=lambda e: e["grad_ratio"])
                ax.scatter([b["eta"]], [b["grad_ratio"]], s=150, marker="*",
                           color=COLORS[depth], edgecolors="black", linewidths=0.8, zorder=6)
            # overflow cliff: first eta exceeding the safe threshold
            unsafe = [e["eta"] for e in fin if e["peak_act_rms"] >= SAFE_THRESHOLD]
            if unsafe:
                ax.axvline(min(unsafe), color=COLORS[depth], ls=":", alpha=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("eta")
        ax.set_ylabel("empirical gradient ratio (max/min hidden grad row norm)")
        ax.set_title(f"{dataset}  (V2 NoBN, width 500)\n"
                     f"star = safe optimum; dotted line = overflow onset (peak_actRMS > {SAFE_THRESHOLD:.0e})")
        ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = ROOT / "reports/figures/eta_sweep_safe_tradeoff.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
