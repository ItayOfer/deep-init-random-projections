#!/usr/bin/env python3
"""Decompose the rcfwd recipe into its two independent interventions.

Campaign 09's "corrected recipe" bundles TWO changes, and campaigns 09/10 only
ever ran two of the four corners:

  (A) initialization: row_centered_he -> row_centered_forward_balanced
      (weights scaled by ~1/r per layer, which flattens the FORWARD pass)
  (B) _GradRescale r after each hidden ReLU
      (identity forward, x r backward, which flattens the BACKWARD pass)

  corner              init                             grad_rescale   ran in
  ------------------  -------------------------------  -------------  ---------------
  raw                 row_centered_he                  None           c09, c10 "raw"
  raw+rescale         row_centered_he                  r              NEVER
  fwdbal              row_centered_forward_balanced    None           NEVER
  rcfwd               row_centered_forward_balanced    r              c09, c10 "rcfwd"

Because both knobs moved at once, campaign 10's conclusion -- that `last3`'s
DEAD -> LEARNING flip was a forward-SCALE artifact rather than a content
difference -- is an inference, not a measurement. This script measures the
2x2 at initialization: per-layer activation RMS (what (A) controls) and
per-layer parameter-gradient norm (what the combination controls).

It also answers the advisor's pre-registered prediction for the backward-only
corner directly: _GradRescale is identity in the forward pass, so `raw+rescale`
has activations bit-identical to `raw` -- "they will probably be small" is true
by construction, and this quantifies how small.

Usage:  python scripts/recipe_decomposition_funnel.py [--depth 100] [--width 500]
Output: reports/results/recipe_decomposition_funnel.json
        reports/figures/rc_frozen_ends/recipe_decomposition_funnel.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.models.classifiers import DeepFCClassifier  # noqa: E402

R = math.sqrt((math.pi - 1.0) / math.pi)  # ~0.8256, the gain-coupling ratio

CORNERS = {
    "raw":         {"init": "row_centered_he",               "grad_rescale": None},
    "raw+rescale": {"init": "row_centered_he",               "grad_rescale": R},
    "fwdbal":      {"init": "row_centered_forward_balanced", "grad_rescale": None},
    "rcfwd":       {"init": "row_centered_forward_balanced", "grad_rescale": R},
}


def measure(corner, depth, width, in_dim, n_classes, batch, seed):
    cfg = CORNERS[corner]
    torch.manual_seed(seed)
    model = DeepFCClassifier(
        in_dim, width, depth, n_classes,
        init_strategy=cfg["init"], use_batch_norm=False,
        grad_rescale=cfg["grad_rescale"],
    )

    torch.manual_seed(0)  # same probe batch for every corner
    x = torch.randn(batch, in_dim)
    y = torch.randint(0, n_classes, (batch,))

    act_rms = []
    with torch.no_grad():
        h = x
        for linear in model.hidden_layers:
            h = torch.relu(linear(h))
            act_rms.append(h.pow(2).mean().sqrt().item())
        logit_rms = model.classifier(h).pow(2).mean().sqrt().item()

    model.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    grad_norm = [model.hidden_layers[i].weight.grad.norm().item() for i in range(depth)]

    positive = [g for g in grad_norm if g > 0]
    ratio = (max(grad_norm) / min(positive)) if positive else float("inf")
    return {
        "init_strategy": cfg["init"],
        "grad_rescale": cfg["grad_rescale"],
        "activation_rms": act_rms,
        "logit_rms": logit_rms,
        "grad_norm_per_layer": grad_norm,
        "grad_max_over_min": ratio,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--input-dim", type=int, default=784)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    results = {
        name: measure(name, args.depth, args.width, args.input_dim,
                      args.num_classes, args.batch, args.seed)
        for name in CORNERS
    }
    payload = {
        "description": "2x2 decomposition of the rcfwd recipe at initialization",
        "config": vars(args) | {"r": R},
        "corners": results,
    }
    out = ROOT / "reports" / "results" / "recipe_decomposition_funnel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    probe = [p for p in (1, 2, 3, 25, 50, 75, 98, 99, 100) if p <= args.depth]
    for title, key in (("ACTIVATION RMS (forward -- grad_rescale cannot change this)", "activation_rms"),
                       ("PARAMETER-GRADIENT NORM ||dL/dW_l||", "grad_norm_per_layer")):
        print(f"\n{title}")
        print(f"{'corner':<14}" + "".join(f"{'L' + str(p):>11}" for p in probe)
              + (f"{'max/min':>12}" if key == "grad_norm_per_layer" else ""))
        for name, res in results.items():
            row = f"{name:<14}" + "".join(f"{res[key][p - 1]:>11.2e}" for p in probe)
            if key == "grad_norm_per_layer":
                row += f"{res['grad_max_over_min']:>12.2e}"
            print(row)
    print(f"\nSaved {out.relative_to(ROOT)}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = range(1, args.depth + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for name, res in results.items():
        axes[0].semilogy(layers, res["activation_rms"], label=name)
        axes[1].semilogy(layers, [max(g, 1e-30) for g in res["grad_norm_per_layer"]], label=name)
    axes[0].set(xlabel="layer $\\ell$", ylabel="activation RMS",
                title="Forward: set by the initialization only")
    axes[1].set(xlabel="layer $\\ell$", ylabel="$\\|\\partial L/\\partial W_\\ell\\|$",
                title="Backward: set by initialization $\\times$ grad_rescale")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(f"rcfwd recipe decomposed, {args.depth}L width {args.width}, at initialization")
    fig.tight_layout()
    fig_path = ROOT / "reports" / "figures" / "rc_frozen_ends" / "recipe_decomposition_funnel.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
