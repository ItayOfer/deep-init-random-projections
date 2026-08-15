#!/usr/bin/env python3
"""Pre-screen for the post-ReLU DC-removal family (campaign 11), at initialization.

Idea under test: ReLU output is non-negative, so E[a] = rms(a)/sqrt(pi) > 0.
That positive DC component -- shared by every sample -- is the engine of the
arc-cosine kernel's rho -> 1 collapse. Subtracting it AFTER the ReLU attacks
the mechanism directly:

    a = relu(W x) - c * rms(relu(W x))

Relation to the row-centered family: the next layer computes
    W (a - c 1) = W a - c (W 1)
so if W is row-centered (W 1 = 0) the subtraction is exactly a no-op. The
shift family is therefore the ACTIVATION-SPACE DUAL of row-centering, applied
to unconstrained He weights: kill the DC on the way out instead of on the way
in. c = 1/sqrt(pi) ~ 0.5642 removes exactly E[a]; c is swept because the
constant is not arbitrary -- overshooting replaces a positive shared DC with a
negative one, which drives cosine back toward 1 just as hard.

Screens each candidate on the three thesis requirements at init, before any
cluster time is spent:
  (i)   no geometric collapse   -> mean pairwise cosine; distance correlation
                                   vs the INPUT (the repo's own metric --
                                   "cosine far from 1" is not the goal,
                                   matching the input geometry is)
  (ii)  gradient/forward health -> activation RMS per layer, implied per-layer
                                   forward gain, dataset-dead unit fraction
  (iii) preserved class content -> cosine k-NN accuracy per layer
                                   (scale-invariant, as in campaign 09)

Usage:  python scripts/relu_shift_geometry_screen.py [--depth 60] [--width 500]
Output: reports/results/relu_shift_geometry_screen.json
        reports/figures/relu_shift/relu_shift_geometry_screen.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.analysis.geometry_metrics import pairwise_distance_correlation  # noqa: E402
from rp_study.data.loaders import load_fashion_mnist, load_cifar10  # noqa: E402
from rp_study.models.initializers import initialize_layer  # noqa: E402

DC = 1.0 / math.sqrt(math.pi)  # E[a]/rms(a) for a half-Gaussian: the exact DC


def build(depth, width, in_dim, init_strategy, seed):
    torch.manual_seed(seed)
    layers = []
    for i in range(depth):
        linear = nn.Linear(in_dim if i == 0 else width, width, bias=True)
        initialize_layer(linear, strategy=init_strategy, layer_index=i, n_layers=depth + 1)
        layers.append(linear)
    return layers


def forward_trace(layers, X, shift_c):
    """Return [(pre_activation, activation)] per layer. shift_c=None -> no shift."""
    trace = []
    with torch.no_grad():
        h = X
        for linear in layers:
            pre = linear(h)
            h = torch.relu(pre)
            if shift_c is not None:
                h = h - shift_c * h.pow(2).mean().sqrt()
            trace.append((pre, h.clone()))
    return trace


def mean_pairwise_cosine(h):
    hn = h / h.norm(dim=1, keepdim=True).clamp(min=1e-30)
    gram = hn @ hn.T
    iu = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1)
    return gram[iu[0], iu[1]].mean().item()


def cosine_knn_accuracy(h, y, k=10, n_classes=10):
    hn = (h / h.norm(dim=1, keepdim=True).clamp(min=1e-30)).numpy()
    sim = hn @ hn.T
    np.fill_diagonal(sim, -np.inf)
    neighbours = np.argsort(-sim, axis=1)[:, :k]
    pred = np.array([np.bincount(y[row], minlength=n_classes).argmax() for row in neighbours])
    return float((pred == y).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", type=float, nargs="*",
                        default=[0.25, DC, 0.75, 1.0])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    loader = load_fashion_mnist if args.dataset == "fashion_mnist" else load_cifar10
    X, y = loader(data_dir=str(ROOT / "data"), num_samples=args.samples,
                  flatten=True, normalize=True, seed=0)
    X, y = X.float(), y.numpy()

    candidates = [("he", "he", None), ("row_centered_he", "row_centered_he", None)]
    candidates += [(f"he_shift_c{c:.4f}", "he", c) for c in args.c_grid]

    layers_cache = {}
    results = {}
    for name, init_strategy, c in candidates:
        if init_strategy not in layers_cache:
            layers_cache[init_strategy] = build(args.depth, args.width, X.shape[1],
                                                init_strategy, args.seed)
        trace = forward_trace(layers_cache[init_strategy], X, c)
        act_rms = [h.pow(2).mean().sqrt().item() for _, h in trace]
        results[name] = {
            "init_strategy": init_strategy,
            "shift_c": c,
            "mean_pairwise_cosine": [mean_pairwise_cosine(h) for _, h in trace],
            "cosine_knn_accuracy": [cosine_knn_accuracy(h, y, args.knn_k) for _, h in trace],
            "activation_rms": act_rms,
            "dataset_dead_fraction": [
                (pre.max(dim=0).values <= 0).float().mean().item() for pre, _ in trace
            ],
            "implied_forward_gain": act_rms[-1] ** (1.0 / args.depth) if act_rms[-1] > 0 else 0.0,
        }

    payload = {
        "description": "Init-time screen of the post-ReLU DC-removal family on the three requirements",
        "config": vars(args) | {"exact_dc_constant": DC},
        "input_baseline": {
            "mean_pairwise_cosine": mean_pairwise_cosine(X),
            "cosine_knn_accuracy": cosine_knn_accuracy(X, y, args.knn_k),
        },
        "candidates": results,
    }
    out = ROOT / "reports" / "results" / "relu_shift_geometry_screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    probe = sorted({1, 2, 5, 10, 20, args.depth // 2, args.depth})
    probe = [p for p in probe if 1 <= p <= args.depth]
    base = payload["input_baseline"]
    print(f"input: mean pairwise cos = {base['mean_pairwise_cosine']:.4f}   "
          f"cosine-{args.knn_k}NN = {base['cosine_knn_accuracy']:.4f}\n")
    panels = [
        ("MEAN PAIRWISE COSINE (1.0 = collapse)", "mean_pairwise_cosine", "{:>9.4f}"),
        (f"COSINE {args.knn_k}-NN ACCURACY (0.10 = chance)", "cosine_knn_accuracy", "{:>9.4f}"),
        ("DATASET-DEAD UNIT FRACTION", "dataset_dead_fraction", "{:>9.3f}"),
        ("ACTIVATION RMS", "activation_rms", "{:>9.2e}"),
    ]
    for title, key, fmt in panels:
        print(title)
        print(f"{'candidate':<22}" + "".join(f"{'L' + str(p):>9}" for p in probe))
        for name, res in results.items():
            print(f"{name:<22}" + "".join(fmt.format(res[key][p - 1]) for p in probe))
        print()
    print("IMPLIED PER-LAYER FORWARD GAIN")
    for name, res in results.items():
        print(f"  {name:<22} {res['implied_forward_gain']:.4f}")

    print("\nDISTANCE CORRELATION vs INPUT (Spearman; 1.0 = input geometry preserved)")
    print(f"{'candidate':<22}" + "".join(f"{'L' + str(p):>9}" for p in probe))
    Xn = X.numpy()
    for name, init_strategy, c in candidates:
        trace = forward_trace(layers_cache[init_strategy], X, c)
        row = [pairwise_distance_correlation(Xn, trace[p - 1][1].numpy(), n_pairs=4000, seed=0)
               for p in probe]
        results[name]["distance_correlation_probe_layers"] = dict(zip(map(str, probe), row))
        print(f"{name:<22}" + "".join(f"{v:>9.4f}" for v in row))
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ell = range(1, args.depth + 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for name, res in results.items():
        axes[0].plot(ell, res["mean_pairwise_cosine"], label=name)
        axes[1].plot(ell, res["cosine_knn_accuracy"], label=name)
        axes[2].plot(ell, res["dataset_dead_fraction"], label=name)
        axes[3].semilogy(ell, [max(v, 1e-30) for v in res["activation_rms"]], label=name)
    axes[0].axhline(base["mean_pairwise_cosine"], ls=":", c="k", lw=1, label="input")
    axes[0].set(xlabel="layer $\\ell$", ylabel="mean pairwise cosine", title="(i) geometry")
    axes[1].axhline(0.1, ls=":", c="k", lw=1, label="chance")
    axes[1].set(xlabel="layer $\\ell$", ylabel=f"cosine {args.knn_k}-NN accuracy",
                title="(iii) class content")
    axes[2].axhline(0.5, ls=":", c="k", lw=1, label="1/2")
    axes[2].set(xlabel="layer $\\ell$", ylabel="dataset-dead fraction", title="dead units")
    axes[3].set(xlabel="layer $\\ell$", ylabel="activation RMS", title="(ii) forward scale")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Post-ReLU DC removal, {args.depth}L width {args.width}, "
                 f"{args.dataset}, at initialization")
    fig.tight_layout()
    fig_path = ROOT / "reports" / "figures" / "relu_shift" / "relu_shift_geometry_screen.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
