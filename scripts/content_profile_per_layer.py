#!/usr/bin/env python3
"""Per-layer content profile at init, width 500 (matching the trained nets).

Final-layer probes (Euclidean, cosine, linear — see content_probe_linear.py)
find chance-level class structure in rcfwd-init representations at every
trained depth, yet fmnist/30L trains to 1.0. This measures WHERE the content
dies: linear-probe accuracy as a function of layer index l.

Because row_centered_forward_balanced (and he) do not depend on total depth,
and the seed fixes the weight stream, the first l layers are identical across
network depths — one profile describes the 30L, 50L and 100L nets alike. The
trainability difference between depths is then the length of the noise tail
past the last informative layer, not the (identical) informative prefix.

Output: reports/results/content_profile_per_layer.json +
reports/figures/rcfwd_rescale/content_profile_per_layer.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from rp_study.data import get_data_loader
from rp_study.projections import multi_layer_rp_with_init

SEED = 42
N = 2000
WIDTH = 500  # matches every rcfwd training run
K = 10
LAYERS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 70, 100]
STRATEGIES = ["he", "row_centered_forward_balanced"]


def cosine_knn_acc(X, y):
    # cosine metric: scale-invariant — Euclidean k-NN is blind on row-centered
    # representations (per-sample scale variation destroys neighborhoods)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.25, random_state=SEED, stratify=y)
    return KNeighborsClassifier(n_neighbors=K, metric="cosine").fit(Xtr, ytr).score(Xte, yte)


def linear_probe_acc(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=SEED, stratify=y)
    return LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr).score(Xte, yte)


rows = []
for dataset in ["fashion_mnist", "cifar10"]:
    X, y = get_data_loader(dataset_name=dataset, data_dir=str(ROOT / "data"), train=True,
                           num_samples=N, flatten=True, as_numpy=True, normalize=True, seed=SEED)
    for strategy in STRATEGIES:
        for l in LAYERS:
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            Xp = multi_layer_rp_with_init(X, l, init_strategy=strategy, width=WIDTH, seed=SEED)
            knn = float(cosine_knn_acc(Xp, y))
            lin = float(linear_probe_acc(Xp, y))
            rows.append({"dataset": dataset, "init_strategy": strategy, "layer": l,
                         "width": WIDTH, "cosine_knn_accuracy": knn, "linear_probe_accuracy": lin})
            print(f"{dataset:14} {strategy:>30} l={l:>3}  cos-kNN={knn:.3f}  linear={lin:.3f}", flush=True)

out = ROOT / "reports/results/content_profile_per_layer.json"
out.write_text(json.dumps(rows, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
COLOR = {"he": "#4363d8", "row_centered_forward_balanced": "#2a9d3a"}
LBL = {"he": "He", "row_centered_forward_balanced": "rcfwd init (row-centered fwd-balanced)"}
for ax, ds in zip(axes, ["fashion_mnist", "cifar10"]):
    for s in STRATEGIES:
        pts = [(r["layer"], r["cosine_knn_accuracy"]) for r in rows
               if r["dataset"] == ds and r["init_strategy"] == s]
        ax.plot(*zip(*pts), marker="o", ms=4, color=COLOR[s], label=LBL[s])
    ax.axhline(0.1, color="gray", ls=":", lw=1)
    for L, c in [(30, "#2a9d3a"), (50, "#e69f00"), (100, "#d62728")]:
        ax.axvline(L, color=c, ls="--", lw=0.9, alpha=0.5)
        ax.text(L, 0.97, f"{L}L", color=c, fontsize=8, ha="center", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100])
    ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("layer index (log)")
    ax.set_title(ds)
axes[0].set_ylabel(f"cosine k-NN accuracy (k={K})")
axes[0].legend(fontsize=8.5, loc="upper right")
fig.suptitle(f"Class structure vs depth at initialization — cosine k-NN (k={K}), width 500", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
figout = ROOT / "reports/figures/rcfwd_rescale/content_profile_per_layer.png"
fig.savefig(figout, dpi=130)
print(f"saved {out.relative_to(ROOT)} and {figout.relative_to(ROOT)}")
