#!/usr/bin/env python3
"""Scale-invariant content probes at the rcfwd-trained depths.

The Euclidean k-NN probe (geometry_content_probe_*.json) is blind to
class structure in row-centered representations: per-sample scale
variation destroys Euclidean neighborhoods even when directional /
linearly-accessible structure survives. This script re-probes the same
representations with two scale-robust measures:

  * cosine k-NN accuracy  (unit-normalize rows, then k-NN)
  * linear probe accuracy (logistic regression, 75/25 split)

Cells: {fashion_mnist, cifar10} x {30, 50, 100} layers x
{he, row_centered_forward_balanced}, num_samples=2000, seed 42,
normalized inputs — matching the Euclidean probe and the training runs.

Output: reports/results/content_probe_linear.json + printed table.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from rp_study.data import get_data_loader
from rp_study.projections import multi_layer_rp_with_init

SEED = 42
N = 2000
DEPTHS = [30, 50, 100]
STRATEGIES = ["he", "row_centered_forward_balanced"]


def cosine_knn_acc(X, y, k=10):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.25, random_state=SEED, stratify=y)
    return KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr, ytr).score(Xte, yte)


def linear_probe_acc(X, y):
    # standardize features so LR converges on wildly-scaled representations
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return clf.fit(Xtr, ytr).score(Xte, yte)


rows = []
for dataset in ["fashion_mnist", "cifar10"]:
    X, y = get_data_loader(dataset_name=dataset, data_dir=str(ROOT / "data"), train=True,
                           num_samples=N, flatten=True, as_numpy=True, normalize=True, seed=SEED)
    for strategy in STRATEGIES:
        for depth in DEPTHS:
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            Xp = multi_layer_rp_with_init(X, depth, init_strategy=strategy, seed=SEED)
            row = {
                "dataset": dataset, "init_strategy": strategy, "depth": depth,
                "cosine_knn_accuracy": float(cosine_knn_acc(Xp, y)),
                "linear_probe_accuracy": float(linear_probe_acc(Xp, y)),
            }
            rows.append(row)
            print(f"{dataset:14} {strategy:>30} L={depth:>3}  "
                  f"cos-kNN={row['cosine_knn_accuracy']:.3f}  linear={row['linear_probe_accuracy']:.3f}",
                  flush=True)

out = ROOT / "reports/results/content_probe_linear.json"
out.write_text(json.dumps(rows, indent=2))
print(f"saved {out.relative_to(ROOT)}")
