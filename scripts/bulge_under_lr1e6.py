#!/usr/bin/env python3
"""Reproduce the lr=1e-6 V2 NoBN runs locally and MEASURE the activation bulge
(per-layer activation RMS) as it evolves under training -- so we see what
actually happened to the bulge when lr was set to 1e-6, rather than assuming
it stayed at init.

For each (dataset, depth, eta): build the V2 NoBN net (seed 42), train with
plain SGD lr=1e-6 (no momentum/wd, no scheduler) for 20 epochs, and snapshot
the per-layer activation RMS on a fixed probe batch at epoch 0, then after
each epoch. Plot the init bulge vs the final bulge, and the peak-bulge
trajectory over epochs.

To stay tractable on CPU (esp. L=100) we train on a 5000-sample subset; the
bulge is a forward/initialisation property and is insensitive to the exact
subset, so this faithfully shows whether lr=1e-6 SGD moves the bulge.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig
from rp_study.data.loaders import get_data_loader
from rp_study.models.classifiers import DeepFCClassifier

WIDTH, LR, EPOCHS, BS, NSUB = 500, 1e-6, 20, 256, 5000
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
CONFIGS = [("cifar10", 30, 0.85), ("cifar10", 50, 0.90), ("cifar10", 100, 0.36)]
COL = {30: "tab:green", 50: "tab:orange", 100: "tab:red"}


def per_layer_act_rms(model, probe):
    """Forward probe batch, return per-hidden-layer activation RMS."""
    rms = []
    x = probe
    with torch.no_grad():
        for i, lin in enumerate(model.hidden_layers):
            x = lin(x)
            if model.use_batch_norm:
                x = model.hidden_norms[i](x)
            x = torch.relu(x)
            rms.append(x.pow(2).mean().sqrt().item())
    return rms


def run(dataset, depth, eta):
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    X, y = get_data_loader(dataset, "data", train=True, num_samples=NSUB,
                           flatten=True, device="cpu")
    y = y.long()
    model = DeepFCClassifier(input_dim=IDIM[dataset], depth=depth,
                             init_strategy="row_centered_layer_balanced_product_base",
                             init_kwargs={"eta": eta}, num_classes=10,
                             use_batch_norm=False, hidden_dim=WIDTH)
    # float64 so the per-layer RMS (which squares the activations) does not
    # overflow when the mid-network bulge reaches 1e22-1e35. The activations
    # themselves are representable in float32, but their SQUARES are not.
    model = model.double(); X = X.double()
    probe = X[:BS]
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    bulges = [per_layer_act_rms(model, probe)]      # epoch 0 = init
    peaks = [max(bulges[0])]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            out = model(X[idx])
            loss = crit(out, y[idx])
            if not torch.isfinite(loss):
                break
            loss.backward(); opt.step()
        bulges.append(per_layer_act_rms(model, probe))
        peaks.append(max(bulges[-1]))
    print(f"{dataset} L={depth} eta={eta}: init peak {peaks[0]:.2e} -> final peak {peaks[-1]:.2e} "
          f"(change {100*(peaks[-1]-peaks[0])/peaks[0]:+.1f}%)")
    return bulges, peaks


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for dataset, depth, eta in CONFIGS:
    bulges, peaks = run(dataset, depth, eta)
    L = len(bulges[0]); x = np.linspace(0, 1, L)
    # left: init bulge (dashed) vs final bulge (solid)
    axes[0].plot(x, bulges[0], "--", color=COL[depth], alpha=0.6, lw=1)
    axes[0].plot(x, bulges[-1], "-", color=COL[depth], lw=1.6,
                 label=f"L={depth} (eta={eta})")
    # right: peak bulge vs epoch
    axes[1].plot(range(len(peaks)), peaks, "o-", color=COL[depth], ms=3,
                 label=f"L={depth}")

axes[0].axhline(3.4e38, ls=":", color="black"); axes[0].text(0.02, 3.4e38, " float32 ceiling", fontsize=8, va="bottom")
axes[0].set_yscale("log"); axes[0].set_ylim(1e-1, 1e42)
axes[0].set_xlabel("fractional depth"); axes[0].set_ylabel("activation RMS")
axes[0].set_title("cifar10: activation bulge under lr=1e-6\n(dashed = epoch 0 / init, solid = epoch 20)")
axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
axes[1].set_yscale("log"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("peak activation RMS")
axes[1].set_title("peak bulge vs epoch under lr=1e-6 (flat = frozen)")
axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
fig.suptitle("What lr=1e-6 did to the activation bulge (measured by reproducing the runs)", y=1.02, fontsize=13)
fig.tight_layout()
out = "reports/figures/v2_nobn_bulge_under_lr1e6_measured.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
