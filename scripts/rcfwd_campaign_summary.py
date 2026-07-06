#!/usr/bin/env python3
"""rcfwd campaign summary figure — the whole story in four panels.

(a,b) 200-epoch audit curves per dataset (depth-graded trainability)
(c)   LR ladder: ep-20 accuracy vs LR, with the divergence wall
(d)   fmnist/30L: rcfwd (fixed LR, no momentum) vs tuned He (onecycle+momentum)

Sources: reports/results/rcfwd_rescale_audit_*.json, rcfwd_rescale_smoke_*.json,
rcfwd_lr*_smoke_*.json, final_audit_merged.json.
Output: reports/figures/rcfwd_rescale/rcfwd_campaign_summary.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
CRIT = 0.995

DEPTH_COLOR = {30: "#2a9d3a", 50: "#e69f00", 100: "#d62728"}


def hist(fname, index=0):
    entry = json.load(open(RES / fname))[index]
    return entry.get("history") or []


def curve(h):
    return [e["epoch"] for e in h], [e["eval_train_accuracy"] for e in h]


fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

# --- (a,b) audit curves ---
for ax, ds, title in [(axes[0, 0], "fmnist", "Fashion-MNIST"), (axes[0, 1], "cifar10", "CIFAR-10")]:
    for depth in [30, 50, 100]:
        h = hist(f"rcfwd_rescale_audit_{ds}_{depth}L.json")
        x, y = curve(h)
        ax.plot(x, y, color=DEPTH_COLOR[depth], lw=1.8, label=f"{depth}L")
    ax.axhline(CRIT, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(4, CRIT - 0.055, "PASS bar 0.995", fontsize=8, alpha=0.7)
    ax.axhline(0.1, color="gray", ls=":", lw=1, alpha=0.6)
    ax.text(4, 0.112, "chance", fontsize=8, alpha=0.6)
    ax.set_title(f"({'ab'[ds == 'cifar10']}) {title} — 200-epoch audits, SGD lr=1e-2 fixed, NoBN")
    ax.set_xlabel("epoch")
    ax.set_ylabel("eval-train accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(title="depth", loc="center right", fontsize=9)

# annotate the PASS
h30 = hist("rcfwd_rescale_audit_fmnist_30L.json")
ep_pass = next(e["epoch"] for e in h30 if e["eval_train_accuracy"] >= CRIT and e["eval_train_loss"] <= 0.10)
axes[0, 0].annotate(f"PASS @ ep{ep_pass}", xy=(ep_pass, CRIT), xytext=(ep_pass + 22, 0.82),
                    arrowprops=dict(arrowstyle="->", color=DEPTH_COLOR[30]), color=DEPTH_COLOR[30], fontsize=9)

# --- (c) LR ladder: categorical bars + the counterfactual anchor ---
ax = axes[1, 0]
LRS = [0.01, 0.03, 0.1, 0.3]
LADDER = {
    "fmnist_100L": {0.01: "rcfwd_rescale_smoke_fmnist_100L.json", 0.03: "rcfwd_lr3e2_smoke_fmnist_100L.json",
                    0.1: "rcfwd_lr1e1_smoke_fmnist_100L.json", 0.3: "rcfwd_lr3e1_smoke_fmnist_100L.json"},
    "cifar10_50L": {0.01: "rcfwd_rescale_smoke_cifar10_50L.json", 0.03: "rcfwd_lr3e2_smoke_cifar10_50L.json",
                    0.1: "rcfwd_lr1e1_smoke_cifar10_50L.json", 0.3: "rcfwd_lr3e1_smoke_cifar10_50L.json"},
}
W = 0.36
for i, (cell, color) in enumerate([("fmnist_100L", DEPTH_COLOR[100]), ("cifar10_50L", DEPTH_COLOR[50])]):
    for j, lr in enumerate(LRS):
        x = j + (i - 0.5) * W
        h = hist(LADDER[cell][lr])
        if h:
            y = next(e["eval_train_accuracy"] for e in h if e["epoch"] == 20)
            ax.bar(x, y, W * 0.92, color=color, alpha=0.85,
                   label=cell.replace("_", " ") if j == 0 else None)
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=8, color=color)
        else:
            ax.bar(x, 0.04, W * 0.92, color="white", edgecolor=color, hatch="////", lw=1.2)
ax.text(2.5, 0.24, "hatched = diverged\n(NaN at epoch 1)", ha="center", fontsize=8.5, color="#555",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#bbb", lw=0.8))
# what fast learning at this depth looks like (He + plain SGD lr=1e-3, fmnist/100L)
he100 = next(e["eval_train_accuracy"] for e in hist("plain_sgd_100L_nobn_w512_fashion_mnist.json") if e["epoch"] == 20)
ax.axhline(he100, color="#4363d8", ls="--", lw=1.4)
ax.text(1.5, he100 - 0.035, f"He + plain SGD (lr=0.001) at fmnist/100L: {he100:.2f} @ ep20\n— the gap no learning rate closes",
        fontsize=8.5, color="#4363d8", ha="center", va="top")
ax.axhline(0.1, color="gray", ls=":", lw=1, alpha=0.7)
ax.text(-0.42, 0.112, "chance", fontsize=8, alpha=0.6)
ax.set_xticks(range(len(LRS)))
ax.set_xticklabels([f"lr = {l}" for l in LRS])
ax.set_title("(c) LR ladder — accuracy @ ep20 barely responds to LR, then diverges")
ax.set_ylabel("eval-train accuracy @ ep20")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=9, loc="center left")

# --- (d) fmnist/30L: rcfwd vs tuned He ---
ax = axes[1, 1]
he = hist("final_audit_merged.json", 6)  # 07_fmnist_30L_nobn, SGD onecycle + momentum
x, y = curve(he)
ax.plot(x, y, color="#4363d8", lw=1.8, label="He, tuned (onecycle, mom 0.9)")
x, y = curve(h30)
ax.plot(x, y, color=DEPTH_COLOR[30], lw=1.8, label="rcfwd (fixed lr, no momentum)")
ax.axhline(CRIT, color="k", ls="--", lw=1, alpha=0.6)
he_pass = next(e["epoch"] for e in he if e["eval_train_accuracy"] >= CRIT and e["eval_train_loss"] <= 0.10)
for ep, c, dy in [(he_pass, "#4363d8", -0.12), (ep_pass, DEPTH_COLOR[30], -0.24)]:
    ax.annotate(f"criterion @ ep{ep}", xy=(ep, CRIT), xytext=(ep + 15, CRIT + dy),
                arrowprops=dict(arrowstyle="->", color=c), color=c, fontsize=9)
ax.set_title("(d) fmnist/30L NoBN — rcfwd matches tuned He to the criterion")
ax.set_xlabel("epoch")
ax.set_ylabel("eval-train accuracy")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=9)

fig.suptitle("rcfwd (row-centered forward-balanced + backward gradient rescale) — campaign results",
             fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = ROOT / "reports/figures/rcfwd_rescale/rcfwd_campaign_summary.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"saved {out.relative_to(ROOT)}")
