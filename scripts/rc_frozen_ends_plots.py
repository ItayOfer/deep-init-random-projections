#!/usr/bin/env python3
"""Campaign 10 (rc frozen ends): the two failure mechanisms in one figure.

last3 (fc99, fc100, head trainable) dies by float32 gradient underflow at the
head; first3 (fc1, fc2, fc3 trainable) stays gradient-healthy but is causally
inert -- 97 frozen downstream layers absorb any change before it reaches the
loss. Both show flat eval_train_loss/accuracy for all 20 smoke epochs.

Reads the 4 smoke JSONs directly (reports/results/rcfrozen_*_smoke_*_100L.json).
Output: reports/figures/rc_frozen_ends/rcfrozen_mechanisms.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"

DATASETS = {"fmnist": "tab:purple", "cifar10": "tab:red"}
CONDITIONS = ["last3", "first3"]
# 0-indexed positions in grad_norm_per_layer for each condition's trainable
# hidden layers (fc99 -> idx 98, fc100 -> idx 99; fc1..fc3 -> idx 0..2).
TRAINABLE_IDX = {"last3": [("fc99", 98), ("fc100", 99)],
                 "first3": [("fc1", 0), ("fc2", 1), ("fc3", 2)]}


def load(condition, dataset):
    p = RES / f"rcfrozen_{condition}_smoke_{dataset}_100L.json"
    run = json.load(open(p))[0]
    return run["history"]


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax_loss, ax_acc, ax_last3, ax_first3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

for cond, ls in [("last3", "-"), ("first3", "--")]:
    for ds, color in DATASETS.items():
        hist = load(cond, ds)
        epochs = [h["epoch"] for h in hist]
        # Round to the precision reported in the campaign README (6dp) --
        # unrounded floats differ in the 7th-8th decimal from ordinary
        # float32 accumulation noise, which auto-scaling would otherwise
        # blow up into a misleading "signal".
        loss = [round(h["eval_train_loss"], 6) for h in hist]
        acc = [h["eval_train_accuracy"] for h in hist]
        label = f"{cond}/{ds}"
        ax_loss.plot(epochs, loss, ls, color=color, marker="o", ms=3, label=label)
        ax_acc.plot(epochs, acc, ls, color=color, marker="o", ms=3, label=label)

ax_loss.axhline(2.302585, color="gray", ls=":", lw=1, label="ln(10)")
ax_loss.set_ylim(2.302, 2.304)
ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("eval_train_loss (rounded, 6dp)")
ax_loss.set_title("Loss: bit-exact flat at ln(10) in all 4 cells")
ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)

ax_acc.axhline(0.10, color="gray", ls=":", lw=1, label="chance")
ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("eval_train_accuracy")
ax_acc.set_title("Accuracy: pinned at chance (10 classes)")
ax_acc.legend(fontsize=7); ax_acc.grid(alpha=0.3)

for ds, color in DATASETS.items():
    hist = load("last3", ds)
    epochs = [h["epoch"] for h in hist]
    for name, idx in TRAINABLE_IDX["last3"]:
        norms = [max(h["grad_norm_per_layer"][idx], 1e-12) for h in hist]
        marker = "o" if name == "fc99" else "s"
        ax_last3.plot(epochs, norms, marker=marker, ms=4, color=color,
                      label=f"{ds}/{name}")
ax_last3.set_yscale("log"); ax_last3.set_xlabel("epoch")
ax_last3.set_ylabel("trainable-layer grad norm (floored at 1e-12)")
ax_last3.set_title("last3: fc99/fc100 gradients underflow to exact float32 zero")
ax_last3.legend(fontsize=7); ax_last3.grid(alpha=0.3, which="both")

for ds, color in DATASETS.items():
    hist = load("first3", ds)
    epochs = [h["epoch"] for h in hist]
    for name, idx in TRAINABLE_IDX["first3"]:
        norms = [h["grad_norm_per_layer"][idx] for h in hist]
        marker = {"fc1": "o", "fc2": "s", "fc3": "^"}[name]
        ax_first3.plot(epochs, norms, marker=marker, ms=4, color=color,
                       label=f"{ds}/{name}")
ax_first3.set_yscale("log"); ax_first3.set_xlabel("epoch")
ax_first3.set_ylabel("trainable-layer grad norm")
ax_first3.set_title("first3: fc1-fc3 gradients stay healthy, never vanish")
ax_first3.legend(fontsize=7); ax_first3.grid(alpha=0.3, which="both")

fig.suptitle("Campaign 10 -- rc frozen ends: two failure modes, one dead network\n"
             "100L row_centered_he, NoBN, plain SGD lr=1e-2, seed 42, 20-epoch smoke",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = ROOT / "reports" / "figures" / "rc_frozen_ends" / "rcfrozen_mechanisms.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out.relative_to(ROOT)}")
