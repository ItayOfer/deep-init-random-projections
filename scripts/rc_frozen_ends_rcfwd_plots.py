#!/usr/bin/env python3
"""Campaign 10 follow-up: the rcfwd (corrected) recipe results, and the direct
raw-vs-rcfwd comparison that resolves the H1-vs-H2 question (see
cluster/10_rc_frozen_ends/README.md sec. "H1 vs H2").

Two figures:
  1. rcfrozen_rcfwd_mechanisms.png -- same 4-panel layout as the raw-recipe
     figure (loss, accuracy, last3 grad norms, first3 grad norms), but for
     the row_centered_forward_balanced + grad_rescale recipe. Unlike the raw
     recipe, loss/accuracy genuinely move here, so no rounding/clamped ylim
     is needed -- the changes are real, not float noise.
  2. rcfrozen_recipe_comparison.png -- eval_train_accuracy vs epoch, raw vs
     rcfwd, side by side for last3 and first3. This is the figure that
     answers H1 vs H2: last3 starts learning once the recipe is corrected
     (supports H1's mechanism for the tail -- the raw recipe's forward-scale
     collapse, not the backward gradient per se, was what killed it); first3
     still fails under either recipe (supports H2 for the front -- healthy
     gradients both times, content still can't get through 97 frozen layers).

Reads all 8 smoke JSONs (reports/results/rcfrozen_*_smoke_*_100L[_rcfwd].json).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
FIG_DIR = ROOT / "reports" / "figures" / "rc_frozen_ends"

DATASETS = {"fmnist": "tab:purple", "cifar10": "tab:red"}
TRAINABLE_IDX = {"last3": [("fc98", 97), ("fc99", 98), ("fc100", 99)],
                 "first3": [("fc1", 0), ("fc2", 1), ("fc3", 2)]}


def load(condition, dataset, recipe):
    suffix = "_rcfwd" if recipe == "rcfwd" else ""
    p = RES / f"rcfrozen_{condition}_smoke_{dataset}_100L{suffix}.json"
    return json.load(open(p))[0]["history"]


# ---------------------------------------------------------------------------
# Figure 1: rcfwd-recipe mechanisms (4-panel, same layout as the raw figure)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax_loss, ax_acc, ax_last3, ax_first3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

for cond, ls in [("last3", "-"), ("first3", "--")]:
    for ds, color in DATASETS.items():
        hist = load(cond, ds, "rcfwd")
        epochs = [h["epoch"] for h in hist]
        loss = [h["eval_train_loss"] for h in hist]
        acc = [h["eval_train_accuracy"] for h in hist]
        label = f"{cond}/{ds}"
        ax_loss.plot(epochs, loss, ls, color=color, marker="o", ms=3, label=label)
        ax_acc.plot(epochs, acc, ls, color=color, marker="o", ms=3, label=label)

ax_loss.axhline(2.302585, color="gray", ls=":", lw=1, label="ln(10)")
ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("eval_train_loss")
ax_loss.set_title("Loss: last3 falls, first3 climbs well past ln(10)")
ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)

ax_acc.axhline(0.10, color="gray", ls=":", lw=1, label="chance")
ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("eval_train_accuracy")
ax_acc.set_title("Accuracy: last3 climbs steadily, first3 stays at chance")
ax_acc.legend(fontsize=7); ax_acc.grid(alpha=0.3)

for ds, color in DATASETS.items():
    hist = load("last3", ds, "rcfwd")
    epochs = [h["epoch"] for h in hist]
    for name, idx in TRAINABLE_IDX["last3"]:
        norms = [h["grad_norm_per_layer"][idx] for h in hist]
        marker = {"fc98": "o", "fc99": "s", "fc100": "^"}[name]
        ax_last3.plot(epochs, norms, marker=marker, ms=4, color=color,
                      label=f"{ds}/{name}")
ax_last3.set_xlabel("epoch"); ax_last3.set_ylabel("trainable-layer grad norm")
ax_last3.set_title("last3: fc98/fc99/fc100 gradients healthy -- no underflow this time")
ax_last3.legend(fontsize=7); ax_last3.grid(alpha=0.3)

for ds, color in DATASETS.items():
    hist = load("first3", ds, "rcfwd")
    epochs = [h["epoch"] for h in hist]
    for name, idx in TRAINABLE_IDX["first3"]:
        norms = [h["grad_norm_per_layer"][idx] for h in hist]
        marker = {"fc1": "o", "fc2": "s", "fc3": "^"}[name]
        ax_first3.plot(epochs, norms, marker=marker, ms=4, color=color,
                       label=f"{ds}/{name}")
ax_first3.set_xlabel("epoch"); ax_first3.set_ylabel("trainable-layer grad norm")
ax_first3.set_title("first3: fc1-fc3 gradients healthy too -- still no progress")
ax_first3.legend(fontsize=7); ax_first3.grid(alpha=0.3)

fig.suptitle("Campaign 10 follow-up -- rcfwd recipe (row_centered_forward_balanced "
             "+ grad_rescale)\nlast3 starts learning; first3 still fails, "
             "loss actively worsens instead of staying flat",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])

out1 = FIG_DIR / "rcfrozen_rcfwd_mechanisms.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"saved {out1.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Figure 2: the H1-vs-H2 resolution -- raw vs rcfwd accuracy, side by side
# ---------------------------------------------------------------------------
fig2, (ax_l3, ax_f3) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, cond, title in [(ax_l3, "last3", "last3: fc98, fc99, fc100 trainable (head frozen)"),
                        (ax_f3, "first3", "first3: fc1, fc2, fc3 trainable (head frozen)")]:
    for recipe, ls in [("raw", "--"), ("rcfwd", "-")]:
        for ds, color in DATASETS.items():
            hist = load(cond, ds, recipe)
            epochs = [h["epoch"] for h in hist]
            acc = [h["eval_train_accuracy"] for h in hist]
            ax.plot(epochs, acc, ls, color=color, marker="o", ms=3,
                    label=f"{recipe}/{ds}")
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="chance")
    ax.set_xlabel("epoch"); ax.set_title(title)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax_l3.set_ylabel("eval_train_accuracy")

fig2.suptitle("Does correcting the recipe (row_centered_he -> forward_balanced+"
              "grad_rescale) unlock training?\nlast3: yes, slowly (H1-consistent -- "
              "the raw recipe's forward-scale collapse was the cause) | "
              "first3: no (H2-consistent -- content, not gradient conditioning, "
              "is the bottleneck)", fontsize=10)
fig2.tight_layout(rect=[0, 0, 1, 0.88])

out2 = FIG_DIR / "rcfrozen_recipe_comparison.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"saved {out2.relative_to(ROOT)}")
