#!/usr/bin/env python3
"""Early learning speed vs depth: He vs V2 vs rcfwd (NoBN cells).

Motivation: the rcfwd smokes train stably at all depths but slowly.
This quantifies "slow" — eval_train_accuracy at epoch 20 and epochs to
reach 50% — as a function of depth, from existing result JSONs only.

Comparability notes (recipes differ by design):
  * rcfwd smokes:      plain SGD lr=1e-2, fixed LR, width 500
  * He replication:    plain SGD lr=1e-3, fixed LR, width 512 (100L only)
  * He recovery:       plain SGD lr=1e-3 + plateau, width 500 (100L only)
  * He final audit:    tuned per-cell recipes (SGD+onecycle / Adam), width 500
  * V2 round-1 smokes: He-passing recipes, width 500 (100L aborted at ep1)

Output: printed table + reports/figures/rcfwd_rescale/learning_speed_vs_depth.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"


def acc_at(hist, ep):
    for e in hist:
        if e["epoch"] == ep:
            return e["eval_train_accuracy"]
    return None


def epochs_to(hist, thr):
    for e in hist:
        if e["eval_train_accuracy"] >= thr:
            return e["epoch"]
    return None


def load(fname, index=0):
    p = RES / fname
    if not p.exists():
        return None
    entry = json.load(open(p))[index]
    return entry.get("history") or None


# (family, dataset, depth) -> history source
SOURCES = {
    # rcfwd smokes (plain SGD 1e-2)
    ("rcfwd", "fmnist", 30): ("rcfwd_rescale_smoke_fmnist_30L.json", 0),
    ("rcfwd", "fmnist", 50): ("rcfwd_rescale_smoke_fmnist_50L.json", 0),
    ("rcfwd", "fmnist", 100): ("rcfwd_rescale_smoke_fmnist_100L.json", 0),
    ("rcfwd", "cifar10", 30): ("rcfwd_rescale_smoke_cifar10_30L.json", 0),
    ("rcfwd", "cifar10", 50): ("rcfwd_rescale_smoke_cifar10_50L.json", 0),
    ("rcfwd", "cifar10", 100): ("rcfwd_rescale_smoke_cifar10_100L.json", 0),
    # V2 round-1 smokes, NoBN (He-passing recipes; 100L NoBN aborted ep1)
    ("v2", "fmnist", 30): ("row_centered_smoke_fmnist_30L_nobn.json", 0),
    ("v2", "fmnist", 50): ("row_centered_smoke_fmnist_50L_nobn.json", 0),
    ("v2", "cifar10", 30): ("row_centered_smoke_cifar10_30L_nobn.json", 0),
    ("v2", "cifar10", 50): ("row_centered_smoke_cifar10_50L_nobn.json", 0),
    # He final audit NoBN cells (tuned recipes, 200 ep)
    ("he_audit", "cifar10", 30): ("final_audit_merged.json", 0),
    ("he_audit", "cifar10", 50): ("final_audit_merged.json", 2),
    ("he_audit", "cifar10", 100): ("final_audit_merged.json", 4),
    ("he_audit", "fmnist", 30): ("final_audit_merged.json", 6),
    ("he_audit", "fmnist", 50): ("final_audit_merged.json", 8),
    ("he_audit", "fmnist", 100): ("final_audit_merged.json", 10),
    # He + plain SGD at 100L NoBN (closest recipe match to rcfwd)
    ("he_plain_sgd", "fmnist", 100): ("plain_sgd_100L_nobn_w512_fashion_mnist.json", 0),
    ("he_plain_sgd", "cifar10", 100): ("plain_sgd_100L_nobn_w512_cifar10.json", 0),
}

FAMILIES = ["he_audit", "he_plain_sgd", "v2", "rcfwd"]
LABELS = {
    "he_audit": "He (tuned audit recipe)",
    "he_plain_sgd": "He (plain SGD 1e-3)",
    "v2": "V2 (round-1 smoke)",
    "rcfwd": "rcfwd (plain SGD 1e-2)",
}

rows = {}
for (fam, ds, depth), (fname, idx) in SOURCES.items():
    hist = load(fname, idx)
    if hist is None:
        rows[(fam, ds, depth)] = ("aborted/absent", None)
        continue
    rows[(fam, ds, depth)] = (acc_at(hist, 20), epochs_to(hist, 0.5))

print(f"{'dataset':8} {'L':>4} | " + " | ".join(f"{LABELS[f]:>26}" for f in FAMILIES))
print("-" * 130)
print("eval_train_accuracy @ epoch 20  (epochs to reach 50% in parentheses)")
for ds in ["fmnist", "cifar10"]:
    for depth in [30, 50, 100]:
        cells = []
        for fam in FAMILIES:
            v = rows.get((fam, ds, depth))
            if v is None:
                cells.append(f"{'—':>26}")
            elif v[0] == "aborted/absent":
                cells.append(f"{'diverged ep1':>26}")
            else:
                a20 = f"{v[0]:.3f}" if v[0] is not None else "n/a"
                e50 = f"(ep{v[1]})" if v[1] else "(>run)"
                cells.append(f"{a20 + ' ' + e50:>26}")
        print(f"{ds:8} {depth:>4} | " + " | ".join(cells))

# figure: acc@20 vs depth
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, ds in zip(axes, ["fmnist", "cifar10"]):
    for fam, marker in zip(FAMILIES, ["o", "s", "^", "D"]):
        xs, ys = [], []
        for depth in [30, 50, 100]:
            v = rows.get((fam, ds, depth))
            if v and v[0] not in (None, "aborted/absent"):
                xs.append(depth)
                ys.append(v[0])
        if xs:
            ax.plot(xs, ys, marker=marker, label=LABELS[fam])
    ax.axhline(0.1, color="gray", ls=":", lw=1, label="chance")
    ax.set_title(f"{ds} · NoBN · eval-train acc @ epoch 20")
    ax.set_xlabel("depth L")
    ax.set_xticks([30, 50, 100])
axes[0].set_ylabel("eval_train_accuracy @ ep20")
axes[1].legend(fontsize=8)
out = ROOT / "reports/figures/rcfwd_rescale/learning_speed_vs_depth.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"\nsaved {out.relative_to(ROOT)}")
