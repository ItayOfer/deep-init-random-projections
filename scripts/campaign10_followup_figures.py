#!/usr/bin/env python3
"""Campaign 10 follow-up (2026-08-15): four figures over one day's frozen-readout,
relu-shift, and LR-ladder experiments (campaigns 10-12).

Figure 1 -- frozen_readout_audits.png
    `rcfrozen_{last3,last2}_audit_{fmnist,cifar10}_100L_rcfwd` -- 400-epoch
    frozen-readout audits (rcfwd recipe, 100L, only the named window
    trainable, head frozen). eval_train_accuracy vs test_accuracy per epoch:
    train climbs, test stays at chance -- the generalization gap is campaign
    12's central observation about this protocol.

Figure 2 -- relushift_30L_arms.png
    `relushift_{he,rc,c010,c025,c070}_smoke_{fmnist,cifar10}_30L` plus the
    `c025 ... _diff` (differentiable-gradient fork) control -- campaign 11's
    30-layer end-to-end training grid. Final eval_train_accuracy / test_accuracy
    per arm, grouped by dataset, deltas vs `he` annotated.

Figure 3 -- frozen_readout_ranking.png
    `frozenro_{he,rc,rcfwd,c010,c025,c070}_smoke_{fmnist,cifar10}_{30,100}L`
    -- campaign 12's frozen-readout ranking (fc99-fc100 trainable, head
    frozen). The `rcfwd` arm was only submitted at 30L; at 100L this figure
    substitutes `rcfrozen_last2_smoke_{fmnist,cifar10}_100L_rcfwd`, which
    campaign 12's README documents as the identical configuration run under
    campaign 10. Final train/test accuracy per arm, grouped by depth.

Figure 4 -- front_window_lr_ladder.png
    `rcfrozen_first2_smoke_{fmnist,cifar10}_100L_rawrescale[_lr1e2|_lr1e4|
    _lr1e6|_lr1e7]` -- campaign 10 W5's LR ladder on the front (fc1,fc2)
    trainable window under the `rawrescale` recipe (row_centered_he +
    grad_rescale, no forward-balanced init). eval_train_loss per epoch,
    five rungs spanning nine orders of magnitude in LR, all pinned at ln(10).

Every number plotted is read from reports/results/*.json at run time -- none
are hardcoded. Output: reports/figures/campaign10_followup/*.png.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # convention parity with the other scripts/*.py; unused directly here

CHANCE = 0.10
LN10 = math.log(10)

# --- Colorblind-safe palette -------------------------------------------------
# 4-hue categorical set (he/rc/rcfwd/shift-family) validated with the dataviz
# skill's scripts/validate_palette.js against light-surface CVD, normal-vision
# and contrast gates for the adjacencies actually used below (all PASS; the
# rcfwd/aqua slot carries a contrast WARN against white, mitigated here with
# a black bar edge + direct value/legend labels rather than color alone).
BLUE = "#2a78d6"  # he -- the baseline every other arm is measured against
VIOLET = "#4a3aa7"  # rc / row_centered_he -- weight-space DC removal
AQUA = "#1baf7a"  # rcfwd -- row_centered_forward_balanced + grad_rescale
# c010/c025/c070 are steps of one ORDINAL variable (the shift constant c), so
# they get a single-hue sequential ramp (light->dark) rather than unrelated
# categorical hues -- the read is "same family, increasing c", not "4 things".
SHIFT_LIGHT = "#f3b59c"  # c = 0.10
SHIFT_MID = "#eb6834"  # c = 0.25 (canonical palette orange; c025_diff reuses this hue + a hatch)
SHIFT_DARK = "#a3370c"  # c = 0.70
GRAY = "#7a7a7a"
INK = "#222222"

ARM_COLOR = {
    "he": BLUE, "rc": VIOLET, "rcfwd": AQUA,
    "c010": SHIFT_LIGHT, "c025": SHIFT_MID, "c025_diff": SHIFT_MID, "c070": SHIFT_DARK,
}
DATASETS = [("fmnist", "Fashion-MNIST"), ("cifar10", "CIFAR-10")]


def load(res_dir, stem):
    """Load reports/results/<stem>.json -> (payload dict, history list)."""
    payload = json.load(open(res_dir / f"{stem}.json"))[0]
    return payload, payload.get("history") or []


def report(fig_name, files_read, lines):
    print(f"\n=== {fig_name} ===")
    print(f"JSON files read ({len(files_read)}):")
    for f in files_read:
        print(f"  reports/results/{f}")
    print("Key numbers plotted:")
    for l in lines:
        print(f"  {l}")


# =============================================================================
# Figure 1 -- frozen_readout_audits.png
# =============================================================================

def figure1(res_dir, out_dir, dpi):
    fname = "frozen_readout_audits.png"
    conditions = [
        ("last3", "last3 (fc98–fc100 trainable)", BLUE),
        ("last2", "last2 (fc99–fc100 trainable)", SHIFT_MID),
    ]
    files_read, lines = [], []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (ds, ds_title) in zip(axes, DATASETS):
        for cond, cond_label, color in conditions:
            stem = f"rcfrozen_{cond}_audit_{ds}_100L_rcfwd"
            files_read.append(f"{stem}.json")
            _, hist = load(res_dir, stem)
            ep = [e["epoch"] for e in hist]
            tr = [e["eval_train_accuracy"] for e in hist]
            te = [e["test_accuracy"] for e in hist]
            ax.plot(ep, tr, "-", color=color, lw=1.9, label=f"{cond_label} — train")
            ax.plot(ep, te, "--", color=color, lw=1.6, label=f"{cond_label} — test")
            lines.append(
                f"{stem}.json: n_epochs={len(hist)}, final train_acc={tr[-1]:.4f}, "
                f"final test_acc={te[-1]:.4f}, max test_acc over run={max(te):.4f}"
            )
        ax.axhline(CHANCE, color=GRAY, ls=":", lw=1.3, zorder=1)
        ax.text(8, CHANCE + 0.02, "chance", fontsize=8.5, color=GRAY)
        ax.set_title(f"{ds_title} — 100L, 400 epochs", fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.3, loc="upper left", framealpha=0.9)
    axes[0].set_ylabel("accuracy (fraction of examples correct)")

    fig.suptitle(
        "Frozen-readout audits — 100L FC, rcfwd recipe (row_centered_forward_balanced + grad_rescale),\n"
        "SGD lr=1e-2 fixed, NoBN, last3 vs last2 trainable windows (head always frozen)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")
    report(fname, files_read, lines)
    return out_path


# =============================================================================
# Figure 2 -- relushift_30L_arms.png
# =============================================================================

ARM_ORDER_FIG2 = ["he", "rc", "c010", "c025", "c025_diff", "c070"]
ARM_LABEL_FIG2 = {
    "he": "he", "rc": "rc", "c010": "shift c=0.10", "c025": "shift c=0.25",
    "c025_diff": "shift c=0.25\n(diff. grad)", "c070": "shift c=0.70",
}


def relushift_file(arm, ds):
    if arm == "c025_diff":
        return f"relushift_c025_smoke_{ds}_30L_diff"
    return f"relushift_{arm}_smoke_{ds}_30L"


def figure2(res_dir, out_dir, dpi):
    fname = "relushift_30L_arms.png"
    files_read, lines = [], []
    data = {}  # (ds, arm) -> (train_acc, test_acc)

    for ds, _ in DATASETS:
        for arm in ARM_ORDER_FIG2:
            stem = relushift_file(arm, ds)
            files_read.append(f"{stem}.json")
            _, hist = load(res_dir, stem)
            tr, te = hist[-1]["eval_train_accuracy"], hist[-1]["test_accuracy"]
            data[(ds, arm)] = (tr, te)
            lines.append(f"{stem}.json: final train_acc={tr:.4f}, test_acc={te:.4f} (n_epochs={len(hist)})")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    x = np.arange(len(ARM_ORDER_FIG2))
    w = 0.34

    for ax, (ds, ds_title) in zip(axes, DATASETS):
        he_tr, he_te = data[(ds, "he")]
        for i, arm in enumerate(ARM_ORDER_FIG2):
            tr, te = data[(ds, arm)]
            color = ARM_COLOR[arm]
            hatch = "////" if arm == "c025_diff" else None
            ew = 2.2 if arm == "he" else 0.6
            ax.bar(i - w / 2, tr, w, color=color, alpha=1.0, hatch=hatch, edgecolor="black", linewidth=ew)
            ax.bar(i + w / 2, te, w, color=color, alpha=0.55, hatch=hatch, edgecolor="black", linewidth=ew)
            if arm != "he":
                d_tr, d_te = (tr - he_tr) * 100, (te - he_te) * 100
                ax.text(i - w / 2, tr + 0.015, f"{d_tr:+.1f}", ha="center", fontsize=7, color=INK)
                ax.text(i + w / 2, te + 0.015, f"{d_te:+.1f}", ha="center", fontsize=7, color=INK)
        ax.axhline(he_tr, color=BLUE, ls=":", lw=1.0, alpha=0.55, zorder=0)
        ax.axhline(he_te, color=BLUE, ls=":", lw=1.0, alpha=0.55, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL_FIG2[a] for a in ARM_ORDER_FIG2], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy (fraction correct)")
        ax.set_title(f"{ds_title} — 30L, 20 epochs", fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)

    # headline callout: c010 vs he on cifar10 test accuracy
    ax_c = axes[1]
    he_te_c = data[("cifar10", "he")][1]
    c010_te_c = data[("cifar10", "c010")][1]
    i_c010 = ARM_ORDER_FIG2.index("c010")
    delta_pp = (c010_te_c - he_te_c) * 100
    ax_c.annotate(
        f"c010 test: {c010_te_c:.3f}\nvs he test: {he_te_c:.3f}\nΔ = {delta_pp:+.1f}pp",
        xy=(i_c010 + w / 2, c010_te_c), xytext=(i_c010 + 1.5, c010_te_c + 0.18),
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), fontsize=8.3, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bbb", lw=0.8),
    )

    train_patch = mpatches.Patch(facecolor="white", alpha=1.0, edgecolor="black", linewidth=0.6, label="train (eval_train_accuracy)")
    test_patch = mpatches.Patch(facecolor="white", alpha=0.55, edgecolor="black", linewidth=0.6, label="test (test_accuracy)")
    he_patch = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.2, label="he = baseline (thick edge)")
    diff_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="differentiable-grad fork")
    fig.legend(handles=[train_patch, test_patch, he_patch, diff_patch], loc="lower center",
               ncol=4, fontsize=8.3, bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        "Campaign 11 — 30-layer FC, NoBN, SGD lr=1e-2, 20-epoch end-to-end training\n"
        "final accuracy by initialization arm (numbers above bars = percentage points vs he)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")
    report(fname, files_read, lines)
    return out_path


# =============================================================================
# Figure 3 -- frozen_readout_ranking.png
# =============================================================================

ARM_ORDER_FIG3 = ["he", "rc", "rcfwd", "c010", "c025", "c070"]
ARM_LABEL_FIG3 = {
    "he": "he", "rc": "rc", "rcfwd": "rcfwd",
    "c010": "shift c=0.10", "c025": "shift c=0.25", "c070": "shift c=0.70",
}


def frozenro_file(arm, ds, depth):
    if arm == "rcfwd" and depth == 100:
        # frozenro_rcfwd_smoke_*_100L was never submitted (campaign 12 README:
        # "do not resubmit the rcfwd arm if campaign 10's rcfrozen_last2_smoke
        # is already ... committed -- it is the same run"). Substitute it.
        return f"rcfrozen_last2_smoke_{ds}_100L_rcfwd"
    return f"frozenro_{arm}_smoke_{ds}_{depth}L"


def figure3(res_dir, out_dir, dpi):
    fname = "frozen_readout_ranking.png"
    depths = [30, 100]
    files_read, lines = [], []
    data = {}  # (ds, depth, arm) -> (train_acc, test_acc)

    for ds, _ in DATASETS:
        for depth in depths:
            for arm in ARM_ORDER_FIG3:
                stem = frozenro_file(arm, ds, depth)
                files_read.append(f"{stem}.json")
                _, hist = load(res_dir, stem)
                tr, te = hist[-1]["eval_train_accuracy"], hist[-1]["test_accuracy"]
                data[(ds, depth, arm)] = (tr, te)
                lines.append(
                    f"{stem}.json: {ds}/{depth}L/{arm}: final train_acc={tr:.4f}, "
                    f"test_acc={te:.4f} (n_epochs={len(hist)})"
                )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), sharey="col")
    x = np.arange(len(ARM_ORDER_FIG3))
    w = 0.34
    depth_hatch = {30: None, 100: "////"}

    for col, (ds, ds_title) in enumerate(DATASETS):
        for row, (metric_key, metric_label) in enumerate([("train", "train accuracy"), ("test", "test accuracy")]):
            ax = axes[row, col]
            for i, arm in enumerate(ARM_ORDER_FIG3):
                color = ARM_COLOR[arm]
                for j, depth in enumerate(depths):
                    tr, te = data[(ds, depth, arm)]
                    val = tr if metric_key == "train" else te
                    xpos = i + (j - 0.5) * w
                    ax.bar(xpos, val, w * 0.95, color=color, edgecolor="black",
                           linewidth=0.5, hatch=depth_hatch[depth])
            ax.axhline(CHANCE, color=GRAY, ls=":", lw=1.0, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([ARM_LABEL_FIG3[a] for a in ARM_ORDER_FIG3], fontsize=8.5)
            ax.set_ylabel(f"{metric_label}\n(fraction correct)", fontsize=9)
            ax.set_title(f"{ds_title} — {metric_label}", fontsize=10.2)
            ax.grid(axis="y", alpha=0.25)

    for col in range(2):
        top = axes[0, col].get_ylim()[1]
        axes[0, col].set_ylim(0, top)

    depth_patch_30 = mpatches.Patch(facecolor="white", edgecolor="black", label="30L")
    depth_patch_100 = mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="100L")
    fig.legend(handles=[depth_patch_30, depth_patch_100], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), title="depth", frameon=False)

    fig.suptitle(
        "Campaign 12 — frozen-readout ranking: fc99–fc100 trainable, head frozen,\n"
        "NoBN, SGD lr=1e-2, 20-epoch smokes (all cells), 30L vs 100L",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")
    report(fname, files_read, lines)
    return out_path


# =============================================================================
# Figure 4 -- front_window_lr_ladder.png
# =============================================================================

LR_RUNGS = [
    ("", "#86b6ef", "o"),
    ("_lr1e2", "#5598e7", "s"),
    ("_lr1e4", "#2a78d6", "^"),
    ("_lr1e6", "#1c5cab", "D"),
    ("_lr1e7", "#104281", "v"),
]
OFFSET_STEP = 0.0005  # artificial vertical offset per rung, for visibility ONLY -- see annotation


def figure4(res_dir, out_dir, dpi):
    fname = "front_window_lr_ladder.png"
    files_read, lines = [], []
    all_dev = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, (ds, ds_title) in zip(axes, DATASETS):
        for k, (suf, color, marker) in enumerate(LR_RUNGS):
            stem = f"rcfrozen_first2_smoke_{ds}_100L_rawrescale{suf}"
            files_read.append(f"{stem}.json")
            payload, hist = load(res_dir, stem)
            lr_val = payload["training_config"]["learning_rate"]
            ep = [e["epoch"] for e in hist]
            loss = [e["eval_train_loss"] for e in hist]
            dev = [abs(v - LN10) for v in loss]
            all_dev.extend(dev)
            offset = k * OFFSET_STEP
            plotted = [v + offset for v in loss]
            ax.plot(ep, plotted, "-", color=color, lw=1.3, marker=marker, ms=5.5,
                     markevery=2, label=f"lr={lr_val:g}")
            lines.append(
                f"{stem}.json: lr={lr_val:g}, n_epochs={len(hist)}, "
                f"eval_train_loss range=[{min(loss):.8f}, {max(loss):.8f}], "
                f"max|loss-ln10|={max(dev):.3e}"
            )
        ax.axhline(LN10, color="black", ls="--", lw=1.2, zorder=0)
        ax.set_title(f"{ds_title} — 100L, first2 window, rawrescale recipe", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(LN10 - 0.0006, LN10 + 4 * OFFSET_STEP + 0.0008)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("eval_train_loss (nats)\noffset per rung for visibility — see annotation")
    axes[0].text(1, LN10 - 0.00045, "ln 10 = 2.302585 (true value, no offset)", fontsize=7.8, color="#333333")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.015), title="learning rate (fc1,fc2 trainable; 1e-2 → 1e7)", frameon=False)

    max_dev = max(all_dev)
    fig.suptitle(
        "Campaign 10 W5 — front-window LR ladder: 100L FC, rawrescale recipe\n"
        "(row_centered_he + grad_rescale), NoBN, 20-epoch smokes, LR swept 1e-2–1e7 (9 orders of magnitude)",
        fontsize=11.3,
    )
    fig.text(
        0.5, 0.90,
        f"max |eval_train_loss − ln 10| = {max_dev:.2e} across all 5 rungs × 20 epochs × 2 datasets "
        "(curves offset vertically by 0.0005/rung for visibility only — real values are float32-noise-identical)",
        ha="center", fontsize=8.6, color=INK,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.relative_to(ROOT)}")
    report(fname, files_read, lines)
    return out_path


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports" / "results",
                     help="directory containing the source result JSONs")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures" / "campaign10_followup",
                     help="directory to write the four PNGs to")
    ap.add_argument("--dpi", type=int, default=140, help="figure DPI")
    ap.add_argument("--only", type=str, default="1,2,3,4",
                     help="comma-separated subset of figure numbers to (re)generate, e.g. '1,3'")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.only.split(",") if x.strip()}

    fig_fns = {1: figure1, 2: figure2, 3: figure3, 4: figure4}
    out_paths = []
    for n in sorted(wanted):
        out_paths.append(fig_fns[n](args.results_dir, args.output_dir, args.dpi))

    print(f"\n=== verification ===")
    all_ok = True
    for p in out_paths:
        size = p.stat().st_size if p.exists() else 0
        ok = size > 10_000
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'} {p.relative_to(ROOT)}: {size:,} bytes")
    if not all_ok:
        raise SystemExit("one or more figures failed the >10KB non-empty check")
    print("all figures rendered and non-empty.")


if __name__ == "__main__":
    main()
