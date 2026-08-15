#!/usr/bin/env python3
"""Campaign 10 follow-up (2026-08-15): four figures over one day's frozen-readout,
relu-shift, and LR-ladder experiments (campaigns 10-12).

Figure 1 -- frozen_readout_audits.png
    `rcfrozen_{last3,last2}_audit_{fmnist,cifar10}_100L_rcfwd` -- 400-epoch
    frozen-readout audits (rcfwd recipe, 100L, only the named window
    trainable, head frozen). eval_train_accuracy vs test_accuracy per epoch:
    train climbs, test stays at chance -- the generalization gap is campaign
    12's central observation about this protocol.

Figure 2 -- relushift_<depth>L_arms.png (depth via --depth, default 30)
    `relushift_{he,rc,c010,c025,c070}_smoke_{fmnist,cifar10}_<depth>L` plus
    the `c025 ..._diff` (differentiable-gradient fork) control -- campaign
    11's end-to-end training grid. Final eval_train_accuracy / test_accuracy
    per arm, grouped by dataset, deltas vs `he` annotated. The 30L grid is
    committed as of 2026-08-15; a 100L grid (`relushift_{he,rc,c010,c025,
    c070}_smoke_{fmnist,cifar10}_100L`, no diff control) was queued the same
    day and is NOT yet on disk -- `--depth 100` renders whatever subset has
    landed and prints a coverage line for what is still missing, rather than
    failing. Re-running this script unchanged after those jobs land will
    pick them up.

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
are hardcoded. Each figure's (arm, dataset, depth) grid is declared as an
explicit request list and loaded defensively: a label whose JSON does not
exist yet is skipped with a printed note (not an error), and each figure
prints a "coverage" line -- how many of the requested labels were actually
found -- so a partially-landed cluster grid degrades gracefully and a
re-run picks up newly-landed JSONs with no code change.

Output: reports/figures/campaign10_followup/*.png.
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


def try_load(res_dir, stem):
    """Load reports/results/<stem>.json -> (payload, history).

    Returns (None, None) if the file does not exist yet. Callers MUST treat
    that as "skip this label", never as an error -- the 100L relushift grid
    (10 labels) was queued but had not landed as of this script's writing.
    """
    path = res_dir / f"{stem}.json"
    if not path.exists():
        return None, None
    payload = json.load(open(path))[0]
    return payload, payload.get("history") or []


def rel_or_abs(p):
    """str(p relative to ROOT), falling back to the absolute path if p lies
    outside ROOT (e.g. a caller pointed --output-dir elsewhere)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def report(fig_name, requested_stems, found_stems, lines):
    """Print the files read, a found/missing coverage line, and the plotted numbers."""
    missing = [s for s in requested_stems if s not in found_stems]
    print(f"\n=== {fig_name} ===")
    print(f"JSON files read ({len(found_stems)}):")
    for f in found_stems:
        print(f"  reports/results/{f}.json")
    print(f"coverage: {len(found_stems)}/{len(requested_stems)} requested labels found, {len(missing)} missing")
    if missing:
        print("  missing (not yet run on the cluster -- skipped, not an error):")
        for m in missing:
            print(f"    reports/results/{m}.json")
    print("Key numbers plotted:")
    for l in lines:
        print(f"  {l}")


# =============================================================================
# Figure 1 -- frozen_readout_audits.png
# =============================================================================

FIG1_CONDITIONS = [
    ("last3", "last3 (fc98–fc100 trainable)", BLUE),
    ("last2", "last2 (fc99–fc100 trainable)", SHIFT_MID),
]


def figure1(res_dir, out_dir, dpi):
    fname = "frozen_readout_audits.png"
    requested = [f"rcfrozen_{cond}_audit_{ds}_100L_rcfwd" for ds, _ in DATASETS for cond, _, _ in FIG1_CONDITIONS]
    found, lines = [], []
    series = {}  # (ds, cond) -> history

    for ds, _ in DATASETS:
        for cond, cond_label, color in FIG1_CONDITIONS:
            stem = f"rcfrozen_{cond}_audit_{ds}_100L_rcfwd"
            _, hist = try_load(res_dir, stem)
            if hist is None:
                continue
            found.append(stem)
            series[(ds, cond)] = hist
            tr = [e["eval_train_accuracy"] for e in hist]
            te = [e["test_accuracy"] for e in hist]
            lines.append(
                f"{stem}.json: n_epochs={len(hist)}, final train_acc={tr[-1]:.4f}, "
                f"final test_acc={te[-1]:.4f}, max test_acc over run={max(te):.4f}"
            )

    if not series:
        print(f"figure1: no data found -- skipping {fname}")
        report(fname, requested, found, lines)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    for ax, (ds, ds_title) in zip(axes, DATASETS):
        plotted = False
        for cond, cond_label, color in FIG1_CONDITIONS:
            hist = series.get((ds, cond))
            if hist is None:
                continue
            plotted = True
            ep = [e["epoch"] for e in hist]
            tr = [e["eval_train_accuracy"] for e in hist]
            te = [e["test_accuracy"] for e in hist]
            ax.plot(ep, tr, "-", color=color, lw=1.9, label=f"{cond_label} — train")
            ax.plot(ep, te, "--", color=color, lw=1.6, label=f"{cond_label} — test")
        ax.axhline(CHANCE, color=GRAY, ls=":", lw=1.3, zorder=1)
        ax.text(8, CHANCE + 0.02, "chance", fontsize=8.5, color=GRAY)
        ax.set_title(f"{ds_title} — 100L, 400 epochs", fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=8.3, loc="upper left", framealpha=0.9)
    axes[0].set_ylabel("accuracy (fraction of examples correct)")

    fig.suptitle(
        "Frozen-readout audits — rcfwd recipe (row_centered_forward_balanced + grad_rescale), "
        "SGD lr=1e-2 fixed, NoBN",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel_or_abs(out_path)}")
    report(fname, requested, found, lines)
    return out_path


# =============================================================================
# Figure 2 -- relushift_<depth>L_arms.png
# =============================================================================

FIG2_ARMS = ["he", "rc", "c010", "c025", "c025_diff", "c070"]
ARM_LABEL_FIG2 = {
    "he": "he", "rc": "rc", "c010": "shift c=0.10", "c025": "shift c=0.25",
    "c025_diff": "shift c=0.25\n(diff. grad)", "c070": "shift c=0.70",
}


def relushift_file(arm, ds, depth):
    if arm == "c025_diff":
        return f"relushift_c025_smoke_{ds}_{depth}L_diff"
    return f"relushift_{arm}_smoke_{ds}_{depth}L"


def _abort_note(payload, hist, expected_epochs=20):
    """Return a short marker when a run did not complete, else ''.

    Load-bearing: eight of the ten 100L relushift runs aborted on
    abort_on_explosion (some after 2 epochs), and their final-epoch values are
    NOT comparable with a completed run's. Any figure that plots a final value
    must say so on the face of the chart.
    """
    if payload is None or not hist:
        return ""
    if payload.get("abort_reason") or len(hist) < expected_epochs:
        return f"  ABORTED @ep{len(hist)}"
    return ""


def figure2(res_dir, out_dir, dpi, depth=30):
    fname = f"relushift_{depth}L_arms.png"
    requested = [relushift_file(arm, ds, depth) for ds, _ in DATASETS for arm in FIG2_ARMS]
    found, lines = [], []
    data = {}    # (ds, arm) -> (train_acc, test_acc)
    aborts = {}  # (ds, arm) -> "" or " ABORTED@epN"; see _abort_note

    for ds, _ in DATASETS:
        for arm in FIG2_ARMS:
            stem = relushift_file(arm, ds, depth)
            payload, hist = try_load(res_dir, stem)
            if hist is None:
                continue
            found.append(stem)
            aborts[(ds, arm)] = _abort_note(payload, hist)
            tr, te = hist[-1]["eval_train_accuracy"], hist[-1]["test_accuracy"]
            data[(ds, arm)] = (tr, te)
            lines.append(f"{stem}.json: final train_acc={tr:.4f}, test_acc={te:.4f} (n_epochs={len(hist)}){aborts[(ds, arm)]}")

    arms_present = [a for a in FIG2_ARMS if any((ds, a) in data for ds, _ in DATASETS)]
    if not arms_present:
        print(f"figure2 (depth={depth}L): no arm data found yet -- skipping {fname}")
        report(fname, requested, found, lines)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    x = np.arange(len(arms_present))
    w = 0.34
    best = None  # (delta_pp, ds, arm, arm_te, he_te) -- computed from data, never assumed

    for ax, (ds, ds_title) in zip(axes, DATASETS):
        he_pair = data.get((ds, "he"))
        for i, arm in enumerate(arms_present):
            pair = data.get((ds, arm))
            if pair is None:
                continue
            tr, te = pair
            color = ARM_COLOR[arm]
            aborted = bool(aborts.get((ds, arm)))
            hatch = "xxx" if aborted else ("////" if arm == "c025_diff" else None)
            ew = 2.2 if arm == "he" else 0.6
            if aborted:
                ax.text(i, 0.03, aborts[(ds, arm)].strip(), ha="center", fontsize=6.8,
                        color="#b3261e", fontweight="bold", rotation=90, va="bottom")
            ax.bar(i - w / 2, tr, w, color=color, alpha=1.0, hatch=hatch, edgecolor="black", linewidth=ew)
            ax.bar(i + w / 2, te, w, color=color, alpha=0.55, hatch=hatch, edgecolor="black", linewidth=ew)
            if arm != "he" and he_pair is not None:
                d_tr, d_te = (tr - he_pair[0]) * 100, (te - he_pair[1]) * 100
                ax.text(i - w / 2, tr + 0.015, f"{d_tr:+.1f}", ha="center", fontsize=7, color=INK)
                ax.text(i + w / 2, te + 0.015, f"{d_te:+.1f}", ha="center", fontsize=7, color=INK)
                if best is None or d_te > best[0]:
                    best = (d_te, ds, arm, te, he_pair[1])
        if he_pair is not None:
            ax.axhline(he_pair[0], color=BLUE, ls=":", lw=1.0, alpha=0.55, zorder=0)
            ax.axhline(he_pair[1], color=BLUE, ls=":", lw=1.0, alpha=0.55, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL_FIG2[a] for a in arms_present], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy (fraction correct)")
        n_ab = sum(1 for a in arms_present if aborts.get((ds, a)))
        suffix = f" — {n_ab}/{len(arms_present)} ABORTED early" if n_ab else ""
        ax.set_title(f"{ds_title} — {depth}L, 20 epochs{suffix}",
                     fontsize=10.5, color=("#b3261e" if n_ab else INK))
        ax.grid(axis="y", alpha=0.25)

    axes[0].text(0.015, 0.985, "bar-top numbers = Δ vs he (percentage points)",
                 transform=axes[0].transAxes, va="top", fontsize=7.6, color=GRAY)

    if any(aborts.values()):
        best = None   # a delta against an aborted run is not a result
    # Headline callout: whichever (dataset, arm) has the single largest test-accuracy
    # delta vs he, found dynamically -- not assumed to be any particular arm, so this
    # stays correct whether it lands on c010 (the 30L headline) or something else once
    # the 100L grid is in.
    if best is not None:
        d_te, best_ds, best_arm, arm_te, he_te = best
        ax_idx = [ds for ds, _ in DATASETS].index(best_ds)
        i_arm = arms_present.index(best_arm)
        label_line1 = ARM_LABEL_FIG2[best_arm].splitlines()[0]
        xt = min(i_arm + 1.4, len(arms_present) - 0.7)
        axes[ax_idx].annotate(
            f"{label_line1} test: {arm_te:.3f}\nvs he test: {he_te:.3f}\nΔ = {d_te:+.1f}pp",
            xy=(i_arm + w / 2, arm_te), xytext=(xt, min(arm_te + 0.22, 0.95)),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), fontsize=8.3, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bbb", lw=0.8),
        )

    train_patch = mpatches.Patch(facecolor="white", alpha=1.0, edgecolor="black", linewidth=0.6, label="train (eval_train_accuracy)")
    test_patch = mpatches.Patch(facecolor="white", alpha=0.55, edgecolor="black", linewidth=0.6, label="test (test_accuracy)")
    he_patch = mpatches.Patch(facecolor="white", edgecolor="black", linewidth=2.2, label="he = baseline (thick edge)")
    legend_handles = [train_patch, test_patch, he_patch]
    if "c025_diff" in arms_present:
        legend_handles.append(mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="differentiable-grad fork"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=8.3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        f"Campaign 11 — {depth}-layer FC, NoBN, SGD lr=1e-2, 20-epoch end-to-end training: "
        "final accuracy by initialization arm",
        fontsize=11.3,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel_or_abs(out_path)}")
    report(fname, requested, found, lines)
    return out_path


# =============================================================================
# Figure 3 -- frozen_readout_ranking.png
# =============================================================================

ARM_ORDER_FIG3 = ["he", "rc", "rcfwd", "c010", "c025", "c070"]
ARM_LABEL_FIG3 = {
    "he": "he", "rc": "rc", "rcfwd": "rcfwd",
    "c010": "shift c=0.10", "c025": "shift c=0.25", "c070": "shift c=0.70",
}
FIG3_DEPTHS = [30, 100]


def frozenro_file(arm, ds, depth):
    if arm == "rcfwd" and depth == 100:
        # frozenro_rcfwd_smoke_*_100L was never submitted (campaign 12 README:
        # "do not resubmit the rcfwd arm if campaign 10's rcfrozen_last2_smoke
        # is already ... committed -- it is the same run"). Substitute it.
        return f"rcfrozen_last2_smoke_{ds}_100L_rcfwd"
    return f"frozenro_{arm}_smoke_{ds}_{depth}L"


def figure3(res_dir, out_dir, dpi):
    fname = "frozen_readout_ranking.png"
    requested = [frozenro_file(arm, ds, depth) for ds, _ in DATASETS for depth in FIG3_DEPTHS for arm in ARM_ORDER_FIG3]
    found, lines = [], []
    data = {}  # (ds, depth, arm) -> (train_acc, test_acc)

    for ds, _ in DATASETS:
        for depth in FIG3_DEPTHS:
            for arm in ARM_ORDER_FIG3:
                stem = frozenro_file(arm, ds, depth)
                _, hist = try_load(res_dir, stem)
                if hist is None:
                    continue
                found.append(stem)
                tr, te = hist[-1]["eval_train_accuracy"], hist[-1]["test_accuracy"]
                data[(ds, depth, arm)] = (tr, te)
                lines.append(
                    f"{stem}.json: {ds}/{depth}L/{arm}: final train_acc={tr:.4f}, "
                    f"test_acc={te:.4f} (n_epochs={len(hist)})"
                )

    if not data:
        print(f"figure3: no data found -- skipping {fname}")
        report(fname, requested, found, lines)
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.4), sharey="col")
    x = np.arange(len(ARM_ORDER_FIG3))
    w = 0.34
    depth_hatch = {30: None, 100: "////"}

    for col, (ds, ds_title) in enumerate(DATASETS):
        for row, (metric_key, metric_label) in enumerate([("train", "train accuracy"), ("test", "test accuracy")]):
            ax = axes[row, col]
            for i, arm in enumerate(ARM_ORDER_FIG3):
                color = ARM_COLOR[arm]
                for j, depth in enumerate(FIG3_DEPTHS):
                    pair = data.get((ds, depth, arm))
                    if pair is None:
                        continue
                    val = pair[0] if metric_key == "train" else pair[1]
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
        "Campaign 12 — frozen-readout ranking: fc99–fc100 trainable, head frozen, "
        "NoBN, SGD lr=1e-2, 20-epoch smokes, 30L vs 100L",
        fontsize=11.3,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel_or_abs(out_path)}")
    report(fname, requested, found, lines)
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
    requested = [f"rcfrozen_first2_smoke_{ds}_100L_rawrescale{suf}" for ds, _ in DATASETS for suf, _, _ in LR_RUNGS]
    found, lines = [], []
    series = {}  # (ds, suf) -> (hist, lr_val)
    all_dev = []

    for ds, _ in DATASETS:
        for suf, _, _ in LR_RUNGS:
            stem = f"rcfrozen_first2_smoke_{ds}_100L_rawrescale{suf}"
            payload, hist = try_load(res_dir, stem)
            if hist is None:
                continue
            found.append(stem)
            lr_val = payload["training_config"]["learning_rate"]
            series[(ds, suf)] = (hist, lr_val)
            loss = [e["eval_train_loss"] for e in hist]
            dev = [abs(v - LN10) for v in loss]
            all_dev.extend(dev)
            lines.append(
                f"{stem}.json: lr={lr_val:g}, n_epochs={len(hist)}, "
                f"eval_train_loss range=[{min(loss):.8f}, {max(loss):.8f}], "
                f"max|loss-ln10|={max(dev):.3e}"
            )

    if not series:
        print(f"figure4: no data found -- skipping {fname}")
        report(fname, requested, found, lines)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    for ax, (ds, ds_title) in zip(axes, DATASETS):
        for k, (suf, color, marker) in enumerate(LR_RUNGS):
            entry = series.get((ds, suf))
            if entry is None:
                continue
            hist, lr_val = entry
            ep = [e["epoch"] for e in hist]
            loss = [e["eval_train_loss"] for e in hist]
            offset = k * OFFSET_STEP
            plotted = [v + offset for v in loss]
            ax.plot(ep, plotted, "-", color=color, lw=1.3, marker=marker, ms=5.5,
                     markevery=2, label=f"lr={lr_val:g}")
        ax.axhline(LN10, color="black", ls="--", lw=1.2, zorder=0)
        ax.set_title(f"{ds_title} — 100L, first2 window, rawrescale recipe", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(LN10 - 0.0006, LN10 + 4 * OFFSET_STEP + 0.0008)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("eval_train_loss (nats)\noffset per rung for visibility — see annotation")
    axes[0].text(1, LN10 - 0.00045, "ln 10 = 2.302585 (true value, no offset)", fontsize=7.8, color="#333333")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.015), title="learning rate (fc1,fc2 trainable; 1e-2 → 1e7)", frameon=False)

    max_dev_str = f"{max(all_dev):.2e}" if all_dev else "n/a"
    fig.suptitle(
        "Campaign 10 W5 — front-window LR ladder: 100L FC, rawrescale recipe "
        "(row_centered_he + grad_rescale), NoBN, 20-epoch smokes, LR swept 1e-2–1e7\n"
        f"max |eval_train_loss − ln 10| = {max_dev_str} across all rungs × epochs × datasets "
        "(curves offset by 0.0005/rung for visibility only)",
        fontsize=10.8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel_or_abs(out_path)}")
    report(fname, requested, found, lines)
    return out_path


# =============================================================================
# main
# =============================================================================


# =============================================================================
# Figure 5 -- epoch_choice_artifact.png
# Why the 30L "win" was not one: it depends entirely on which epoch you read.
# =============================================================================

FIG5_ARMS = [("c010", "c = 0.10", SHIFT_LIGHT), ("c025", "c = 0.25", SHIFT_MID),
             ("c070", "c = 0.70", SHIFT_DARK), ("rc", "row_centered_he", VIOLET)]


def figure5(res_dir, out_dir, dpi):
    """Left: He's raw CIFAR-10 test curve at 30L, showing it lands on its worst
    value at the last epoch. Right: every arm's delta vs He under three
    estimators. The +6.6 pp headline survives only the final-epoch reading."""
    fname = "epoch_choice_artifact.png"
    requested, found, lines = [], [], []

    def series(stem):
        requested.append(stem)
        _, h = try_load(res_dir, stem)
        if not h:
            return None
        found.append(stem)
        return [x["test_accuracy"] for x in h]

    he = series("relushift_he_smoke_cifar10_30L")
    arms = [(lab, col, series(f"relushift_{a}_smoke_cifar10_30L")) for a, lab, col in FIG5_ARMS]
    if he is None:
        report(fname, requested, found, ["SKIPPED -- He baseline missing"])
        return None

    est = {"final epoch": lambda v: v[-1],
           "mean, last 5": lambda v: float(np.mean(v[-5:])),
           "best epoch": lambda v: max(v)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    ep = np.arange(1, len(he) + 1)
    axL.plot(ep, he, color=BLUE, lw=2, marker="o", ms=3.4, label="he (baseline)")
    for lab, col, v in arms:
        if v:
            axL.plot(ep, v, color=col, lw=1.4, alpha=.85, label=lab)
    bi = int(np.argmax(he))
    axL.scatter([bi + 1], [he[bi]], s=90, facecolor="none", edgecolor=INK, zorder=5, lw=1.6)
    axL.annotate(f"He best  {he[bi]:.4f}\n(epoch {bi+1})", (bi + 1, he[bi]),
                 textcoords="offset points", xytext=(-4, 16), ha="right", fontsize=8.5, color=INK)
    axL.scatter([len(he)], [he[-1]], s=90, facecolor="none", edgecolor="#b3261e", zorder=5, lw=1.6)
    axL.annotate(f"He final  {he[-1]:.4f}\n(read here)", (len(he), he[-1]),
                 textcoords="offset points", xytext=(-6, -30), ha="right", fontsize=8.5, color="#b3261e")
    axL.set(xlabel="epoch", ylabel="test accuracy",
            title="CIFAR-10 test accuracy, 30 layers, end-to-end")
    axL.grid(alpha=.25, lw=.6)
    axL.legend(fontsize=8, loc="lower left", framealpha=.9)

    names = list(est)
    x = np.arange(len(names)); w = 0.2
    for i, (lab, col, v) in enumerate(arms):
        if not v:
            continue
        d = [(est[n](v) - est[n](he)) * 100 for n in names]
        axR.bar(x + (i - 1.5) * w, d, w, color=col, edgecolor=INK, lw=.5, label=lab)
        for xi, di in zip(x + (i - 1.5) * w, d):
            axR.annotate(f"{di:+.1f}", (xi, di), textcoords="offset points",
                         xytext=(0, 3 if di >= 0 else -11), ha="center", fontsize=7.4)
        lines.append(f"{lab}: " + "  ".join(f"{n}={dd:+.1f}pp" for n, dd in zip(names, d)))
    axR.axhline(0, color=INK, lw=1)
    axR.set(xticks=x, ylabel="test accuracy vs he (percentage points)",
            title="The same comparison, three ways of reading it")
    axR.set_xticklabels(names)
    axR.grid(axis="y", alpha=.25, lw=.6)
    axR.legend(fontsize=8, ncol=2, framealpha=.9)

    fig.suptitle("Why the 30-layer result is not a win: the effect exists only at the final epoch",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    lines.append(f"He CIFAR-10 30L: best={max(he):.4f} @ep{bi+1}, final={he[-1]:.4f}, drop={max(he)-he[-1]:.4f}")
    report(fname, requested, found, lines)
    print(f"  wrote {rel_or_abs(out)} ({out.stat().st_size/1024:.0f} KB)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports" / "results",
                     help="directory containing the source result JSONs")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures" / "campaign10_followup",
                     help="directory to write the PNGs to")
    ap.add_argument("--dpi", type=int, default=140, help="figure DPI")
    ap.add_argument("--only", type=str, default="1,2,3,4,5",
                     help="comma-separated subset of figure numbers to (re)generate, e.g. '1,3'")
    ap.add_argument("--depth", type=int, default=30, choices=[30, 100],
                     help="depth for figure 2 (campaign 11 end-to-end relushift arms). "
                          "The 100L grid was queued 2026-08-15 and may still be landing; "
                          "figure2 renders whatever subset is present and reports coverage.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {int(x) for x in args.only.split(",") if x.strip()}

    out_paths = []
    if 1 in wanted:
        out_paths.append(figure1(args.results_dir, args.output_dir, args.dpi))
    if 2 in wanted:
        out_paths.append(figure2(args.results_dir, args.output_dir, args.dpi, depth=args.depth))
    if 3 in wanted:
        out_paths.append(figure3(args.results_dir, args.output_dir, args.dpi))
    if 4 in wanted:
        out_paths.append(figure4(args.results_dir, args.output_dir, args.dpi))
    if 5 in wanted:
        out_paths.append(figure5(args.results_dir, args.output_dir, args.dpi))
    out_paths = [p for p in out_paths if p is not None]

    print("\n=== verification ===")
    all_ok = True
    for p in out_paths:
        size = p.stat().st_size if p.exists() else 0
        ok = size > 10_000
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'} {rel_or_abs(p)}: {size:,} bytes")
    if not out_paths:
        print("  (no figures were generated -- nothing to verify)")
    elif not all_ok:
        raise SystemExit("one or more figures failed the >10KB non-empty check")
    else:
        print("all figures rendered and non-empty.")


if __name__ == "__main__":
    main()
