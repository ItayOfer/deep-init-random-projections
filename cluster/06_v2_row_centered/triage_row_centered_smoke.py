#!/usr/bin/env python3
"""Triage the V2 row-centered smoke results.

Loads all reports/results/row_centered_smoke_*.json files and classifies each
architecture by training dynamics, then prints a per-arch summary plus a
suggested action (extend to 200 ep / modify recipe / mark as known-failure).

Run from the project root after the 10 smoke jobs complete:
    python3 cluster/triage_row_centered_smoke.py

Triage categories:
  - ABORTED        — explosion guard or non-finite loss caught it; recipe needs change.
  - DEAD           — all per-layer grads at floor; LR is irrelevant; needs init or BN.
  - PASS           — eval_train_accuracy >= 0.995 by smoke end (well-converged).
  - LEARNING       — eval_train_loss strictly decreasing across the 20 ep; promising.
  - STUCK          — eval_train_loss flat near initial; gradient profile may show
                    why (vanishing or ill-conditioning).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "reports" / "results"

ARCHITECTURE_KEYS = [
    "cifar10_30L_nobn",
    "cifar10_30L_bn",
    "cifar10_50L_nobn",
    "cifar10_50L_bn",
    "cifar10_100L_nobn",
    "fmnist_30L_nobn",
    "fmnist_30L_bn",
    "fmnist_50L_nobn",
    "fmnist_50L_bn",
    "fmnist_100L_nobn",
]


def _classify(payload: dict) -> dict:
    """Return triage verdict + diagnostics."""
    history = payload.get("history", [])
    status = payload.get("status", "?")
    stop_reason = payload.get("stop_reason", "")
    abort_reason = payload.get("abort_reason")
    initial_batch_loss = payload.get("initial_batch_loss")

    verdict = "?"
    notes = []
    suggested = "?"

    if status == "diverged":
        verdict = "ABORTED"
        if abort_reason:
            notes.append(f"abort: {abort_reason}")
        if stop_reason == "loss_exploded":
            suggested = "lower LR or add warmup; smoke at half the LR"
        elif stop_reason == "non_finite_loss":
            suggested = "drastic LR cut or grad-clip; smoke at LR/10 + grad_clip_max_norm=1.0"
        else:
            suggested = "modify recipe (unknown failure mode)"
    elif not history:
        verdict = "NO_HISTORY"
        notes.append("no epochs completed — early abort before first eval")
        suggested = "investigate logs"
    else:
        first_loss = history[0].get("eval_train_loss")
        last_loss = history[-1].get("eval_train_loss")
        last_acc = history[-1].get("eval_train_accuracy", 0.0) or 0.0
        peak_acc = max((h.get("eval_train_accuracy", 0) or 0) for h in history)

        # Per-layer gradient diagnostic at the last available checkpoint
        grad_min, grad_max, grad_ratio = None, None, None
        diag = [h for h in history if h.get("grad_norm_per_layer")]
        if diag:
            g = diag[-1]["grad_norm_per_layer"]
            nonzero = [x for x in g if x > 0]
            grad_min = min(nonzero) if nonzero else 0.0
            grad_max = max(g) if g else 0.0
            grad_ratio = (grad_max / grad_min) if grad_min > 0 else float("inf")

        if last_acc >= 0.995:
            verdict = "PASS"
            suggested = "extend to 200 ep with same recipe"
        elif grad_min is not None and grad_min == 0.0:
            verdict = "DEAD"
            suggested = "init issue — try adding BN or different eta"
        elif first_loss is not None and last_loss is not None and last_loss < 0.95 * first_loss:
            verdict = "LEARNING"
            suggested = "extend to 200 ep with same recipe"
        elif first_loss is not None and last_loss is not None and last_loss < first_loss:
            verdict = "SLOW_LEARNING"
            suggested = "extend to 200 ep; tighten plateau patience or boost LR slightly"
        else:
            verdict = "STUCK"
            if grad_ratio is not None and grad_ratio > 1e3:
                suggested = "ill-conditioned — try Adam if SGD, or grad clip"
            elif grad_min is not None and grad_min < 1e-6:
                suggested = "vanishing — try a smaller eta or different init"
            else:
                suggested = "modify recipe"

        notes.append(
            f"loss {first_loss:.3f}→{last_loss:.3f} acc peak={peak_acc:.3f} last={last_acc:.3f}"
        )
        if grad_ratio is not None:
            notes.append(
                f"grad@last: min={grad_min:.2e} max={grad_max:.2e} ratio={grad_ratio:.1e}"
            )

    return {
        "verdict": verdict,
        "stop_reason": stop_reason,
        "epochs": len(history),
        "initial_batch_loss": initial_batch_loss,
        "notes": notes,
        "suggested": suggested,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Looking for V2 smoke JSONs in {results_dir}")
    print()

    found = []
    missing = []
    for arch in ARCHITECTURE_KEYS:
        path = results_dir / f"row_centered_smoke_{arch}.json"
        if path.exists():
            found.append((arch, path))
        else:
            missing.append(arch)

    if missing:
        print(f"MISSING ({len(missing)}/{len(ARCHITECTURE_KEYS)}):")
        for arch in missing:
            print(f"  - row_centered_smoke_{arch}.json")
        print()

    if not found:
        print("No smoke JSONs found yet. Returning.")
        return

    print(f"FOUND {len(found)}/{len(ARCHITECTURE_KEYS)} smoke results.")
    print()
    print("=" * 110)
    print(f"{'Architecture':<22} {'Verdict':<14} {'Eps':>4} {'InitLoss':>10}  Notes")
    print("=" * 110)

    verdict_counts = {}
    for arch, path in found:
        with open(path) as f:
            data = json.load(f)
        payload = data[0] if isinstance(data, list) else data
        result = _classify(payload)
        verdict_counts[result["verdict"]] = verdict_counts.get(result["verdict"], 0) + 1
        init_loss_str = f"{result['initial_batch_loss']:.4f}" if result["initial_batch_loss"] else "n/a"
        notes_str = " | ".join(result["notes"]) if result["notes"] else ""
        print(f"{arch:<22} {result['verdict']:<14} {result['epochs']:>4} {init_loss_str:>10}  {notes_str}")

    print()
    print("=" * 110)
    print("SUGGESTED ACTIONS")
    print("=" * 110)
    for arch, path in found:
        with open(path) as f:
            data = json.load(f)
        payload = data[0] if isinstance(data, list) else data
        result = _classify(payload)
        print(f"  {arch:<22} ({result['verdict']:<14}) → {result['suggested']}")

    print()
    print("Verdict tally:", verdict_counts)


if __name__ == "__main__":
    main()
