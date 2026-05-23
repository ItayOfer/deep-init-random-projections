#!/usr/bin/env python3
"""Run short diagnostic experiments with enhanced logging.

Designed for the iterative tuning workflow: run 30 epochs, inspect gradient
norms / BN stats / LR schedule, form a hypothesis, adjust, repeat.

Each experiment is specified inline in a config list (not loaded from JSON)
so that hypotheses are self-documenting.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import (
    EpochMetrics,
    SupervisedTrainingResult,
    run_supervised_experiment,
)


def build_phase1_configs(
    epochs: int = 30,
    diagnostics_every: int = 5,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    """Phase 1 diagnostic experiments for the 7 failing architectures.

    Returns list of (hypothesis_label, classifier_config, training_config).
    """
    configs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    # --- fmnist/50L/NoBN: LR overshoot hypothesis ---
    # Original: SGD+OneCycle lr=0.01 -> crash at epoch 22 when LR ramps up
    # Hypothesis A: cosine decay avoids the ramp-up destabilization
    configs.append((
        "fmnist_50L_nobn_sgd_cosine",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="sgd", learning_rate=0.01, momentum=0.9,
                       scheduler="cosine",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # Hypothesis B: constant lower LR avoids overshoot entirely
    configs.append((
        "fmnist_50L_nobn_sgd_const_lr003",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="sgd", learning_rate=0.003, momentum=0.9,
                       scheduler="none",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # Hypothesis C: Adam + cosine (different optimizer)
    configs.append((
        "fmnist_50L_nobn_adam_cosine",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # --- cifar10/50L/BN: BN running stats drift hypothesis ---
    # Original: Adam+Cosine lr=0.001, bn_m=0.05 -> collapse at epoch ~40
    # Hypothesis D: even slower running stats (bn_m=0.01)
    configs.append((
        "cifar10_50L_bn_adam_cosine_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500,
                         bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # Hypothesis E: lower LR for stability + slow BN
    configs.append((
        "cifar10_50L_bn_adam_cosine_lr0003_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500,
                         bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.0003,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # --- fmnist/50L/BN: same BN drift, different dataset ---
    # Hypothesis F: slow BN + cosine on fmnist
    configs.append((
        "fmnist_50L_bn_adam_cosine_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500,
                         bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # --- 100L/NoBN: confirm gradient death ---
    # Hypothesis G: gradient logging to confirm zero gradients
    configs.append((
        "cifar10_100L_nobn_gradient_death_check",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="cifar10", epochs=min(epochs, 10),
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine",
                       log_every_epoch=True, diagnostics_every=1,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    return configs


def _result_to_payload(label: str, result: SupervisedTrainingResult) -> dict:
    payload = asdict(result)
    payload["history"] = [asdict(epoch) for epoch in result.history]
    payload["hypothesis_label"] = label
    return payload


def print_diagnostic_summary(label: str, result: SupervisedTrainingResult) -> None:
    """Print a concise diagnostic summary of one run."""
    hist = result.history
    if not hist:
        print(f"\n{'='*60}\n{label}: NO HISTORY (diverged immediately?)\n{'='*60}")
        return

    best = max(hist, key=lambda h: h.eval_train_accuracy or 0)
    final = hist[-1]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Status: {result.status} ({result.stop_reason})")
    print(f"  Epochs: {result.epochs_ran}")
    print(f"  Best eval_train: {best.eval_train_accuracy:.4f} @ epoch {best.epoch}")
    print(f"  Final eval_train: {final.eval_train_accuracy:.4f} (loss={final.eval_train_loss:.4f})")
    print(f"  Final test: {final.test_accuracy:.4f}")
    print(f"  LR range: {hist[0].learning_rate:.6f} -> {final.learning_rate:.6f}")

    improving = (
        final.eval_train_accuracy is not None
        and best.eval_train_accuracy is not None
        and final.eval_train_accuracy >= best.eval_train_accuracy * 0.95
    )
    collapsed = (
        best.eval_train_accuracy is not None
        and final.eval_train_accuracy is not None
        and best.eval_train_accuracy - final.eval_train_accuracy > 0.1
    )
    dead = final.eval_train_accuracy is not None and final.eval_train_accuracy <= 0.11

    if dead:
        print("  Verdict: DEAD (stuck at random chance)")
    elif collapsed:
        delta = best.eval_train_accuracy - final.eval_train_accuracy
        print(f"  Verdict: COLLAPSED (delta={delta:.4f} from best)")
    elif improving:
        print("  Verdict: LEARNING (stable or improving)")
    else:
        print("  Verdict: PARTIAL (some learning, not converging)")

    diag_epochs = [h for h in hist if h.grad_norm_per_layer is not None]
    if diag_epochs:
        print(f"\n  Gradient norms (sampled at {len(diag_epochs)} epochs):")
        for h in diag_epochs[:5]:
            norms = h.grad_norm_per_layer
            print(
                f"    epoch {h.epoch:3d}: "
                f"min={min(norms):.2e} max={max(norms):.2e} "
                f"ratio={max(norms)/max(min(norms), 1e-30):.1f}"
            )
        if len(diag_epochs) > 5:
            h = diag_epochs[-1]
            norms = h.grad_norm_per_layer
            print(
                f"    epoch {h.epoch:3d}: "
                f"min={min(norms):.2e} max={max(norms):.2e} "
                f"ratio={max(norms)/max(min(norms), 1e-30):.1f}"
            )

    bn_epochs = [h for h in hist if h.bn_running_var_mean is not None]
    if bn_epochs:
        print(f"\n  BN running var (mean across layers, sampled at {len(bn_epochs)} epochs):")
        for h in bn_epochs[:3]:
            avg_var = sum(h.bn_running_var_mean) / len(h.bn_running_var_mean)
            print(f"    epoch {h.epoch:3d}: avg_var={avg_var:.4f}")
        if len(bn_epochs) > 3:
            h = bn_epochs[-1]
            avg_var = sum(h.bn_running_var_mean) / len(h.bn_running_var_mean)
            print(f"    epoch {h.epoch:3d}: avg_var={avg_var:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--diagnostics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated list of experiment labels to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "diagnostic_phase1.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_phase1_configs(
        epochs=args.epochs,
        diagnostics_every=args.diagnostics_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    if args.experiments:
        selected = {s.strip() for s in args.experiments.split(",")}
        all_configs = [(l, cc, tc) for l, cc, tc in all_configs if l in selected]
        if not all_configs:
            print(f"No experiments matched: {args.experiments}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_payload: List[dict] = []
    total = len(all_configs)

    for run_idx, (label, classifier_config, training_config) in enumerate(all_configs, 1):
        print(f"\n{'#'*60}")
        print(f"# [{run_idx}/{total}] {label}")
        print(f"{'#'*60}\n")

        exp_config.setup_seeds()
        result = run_supervised_experiment(exp_config, classifier_config, training_config)
        print_diagnostic_summary(label, result)

        all_payload.append(_result_to_payload(label, result))
        output_path.write_text(json.dumps(all_payload, indent=2))

    print(f"\n\nSaved {len(all_payload)} diagnostic runs to {output_path}")

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    for entry in all_payload:
        hist = entry.get("history", [])
        if not hist:
            continue
        best_eta = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_eta = hist[-1].get("eval_train_accuracy", 0) or 0
        label = entry["hypothesis_label"]
        print(f"  {label:50s} best={best_eta:.4f} final={final_eta:.4f}")


if __name__ == "__main__":
    main()
