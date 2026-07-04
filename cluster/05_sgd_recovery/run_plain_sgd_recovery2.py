#!/usr/bin/env python3
"""Plain-SGD recovery — round 2 — addressing the 3 remaining 100L failures.

Recovery round 1 results:
  ✅ fmnist /100L/NoBN  PASS (epoch 152, 0.9953 train, 0.0157 loss)
  ✗  cifar10/100L/NoBN  near-miss (0.9865 train, 0.0458 loss — passes loss bar,
                          missed train-acc bar 0.995 — LR didn't drop aggressively
                          enough; plateau patience=10 only triggered once at ~ep 150)
  ✗  fmnist /100L/BN    BN running stats blew up (eval_train_loss = 5.5e8)
                          → bn_momentum=0.1 too aggressive; need bn_momentum=0.01
  ✗  cifar10/100L/BN    NaN at epoch 0 (BN running stats overflowed immediately)
                          → same fix: bn_momentum=0.01

Round 2 changes per architecture:
  1. cifar10/100L/NoBN:  plateau patience 10→5, min_lr 1e-6→1e-7
                          (more frequent LR halvings within the 200-epoch budget)
  2. fmnist /100L/BN:    bn_momentum 0.1→0.01 (matches our previously-best BN setup)
  3. cifar10/100L/BN:    bn_momentum 0.1→0.01

Everything else stays identical to recovery1: SGD lr=1e-3, momentum=0,
weight_decay=0, batch_size=128, depth=100, width=500, He init, 200 epochs,
early-stop on eval_train_acc ≥ 0.995.

200 epochs across all archs keeps the comparison unified.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def _build(label: str) -> tuple:
    """Per-architecture recipe. Returns (ClassifierConfig, TrainingConfig)."""

    if label == "recovery2_plain_sgd_cifar10_100L_nobn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=False, fc_hidden_dim=500,
        )
        tc = TrainingConfig(
            dataset="cifar10", epochs=200, batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5,     # was 10 in recovery1 — drop LR more often
            plateau_factor=0.5,
            plateau_min_lr=1e-7,    # was 1e-6 — allow finer late-stage descent
            plateau_metric="eval_train_loss",
            normalize_inputs=True,
            log_every_epoch=True, diagnostics_every=10,
            target_train_accuracy=0.995, target_patience=1,
            target_metric="eval_train_accuracy",
        )
        return cc, tc

    if label == "recovery2_plain_sgd_fmnist_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,       # was 0.1 (PyTorch default) — too aggressive at depth 100
        )
        tc = TrainingConfig(
            dataset="fashion_mnist", epochs=200, batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=10, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            normalize_inputs=True,
            log_every_epoch=True, diagnostics_every=10,
            target_train_accuracy=0.995, target_patience=1,
            target_metric="eval_train_accuracy",
        )
        return cc, tc

    if label == "recovery2_plain_sgd_cifar10_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,       # same fix as fmnist BN
        )
        tc = TrainingConfig(
            dataset="cifar10", epochs=200, batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=10, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            normalize_inputs=True,
            log_every_epoch=True, diagnostics_every=10,
            target_train_accuracy=0.995, target_patience=1,
            target_metric="eval_train_accuracy",
        )
        return cc, tc

    raise ValueError(f"Unknown experiment label: {label}")


EXPERIMENTS = [
    "recovery2_plain_sgd_cifar10_100L_nobn",
    "recovery2_plain_sgd_fmnist_100L_bn",
    "recovery2_plain_sgd_cifar10_100L_bn",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS,
                        help="Which single architecture to run (one job runs one architecture).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None,
                        help="JSON output path. Default: reports/results/<experiment>.json")
    args = parser.parse_args()

    label = args.experiment
    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(label)

    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm} "
          f"bn_momentum={classifier_config.bn_momentum}")
    print(f"# optimizer=SGD lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (patience={training_config.plateau_patience} "
          f"factor={training_config.plateau_factor} min_lr={training_config.plateau_min_lr})")
    print(f"# epochs={training_config.epochs} early_stop_target={training_config.target_train_accuracy}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n")

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    payload = [_result_to_payload(label, result)]
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved 1 run to {output_path}")

    hist = payload[0].get("history", [])
    if hist:
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_test = hist[-1].get("test_accuracy", 0) or 0
        passes = final_acc >= 0.995 and final_loss <= 0.10
        flag = "PASS" if passes else "fail"
        print(f"\nSUMMARY {label} | {flag} | best_train={best_acc:.4f} "
              f"final_train={final_acc:.4f} final_loss={final_loss:.4f} "
              f"final_test={final_test:.4f} epochs_ran={len(hist)}")


if __name__ == "__main__":
    main()
