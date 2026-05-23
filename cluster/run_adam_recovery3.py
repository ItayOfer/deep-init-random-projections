#!/usr/bin/env python3
"""Recovery round 3 — back to Adam for the 2 remaining 100L/BN failures.

Background: rounds 1 and 2 used plain SGD (the rescue for 100L/NoBN). For BN
cases, plain SGD turned out to be too weak — fmnist/100L/BN's train_loss
stuck at 2.29 (uniform output) for all 200 epochs; cifar10/100L/BN diverged
at epoch 0 (bn_momentum was irrelevant because the NaN happened before any
running-stat update). The audit-best optimizer for both BN cases was Adam
anyway (fmnist 0.38, cifar10 0.15), so this round returns to Adam with
stricter safeguards.

Recipe (common to both jobs):
  - Adam, lr = 1e-3
  - depth = 100, width = 500, He init
  - bn_momentum = 0.01  (avoid running-statistic blowup)
  - batch_size = 256    (matches the audit recipe for BN runs)
  - plateau scheduler:  patience = 5, factor = 0.5, min_lr = 1e-7
  - gradient clipping:  max_norm = 1.0  (prevents single-batch overflow)
  - 200 epochs, early-stop on eval_train_acc >= 0.995
  - normalize_inputs = True

Per-architecture differences:
  - cifar10/100L/BN additionally gets LR warmup (linear ramp from
    lr * 0.01 to lr over the first 20 epochs) AND plateau_warmup_epochs=20
    so the plateau scheduler doesn't try to halve LR before warmup finishes.
    Rationale: cifar10's epoch-0 NaN happens in the very first forward+backward
    pass — only a small initial LR avoids it.
  - fmnist/100L/BN starts at lr=1e-3 directly; the audit run reached 0.38
    without any warmup, so this run probes whether tighter plateau gets us
    further.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def _build(label: str) -> Tuple[ClassifierConfig, TrainingConfig]:
    if label == "recovery3_adam_fmnist_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,
        )
        tc = TrainingConfig(
            dataset="fashion_mnist", epochs=200, batch_size=256,
            optimizer="adam", learning_rate=1e-3,
            weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            grad_clip_max_norm=1.0,
            lr_warmup_epochs=0,           # no warmup needed — audit ran fine without it
            normalize_inputs=True,
            log_every_epoch=True, diagnostics_every=10,
            target_train_accuracy=0.995, target_patience=1,
            target_metric="eval_train_accuracy",
        )
        return cc, tc

    if label == "recovery3_adam_cifar10_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,
        )
        tc = TrainingConfig(
            dataset="cifar10", epochs=200, batch_size=256,
            optimizer="adam", learning_rate=1e-3,
            weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            grad_clip_max_norm=1.0,
            lr_warmup_epochs=20,                 # linear ramp 1e-5 -> 1e-3 over 20 epochs
            lr_warmup_start_factor=0.01,         # start LR = 1e-5
            plateau_warmup_epochs=20,            # don't trigger plateau during warmup
            normalize_inputs=True,
            log_every_epoch=True, diagnostics_every=10,
            target_train_accuracy=0.995, target_patience=1,
            target_metric="eval_train_accuracy",
        )
        return cc, tc

    raise ValueError(f"Unknown experiment label: {label}")


EXPERIMENTS = [
    "recovery3_adam_fmnist_100L_bn",
    "recovery3_adam_cifar10_100L_bn",
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
    print(f"# optimizer=Adam lr={training_config.learning_rate} "
          f"wd={training_config.weight_decay} bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (patience={training_config.plateau_patience} "
          f"factor={training_config.plateau_factor} min_lr={training_config.plateau_min_lr})")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm}")
    print(f"# lr_warmup_epochs={training_config.lr_warmup_epochs} "
          f"start_factor={training_config.lr_warmup_start_factor}")
    print(f"# plateau_warmup_epochs={training_config.plateau_warmup_epochs}")
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
