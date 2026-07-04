#!/usr/bin/env python3
"""Plain-SGD recovery campaign for the 4 failing 100L architectures.

Background: the advisor-replication runs proved that plain SGD (lr=1e-3,
no momentum, no weight decay) trains 100L NoBN networks where our Adam
recipes died from gradient underflow. This campaign extends that recipe
to all 4 failing architectures in the final audit:

  ⑤ cifar10/100L/NoBN  (Adam-cosine died, loss stuck at 2.3026)
  ⑥ cifar10/100L/BN    (Adam best 21% train acc)
  ⑪ fmnist /100L/NoBN  (Adam-cosine died, loss stuck at 2.3026)
  ⑫ fmnist /100L/BN    (Adam best 34% train acc)

Recipe (identical across the 4 runs — only dataset/BN flag differs):
  - depth = 100, width = 500   (matches the failing audit runs exactly)
  - optimizer = SGD, lr = 1e-3, momentum = 0, weight_decay = 0
  - batch_size = 128
  - scheduler = plateau (patience=10, factor=0.5, min_lr=1e-6,
                         monitor=eval_train_loss)
      reason: plain SGD reaches a gross optimum quickly; plateau halves
      the LR when progress stalls so the model can find the fine optimum.
  - bn_momentum = 0.1 (PyTorch default) for BN runs — matches the
                       "no sophisticated tuning" spirit of the advisor's recipe
  - normalize_inputs = True (per-channel z-score)
  - epochs = 200, with target_train_accuracy = 0.995 early-stop
      reason: stop early once we cross the thesis bar
              (eval_train_acc >= 0.995 AND eval_train_loss <= 0.10)

Select one architecture per job: --experiment recovery_plain_sgd_<arch>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


# (label, dataset, use_batch_norm). The label IS the architecture identifier
# and matches the filename stem — analysis code can key on it directly.
EXPERIMENTS: Dict[str, Tuple[str, bool]] = {
    "recovery_plain_sgd_cifar10_100L_nobn": ("cifar10", False),
    "recovery_plain_sgd_cifar10_100L_bn":   ("cifar10", True),
    "recovery_plain_sgd_fmnist_100L_nobn":  ("fashion_mnist", False),
    "recovery_plain_sgd_fmnist_100L_bn":    ("fashion_mnist", True),
}


def _classifier_config(use_bn: bool) -> ClassifierConfig:
    return ClassifierConfig(
        architecture="fc",
        depth=100,
        init_strategy="he",
        use_batch_norm=use_bn,
        fc_hidden_dim=500,           # match failing audit width
        bn_momentum=0.1,             # PyTorch default; only used when BN is on
    )


def _training_config(dataset: str, epochs: int, diagnostics_every: int,
                     checkpoint_dir: str, checkpoint_every: int) -> TrainingConfig:
    return TrainingConfig(
        dataset=dataset,
        epochs=epochs,
        batch_size=128,
        optimizer="sgd",
        learning_rate=1e-3,
        momentum=0.0,                # plain SGD
        weight_decay=0.0,
        scheduler="plateau",
        plateau_patience=10,
        plateau_factor=0.5,
        plateau_min_lr=1e-6,
        plateau_metric="eval_train_loss",
        normalize_inputs=True,
        log_every_epoch=True,
        diagnostics_every=diagnostics_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        # Thesis pass bar -> early stop
        target_train_accuracy=0.995,
        target_patience=1,
        target_metric="eval_train_accuracy",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment", required=True, choices=sorted(EXPERIMENTS.keys()),
        help="Which single architecture to run (one job runs one architecture).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--diagnostics-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--output", default=None,
        help="JSON output path. Default: reports/results/<experiment>.json",
    )
    args = parser.parse_args()

    label = args.experiment
    dataset, use_bn = EXPERIMENTS[label]

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config = _classifier_config(use_bn=use_bn)
    training_config = _training_config(
        dataset=dataset,
        epochs=args.epochs,
        diagnostics_every=args.diagnostics_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# dataset={dataset} depth={classifier_config.depth} "
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
