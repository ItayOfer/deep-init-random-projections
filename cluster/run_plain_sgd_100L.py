#!/usr/bin/env python3
"""Plain-SGD 100-layer FC baseline — replicates the advisor's recipe.

Advisor's email reported the following 100-layer training succeeding where
ours failed:

  width=512, depth=100, batch_size=128, lr=1e-3,
  SGD (no momentum, no weight decay), He init, normalized inputs, 20 epochs
  → Epoch 5: Train Acc 75.40%, Test Acc 77.40%
  → 26,147,338 parameters  =>  D_in=784 (MNIST/FMNIST), NO BatchNorm

He did not specify the dataset, so we run his recipe on three datasets
that bracket the question. Each is a separate SLURM job with its own JSON
output (so they can run in parallel on the cluster, and so post-hoc
analysis can never mix up architectures):

  plain_sgd_100L_nobn_w512_fashion_mnist   (likely his dataset; vs our failing ⑪)
  plain_sgd_100L_nobn_w512_mnist           (same input dim, easier — sanity check)
  plain_sgd_100L_nobn_w512_cifar10         (different input dim; vs our failing ⑤)

Select which one to run with --experiment <label>. One label per job.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def _classifier_config() -> ClassifierConfig:
    """Architecture is fixed: 100 layers, width 512, NoBN, He init."""
    return ClassifierConfig(
        architecture="fc",
        depth=100,
        init_strategy="he",
        use_batch_norm=False,
        fc_hidden_dim=512,
    )


def _training_config(dataset: str, epochs: int, diagnostics_every: int,
                     checkpoint_dir: str, checkpoint_every: int) -> TrainingConfig:
    """Recipe is fixed: plain SGD, lr=1e-3, bs=128, no momentum, no scheduler."""
    return TrainingConfig(
        dataset=dataset,
        epochs=epochs,
        batch_size=128,
        optimizer="sgd",
        learning_rate=1e-3,
        momentum=0.0,           # plain SGD — no momentum
        weight_decay=0.0,       # no weight decay
        scheduler="none",       # no LR schedule
        normalize_inputs=True,
        log_every_epoch=True,
        diagnostics_every=diagnostics_every,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        target_train_accuracy=None,
    )


# label -> dataset. The label IS the architecture identifier; analysis code
# should match against this exact string to avoid mixing up runs.
EXPERIMENTS: Dict[str, str] = {
    "plain_sgd_100L_nobn_w512_fashion_mnist": "fashion_mnist",
    "plain_sgd_100L_nobn_w512_mnist":         "mnist",
    "plain_sgd_100L_nobn_w512_cifar10":       "cifar10",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment", required=True, choices=sorted(EXPERIMENTS.keys()),
        help="Which single architecture to run (one job runs one architecture).",
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Default 20 matches the advisor's report.")
    parser.add_argument("--diagnostics-every", type=int, default=2,
                        help="Per-layer grad-norm logging cadence (epochs).")
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
    dataset = EXPERIMENTS[label]

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config = _classifier_config()
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
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm}")
    print(f"# optimizer=SGD lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size} scheduler={training_config.scheduler} "
          f"epochs={training_config.epochs}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n")

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    # Single-entry list so the JSON shape matches the other audit files
    # (final_audit_merged.json etc.) and can be loaded uniformly.
    payload = [_result_to_payload(label, result)]
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved 1 run to {output_path}")

    # One-line summary for the SLURM log
    hist = payload[0].get("history", [])
    if hist:
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_test = hist[-1].get("test_accuracy", 0) or 0
        print(f"\nSUMMARY {label} | best_train={best_acc:.4f} "
              f"final_train={final_acc:.4f} final_loss={final_loss:.4f} "
              f"final_test={final_test:.4f}")


if __name__ == "__main__":
    main()
