#!/usr/bin/env python3
"""Rerun supervised experiments from selected config JSON files."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import (
    run_supervised_experiment,
    supervised_results_to_rows,
)


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _maybe_override(value, override):
    return value if override is None else override


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-jsons", required=True, help="Comma-separated selected-config JSON paths")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-train-samples", type=int, default=None)
    parser.add_argument("--num-test-samples", type=int, default=None)
    parser.add_argument("--target-train-accuracy", type=float, default=None)
    parser.add_argument("--target-patience", type=int, default=None)
    parser.add_argument(
        "--target-metric",
        default=None,
        choices=["train_accuracy", "eval_train_accuracy"],
    )
    parser.add_argument("--log-every-epoch", action="store_true")
    parser.add_argument("--diagnostics-every", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "supervised_selected_rerun.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    config_pairs = []
    for raw_path in _parse_csv_list(args.config_jsons):
        payload = json.loads(Path(raw_path).read_text())
        for item in payload:
            classifier_config = ClassifierConfig(**item["classifier_config"])
            training_config = TrainingConfig(**item["training_config"])
            training_config = TrainingConfig(
                **{
                    **asdict(training_config),
                    "epochs": _maybe_override(training_config.epochs, args.epochs),
                    "min_epochs": _maybe_override(training_config.min_epochs, args.min_epochs),
                    "eval_batch_size": _maybe_override(training_config.eval_batch_size, args.eval_batch_size),
                    "num_train_samples": _maybe_override(training_config.num_train_samples, args.num_train_samples),
                    "num_test_samples": _maybe_override(training_config.num_test_samples, args.num_test_samples),
                    "target_train_accuracy": _maybe_override(
                        training_config.target_train_accuracy, args.target_train_accuracy
                    ),
                    "target_patience": _maybe_override(training_config.target_patience, args.target_patience),
                    "target_metric": _maybe_override(training_config.target_metric, args.target_metric),
                    "log_every_epoch": args.log_every_epoch or training_config.log_every_epoch,
                    "diagnostics_every": _maybe_override(
                        training_config.diagnostics_every, args.diagnostics_every
                    ),
                    "checkpoint_dir": _maybe_override(
                        training_config.checkpoint_dir, args.checkpoint_dir
                    ),
                    "checkpoint_every": _maybe_override(
                        training_config.checkpoint_every, args.checkpoint_every
                    ),
                }
            )
            config_pairs.append((classifier_config, training_config))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    full_payload = []
    total_runs = len(config_pairs)

    for run_idx, (classifier_config, training_config) in enumerate(config_pairs, start=1):
        exp_config.setup_seeds()
        bn_label = "bn" if classifier_config.use_batch_norm else "nobn"
        run_name = (
            f"{training_config.dataset}/{classifier_config.architecture}"
            f"/{classifier_config.depth}L/{classifier_config.init_strategy}/{bn_label}"
        )
        print(
            f"[run {run_idx:02d}/{total_runs:02d}] start {run_name} "
            f"(hidden={classifier_config.fc_hidden_dim}, epochs={training_config.epochs})",
            flush=True,
        )
        result = run_supervised_experiment(exp_config, classifier_config, training_config)
        print(
            f"[run {run_idx:02d}/{total_runs:02d}] done  {run_name} "
            f"status={result.status} stop={result.stop_reason} "
            f"epochs_ran={result.epochs_ran} "
            f"best_test={result.best_test_accuracy:.4f} "
            f"final_eval_train={result.final_eval_train_accuracy:.4f} "
            f"final_eval_train_loss={result.final_eval_train_loss:.4f}",
            flush=True,
        )
        results.append(result)
        payload = asdict(result)
        payload["history"] = [asdict(epoch) for epoch in result.history]
        full_payload.append(payload)
        output_path.write_text(json.dumps(full_payload, indent=2))

    rows = supervised_results_to_rows(results)
    print(f"\nSaved {len(results)} rerun results to {output_path}", flush=True)
    for row in rows:
        print(
            f'{row["dataset"]:>14s} {row["architecture"]:>3s} {row["depth"]:>3d}L '
            f'{"BN" if row["use_batch_norm"] else "NoBN":>4s} '
            f'opt={row["optimizer"]:>4s} sch={row["scheduler"]:>8s} '
            f'lr={row["learning_rate"]:.4g} bs={row["batch_size"]:>3d} '
            f'eval_train={row["final_eval_train_accuracy"]:.4f} '
            f'eval_loss={row["final_eval_train_loss"]:.4f} '
            f'final_test={row["final_test_accuracy"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
