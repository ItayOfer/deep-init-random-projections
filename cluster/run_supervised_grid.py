#!/usr/bin/env python3
"""Run the FC/CNN initialization comparison grid from the command line."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import (
    run_supervised_grid,
    supervised_results_to_rows,
)


def _parse_csv_list(raw: str, cast=str):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="fashion_mnist,cifar10")
    parser.add_argument("--architectures", default="cnn,fc")
    parser.add_argument("--depths", default="50,100")
    parser.add_argument("--init-strategies", default="he,row_centered_he")
    parser.add_argument("--batch-norm", default="false,true", help="Comma-separated booleans")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "step", "onecycle"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1e4)
    parser.add_argument("--num-train-samples", type=int, default=None)
    parser.add_argument("--num-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fc-hidden-dim", type=int, default=512)
    parser.add_argument("--cnn-base-channels", type=int, default=32)
    parser.add_argument("--cnn-max-channels", type=int, default=256)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--target-train-accuracy", type=float, default=None)
    parser.add_argument("--target-patience", type=int, default=1)
    parser.add_argument(
        "--target-metric",
        default="eval_train_accuracy",
        choices=["train_accuracy", "eval_train_accuracy"],
    )
    parser.add_argument("--log-every-epoch", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "supervised_grid.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))
    datasets = _parse_csv_list(args.datasets, str)
    architectures = _parse_csv_list(args.architectures, str)
    depths = _parse_csv_list(args.depths, int)
    init_strategies = _parse_csv_list(args.init_strategies, str)
    batch_norm_options = [item.lower() == "true" for item in _parse_csv_list(args.batch_norm, str)]

    config_pairs = []
    for dataset in datasets:
        training_config = TrainingConfig(
            dataset=dataset,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            step_size=args.step_size,
            gamma=args.gamma,
            onecycle_pct_start=args.onecycle_pct_start,
            onecycle_div_factor=args.onecycle_div_factor,
            onecycle_final_div_factor=args.onecycle_final_div_factor,
            num_train_samples=args.num_train_samples,
            num_test_samples=args.num_test_samples,
            normalize_inputs=args.normalize_inputs,
            target_train_accuracy=args.target_train_accuracy,
            target_patience=args.target_patience,
            target_metric=args.target_metric,
            log_every_epoch=args.log_every_epoch,
        )
        for architecture in architectures:
            for depth in depths:
                for init_strategy in init_strategies:
                    for use_batch_norm in batch_norm_options:
                        classifier_config = ClassifierConfig(
                            architecture=architecture,
                            depth=depth,
                            init_strategy=init_strategy,
                            use_batch_norm=use_batch_norm,
                            fc_hidden_dim=args.fc_hidden_dim,
                            cnn_base_channels=args.cnn_base_channels,
                            cnn_max_channels=args.cnn_max_channels,
                            bn_momentum=args.bn_momentum,
                            bn_eps=args.bn_eps,
                        )
                        config_pairs.append((classifier_config, training_config))

    results = run_supervised_grid(exp_config, config_pairs)
    rows = supervised_results_to_rows(results)
    full_payload = []
    for result in results:
        payload = asdict(result)
        payload["history"] = [asdict(epoch) for epoch in result.history]
        full_payload.append(payload)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(full_payload, indent=2))

    print(f"Saved {len(results)} training runs to {output_path}", flush=True)
    for row in rows:
        bn_label = "BN" if row["use_batch_norm"] else "NoBN"
        print(
            f'{row["dataset"]:>14s} {row["architecture"]:>3s} '
            f'{row["depth"]:>3d}L {row["init_strategy"]:>18s} {bn_label:>4s} '
            f'status={row["status"]:>14s} epochs={row["epochs_ran"]:>3d} '
            f'best={row["best_test_accuracy"]:.4f} '
            f'eval_train={row["final_eval_train_accuracy"]:.4f} '
            f'eval_train_loss={row["final_eval_train_loss"]:.4f}'
        )


if __name__ == "__main__":
    main()
