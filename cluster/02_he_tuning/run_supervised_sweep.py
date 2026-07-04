#!/usr/bin/env python3
"""Run a conditional hyperparameter sweep and select best configs per architecture.

Supports incremental saving (results written after every run) and --resume
to pick up from a timed-out or interrupted job.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import (
    EpochMetrics,
    SupervisedTrainingResult,
    run_supervised_experiment,
    supervised_results_to_rows,
)


def _parse_csv_list(raw: str, cast=str) -> List:
    if raw is None:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _result_to_payload(result: SupervisedTrainingResult) -> dict:
    payload = asdict(result)
    payload["history"] = [asdict(epoch) for epoch in result.history]
    return payload


def _payload_to_result(entry: dict) -> SupervisedTrainingResult:
    history = [EpochMetrics(**h) for h in entry.get("history", [])]
    return SupervisedTrainingResult(
        classifier_config=entry["classifier_config"],
        training_config=entry["training_config"],
        status=entry["status"],
        stop_reason=entry["stop_reason"],
        best_epoch=entry["best_epoch"],
        epochs_ran=entry["epochs_ran"],
        best_test_accuracy=entry["best_test_accuracy"],
        final_train_accuracy=entry["final_train_accuracy"],
        final_test_accuracy=entry["final_test_accuracy"],
        history=history,
        parameter_count=entry["parameter_count"],
        final_eval_train_accuracy=entry.get("final_eval_train_accuracy"),
        final_eval_train_loss=entry.get("final_eval_train_loss"),
    )


def _format_run_name(cc: ClassifierConfig, tc: TrainingConfig) -> str:
    bn_label = "bn" if cc.use_batch_norm else "nobn"
    return f"{tc.dataset}/{cc.architecture}/{cc.depth}L/{cc.init_strategy}/{bn_label}"


def _group_key(result: SupervisedTrainingResult) -> Tuple:
    c = result.classifier_config
    t = result.training_config
    return (
        t["dataset"],
        c["architecture"],
        c["depth"],
        c["init_strategy"],
        c["use_batch_norm"],
        c.get("fc_hidden_dim"),
        c.get("cnn_base_channels"),
        c.get("cnn_max_channels"),
    )


def _result_selection_key(
    result: SupervisedTrainingResult,
    train_acc_threshold: float,
    train_loss_threshold: float,
) -> Tuple[float, ...]:
    eval_acc = result.final_eval_train_accuracy or 0.0
    eval_loss = result.final_eval_train_loss if result.final_eval_train_loss is not None else float("inf")
    criterion_met = eval_acc >= train_acc_threshold and eval_loss <= train_loss_threshold
    non_diverged = result.status != "diverged"
    finite_loss = 0.0 if eval_loss == float("inf") else eval_loss
    return (
        1.0 if non_diverged else 0.0,
        1.0 if criterion_met else 0.0,
        eval_acc,
        -finite_loss,
        -float(result.epochs_ran),
    )


def _select_best_results(
    results: Sequence[SupervisedTrainingResult],
    train_acc_threshold: float,
    train_loss_threshold: float,
) -> List[dict]:
    best_by_group = {}
    for result in results:
        key = _group_key(result)
        current_best = best_by_group.get(key)
        if current_best is None:
            best_by_group[key] = result
            continue
        if _result_selection_key(result, train_acc_threshold, train_loss_threshold) > _result_selection_key(
            current_best,
            train_acc_threshold,
            train_loss_threshold,
        ):
            best_by_group[key] = result

    payload = []
    for key in sorted(best_by_group):
        result = best_by_group[key]
        eval_acc = result.final_eval_train_accuracy or 0.0
        eval_loss = result.final_eval_train_loss if result.final_eval_train_loss is not None else float("inf")
        payload.append(
            {
                "group": {
                    "dataset": key[0],
                    "architecture": key[1],
                    "depth": key[2],
                    "init_strategy": key[3],
                    "use_batch_norm": key[4],
                    "fc_hidden_dim": key[5],
                    "cnn_base_channels": key[6],
                    "cnn_max_channels": key[7],
                },
                "selection": {
                    "criterion_met": eval_acc >= train_acc_threshold and eval_loss <= train_loss_threshold,
                    "final_eval_train_accuracy": eval_acc,
                    "final_eval_train_loss": eval_loss,
                    "status": result.status,
                    "stop_reason": result.stop_reason,
                },
                "classifier_config": result.classifier_config,
                "training_config": result.training_config,
                "result": {
                    "best_epoch": result.best_epoch,
                    "epochs_ran": result.epochs_ran,
                    "best_test_accuracy": result.best_test_accuracy,
                    "final_train_accuracy": result.final_train_accuracy,
                    "final_test_accuracy": result.final_test_accuracy,
                    "final_eval_train_accuracy": eval_acc,
                    "final_eval_train_loss": eval_loss,
                },
            }
        )
    return payload


def _build_pairs_for_architecture(
    dataset: str,
    architecture: str,
    depth: int,
    init_strategy: str,
    use_batch_norm: bool,
    args: argparse.Namespace,
) -> Iterable[Tuple[ClassifierConfig, TrainingConfig]]:
    if use_batch_norm:
        optimizers = _parse_csv_list(args.bn_optimizers, str)
        schedulers = _parse_csv_list(args.bn_schedulers, str)
        weight_decays = _parse_csv_list(args.bn_weight_decays, float)
        batch_sizes = _parse_csv_list(args.bn_batch_sizes, int)
        bn_momentums = _parse_csv_list(args.bn_norm_momentums, float)
        adam_lrs = _parse_csv_list(args.bn_adam_lrs, float)
        sgd_lrs = _parse_csv_list(args.bn_sgd_lrs, float)
    else:
        optimizers = _parse_csv_list(args.nobn_optimizers, str)
        schedulers = _parse_csv_list(args.nobn_schedulers, str)
        weight_decays = _parse_csv_list(args.nobn_weight_decays, float)
        batch_sizes = _parse_csv_list(args.nobn_batch_sizes, int)
        bn_momentums = [args.default_bn_momentum]
        adam_lrs = _parse_csv_list(args.nobn_adam_lrs, float)
        sgd_lrs = _parse_csv_list(args.nobn_sgd_lrs, float)

    sgd_momentums = _parse_csv_list(args.sgd_momentums, float)

    for optimizer in optimizers:
        if optimizer == "adam":
            learning_rates = adam_lrs
            optimizer_momentums = [args.default_optimizer_momentum]
        elif optimizer == "sgd":
            learning_rates = sgd_lrs
            optimizer_momentums = sgd_momentums
        else:
            raise ValueError(f"Unsupported optimizer in sweep: {optimizer}")

        for learning_rate in learning_rates:
            for scheduler in schedulers:
                for weight_decay in weight_decays:
                    for batch_size in batch_sizes:
                        for optimizer_momentum in optimizer_momentums:
                            for bn_momentum in bn_momentums:
                                classifier_config = ClassifierConfig(
                                    architecture=architecture,
                                    depth=depth,
                                    init_strategy=init_strategy,
                                    use_batch_norm=use_batch_norm,
                                    fc_hidden_dim=args.fc_hidden_dim,
                                    cnn_base_channels=args.cnn_base_channels,
                                    cnn_max_channels=args.cnn_max_channels,
                                    bn_momentum=bn_momentum,
                                    bn_eps=args.bn_eps,
                                )
                                training_config = TrainingConfig(
                                    dataset=dataset,
                                    epochs=args.epochs,
                                    min_epochs=args.min_epochs,
                                    batch_size=batch_size,
                                    eval_batch_size=args.eval_batch_size,
                                    learning_rate=learning_rate,
                                    optimizer=optimizer,
                                    momentum=optimizer_momentum,
                                    scheduler=scheduler,
                                    step_size=args.step_size,
                                    gamma=args.gamma,
                                    onecycle_pct_start=args.onecycle_pct_start,
                                    onecycle_div_factor=args.onecycle_div_factor,
                                    onecycle_final_div_factor=args.onecycle_final_div_factor,
                                    weight_decay=weight_decay,
                                    num_train_samples=args.num_train_samples,
                                    num_test_samples=args.num_test_samples,
                                    normalize_inputs=args.normalize_inputs,
                                    num_workers=args.num_workers,
                                    label_smoothing=args.label_smoothing,
                                    target_train_accuracy=args.target_train_accuracy,
                                    target_patience=args.target_patience,
                                    target_metric=args.target_metric,
                                    log_every_epoch=args.log_every_epoch,
                                )
                                yield classifier_config, training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="fashion_mnist,cifar10")
    parser.add_argument("--architectures", default="fc")
    parser.add_argument("--depths", default="30,50,100")
    parser.add_argument("--init-strategies", default="he")
    parser.add_argument("--batch-norm", default="false,true", help="Comma-separated booleans")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-train-samples", type=int, default=None)
    parser.add_argument("--num-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fc-hidden-dim", type=int, default=500)
    parser.add_argument("--cnn-base-channels", type=int, default=32)
    parser.add_argument("--cnn-max-channels", type=int, default=256)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--target-train-accuracy", type=float, default=0.995)
    parser.add_argument("--target-patience", type=int, default=3)
    parser.add_argument(
        "--target-metric",
        default="eval_train_accuracy",
        choices=["train_accuracy", "eval_train_accuracy"],
    )
    parser.add_argument("--log-every-epoch", action="store_true")
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1e4)
    parser.add_argument("--bn-eps", type=float, default=1e-5)
    parser.add_argument("--default-bn-momentum", type=float, default=0.1)
    parser.add_argument("--default-optimizer-momentum", type=float, default=0.9)
    parser.add_argument("--sgd-momentums", default="0.9")
    parser.add_argument("--bn-optimizers", default="adam,sgd")
    parser.add_argument("--bn-adam-lrs", default="1e-3,3e-3")
    parser.add_argument("--bn-sgd-lrs", default="3e-2,1e-1")
    parser.add_argument("--bn-schedulers", default="cosine,onecycle")
    parser.add_argument("--bn-weight-decays", default="0")
    parser.add_argument("--bn-batch-sizes", default="128,256")
    parser.add_argument("--bn-norm-momentums", default="0.1,0.05")
    parser.add_argument("--nobn-optimizers", default="adam,sgd")
    parser.add_argument("--nobn-adam-lrs", default="3e-4,1e-3")
    parser.add_argument("--nobn-sgd-lrs", default="1e-2,3e-2")
    parser.add_argument("--nobn-schedulers", default="none,cosine")
    parser.add_argument("--nobn-weight-decays", default="0")
    parser.add_argument("--nobn-batch-sizes", default="128")
    parser.add_argument("--selection-train-acc-threshold", type=float, default=0.95)
    parser.add_argument("--selection-train-loss-threshold", type=float, default=0.10)
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "supervised_sweep.json"),
    )
    parser.add_argument(
        "--best-output",
        default=str(ROOT / "reports" / "results" / "supervised_sweep_best.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping already-completed runs",
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
        for architecture in architectures:
            for depth in depths:
                for init_strategy in init_strategies:
                    for use_batch_norm in batch_norm_options:
                        config_pairs.extend(
                            _build_pairs_for_architecture(
                                dataset=dataset,
                                architecture=architecture,
                                depth=depth,
                                init_strategy=init_strategy,
                                use_batch_norm=use_batch_norm,
                                args=args,
                            )
                        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path = Path(args.best_output)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)

    all_payload: List[dict] = []
    all_results: List[SupervisedTrainingResult] = []
    start_idx = 0

    if args.resume and output_path.exists():
        existing_payload = json.loads(output_path.read_text())
        all_payload = existing_payload
        all_results = [_payload_to_result(entry) for entry in existing_payload]
        start_idx = len(existing_payload)
        print(
            f"Resuming: found {start_idx} completed runs, "
            f"skipping to run {start_idx + 1}/{len(config_pairs)}",
            flush=True,
        )

    remaining_pairs = config_pairs[start_idx:]
    total_runs = len(config_pairs)

    for run_offset, (classifier_config, training_config) in enumerate(remaining_pairs):
        run_idx = start_idx + run_offset + 1
        exp_config.setup_seeds()
        run_name = _format_run_name(classifier_config, training_config)
        print(
            f"[run {run_idx:02d}/{total_runs:02d}] start {run_name} "
            f"(hidden={classifier_config.fc_hidden_dim if classifier_config.architecture == 'fc' else 'n/a'}, "
            f"epochs={training_config.epochs})",
            flush=True,
        )
        result = run_supervised_experiment(exp_config, classifier_config, training_config)
        print(
            f"[run {run_idx:02d}/{total_runs:02d}] done  {run_name} "
            f"status={result.status} stop={result.stop_reason} "
            f"epochs_ran={result.epochs_ran} "
            f"best_test={result.best_test_accuracy:.4f} "
            f"final_train={result.final_train_accuracy:.4f} "
            f"final_eval_train={result.final_eval_train_accuracy:.4f} "
            f"final_eval_train_loss={result.final_eval_train_loss:.4f}",
            flush=True,
        )

        all_results.append(result)
        all_payload.append(_result_to_payload(result))
        output_path.write_text(json.dumps(all_payload, indent=2))

    rows = supervised_results_to_rows(all_results)

    best_payload = _select_best_results(
        all_results,
        train_acc_threshold=args.selection_train_acc_threshold,
        train_loss_threshold=args.selection_train_loss_threshold,
    )
    best_output_path.write_text(json.dumps(best_payload, indent=2))

    print(f"\nSaved {len(all_results)} sweep runs to {output_path}", flush=True)
    print(f"Saved {len(best_payload)} best-config summaries to {best_output_path}", flush=True)
    for row in rows:
        bn_label = "BN" if row["use_batch_norm"] else "NoBN"
        print(
            f'{row["dataset"]:>14s} {row["architecture"]:>3s} '
            f'{row["depth"]:>3d}L {bn_label:>4s} '
            f'opt={row["optimizer"]:>4s} sch={row["scheduler"]:>8s} '
            f'lr={row["learning_rate"]:.4g} bs={row["batch_size"]:>3d} '
            f'bn_m={row["bn_momentum"]:.3f} '
            f'eval_train={row["final_eval_train_accuracy"]:.4f} '
            f'eval_loss={row["final_eval_train_loss"]:.4f} '
            f'final_test={row["final_test_accuracy"]:.4f}'
        )

    print("\nSelected best configs by architecture:", flush=True)
    for item in best_payload:
        group = item["group"]
        sel = item["selection"]
        train_cfg = item["training_config"]
        print(
            f'{group["dataset"]:>14s} {group["architecture"]:>3s} '
            f'{group["depth"]:>3d}L {"BN" if group["use_batch_norm"] else "NoBN":>4s} '
            f'opt={train_cfg["optimizer"]:>4s} sch={train_cfg["scheduler"]:>8s} '
            f'lr={train_cfg["learning_rate"]:.4g} bs={train_cfg["batch_size"]:>3d} '
            f'eval_train={sel["final_eval_train_accuracy"]:.4f} '
            f'eval_loss={sel["final_eval_train_loss"]:.4f} '
            f'criterion={"Yes" if sel["criterion_met"] else "No"}',
            flush=True,
        )


if __name__ == "__main__":
    main()
