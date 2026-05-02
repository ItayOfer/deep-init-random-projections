"""Supervised training runners for FC/CNN initialization comparisons."""

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ClassifierConfig, ExperimentConfig, TrainingConfig
from ..data.loaders import create_classification_loaders
from ..models.classifiers import build_classifier, classifier_config_to_dict


DATASET_MODEL_METADATA: Dict[str, Dict[str, int]] = {
    "mnist": {"input_channels": 1, "flattened_dim": 28 * 28, "num_classes": 10},
    "fashion_mnist": {"input_channels": 1, "flattened_dim": 28 * 28, "num_classes": 10},
    "cifar10": {"input_channels": 3, "flattened_dim": 32 * 32 * 3, "num_classes": 10},
}


@dataclass
class EpochMetrics:
    """Per-epoch scalar metrics."""

    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    eval_train_loss: Optional[float] = None
    eval_train_accuracy: Optional[float] = None


@dataclass
class SupervisedTrainingResult:
    """Result of one supervised training run."""

    classifier_config: dict
    training_config: dict
    status: str
    stop_reason: str
    best_epoch: int
    epochs_ran: int
    best_test_accuracy: float
    final_train_accuracy: float
    final_test_accuracy: float
    history: List[EpochMetrics]
    parameter_count: int
    final_eval_train_accuracy: Optional[float] = None
    final_eval_train_loss: Optional[float] = None


def _resolve_classifier_config(
    classifier_config: ClassifierConfig,
    dataset_name: str,
) -> ClassifierConfig:
    """Inject dataset-specific input dimensions/channels into the model config."""
    metadata = DATASET_MODEL_METADATA[dataset_name]
    return replace(
        classifier_config,
        fc_input_dim=metadata["flattened_dim"],
        cnn_input_channels=metadata["input_channels"],
        num_classes=metadata["num_classes"],
    )


def _build_optimizer(
    model: nn.Module,
    training_config: TrainingConfig,
) -> torch.optim.Optimizer:
    if training_config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    if training_config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            momentum=training_config.momentum,
            weight_decay=training_config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {training_config.optimizer}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if training_config.scheduler == "none":
        return None
    if training_config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.step_size,
            gamma=training_config.gamma,
        )
    if training_config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.epochs,
        )
    if training_config.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_config.learning_rate,
            epochs=training_config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=training_config.onecycle_pct_start,
            div_factor=training_config.onecycle_div_factor,
            final_div_factor=training_config.onecycle_final_div_factor,
            anneal_strategy="cos",
        )
    raise ValueError(f"Unknown scheduler: {training_config.scheduler}")


def _evaluate_classifier(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_accuracy = total_correct / max(total_samples, 1)
    return avg_loss, avg_accuracy


def _make_eval_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a deterministic evaluation loader over an existing dataset object."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _format_run_name(classifier_config: ClassifierConfig, training_config: TrainingConfig) -> str:
    bn_label = "bn" if classifier_config.use_batch_norm else "nobn"
    return (
        f"{training_config.dataset}"
        f"/{classifier_config.architecture}"
        f"/{classifier_config.depth}L"
        f"/{classifier_config.init_strategy}"
        f"/{bn_label}"
    )


def run_supervised_experiment(
    exp_config: ExperimentConfig,
    classifier_config: ClassifierConfig,
    training_config: TrainingConfig,
) -> SupervisedTrainingResult:
    """Train one FC/CNN model under the requested initialization setting."""
    exp_config.setup_seeds()
    device = exp_config.get_device()
    resolved_classifier = _resolve_classifier_config(classifier_config, training_config.dataset)
    flatten = resolved_classifier.architecture == "fc"

    train_loader, test_loader = create_classification_loaders(
        dataset_name=training_config.dataset,
        data_dir=exp_config.data_dir,
        batch_size=training_config.batch_size,
        eval_batch_size=training_config.eval_batch_size,
        num_train_samples=training_config.num_train_samples,
        num_test_samples=training_config.num_test_samples,
        flatten=flatten,
        normalize=training_config.normalize_inputs,
        seed=exp_config.seed,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = _make_eval_loader(
        train_loader.dataset,
        batch_size=training_config.eval_batch_size,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_classifier(resolved_classifier).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    optimizer = _build_optimizer(model, training_config)
    scheduler = _build_scheduler(optimizer, training_config, steps_per_epoch=len(train_loader))
    scheduler_step_per_batch = training_config.scheduler == "onecycle"

    history: List[EpochMetrics] = []
    best_epoch = 0
    best_test_accuracy = float("-inf")
    status = "completed"
    stop_reason = "max_epochs"
    final_train_accuracy = 0.0
    final_test_accuracy = 0.0
    final_eval_train_accuracy = 0.0
    final_eval_train_loss = 0.0
    epochs_ran = 0
    target_streak = 0
    run_name = _format_run_name(resolved_classifier, training_config)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                status = "diverged"
                stop_reason = "non_finite_loss"
                break

            loss.backward()
            optimizer.step()
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        if status == "diverged":
            break

        train_loss = total_loss / max(total_samples, 1)
        train_accuracy = total_correct / max(total_samples, 1)
        eval_train_loss, eval_train_accuracy = _evaluate_classifier(
            model,
            train_eval_loader,
            criterion,
            device,
        )
        test_loss, test_accuracy = _evaluate_classifier(model, test_loader, criterion, device)
        epochs_ran = epoch
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
                eval_train_loss=eval_train_loss,
                eval_train_accuracy=eval_train_accuracy,
            )
        )

        final_train_accuracy = train_accuracy
        final_test_accuracy = test_accuracy
        final_eval_train_accuracy = eval_train_accuracy
        final_eval_train_loss = eval_train_loss
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        if training_config.log_every_epoch:
            print(
                f"[{run_name}] "
                f"epoch={epoch:03d}/{training_config.epochs:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"eval_train_loss={eval_train_loss:.4f} eval_train_acc={eval_train_accuracy:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}",
                flush=True,
            )

        target_value = (
            eval_train_accuracy
            if training_config.target_metric == "eval_train_accuracy"
            else train_accuracy
        )
        target_reached = (
            training_config.target_train_accuracy is not None
            and target_value >= training_config.target_train_accuracy
            and epoch >= training_config.min_epochs
        )
        if target_reached:
            target_streak += 1
        else:
            target_streak = 0

        if (
            training_config.target_train_accuracy is not None
            and target_streak >= training_config.target_patience
        ):
            status = "target_reached"
            stop_reason = (
                f"{training_config.target_metric}>={training_config.target_train_accuracy:.4f} "
                f"for {training_config.target_patience} epoch(s)"
            )
            break

    if best_test_accuracy == float("-inf"):
        best_test_accuracy = 0.0

    return SupervisedTrainingResult(
        classifier_config=classifier_config_to_dict(resolved_classifier),
        training_config=asdict(training_config),
        status=status,
        stop_reason=stop_reason,
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_test_accuracy=best_test_accuracy,
        final_train_accuracy=final_train_accuracy,
        final_test_accuracy=final_test_accuracy,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        final_eval_train_accuracy=final_eval_train_accuracy,
        final_eval_train_loss=final_eval_train_loss,
    )


def build_supervised_comparison_configs(
    datasets: Sequence[str] = ("fashion_mnist", "cifar10"),
    architectures: Sequence[str] = ("cnn", "fc"),
    depths: Sequence[int] = (50, 100),
    init_strategies: Sequence[str] = ("he", "row_centered_he"),
    batch_norm_options: Sequence[bool] = (False, True),
    fc_hidden_dim: int = 512,
    cnn_base_channels: int = 32,
    cnn_max_channels: int = 256,
    shared_init_kwargs: Optional[Dict[str, dict]] = None,
) -> List[Tuple[ClassifierConfig, TrainingConfig]]:
    """Construct the experiment grid requested in the supervisor meeting."""
    shared_init_kwargs = shared_init_kwargs or {}
    configs: List[Tuple[ClassifierConfig, TrainingConfig]] = []
    for dataset in datasets:
        for architecture in architectures:
            for depth in depths:
                for init_strategy in init_strategies:
                    for use_batch_norm in batch_norm_options:
                        classifier_config = ClassifierConfig(
                            architecture=architecture,
                            depth=depth,
                            init_strategy=init_strategy,
                            init_kwargs=dict(shared_init_kwargs.get(init_strategy, {})),
                            use_batch_norm=use_batch_norm,
                            fc_hidden_dim=fc_hidden_dim,
                            cnn_base_channels=cnn_base_channels,
                            cnn_max_channels=cnn_max_channels,
                        )
                        training_config = TrainingConfig(dataset=dataset)
                        configs.append((classifier_config, training_config))
    return configs


def run_supervised_grid(
    exp_config: ExperimentConfig,
    config_pairs: Iterable[Tuple[ClassifierConfig, TrainingConfig]],
) -> List[SupervisedTrainingResult]:
    """Run a list of supervised training configurations with seed reset each time."""
    results: List[SupervisedTrainingResult] = []
    config_pairs = list(config_pairs)
    total_runs = len(config_pairs)
    for run_idx, (classifier_config, training_config) in enumerate(config_pairs, start=1):
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
        results.append(result)
    return results


def supervised_results_to_rows(
    results: Sequence[SupervisedTrainingResult],
) -> List[dict]:
    """Flatten supervised results for JSON export or tabular summaries."""
    rows: List[dict] = []
    for result in results:
        row = {
            "dataset": result.training_config["dataset"],
            "architecture": result.classifier_config["architecture"],
            "depth": result.classifier_config["depth"],
            "init_strategy": result.classifier_config["init_strategy"],
            "use_batch_norm": result.classifier_config["use_batch_norm"],
            "bn_momentum": result.classifier_config.get("bn_momentum", 0.1),
            "bn_eps": result.classifier_config.get("bn_eps", 1e-5),
            "optimizer": result.training_config["optimizer"],
            "scheduler": result.training_config["scheduler"],
            "learning_rate": result.training_config["learning_rate"],
            "weight_decay": result.training_config["weight_decay"],
            "batch_size": result.training_config["batch_size"],
            "status": result.status,
            "stop_reason": result.stop_reason,
            "best_epoch": result.best_epoch,
            "epochs_ran": result.epochs_ran,
            "best_test_accuracy": result.best_test_accuracy,
            "final_train_accuracy": result.final_train_accuracy,
            "final_test_accuracy": result.final_test_accuracy,
            "final_eval_train_accuracy": result.final_eval_train_accuracy,
            "final_eval_train_loss": result.final_eval_train_loss,
            "parameter_count": result.parameter_count,
        }
        rows.append(row)
    return rows
