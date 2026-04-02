"""Supervised training runners for FC/CNN initialization comparisons."""

from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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


@dataclass
class SupervisedTrainingResult:
    """Result of one supervised training run."""

    classifier_config: dict
    training_config: dict
    status: str
    best_epoch: int
    best_test_accuracy: float
    final_train_accuracy: float
    final_test_accuracy: float
    history: List[EpochMetrics]
    parameter_count: int


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

    model = build_classifier(resolved_classifier).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    optimizer = _build_optimizer(model, training_config)
    scheduler = _build_scheduler(optimizer, training_config)

    history: List[EpochMetrics] = []
    best_epoch = 0
    best_test_accuracy = float("-inf")
    status = "completed"
    final_train_accuracy = 0.0
    final_test_accuracy = 0.0

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
                break

            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        if status == "diverged":
            break

        train_loss = total_loss / max(total_samples, 1)
        train_accuracy = total_correct / max(total_samples, 1)
        test_loss, test_accuracy = _evaluate_classifier(model, test_loader, criterion, device)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
            )
        )

        final_train_accuracy = train_accuracy
        final_test_accuracy = test_accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch

        if scheduler is not None:
            scheduler.step()

    if best_test_accuracy == float("-inf"):
        best_test_accuracy = 0.0

    return SupervisedTrainingResult(
        classifier_config=classifier_config_to_dict(resolved_classifier),
        training_config=asdict(training_config),
        status=status,
        best_epoch=best_epoch,
        best_test_accuracy=best_test_accuracy,
        final_train_accuracy=final_train_accuracy,
        final_test_accuracy=final_test_accuracy,
        history=history,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
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
    for classifier_config, training_config in config_pairs:
        exp_config.setup_seeds()
        results.append(run_supervised_experiment(exp_config, classifier_config, training_config))
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
            "status": result.status,
            "best_epoch": result.best_epoch,
            "best_test_accuracy": result.best_test_accuracy,
            "final_train_accuracy": result.final_train_accuracy,
            "final_test_accuracy": result.final_test_accuracy,
            "parameter_count": result.parameter_count,
        }
        rows.append(row)
    return rows
