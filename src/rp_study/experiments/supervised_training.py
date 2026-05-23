"""Supervised training runners for FC/CNN initialization comparisons."""

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
    learning_rate: Optional[float] = None
    grad_norm_per_layer: Optional[List[float]] = None
    bn_running_mean_norm: Optional[List[float]] = None
    bn_running_var_mean: Optional[List[float]] = None


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
    abort_reason: Optional[str] = None
    initial_batch_loss: Optional[float] = None


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
    if training_config.scheduler == "plateau":
        mode = "min" if training_config.plateau_metric == "eval_train_loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=training_config.plateau_factor,
            patience=training_config.plateau_patience,
            min_lr=training_config.plateau_min_lr,
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


def _collect_grad_norms(model: nn.Module) -> List[float]:
    """Collect per-layer gradient L2 norms from hidden layers."""
    norms: List[float] = []
    hidden = getattr(model, "hidden_layers", [])
    for layer in hidden:
        if hasattr(layer, "weight") and layer.weight.grad is not None:
            norms.append(layer.weight.grad.data.norm(2).item())
        else:
            norms.append(0.0)
    return norms


def _collect_bn_stats(model: nn.Module) -> Tuple[List[float], List[float]]:
    """Collect BN running statistics summaries (mean norm, var mean per layer)."""
    mean_norms: List[float] = []
    var_means: List[float] = []
    hidden_norms = getattr(model, "hidden_norms", [])
    for bn in hidden_norms:
        if hasattr(bn, "running_mean") and bn.running_mean is not None:
            mean_norms.append(bn.running_mean.norm(2).item())
            var_means.append(bn.running_var.mean().item())
    return mean_norms, var_means


def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _save_checkpoint(
    checkpoint_dir: str,
    run_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_test_accuracy: float,
    best_epoch: int,
) -> None:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    safe_name = run_name.replace("/", "_")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_test_accuracy": best_test_accuracy,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path / f"{safe_name}_ep{epoch:04d}.pt")


def _load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Tuple[int, float, int]:
    """Load checkpoint; returns (start_epoch, best_test_accuracy, best_epoch)."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        checkpoint["epoch"],
        checkpoint.get("best_test_accuracy", 0.0),
        checkpoint.get("best_epoch", 0),
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
    scheduler_kind = training_config.scheduler
    scheduler_step_per_batch = scheduler_kind == "onecycle"
    scheduler_needs_metric = scheduler_kind == "plateau"

    history: List[EpochMetrics] = []
    best_epoch = 0
    best_test_accuracy = float("-inf")
    status = "completed"
    stop_reason = "max_epochs"
    abort_reason: Optional[str] = None
    initial_batch_loss: Optional[float] = None
    final_train_accuracy = 0.0
    final_test_accuracy = 0.0
    final_eval_train_accuracy = 0.0
    final_eval_train_loss = 0.0
    epochs_ran = 0
    target_streak = 0
    run_name = _format_run_name(resolved_classifier, training_config)
    start_epoch = 1

    if training_config.resume_checkpoint:
        start_epoch_loaded, best_test_accuracy, best_epoch = _load_checkpoint(
            training_config.resume_checkpoint, model, optimizer, scheduler,
        )
        start_epoch = start_epoch_loaded + 1
        print(f"[{run_name}] resumed from checkpoint epoch {start_epoch_loaded}", flush=True)

    diag_every = training_config.diagnostics_every

    for epoch in range(start_epoch, training_config.epochs + 1):
        if training_config.lr_warmup_epochs > 0 and epoch <= training_config.lr_warmup_epochs:
            sf = training_config.lr_warmup_start_factor
            frac = epoch / training_config.lr_warmup_epochs
            scale = sf + (1.0 - sf) * frac
            for pg in optimizer.param_groups:
                pg["lr"] = training_config.learning_rate * scale

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss_value = loss.item()

            if not torch.isfinite(loss):
                status = "diverged"
                stop_reason = "non_finite_loss"
                abort_reason = (
                    f"non-finite loss ({loss_value}) at epoch {epoch} batch {batch_idx}"
                )
                print(f"[{run_name}] ABORT: {abort_reason}", flush=True)
                break

            if initial_batch_loss is None:
                initial_batch_loss = loss_value

            if (
                training_config.abort_on_explosion
                and batch_idx > 0
                and loss_value > training_config.explosion_loss_factor * initial_batch_loss
            ):
                status = "diverged"
                stop_reason = "loss_exploded"
                threshold = training_config.explosion_loss_factor * initial_batch_loss
                abort_reason = (
                    f"loss={loss_value:.4f} exceeded "
                    f"{training_config.explosion_loss_factor:.2f}x initial "
                    f"({initial_batch_loss:.4f}, threshold={threshold:.4f}) "
                    f"at epoch {epoch} batch {batch_idx}"
                )
                print(f"[{run_name}] EXPLOSION ABORT: {abort_reason}", flush=True)
                break

            loss.backward()

            if training_config.log_per_batch_first_epoch and epoch == 1:
                per_layer = _collect_grad_norms(model)
                if per_layer:
                    total_norm = math.sqrt(sum(g * g for g in per_layer))
                    nonzero = [g for g in per_layer if g > 0]
                    min_g = min(nonzero) if nonzero else 0.0
                    max_g = max(per_layer)
                    print(
                        f"[{run_name}] ep1 batch={batch_idx:04d} "
                        f"loss={loss_value:.4f} total_grad={total_norm:.3e} "
                        f"layer_grad=[{min_g:.3e},{max_g:.3e}]",
                        flush=True,
                    )

            if training_config.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=training_config.grad_clip_max_norm,
                )
            optimizer.step()
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

            batch_size = targets.size(0)
            total_loss += loss_value * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        if status == "diverged":
            break

        train_loss = total_loss / max(total_samples, 1)
        train_accuracy = total_correct / max(total_samples, 1)

        is_diag_epoch = diag_every > 0 and epoch % diag_every == 0
        grad_norms = _collect_grad_norms(model) if is_diag_epoch else None
        current_lr = _get_current_lr(optimizer)
        bn_mean_norms = None
        bn_var_means = None
        if is_diag_epoch and resolved_classifier.use_batch_norm:
            bn_mean_norms, bn_var_means = _collect_bn_stats(model)

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
                learning_rate=current_lr,
                grad_norm_per_layer=grad_norms,
                bn_running_mean_norm=bn_mean_norms,
                bn_running_var_mean=bn_var_means,
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
            if scheduler_needs_metric:
                in_warmup = epoch <= training_config.plateau_warmup_epochs
                if not in_warmup:
                    metric_value = (
                        eval_train_loss
                        if training_config.plateau_metric == "eval_train_loss"
                        else eval_train_accuracy
                    )
                    scheduler.step(metric_value)
            else:
                scheduler.step()

        if training_config.log_every_epoch:
            log_parts = [
                f"[{run_name}] ",
                f"epoch={epoch:03d}/{training_config.epochs:03d} ",
                f"lr={current_lr:.6f} ",
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} ",
                f"eval_train_loss={eval_train_loss:.4f} eval_train_acc={eval_train_accuracy:.4f} ",
                f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}",
            ]
            if is_diag_epoch and grad_norms:
                log_parts.append(
                    f" grad_norm_range=[{min(grad_norms):.2e},{max(grad_norms):.2e}]"
                )
            print("".join(log_parts), flush=True)

        if (
            training_config.checkpoint_dir
            and training_config.checkpoint_every > 0
            and epoch % training_config.checkpoint_every == 0
        ):
            _save_checkpoint(
                training_config.checkpoint_dir, run_name, epoch,
                model, optimizer, scheduler, best_test_accuracy, best_epoch,
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
        abort_reason=abort_reason,
        initial_batch_loss=initial_batch_loss,
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
