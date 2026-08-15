"""
Configuration dataclasses for experiments.

These provide type-safe, IDE-friendly configuration objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import torch


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""

    seed: int = 42
    device: Literal["auto", "cuda", "cpu"] = "auto"
    data_dir: str = "./data"

    def get_device(self) -> torch.device:
        """Resolve the device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def setup_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        import numpy as np
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture and initialization."""

    layer_sizes: List[int] = field(default_factory=lambda: [784, 784, 1])
    init_strategy: str = "he"  # "he", "row_centered_he", "custom_variance", or custom
    weight_variance: Optional[float] = None  # For custom_variance init
    weight_mean: float = 0.0  # For custom_variance init
    use_bias: bool = True
    init_kwargs: dict = field(default_factory=dict)  # Extra kwargs for initializer (e.g., eta)

    def __post_init__(self):
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")


@dataclass
class GradientExperimentConfig:
    """Configuration for gradient analysis experiments."""

    num_samples: int = 1000
    dataset: Literal["mnist", "fashion_mnist", "cifar10"] = "fashion_mnist"
    num_bins: int = 50  # For histogram plotting


@dataclass
class ClassifierConfig:
    """Configuration for supervised classifier architecture and initialization."""

    architecture: Literal["fc", "cnn"] = "fc"
    depth: int = 50
    init_strategy: str = "he"
    init_kwargs: dict = field(default_factory=dict)
    use_batch_norm: bool = False
    use_bias: bool = True
    num_classes: int = 10
    fc_input_dim: int = 784
    fc_hidden_dim: int = 512
    cnn_input_channels: int = 1
    cnn_base_channels: int = 32
    cnn_max_channels: int = 256
    bn_momentum: float = 0.1
    bn_eps: float = 1e-5
    # Per-layer BACKWARD gradient rescale (FC only). If set, a GradRescale op
    # (identity forward, multiply gradient by this factor in backward) is
    # inserted after each hidden ReLU. Used to cancel the row-centered
    # forward-balanced backward amplification g_bwd = 1/r with r = sqrt((pi-1)/pi)
    # ~ 0.826. None disables it (standard backprop).
    grad_rescale: Optional[float] = None
    # Post-ReLU DC removal (FC only), campaign 11: after each hidden ReLU,
    #     a <- a - relu_shift * rms(a),   rms(a) = a.pow(2).mean().sqrt()
    # the SCALE-RELATIVE shift. relu_shift = c is the coefficient; c = 1/sqrt(pi)
    # ~ 0.5642 removes exactly E[a] for a rectified Gaussian. None disables it,
    # which is a bit-exact no-op (no tensor op is inserted at all).
    # NOTE the rms is a single scalar over the WHOLE tensor (batch x units), so
    # like BatchNorm this makes the forward pass batch-dependent; unlike
    # BatchNorm there are no running statistics, so eval uses the eval batch's
    # own rms. Keep eval_batch_size == batch_size for like-for-like numbers.
    relu_shift: Optional[float] = None
    # Whether rms(a) is treated as a constant in the backward pass.
    # True (default, RECOMMENDED) = detached: the shift is a pure per-layer
    #   additive bias and the backward pass is bit-identical to plain He
    #   backprop, so the intervention is purely a forward-pass/geometry one.
    #   This is the exact activation-space dual of row-centering, which is
    #   likewise a pure forward-side constraint.
    # False = differentiable: the Jacobian picks up the rank-one term
    #   -(c/(N*rms)) * 1 a^T (N = numel), which couples every unit AND every
    #   SAMPLE in the batch -- a BatchNorm-like coupling with no counterpart in
    #   row-centering. Measured at init to move the per-layer weight gradient by
    #   a median 3-32% and up to 71% (reports/results/relu_shift_funnel_fwd_bwd.json,
    #   field `fork_relative_grad_diff`), with no principled scaling.
    # Ignored when relu_shift is None. See cluster/11_relu_shift/README.md.
    relu_shift_detach: bool = True
    # Restrict training to a subset of layers (FC only): requires_grad=False on
    # every Linear layer NOT named here (weight and bias), backprop still runs
    # through frozen layers unchanged. Names: "fc<N>" (1-indexed hidden layer,
    # fc1..fc<depth>) or "head" (final classifier layer). None = all trainable
    # (standard behavior, no freezing).
    trainable_layers: Optional[List[str]] = None


@dataclass
class TrainingConfig:
    """Configuration for supervised training experiments."""

    dataset: Literal["mnist", "fashion_mnist", "cifar10"] = "fashion_mnist"
    epochs: int = 5
    min_epochs: int = 1
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: Literal["adam", "sgd"] = "adam"
    momentum: float = 0.9
    scheduler: Literal["none", "cosine", "step", "onecycle", "plateau"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6
    plateau_metric: Literal["eval_train_loss", "eval_train_accuracy"] = "eval_train_loss"
    plateau_warmup_epochs: int = 0
    # Linear LR warmup: ramp from learning_rate * lr_warmup_start_factor up
    # to learning_rate over the first lr_warmup_epochs epochs.
    # Applied BEFORE any scheduler step; the scheduler's own warmup (e.g.
    # plateau_warmup_epochs) is independent and stacks on top.
    lr_warmup_epochs: int = 0
    lr_warmup_start_factor: float = 0.01
    # Gradient clipping by global L2 norm; None = no clipping.
    grad_clip_max_norm: Optional[float] = None
    # Explosion guard. If abort_on_explosion is True, training aborts when
    # batch loss exceeds explosion_loss_factor * first-ever-batch-loss, OR
    # when batch loss is NaN/Inf (the latter aborts regardless of the flag).
    # The first-batch loss is recorded once at epoch 1 batch 0 and never updated.
    abort_on_explosion: bool = False
    explosion_loss_factor: float = 5.0
    # If True, prints per-batch loss + per-layer grad-norm summary every batch
    # during epoch 1 only. Useful for diagnosing initialization-induced
    # explosion/vanishing in the very first training step.
    log_per_batch_first_epoch: bool = False
    # If True, prints the FULL per-layer gradient L2 norm vector (one value per
    # hidden layer) every diagnostic epoch, instead of only the [min, max]
    # range. Use with diagnostics_every=1 to log every layer every epoch.
    log_grad_per_layer: bool = False
    num_train_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    normalize_inputs: bool = True
    num_workers: int = 0
    label_smoothing: float = 0.0
    target_train_accuracy: Optional[float] = None
    target_patience: int = 1
    target_metric: Literal["train_accuracy", "eval_train_accuracy"] = "eval_train_accuracy"
    log_every_epoch: bool = False
    diagnostics_every: int = 0
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 0
    resume_checkpoint: Optional[str] = None


@dataclass
class GeometryBenchmarkConfig:
    """Configuration for dataset-based geometry benchmarking."""

    dataset: Literal["mnist", "fashion_mnist", "cifar10"] = "fashion_mnist"
    num_samples: int = 2000
    depths: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    init_strategies: List[str] = field(
        default_factory=lambda: ["he", "orthogonal_he", "row_centered_he_var_adj"]
    )
    knn_k: int = 5
    n_pairs: int = 2000
    flatten: bool = True
    normalize_inputs: bool = False


@dataclass
class ProjectionConfig:
    """Configuration for random projection experiments."""

    num_layers: int = 1
    mode: Literal["square", "rectangular"] = "square"
    output_dim: int = 2  # For rectangular mode
    variance_scale: float = 2.0  # Multiplier for variance (2/d * scale)


@dataclass
class ShapeConfig:
    """Configuration for synthetic shape generation."""

    n_points: int = 200
    shape_type: Literal["circle", "ellipse", "square", "rectangle"] = "circle"
    # Circle/ellipse params
    radius: float = 1.0
    a: float = 2.0  # horizontal radius for ellipse
    b: float = 1.0  # vertical radius for ellipse
    # Square/rectangle params
    side: float = 2.0
    width: float = 4.0
    height: float = 2.0
    # Common
    center: tuple = (0.0, 0.0)
    rotation_degrees: float = 0.0  # Optional rotation
