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
    scheduler: Literal["none", "cosine", "step", "onecycle"] = "none"
    step_size: int = 10
    gamma: float = 0.1
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    num_train_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    normalize_inputs: bool = True
    num_workers: int = 0
    label_smoothing: float = 0.0
    target_train_accuracy: Optional[float] = None
    target_patience: int = 1
    target_metric: Literal["train_accuracy", "eval_train_accuracy"] = "eval_train_accuracy"
    log_every_epoch: bool = False


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
