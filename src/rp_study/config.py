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
    dataset: Literal["mnist", "fashion_mnist"] = "fashion_mnist"
    num_bins: int = 50  # For histogram plotting


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
