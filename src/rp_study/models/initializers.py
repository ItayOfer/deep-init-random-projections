"""
Extensible weight initialization strategies.

This module uses a registry pattern to make adding new initialization
strategies trivial. To add a new initializer:

1. Define a function that takes (layer: nn.Linear, **kwargs)
2. Decorate it with @register_initializer("your_name")

Example:
    @register_initializer("lecun")
    def lecun_init(layer: nn.Linear, **kwargs):
        fan_in = layer.weight.shape[1]
        std = math.sqrt(1.0 / fan_in)
        with torch.no_grad():
            layer.weight.normal_(0.0, std)
            if layer.bias is not None:
                layer.bias.zero_()
"""

import math
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn


# Registry for initialization strategies
INITIALIZERS: Dict[str, Callable] = {}


def register_initializer(name: str):
    """Decorator to register a new initialization strategy.

    Args:
        name: Name to register the initializer under.

    Example:
        @register_initializer("my_init")
        def my_init(layer: nn.Linear, **kwargs):
            # Initialize layer.weight and layer.bias
            pass
    """
    def decorator(fn: Callable) -> Callable:
        INITIALIZERS[name] = fn
        return fn
    return decorator


def initialize_layer(
    layer: nn.Linear,
    strategy: str = "he",
    **kwargs,
) -> None:
    """Initialize a linear layer using the specified strategy.

    Args:
        layer: The nn.Linear layer to initialize.
        strategy: Name of the initialization strategy.
        **kwargs: Additional arguments passed to the initializer.

    Raises:
        ValueError: If the strategy is not registered.
    """
    if strategy not in INITIALIZERS:
        available = list(INITIALIZERS.keys())
        raise ValueError(f"Unknown initialization strategy: {strategy}. Available: {available}")

    INITIALIZERS[strategy](layer, **kwargs)


def list_initializers() -> list:
    """Return a list of available initialization strategies."""
    return list(INITIALIZERS.keys())


# =============================================================================
# Built-in initialization strategies
# =============================================================================


@register_initializer("he")
def he_init(layer: nn.Linear, **kwargs) -> None:
    """He (Kaiming) initialization for ReLU networks.

    Uses fan_in mode: Var(W) = 2/fan_in

    This is the standard initialization for ReLU networks from:
    "Delving Deep into Rectifiers" (He et al., 2015)
    """
    nn.init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("row_centered_he")
def row_centered_he_init(layer: nn.Linear, **kwargs) -> None:
    """Row-centered He initialization.

    Draws from N(0, 2/fan_in) then subtracts the mean of each row,
    ensuring each row sums to zero. Biases are set to zero.

    This variant may have different gradient flow properties.
    """
    fan_in = layer.weight.shape[1]
    std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=std)
        # Subtract row mean to center each row
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("custom_variance")
def custom_variance_init(
    layer: nn.Linear,
    variance: Optional[float] = None,
    mean: float = 0.0,
    **kwargs,
) -> None:
    """Custom variance initialization.

    Args:
        layer: Layer to initialize.
        variance: Variance of the weight distribution. If None, uses 2/fan_in.
        mean: Mean of the weight distribution.

    This allows experiments with different variance scales (e.g., 2/d, 2.5/d, 4/d).
    """
    fan_in = layer.weight.shape[1]
    if variance is None:
        variance = 2.0 / fan_in

    std = math.sqrt(variance)

    with torch.no_grad():
        layer.weight.normal_(mean=mean, std=std)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("xavier")
def xavier_init(layer: nn.Linear, **kwargs) -> None:
    """Xavier (Glorot) initialization.

    Uses the average of fan_in and fan_out: Var(W) = 2/(fan_in + fan_out)

    Suitable for sigmoid/tanh activations.
    """
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("uniform_he")
def uniform_he_init(layer: nn.Linear, **kwargs) -> None:
    """Uniform distribution with He-like variance.

    Uses U(-a, a) where a = sqrt(3 * 2/fan_in) so that Var = 2/fan_in.

    This matches the original notebook's uniform initialization style.
    """
    fan_in = layer.weight.shape[1]
    a = math.sqrt(3 * 2 / fan_in)  # Var(U(-a,a)) = a^2/3

    with torch.no_grad():
        layer.weight.uniform_(-a, a)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("orthogonal")
def orthogonal_init(layer: nn.Linear, gain: float = 1.0, **kwargs) -> None:
    """Orthogonal initialization.

    Initializes the weight matrix to be orthogonal (or semi-orthogonal
    if not square). Can help with gradient flow in deep networks.

    Args:
        layer: Layer to initialize.
        gain: Multiplicative factor for the weights.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
