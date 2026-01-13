"""
Neural network architectures for gradient analysis experiments.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .initializers import initialize_layer, INITIALIZERS


class FeedForward(nn.Module):
    """Fully-connected ReLU network with configurable architecture and initialization.

    The network has the structure:
        input -> [Linear -> ReLU] * (n-1) -> Linear -> output

    All hidden layers use ReLU activation. The final layer is linear (no activation).

    Args:
        layer_sizes: List of layer dimensions. First element is input dim,
                    last is output dim, middle elements are hidden layer sizes.
                    Example: [784, 512, 256, 1] creates a network with 2 hidden layers.
        init_strategy: Initialization strategy name (see models.initializers).
        use_bias: Whether to use bias terms in linear layers.
        **init_kwargs: Additional arguments passed to the initializer.

    Example:
        # Standard He-initialized network
        net = FeedForward([784, 784, 512, 256, 1], init_strategy="he")

        # Row-centered He initialization
        net = FeedForward([784, 784, 1], init_strategy="row_centered_he")

        # Custom variance (e.g., 4/d instead of 2/d)
        net = FeedForward([784, 784, 1], init_strategy="custom_variance", variance=4.0/784)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        init_strategy: str = "he",
        use_bias: bool = True,
        **init_kwargs,
    ):
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")

        if init_strategy not in INITIALIZERS:
            raise ValueError(
                f"Unknown init_strategy: {init_strategy}. "
                f"Available: {list(INITIALIZERS.keys())}"
            )

        self.layer_sizes = layer_sizes
        self.init_strategy = init_strategy
        self.init_kwargs = init_kwargs

        # Build layers
        self.layers = nn.ModuleList()
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer = nn.Linear(fan_in, fan_out, bias=use_bias)
            initialize_layer(layer, strategy=init_strategy, **init_kwargs)
            self.layers.append(layer)

    def forward(
        self,
        x: torch.Tensor,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            return_activations: If True, also return intermediate activations.

        Returns:
            If return_activations is False: Output tensor.
            If return_activations is True: Tuple of (output, activations) where
                activations is a list of tensors after each layer (including ReLU).
        """
        activations = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU to all but last layer
                x = F.relu(x)
            if return_activations:
                activations.append(x)

        if return_activations:
            return x, activations
        return x

    @property
    def num_layers(self) -> int:
        """Number of weight layers in the network."""
        return len(self.layers)

    @property
    def num_hidden_layers(self) -> int:
        """Number of hidden layers (excluding output layer)."""
        return len(self.layers) - 1

    def get_layer_info(self) -> List[dict]:
        """Get information about each layer.

        Returns:
            List of dicts with 'input_dim', 'output_dim', and 'num_params' for each layer.
        """
        info = []
        for layer in self.layers:
            num_params = layer.weight.numel()
            if layer.bias is not None:
                num_params += layer.bias.numel()
            info.append({
                "input_dim": layer.in_features,
                "output_dim": layer.out_features,
                "num_params": num_params,
            })
        return info

    def __repr__(self) -> str:
        return (
            f"FeedForward(layer_sizes={self.layer_sizes}, "
            f"init_strategy='{self.init_strategy}', "
            f"num_params={sum(p.numel() for p in self.parameters())})"
        )


def create_deep_network(
    input_dim: int = 784,
    output_dim: int = 1,
    num_hidden_layers: int = 100,
    hidden_widths: Union[int, List[int], str] = "random",
    width_range: Tuple[int, int] = (100, 1000),
    init_strategy: str = "he",
    seed: Optional[int] = None,
    **init_kwargs,
) -> FeedForward:
    """Factory function to create deep networks with various width patterns.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        num_hidden_layers: Number of hidden layers.
        hidden_widths: Either:
            - An int: all hidden layers have this width
            - A list of ints: specific width for each hidden layer
            - "random": randomly sample widths from width_range
        width_range: (min, max) for random width sampling.
        init_strategy: Initialization strategy.
        seed: Random seed for width sampling (if hidden_widths="random").
        **init_kwargs: Additional arguments for initialization.

    Returns:
        Configured FeedForward network.

    Example:
        # 100-layer network with random widths between 100-1000
        net = create_deep_network(num_hidden_layers=100, hidden_widths="random")

        # All hidden layers with width 512
        net = create_deep_network(num_hidden_layers=50, hidden_widths=512)
    """
    import numpy as np

    if seed is not None:
        np.random.seed(seed)

    if isinstance(hidden_widths, int):
        widths = [hidden_widths] * num_hidden_layers
    elif isinstance(hidden_widths, list):
        if len(hidden_widths) != num_hidden_layers:
            raise ValueError(
                f"hidden_widths list length ({len(hidden_widths)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
        widths = hidden_widths
    elif hidden_widths == "random":
        widths = np.random.randint(width_range[0], width_range[1] + 1, size=num_hidden_layers).tolist()
    else:
        raise ValueError(f"Invalid hidden_widths: {hidden_widths}")

    layer_sizes = [input_dim] + widths + [output_dim]

    return FeedForward(layer_sizes, init_strategy=init_strategy, **init_kwargs)
