"""
Random projection utilities.

Functions for creating random projection matrices and applying
multi-layer random projections with ReLU activations.
"""

from typing import Literal, Optional, Union
import math
import numpy as np
import torch


def relu(data: np.ndarray) -> np.ndarray:
    """Apply ReLU activation (numpy version)."""
    return np.maximum(0, data)


def random_projection_matrix(
    d_in: int,
    d_out: int,
    variance: Union[float, Literal["he", "jl"]] = "he",
    distribution: Literal["normal", "uniform"] = "uniform",
    device: Optional[torch.device] = None,
    as_numpy: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Create a random projection matrix.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
        variance: Variance of the distribution. Can be:
            - A float: explicit variance value
            - "he": variance = 2/d_in (He initialization)
            - "jl": variance = 1/d_out (Johnson-Lindenstrauss)
        distribution: "normal" for Gaussian, "uniform" for uniform.
        device: Torch device (only used if as_numpy=False).
        as_numpy: If True, return numpy array. If False, return torch tensor.

    Returns:
        Random projection matrix of shape (d_in, d_out).
    """
    # Resolve variance
    if variance == "he":
        var = 2.0 / d_in
    elif variance == "jl":
        var = 1.0 / d_out
    elif isinstance(variance, (int, float)):
        var = float(variance)
    else:
        raise ValueError(f"Unknown variance specification: {variance}")

    if as_numpy:
        if distribution == "normal":
            std = math.sqrt(var)
            return np.random.normal(0, std, size=(d_in, d_out))
        else:  # uniform
            a = math.sqrt(3 * var)  # Var(U(-a,a)) = a^2/3
            return np.random.uniform(-a, a, size=(d_in, d_out))
    else:
        if distribution == "normal":
            std = math.sqrt(var)
            R = torch.randn(d_in, d_out) * std
        else:  # uniform
            a = math.sqrt(3 * var)
            R = torch.empty(d_in, d_out).uniform_(-a, a)

        if device is not None:
            R = R.to(device)
        return R


def apply_random_projection(
    X: np.ndarray,
    d_out: int,
    variance: Union[float, Literal["he", "jl"]] = "he",
    distribution: Literal["normal", "uniform"] = "uniform",
    apply_relu: bool = True,
) -> np.ndarray:
    """Apply a single random projection (optionally with ReLU).

    Args:
        X: Input data of shape (n_samples, d_in).
        d_out: Output dimension.
        variance: Variance specification (see random_projection_matrix).
        distribution: Distribution type.
        apply_relu: Whether to apply ReLU after projection.

    Returns:
        Projected data of shape (n_samples, d_out).
    """
    d_in = X.shape[1]
    R = random_projection_matrix(d_in, d_out, variance, distribution, as_numpy=True)
    Y = X @ R

    if apply_relu:
        Y = relu(Y)

    return Y


def multi_layer_projection(
    X: Union[np.ndarray, torch.Tensor],
    num_layers: int,
    mode: Literal["square", "rectangular"] = "square",
    output_dim: int = 2,
    variance: Union[float, Literal["he"]] = "he",
    distribution: Literal["normal", "uniform"] = "uniform",
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Apply multiple layers of random projections with ReLU.

    Args:
        X: Input data of shape (n_samples, d_in).
        num_layers: Number of projection layers to apply.
        mode: "square" keeps dimension constant, "rectangular" projects to output_dim.
        output_dim: Target dimension for rectangular mode.
        variance: Variance specification for each layer.
        distribution: Distribution type for projection matrices.
        device: Torch device for GPU computation.

    Returns:
        Transformed data after all layers.

    Notes:
        - Square mode: Each layer is (d, d) and preserves input dimension.
        - Rectangular mode: First layer projects to output_dim, subsequent layers
          are (output_dim, output_dim).
    """
    is_torch = isinstance(X, torch.Tensor)

    if is_torch:
        return _multi_layer_projection_torch(X, num_layers, mode, output_dim, variance, distribution, device)
    else:
        return _multi_layer_projection_numpy(X, num_layers, mode, output_dim, variance, distribution)


def _multi_layer_projection_numpy(
    X: np.ndarray,
    num_layers: int,
    mode: str,
    output_dim: int,
    variance: Union[float, str],
    distribution: str,
) -> np.ndarray:
    """Numpy implementation of multi-layer projection."""
    X_layer = X.copy()

    if mode == "rectangular":
        # First layer: d_in -> output_dim
        d_in = X_layer.shape[1]
        var = 2.0 / d_in if variance == "he" else variance
        R_first = random_projection_matrix(d_in, output_dim, var, distribution, as_numpy=True)
        X_layer = relu(X_layer @ R_first)

        # Subsequent layers: output_dim -> output_dim
        for _ in range(num_layers - 1):
            var = 2.0 / output_dim if variance == "he" else variance
            R = random_projection_matrix(output_dim, output_dim, var, distribution, as_numpy=True)
            X_layer = relu(X_layer @ R)

    else:  # square mode
        d = X_layer.shape[1]
        for _ in range(num_layers):
            var = 2.0 / d if variance == "he" else variance
            R = random_projection_matrix(d, d, var, distribution, as_numpy=True)
            X_layer = relu(X_layer @ R)

    return X_layer


def _multi_layer_projection_torch(
    X: torch.Tensor,
    num_layers: int,
    mode: str,
    output_dim: int,
    variance: Union[float, str],
    distribution: str,
    device: Optional[torch.device],
) -> torch.Tensor:
    """Torch implementation of multi-layer projection (supports GPU)."""
    if device is not None:
        X = X.to(device)

    X_layer = X.clone()

    if mode == "rectangular":
        # First layer: d_in -> output_dim
        d_in = X_layer.shape[1]
        var = 2.0 / d_in if variance == "he" else variance
        R_first = random_projection_matrix(d_in, output_dim, var, distribution, device=device, as_numpy=False)
        X_layer = torch.relu(X_layer @ R_first)

        # Subsequent layers: output_dim -> output_dim
        for _ in range(num_layers - 1):
            var = 2.0 / output_dim if variance == "he" else variance
            R = random_projection_matrix(output_dim, output_dim, var, distribution, device=device, as_numpy=False)
            X_layer = torch.relu(X_layer @ R)

    else:  # square mode
        d = X_layer.shape[1]
        for _ in range(num_layers):
            var = 2.0 / d if variance == "he" else variance
            R = random_projection_matrix(d, d, var, distribution, device=device, as_numpy=False)
            X_layer = torch.relu(X_layer @ R)

    return X_layer


def multi_layer_rp_with_init(
    X: np.ndarray,
    n_layers: int,
    init_strategy: str = "he",
    width: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **init_kwargs,
) -> np.ndarray:
    """Apply multi-layer RP + ReLU using registry initializers.

    This bridges geometry experiments (numpy data) with the registry-based
    initializers (torch nn.Linear), ensuring a single source of truth for
    initialization logic.

    Args:
        X: Input data of shape (n_samples, d_in), numpy array.
        n_layers: Number of projection layers to apply.
        init_strategy: Name from the initializer registry (e.g., "he",
            "row_centered_he", "orthogonal_he").
        width: Width of each hidden layer. Defaults to d_in (square projections).
        seed: Optional random seed for reproducibility.
        device: Torch device ("cuda", "cpu", or None for auto-detect).
        **init_kwargs: Extra kwargs passed to the initializer (e.g., alpha=0.5).

    Returns:
        Transformed numpy array of shape (n_samples, width) after n_layers
        of (Linear + ReLU).
    """
    import torch.nn as nn
    from ..models.initializers import initialize_layer

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Resolve device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    d_in = X.shape[1]
    w = width or d_in

    X_tensor = torch.from_numpy(X).float().to(dev)

    current_dim = d_in
    for _ in range(n_layers):
        layer = nn.Linear(current_dim, w, bias=False).to(dev)
        # Initialize outside no_grad: some initializers (e.g., kernel_preserving)
        # run an internal optimization loop that requires gradients.
        initialize_layer(layer, strategy=init_strategy, **init_kwargs)
        # Forward pass doesn't need gradients
        with torch.no_grad():
            X_tensor = torch.relu(X_tensor @ layer.weight.t())
        current_dim = w

    return X_tensor.cpu().numpy()


def jl_projection(
    X: np.ndarray,
    target_dim: int,
    apply_relu: bool = False,
) -> np.ndarray:
    """Apply Johnson-Lindenstrauss random projection.

    Uses Gaussian projection with variance 1/target_dim, which approximately
    preserves pairwise distances.

    Args:
        X: Input data of shape (n_samples, d_in).
        target_dim: Target dimension.
        apply_relu: Whether to apply ReLU after projection.

    Returns:
        Projected data of shape (n_samples, target_dim).
    """
    d_in = X.shape[1]
    R = random_projection_matrix(d_in, target_dim, variance="jl", distribution="normal", as_numpy=True)
    Y = X @ R

    if apply_relu:
        Y = relu(Y)

    return Y
