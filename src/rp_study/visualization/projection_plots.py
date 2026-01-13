"""
Visualization utilities for random projection experiments.

Functions for plotting PCA vs RP comparisons, multi-layer transformations,
and shape transformation visualizations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_vs_rp(
    X: np.ndarray,
    X_pca: np.ndarray,
    X_rp: np.ndarray,
    labels: Optional[np.ndarray] = None,
    X_pca_relu: Optional[np.ndarray] = None,
    X_rp_relu: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (14, 12),
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison between PCA and Random Projection transformations.

    Args:
        X: Original data (used for PCA visualization if 2D).
        X_pca: PCA-transformed data (2D).
        X_rp: Random projection transformed data (2D).
        labels: Optional labels for coloring points.
        X_pca_relu: Optional PCA + ReLU transformed data.
        X_rp_relu: Optional RP + ReLU transformed data.
        figsize: Figure size.
        title: Overall figure title.

    Returns:
        Matplotlib figure.
    """
    has_relu = X_pca_relu is not None and X_rp_relu is not None
    nrows = 2 if has_relu else 1
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    scatter_kwargs = dict(s=5, cmap="viridis", alpha=0.7)
    if labels is not None:
        scatter_kwargs["c"] = labels

    # PCA
    axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], **scatter_kwargs)
    axes[0, 0].set_title("PCA (2 Components)")
    axes[0, 0].axis("equal")

    # RP
    axes[0, 1].scatter(X_rp[:, 0], X_rp[:, 1], **scatter_kwargs)
    axes[0, 1].set_title("Random Projection")
    axes[0, 1].axis("equal")

    if has_relu:
        # PCA + ReLU
        axes[1, 0].scatter(X_pca_relu[:, 0], X_pca_relu[:, 1], **scatter_kwargs)
        axes[1, 0].set_title("ReLU on PCA")
        axes[1, 0].axis("equal")

        # RP + ReLU
        axes[1, 1].scatter(X_rp_relu[:, 0], X_rp_relu[:, 1], **scatter_kwargs)
        axes[1, 1].set_title("ReLU on Random Projection")
        axes[1, 1].axis("equal")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_multi_layer_transformations(
    original_data: np.ndarray,
    transformed_data: Dict[int, np.ndarray],
    labels: Optional[np.ndarray] = None,
    mode: str = "rectangular",
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Plot data after different numbers of RP+ReLU layers.

    Args:
        original_data: Original dataset (for reference).
        transformed_data: Dict mapping num_layers -> transformed data (2D).
        labels: Optional labels for coloring.
        mode: "rectangular" or "square" (for title).
        figsize: Figure size.
        suptitle: Overall figure title.

    Returns:
        Matplotlib figure.
    """
    num_plots = len(transformed_data) + 1  # +1 for original
    ncols = min(num_plots, 4)
    nrows = (num_plots + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    scatter_kwargs = dict(s=5, cmap="viridis", alpha=0.7)
    if labels is not None:
        scatter_kwargs["c"] = labels

    # Plot original (if 2D or reduced)
    if original_data.shape[1] == 2:
        axes[0].scatter(original_data[:, 0], original_data[:, 1], **scatter_kwargs)
    else:
        # Reduce for visualization
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(original_data)
        axes[0].scatter(X_vis[:, 0], X_vis[:, 1], **scatter_kwargs)
    axes[0].set_title("Original Data")
    axes[0].axis("equal")

    # Plot transformed data
    for i, (num_layers, X_transformed) in enumerate(sorted(transformed_data.items()), start=1):
        if i < len(axes):
            axes[i].scatter(X_transformed[:, 0], X_transformed[:, 1], **scatter_kwargs)
            axes[i].set_title(f"{num_layers} Layers ({mode})")
            axes[i].axis("equal")

    # Hide unused axes
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    return fig


def plot_shape_transformations(
    shapes: Dict[str, np.ndarray],
    transformed_shapes: Dict[str, np.ndarray],
    num_layers: int,
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """Plot original shapes and their transformed versions.

    Args:
        shapes: Dict mapping shape name to original points.
        transformed_shapes: Dict mapping shape name to transformed points.
        num_layers: Number of RP+ReLU layers applied.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    num_shapes = len(shapes)
    fig, axes = plt.subplots(2, num_shapes, figsize=figsize)

    for i, (name, X) in enumerate(shapes.items()):
        # Original
        axes[0, i].scatter(X[:, 0], X[:, 1], s=15, cmap="viridis")
        axes[0, i].set_title(f"{name} (Original)")
        axes[0, i].axis("equal")

        # Transformed
        X_trans = transformed_shapes[name]
        axes[1, i].scatter(X_trans[:, 0], X_trans[:, 1], s=15, cmap="viridis")
        axes[1, i].set_title(f"{name} ({num_layers} layers)")
        axes[1, i].axis("equal")

    fig.suptitle(f"Shape Transformations after {num_layers} RP+ReLU Layers", fontsize=14)
    plt.tight_layout()
    return fig


def plot_layer_comparison(
    data_by_layer: Dict[str, Dict[int, np.ndarray]],
    labels: Optional[np.ndarray] = None,
    layer_counts: List[int] = [1, 5, 10, 20],
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot comparison of different transformation modes across layers.

    Args:
        data_by_layer: Dict mapping mode name to {num_layers: transformed_data}.
        labels: Optional labels for coloring.
        layer_counts: Which layer counts to show.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    modes = list(data_by_layer.keys())
    num_modes = len(modes)
    num_layers = len(layer_counts)

    if figsize is None:
        figsize = (4 * num_layers, 4 * num_modes)

    fig, axes = plt.subplots(num_modes, num_layers, figsize=figsize)
    if num_modes == 1:
        axes = axes.reshape(1, -1)

    scatter_kwargs = dict(s=5, cmap="viridis", alpha=0.7)
    if labels is not None:
        scatter_kwargs["c"] = labels

    for i, mode in enumerate(modes):
        for j, num_layer in enumerate(layer_counts):
            if num_layer in data_by_layer[mode]:
                X = data_by_layer[mode][num_layer]
                axes[i, j].scatter(X[:, 0], X[:, 1], **scatter_kwargs)
                axes[i, j].set_title(f"{mode} - {num_layer} layers")
                axes[i, j].axis("equal")
            else:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                               transform=axes[i, j].transAxes)

    plt.tight_layout()
    return fig


def plot_jl_vs_rp(
    X_jl: np.ndarray,
    X_rp: np.ndarray,
    X_rp_relu: np.ndarray,
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (15, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison of JL projection vs RP vs RP+ReLU.

    Args:
        X_jl: JL-projected data (2D).
        X_rp: Random projected data (2D).
        X_rp_relu: Random projected + ReLU data (2D).
        labels: Optional labels for coloring.
        figsize: Figure size.
        title: Overall title.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    scatter_kwargs = dict(s=5, cmap="viridis", alpha=0.7)
    if labels is not None:
        scatter_kwargs["c"] = labels

    axes[0].scatter(X_jl[:, 0], X_jl[:, 1], **scatter_kwargs)
    axes[0].set_title("Johnson-Lindenstrauss")
    axes[0].axis("equal")

    axes[1].scatter(X_rp[:, 0], X_rp[:, 1], **scatter_kwargs)
    axes[1].set_title("Random Projection")
    axes[1].axis("equal")

    axes[2].scatter(X_rp_relu[:, 0], X_rp_relu[:, 1], **scatter_kwargs)
    axes[2].set_title("RP + ReLU")
    axes[2].axis("equal")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig
