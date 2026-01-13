"""
Gradient analysis visualization utilities.

Functions for plotting gradient distributions, zero-gradient statistics,
activation histograms, and related analyses.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..experiments.gradient_analysis import ExperimentResults, LayerGradientStats, ActivationStats


# Default color for highlighting zeros
ZERO_COLOR = "red"
DEFAULT_COLOR = "steelblue"


def plot_gradient_histograms(
    results: ExperimentResults,
    bins: int = 50,
    highlight_zeros: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot histograms of gradient entries for each layer.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        bins: Number of histogram bins.
        highlight_zeros: Whether to highlight the zero bin in red.
        figsize: Figure size (auto-computed if None).
        title: Optional overall title.

    Returns:
        Matplotlib figure.
    """
    grad_stats = results.grad_stats
    num_layers = len(grad_stats)

    if figsize is None:
        figsize = (4 * min(num_layers, 4), 3 * ((num_layers + 3) // 4))

    ncols = min(num_layers, 4)
    nrows = (num_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, (layer_name, stats) in enumerate(grad_stats.items()):
        ax = axes[i]
        _plot_histogram_with_zeros(
            ax, stats.matrix.flatten(), bins,
            highlight_zeros=highlight_zeros,
            title=f"{layer_name} (zeros: {stats.zero_proportion:.1%})"
        )

    # Hide unused axes
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_row_norm_histograms(
    results: ExperimentResults,
    bins: int = 50,
    highlight_zeros: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot histograms of row-wise gradient norms for each layer.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        bins: Number of histogram bins.
        highlight_zeros: Whether to highlight the zero bin in red.
        figsize: Figure size.
        title: Optional overall title.

    Returns:
        Matplotlib figure.
    """
    grad_stats = results.grad_stats
    num_layers = len(grad_stats)

    if figsize is None:
        figsize = (4 * min(num_layers, 4), 3 * ((num_layers + 3) // 4))

    ncols = min(num_layers, 4)
    nrows = (num_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, (layer_name, stats) in enumerate(grad_stats.items()):
        ax = axes[i]
        _plot_histogram_with_zeros(
            ax, stats.row_norms, bins,
            highlight_zeros=highlight_zeros,
            title=f"{layer_name} row norms (zero rows: {stats.zero_row_proportion:.1%})"
        )
        ax.set_xlabel("Row L2 Norm")

    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_activation_histograms(
    results: ExperimentResults,
    bins: int = 50,
    highlight_zeros: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot histograms of activations for each layer.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        bins: Number of histogram bins.
        highlight_zeros: Whether to highlight the zero bin in red.
        figsize: Figure size.
        title: Optional overall title.

    Returns:
        Matplotlib figure.
    """
    activation_stats = results.activation_stats
    num_layers = len(activation_stats)

    if figsize is None:
        figsize = (4 * min(num_layers, 4), 3 * ((num_layers + 3) // 4))

    ncols = min(num_layers, 4)
    nrows = (num_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for i, stats in enumerate(activation_stats):
        ax = axes[i]
        _plot_histogram_with_zeros(
            ax, stats.values.flatten(), bins,
            highlight_zeros=highlight_zeros,
            title=f"Layer {stats.layer_idx} (zeros: {stats.zero_proportion:.1%})"
        )
        ax.set_xlabel("Activation Value")

    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    return fig


def plot_zero_gradient_stats(
    results: ExperimentResults,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot zero-gradient statistics across layers.

    Creates two subplots:
    1. Zero gradient counts per layer
    2. Zero gradient proportions per layer

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    grad_stats = results.grad_stats

    layers = list(grad_stats.keys())
    zero_counts = [stats.zero_count for stats in grad_stats.values()]
    zero_props = [stats.zero_proportion for stats in grad_stats.values()]
    zero_row_props = [stats.zero_row_proportion for stats in grad_stats.values()]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Zero counts
    axes[0].bar(layers, zero_counts, color=DEFAULT_COLOR)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Zero Gradient Count")
    axes[0].set_title("Zero Gradient Entries per Layer")
    axes[0].tick_params(axis="x", rotation=45)

    # Zero proportions
    axes[1].bar(layers, zero_props, color=DEFAULT_COLOR)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Zero Gradient Proportion per Layer")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, 1)

    # Zero row proportions
    axes[2].bar(layers, zero_row_props, color=ZERO_COLOR, alpha=0.7)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Proportion")
    axes[2].set_title("Entirely Zero Rows per Layer")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_row_norm_per_layer(
    results: ExperimentResults,
    figsize: Tuple[float, float] = (10, 5),
    show_std: bool = True,
) -> plt.Figure:
    """Plot mean row-norm of gradients per layer.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        figsize: Figure size.
        show_std: Whether to show standard deviation as error bars.

    Returns:
        Matplotlib figure.
    """
    grad_stats = results.grad_stats

    layers = list(grad_stats.keys())
    mean_norms = [stats.mean_row_norm for stats in grad_stats.values()]

    fig, ax = plt.subplots(figsize=figsize)

    if show_std:
        std_norms = [np.std(stats.row_norms) for stats in grad_stats.values()]
        ax.errorbar(layers, mean_norms, yerr=std_norms, fmt="o-", capsize=5, color=DEFAULT_COLOR)
    else:
        ax.plot(layers, mean_norms, "o-", color=DEFAULT_COLOR)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Row L2 Norm")
    ax.set_title("Average Gradient Row Norm per Layer")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_activation_zero_stats(
    results: ExperimentResults,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot zero-activation statistics across layers.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    activation_stats = results.activation_stats

    layers = [f"L{stats.layer_idx}" for stats in activation_stats]
    zero_props = [stats.zero_proportion for stats in activation_stats]
    inactive_props = [stats.truly_inactive_proportion for stats in activation_stats]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Zero activation proportion
    axes[0].bar(layers, zero_props, color=DEFAULT_COLOR)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Zero Activations per Layer")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylim(0, 1)

    # Truly inactive neurons
    axes[1].bar(layers, inactive_props, color=ZERO_COLOR, alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Truly Inactive Neurons (zero for ALL samples)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    return fig


def compare_initializations_plot(
    results_dict: Dict[str, ExperimentResults],
    metric: str = "zero_proportion",
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Compare gradient statistics across different initializations.

    Args:
        results_dict: Dictionary mapping init name to ExperimentResults.
        metric: Metric to compare ("zero_proportion", "mean_row_norm", "zero_row_proportion").
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for init_name, results in results_dict.items():
        layers = list(results.grad_stats.keys())

        if metric == "zero_proportion":
            values = [s.zero_proportion for s in results.grad_stats.values()]
            ylabel = "Zero Gradient Proportion"
        elif metric == "mean_row_norm":
            values = [s.mean_row_norm for s in results.grad_stats.values()]
            ylabel = "Mean Row Norm"
        elif metric == "zero_row_proportion":
            values = [s.zero_row_proportion for s in results.grad_stats.values()]
            ylabel = "Zero Row Proportion"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        ax.plot(layers, values, "o-", label=init_name)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Comparison of {metric} across Initializations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def _plot_histogram_with_zeros(
    ax: plt.Axes,
    data: np.ndarray,
    bins: int,
    highlight_zeros: bool = True,
    title: Optional[str] = None,
    color: str = DEFAULT_COLOR,
) -> None:
    """Helper to plot histogram with optional zero highlighting.

    Args:
        ax: Matplotlib axes.
        data: Data to plot.
        bins: Number of bins.
        highlight_zeros: Whether to highlight zero bin.
        title: Axes title.
        color: Bar color.
    """
    # Filter out NaN/Inf
    data = data[np.isfinite(data)]

    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    counts, edges, patches = ax.hist(data, bins=bins, color=color, alpha=0.7, edgecolor="black")

    if highlight_zeros:
        # Find the bin containing zero
        zero_bin_idx = np.digitize(0, edges) - 1
        if 0 <= zero_bin_idx < len(patches):
            patches[zero_bin_idx].set_facecolor(ZERO_COLOR)
            patches[zero_bin_idx].set_alpha(0.9)

            # Add count label
            zero_count = int(counts[zero_bin_idx])
            if zero_count > 0:
                ax.annotate(
                    f"{zero_count}",
                    xy=(edges[zero_bin_idx] + (edges[zero_bin_idx + 1] - edges[zero_bin_idx]) / 2,
                        counts[zero_bin_idx]),
                    ha="center", va="bottom", fontsize=8, color=ZERO_COLOR
                )

    if title:
        ax.set_title(title, fontsize=10)
