"""
Gradient analysis visualization utilities.

Functions for plotting gradient distributions, zero-gradient statistics,
activation histograms, and related analyses.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..experiments.gradient_analysis import (
    ExperimentResults, LayerGradientStats, ActivationStats, LayerSignalStats,
)


# Default color for highlighting zeros
ZERO_COLOR = "red"
DEFAULT_COLOR = "steelblue"

# Distinct markers for multi-line comparison plots
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def _thin_xticks(ax, labels: List[str], max_ticks: int = 10) -> None:
    """Show at most max_ticks evenly-spaced x-axis labels.

    Keeps the first and last label visible, and distributes the rest
    evenly. When there are <= max_ticks labels, all are shown.
    """
    n = len(labels)
    if n <= max_ticks:
        return  # nothing to thin

    step = max(1, (n - 1) // (max_ticks - 1))
    visible = set(range(0, n, step))
    visible.add(n - 1)  # always show last

    ax.set_xticks([labels[i] for i in sorted(visible)])
    ax.set_xticklabels([labels[i] for i in sorted(visible)])


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

    Creates three subplots:
    1. Zero gradient counts per layer
    2. Zero gradient proportions per layer
    3. Entirely zero rows per layer (dead neurons)

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
    bars0 = axes[0].bar(layers, zero_counts, color=DEFAULT_COLOR)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Zero Gradient Count")
    axes[0].set_title("Zero Gradient Entries per Layer")
    axes[0].tick_params(axis="x", rotation=45)

    # Zero proportions
    bars1 = axes[1].bar(layers, zero_props, color=DEFAULT_COLOR)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Zero Gradient Proportion per Layer")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0, 1)

    # Add text annotation if all values are zero
    if all(p == 0 for p in zero_props):
        axes[1].text(0.5, 0.5, "All values = 0%", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=12, color="gray",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Zero row proportions
    bars2 = axes[2].bar(layers, zero_row_props, color=ZERO_COLOR, alpha=0.7)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Proportion")
    axes[2].set_title("Entirely Zero Rows per Layer")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylim(0, 1)

    # Add text annotation if all values are zero
    if all(p == 0 for p in zero_row_props):
        axes[2].text(0.5, 0.5, "All values = 0%\n(no dead neurons)", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=12, color="gray",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Thin x-axis labels for deep networks
    for ax_item in axes:
        _thin_xticks(ax_item, layers)

    plt.tight_layout()
    return fig


def plot_row_norm_per_layer(
    results: ExperimentResults,
    figsize: Tuple[float, float] = (10, 5),
    show_std: bool = True,
    exclude_output_layer: bool = False,
    use_log_scale: bool = False,
) -> plt.Figure:
    """Plot mean row-norm of gradients per layer.

    Args:
        results: ExperimentResults from a gradient analysis experiment.
        figsize: Figure size.
        show_std: Whether to show standard deviation as error bars.
        exclude_output_layer: If True, exclude the output layer from the plot.
        use_log_scale: If True, use logarithmic y-axis scale.

    Returns:
        Matplotlib figure.
    """
    grad_stats = results.grad_stats

    layers = list(grad_stats.keys())
    stats_list = list(grad_stats.values())

    # Exclude output layer if requested
    if exclude_output_layer and len(layers) > 1:
        layers = layers[:-1]
        stats_list = stats_list[:-1]

    mean_norms = [stats.mean_row_norm for stats in stats_list]

    fig, ax = plt.subplots(figsize=figsize)

    if show_std:
        std_norms = [np.std(stats.row_norms) for stats in stats_list]
        ax.errorbar(layers, mean_norms, yerr=std_norms, fmt="o-", capsize=5, color=DEFAULT_COLOR)
    else:
        ax.plot(layers, mean_norms, "o-", color=DEFAULT_COLOR)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Row L2 Norm")

    title_suffix = " (hidden layers only)" if exclude_output_layer else ""
    ax.set_title(f"Average Gradient Row Norm per Layer{title_suffix}")

    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Thin x-axis labels for deep networks
    _thin_xticks(ax, layers)

    if use_log_scale:
        ax.set_yscale("log")

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

    # Add text annotation if all values are zero
    if all(p == 0 for p in inactive_props):
        axes[1].text(0.5, 0.5, "All values = 0%\n(no dead neurons)", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=12, color="gray",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return fig


def compare_initializations_plot(
    results_dict: Dict[str, ExperimentResults],
    metric: str = "zero_proportion",
    figsize: Tuple[float, float] = (12, 5),
    exclude_output_layer: bool = True,
    use_log_scale: bool = False,
) -> plt.Figure:
    """Compare gradient statistics across different initializations.

    Args:
        results_dict: Dictionary mapping init name to ExperimentResults.
        metric: Metric to compare ("zero_proportion", "mean_row_norm", "zero_row_proportion").
        figsize: Figure size.
        exclude_output_layer: If True, exclude the output layer from plots (default True).
                             The output layer often has much larger gradients, dominating the y-axis.
        use_log_scale: If True, use logarithmic y-axis scale.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_layers = None  # track for x-axis thinning

    for idx, (init_name, results) in enumerate(results_dict.items()):
        layers = list(results.grad_stats.keys())
        stats_list = list(results.grad_stats.values())

        # Exclude output layer if requested (last layer typically has much larger gradients)
        if exclude_output_layer and len(layers) > 1:
            layers = layers[:-1]
            stats_list = stats_list[:-1]

        if all_layers is None:
            all_layers = layers

        if metric == "zero_proportion":
            values = [s.zero_proportion for s in stats_list]
            ylabel = "Zero Gradient Entry Proportion"
        elif metric == "mean_row_norm":
            values = [s.mean_row_norm for s in stats_list]
            ylabel = "Mean Row Norm"
        elif metric == "zero_row_proportion":
            values = [s.zero_row_proportion for s in stats_list]
            ylabel = "Zero Row Proportion"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        marker = _MARKERS[idx % len(_MARKERS)]
        ax.plot(layers, values, marker=marker, linestyle="-", markersize=4,
                label=init_name, linewidth=1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)

    title_suffix = " (hidden layers only)" if exclude_output_layer else ""
    ax.set_title(f"Comparison of {metric} across Initializations{title_suffix}")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Thin x-axis labels for deep networks
    if all_layers is not None:
        _thin_xticks(ax, all_layers)

    if use_log_scale:
        ax.set_yscale("log")

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


# =============================================================================
# Signal statistics & gain decomposition plots
# =============================================================================


def plot_gain_decomposition(
    result: ExperimentResults,
    exclude_last_hidden: bool = True,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Plot forward and backward gain per layer side by side.

    The forward gain measures rms(a^l) / rms(a^{l-1}) — how much the
    per-neuron signal level changes through each layer.

    The backward gain measures rms(δ^l) / rms(δ^{l+1}) — how much the
    error signal changes when propagating backward.

    For stable training, both should be close to 1.0.

    Args:
        result: ExperimentResults with signal_stats populated.
        exclude_last_hidden: If True, exclude the last hidden layer
            (its backward gain involves the output dimension change).
        figsize: Figure size.
    """
    stats = result.signal_stats
    if not stats:
        raise ValueError("No signal_stats in results. Run experiment with updated code.")

    if exclude_last_hidden and len(stats) > 1:
        stats = stats[:-1]

    layers = [f"L{s.layer_idx}" for s in stats]
    fwd_gains = [s.forward_gain for s in stats]
    bwd_gains = [s.backward_gain for s in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Forward gain
    ax1.plot(layers, fwd_gains, "o-", color="tab:blue", markersize=4, linewidth=1.5)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="gain = 1.0")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Forward Gain")
    ax1.set_title("Forward Gain: rms(a^l) / rms(a^{l-1})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)
    _thin_xticks(ax1, layers)

    # Backward gain
    ax2.plot(layers, bwd_gains, "s-", color="tab:red", markersize=4, linewidth=1.5)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="gain = 1.0")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Backward Gain")
    ax2.set_title("Backward Gain: rms(δ^l) / rms(δ^{l+1})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)
    _thin_xticks(ax2, layers)

    plt.tight_layout()
    return fig


def plot_signal_norms(
    result: ExperimentResults,
    exclude_last_hidden: bool = True,
    use_log_scale: bool = True,
    figsize: Tuple[float, float] = (16, 4),
) -> plt.Figure:
    """Plot activation RMS, error signal RMS, and gradient row norms per layer.

    This three-panel decomposition shows:
    - Left: rms(a^l) — how the forward signal evolves with depth
    - Center: rms(δ^l) — how the backward error signal evolves
    - Right: mean ||∂C/∂W^l|| row norm — the product (what we optimize)

    Since ∂C/∂W^l = δ^l · (a^{l-1})^T, the gradient norm is proportional
    to ||δ^l|| · ||a^{l-1}||. These panels show which component dominates.

    Args:
        result: ExperimentResults with signal_stats populated.
        exclude_last_hidden: Exclude last hidden layer from signal plots.
        use_log_scale: Use log y-axis.
        figsize: Figure size.
    """
    stats = result.signal_stats
    if not stats:
        raise ValueError("No signal_stats in results.")

    if exclude_last_hidden and len(stats) > 1:
        stats_trimmed = stats[:-1]
    else:
        stats_trimmed = stats

    layers_sig = [f"L{s.layer_idx}" for s in stats_trimmed]
    act_rms = [s.activation_rms for s in stats_trimmed]
    err_rms = [s.error_signal_rms for s in stats_trimmed]

    # Gradient row norms (from grad_stats, excluding output layer)
    grad_stats = result.grad_stats
    grad_layers = list(grad_stats.keys())
    if exclude_last_hidden and len(grad_layers) > 1:
        grad_layers = grad_layers[:-1]
    grad_norms = [grad_stats[k].mean_row_norm for k in grad_layers]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Activation RMS
    ax1.plot(layers_sig, act_rms, "o-", color="tab:blue", markersize=4)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("rms(a^l)")
    ax1.set_title("Activation RMS (forward signal)")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)
    _thin_xticks(ax1, layers_sig)

    # Error signal RMS
    ax2.plot(layers_sig, err_rms, "s-", color="tab:red", markersize=4)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("rms(δ^l)")
    ax2.set_title("Error Signal RMS (backward signal)")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)
    _thin_xticks(ax2, layers_sig)

    # Gradient row norms
    ax3.plot(grad_layers, grad_norms, "^-", color="tab:green", markersize=4)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Mean Row Norm")
    ax3.set_title("Gradient Row Norms (∝ ||δ^l|| · ||a^{l-1}||)")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)
    _thin_xticks(ax3, grad_layers)

    if use_log_scale:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax3.set_yscale("log")

    plt.tight_layout()
    return fig


def compare_gains_plot(
    results_dict: Dict[str, ExperimentResults],
    gain_type: str = "forward",
    exclude_last_hidden: bool = True,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Compare forward or backward gains across initializations.

    Args:
        results_dict: Dict mapping init name to ExperimentResults.
        gain_type: "forward" or "backward".
        exclude_last_hidden: Exclude last hidden layer.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    all_layers = None

    for idx, (name, result) in enumerate(results_dict.items()):
        stats = result.signal_stats
        if not stats:
            continue
        if exclude_last_hidden and len(stats) > 1:
            stats = stats[:-1]

        layers = [f"L{s.layer_idx}" for s in stats]
        if all_layers is None:
            all_layers = layers

        if gain_type == "forward":
            values = [s.forward_gain for s in stats]
            ylabel = "Forward Gain: rms(a^l) / rms(a^{l-1})"
        elif gain_type == "backward":
            values = [s.backward_gain for s in stats]
            ylabel = "Backward Gain: rms(δ^l) / rms(δ^{l+1})"
        else:
            raise ValueError(f"Unknown gain_type: {gain_type}")

        marker = _MARKERS[idx % len(_MARKERS)]
        ax.plot(layers, values, marker=marker, linestyle="-", markersize=4,
                label=name, linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{gain_type.capitalize()} Gain Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    if all_layers is not None:
        _thin_xticks(ax, all_layers)

    plt.tight_layout()
    return fig


def plot_relu_survival(
    results_dict: Dict[str, ExperimentResults],
    exclude_last_hidden: bool = True,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot ReLU survival rate per layer for multiple initializations.

    The survival rate is the fraction of pre-activation entries z^l > 0.
    For standard He initialization, this should be ~50%.

    Args:
        results_dict: Dict mapping init name to ExperimentResults.
        exclude_last_hidden: Exclude last hidden layer.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    all_layers = None

    for idx, (name, result) in enumerate(results_dict.items()):
        stats = result.signal_stats
        if not stats:
            continue
        if exclude_last_hidden and len(stats) > 1:
            stats = stats[:-1]

        layers = [f"L{s.layer_idx}" for s in stats]
        if all_layers is None:
            all_layers = layers
        survival = [s.relu_survival_rate for s in stats]

        marker = _MARKERS[idx % len(_MARKERS)]
        ax.plot(layers, survival, marker=marker, linestyle="-", markersize=4,
                label=name, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="50%")
    ax.set_xlabel("Layer")
    ax.set_ylabel("ReLU Survival Rate")
    ax.set_title("Fraction of z^l > 0 per Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    if all_layers is not None:
        _thin_xticks(ax, all_layers)

    plt.tight_layout()
    return fig


def plot_centering_ratio(
    results_dict: Dict[str, ExperimentResults],
    exclude_last_hidden: bool = True,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot centering ratio Var(a)/E[a²] per layer for each initializer.

    This ratio measures the forward gain reduction caused by row-centering.
    When weights have zero-sum rows, the effective input to each neuron is
    (a_k - mean(a)), so the forward gain is proportional to Var(a)/E[a²]
    instead of E[a²]/E[a²] = 1.

    For post-ReLU activations (half-Gaussian): ratio ≈ (π-1)/π ≈ 0.68.
    For symmetric distributions (zero mean): ratio ≈ 1.0.

    The centering ratio is a property of the ACTIVATIONS, not the weights.
    It is the same for all initializers at the same layer. What differs
    between initializers is whether this ratio affects the forward gain
    (it does for row-centered weights, but not for standard He).

    Args:
        results_dict: Dict mapping init name to ExperimentResults.
        exclude_last_hidden: Exclude last hidden layer.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    all_layers = None

    for idx, (name, result) in enumerate(results_dict.items()):
        stats = result.signal_stats
        if not stats:
            continue
        if exclude_last_hidden and len(stats) > 1:
            stats = stats[:-1]

        layers = [f"L{s.layer_idx}" for s in stats]
        if all_layers is None:
            all_layers = layers
        ratios = [s.centering_ratio for s in stats]

        marker = _MARKERS[idx % len(_MARKERS)]
        ax.plot(layers, ratios, marker=marker, linestyle="-", markersize=4,
                label=name, linewidth=1.5)

    # Theoretical value for half-Gaussian
    import math
    theoretical = (math.pi - 1) / math.pi
    ax.axhline(y=theoretical, color="gray", linestyle="--", alpha=0.7,
               label=f"(π-1)/π ≈ {theoretical:.3f}")
    ax.axhline(y=1.0, color="black", linestyle=":", alpha=0.5, label="1.0 (no reduction)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Var(a) / E[a²]")
    ax.set_title("Centering Ratio per Layer (forward gain reduction factor)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=45)
    if all_layers is not None:
        _thin_xticks(ax, all_layers)

    plt.tight_layout()
    return fig
