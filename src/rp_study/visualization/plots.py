"""
General plotting utilities and style configuration.
"""

from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style(style: str = "default") -> None:
    """Configure matplotlib style for consistent plotting.

    Args:
        style: Style preset name.
    """
    if style == "paper":
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
        })
    elif style == "presentation":
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.figsize": (12, 8),
            "figure.dpi": 100,
        })
    else:
        # Default style
        plt.rcParams.update(plt.rcParamsDefault)


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with subplots.

    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        figsize: Figure size (width, height). Auto-computed if None.
        **kwargs: Additional arguments to plt.subplots.

    Returns:
        Tuple of (figure, axes array).
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def add_pi_ticks(ax: plt.Axes, axis: str = "x", num_ticks: int = 5) -> None:
    """Add pi-fraction tick labels to an axis.

    Args:
        ax: Matplotlib axes.
        axis: "x" or "y".
        num_ticks: Number of tick points.
    """
    ticks = np.linspace(0, np.pi, num_ticks)
    labels = _generate_pi_labels(num_ticks)

    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def _generate_pi_labels(num_ticks: int) -> list:
    """Generate pi-fraction labels for ticks from 0 to pi."""
    if num_ticks == 5:
        return [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
    elif num_ticks == 3:
        return [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    elif num_ticks == 9:
        return [
            r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$",
            r"$\frac{\pi}{2}$", r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"
        ]
    else:
        # Generic: just use numerical values
        return [f"{t:.2f}" for t in np.linspace(0, np.pi, num_ticks)]


def save_figure(
    fig: plt.Figure,
    filename: str,
    dpi: int = 150,
    bbox_inches: str = "tight",
    **kwargs,
) -> None:
    """Save figure to file.

    Args:
        fig: Figure to save.
        filename: Output filename.
        dpi: Resolution.
        bbox_inches: Bounding box setting.
        **kwargs: Additional arguments to savefig.
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


def format_scientific(value: float, precision: int = 2) -> str:
    """Format a number in scientific notation.

    Args:
        value: Number to format.
        precision: Number of decimal places.

    Returns:
        Formatted string.
    """
    return np.format_float_scientific(value, precision=precision)
