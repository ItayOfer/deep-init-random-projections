"""
Arc-cosine kernel and analytical functions.

Functions related to the K(alpha) kernel from the thesis proposal,
which describes the behavior of inner products under ReLU random projections.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..visualization.plots import add_pi_ticks


def k_alpha(alpha: np.ndarray) -> np.ndarray:
    """Compute the arc-cosine kernel K(alpha).

    K(alpha) = (sin(alpha) + (pi - alpha) * cos(alpha)) / (2 * pi)

    This kernel describes the expected inner product between two vectors
    after applying a random projection followed by ReLU activation.

    Args:
        alpha: Input angles in radians (can be array).

    Returns:
        K(alpha) values.

    Reference:
        Section 1.6 of the thesis proposal document.
    """
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / (2 * np.pi)


def compute_output_angle(alpha: np.ndarray) -> np.ndarray:
    """Compute the output angle theta_out from input angle alpha.

    The output angle is: theta_out = arccos(2 * K(alpha))

    This represents the angle between two vectors after ReLU projection,
    given that their input angle was alpha.

    Args:
        alpha: Input angles in radians.

    Returns:
        Output angles in radians.
    """
    k_values = k_alpha(alpha)
    # Clip to [-1, 1] to avoid numerical issues with arccos
    cos_theta_out = np.clip(2 * k_values, -1, 1)
    return np.arccos(cos_theta_out)


def compute_kernel_values(
    alpha: np.ndarray,
    d: int = 100,
) -> dict:
    """Compute various kernel-related quantities.

    Args:
        alpha: Input angles in radians.
        d: Dimension of the projection (number of ReLU units).

    Returns:
        Dictionary with:
        - k_alpha: The K(alpha) kernel values
        - output_angle: The output angle theta_out
        - input_cos: cos(alpha), the input inner product (normalized)
        - raw_inner_product: d * K(alpha), unnormalized expected inner product
        - normalized_inner_product: 2 * K(alpha), normalized expected inner product
    """
    k_values = k_alpha(alpha)

    return {
        "k_alpha": k_values,
        "output_angle": compute_output_angle(alpha),
        "input_cos": np.cos(alpha),
        "raw_inner_product": d * k_values,
        "normalized_inner_product": 2 * k_values,
    }


def plot_k_alpha(
    n_points: int = 400,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot the K(alpha) kernel function.

    Args:
        n_points: Number of points to plot.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    alpha = np.linspace(0, np.pi, n_points)
    k_values = k_alpha(alpha)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(alpha, k_values,
            label=r"$K(\alpha)=\frac{1}{2\pi}(\sin\alpha+(\pi-\alpha)\cos\alpha)$")

    add_pi_ticks(ax, axis="x", num_ticks=5)

    ax.set_xlabel(r"$\alpha$ (input angle)")
    ax.set_ylabel(r"$K(\alpha)$")
    ax.set_title("Arc-Cosine Kernel Behavior with ReLU")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_angle_transformation(
    n_points: int = 400,
    d: int = 100,
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Plot the relationship between input and output angles/inner products.

    Creates two subplots:
    1. Input angle -> Output angle
    2. Input cos(alpha) -> Output K(alpha)

    Args:
        n_points: Number of points to plot.
        d: Dimension (for labeling purposes).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    alpha = np.linspace(0, np.pi, n_points)
    kernel_data = compute_kernel_values(alpha, d)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Input angle -> Output angle
    axes[0].plot(alpha, kernel_data["output_angle"], lw=2, color="blue",
                 label=r"$\theta_{\text{out}}=\arccos(2K(\alpha))$")

    add_pi_ticks(axes[0], axis="x", num_ticks=5)

    # Y-axis ticks for output angle (0 to pi/2)
    ticks_y = np.linspace(0, np.pi / 2, 5)
    labels_y = [r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$",
                r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"]
    axes[0].set_yticks(ticks_y)
    axes[0].set_yticklabels(labels_y)

    axes[0].set_xlabel(r"Input angle $\alpha$")
    axes[0].set_ylabel(r"Output angle $\theta_{\text{out}}$")
    axes[0].set_title(r"Input angle $\longrightarrow$ Output angle")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Input inner product -> Output inner product
    axes[1].plot(kernel_data["input_cos"], kernel_data["normalized_inner_product"],
                 lw=2, color="green", label=r"$K(\alpha)$")

    axes[1].set_xlabel(r"Input inner product $\cos(\alpha)$")
    axes[1].set_ylabel(r"Output inner product $K(\alpha)$")
    axes[1].set_title(r"Input $\cos(\alpha)$ $\longrightarrow$ Output similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_angle_preservation(
    alpha: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Analyze how well angles are preserved under ReLU projection.

    Args:
        alpha: Input angles to analyze.
        verbose: Whether to print analysis summary.

    Returns:
        Dictionary with analysis results.
    """
    output_angles = compute_output_angle(alpha)

    # Compute various statistics
    angle_ratio = output_angles / (alpha + 1e-10)  # Avoid division by zero
    angle_diff = output_angles - alpha

    results = {
        "input_angles": alpha,
        "output_angles": output_angles,
        "angle_ratio": angle_ratio,
        "angle_difference": angle_diff,
        "max_contraction": float(np.min(angle_ratio[alpha > 0.1])),  # Avoid near-zero
        "mean_ratio": float(np.mean(angle_ratio[(alpha > 0.1) & (alpha < 3.0)])),
    }

    if verbose:
        print("Angle Preservation Analysis")
        print("=" * 40)
        print(f"Input angle range: [{alpha.min():.3f}, {alpha.max():.3f}]")
        print(f"Output angle range: [{output_angles.min():.3f}, {output_angles.max():.3f}]")
        print(f"Mean angle ratio (theta_out/alpha): {results['mean_ratio']:.3f}")
        print(f"Max contraction: {results['max_contraction']:.3f}")
        print(f"Note: Output angles are generally contracted toward 0")

    return results
