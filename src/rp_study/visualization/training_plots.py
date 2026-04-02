"""Plots for supervised training comparisons."""

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..experiments.supervised_training import SupervisedTrainingResult


def _label_for_result(result: SupervisedTrainingResult) -> str:
    cfg = result.classifier_config
    bn_label = "BN" if cfg["use_batch_norm"] else "No BN"
    return f'{cfg["init_strategy"]} / {bn_label}'


def plot_training_histories(
    results: Sequence[SupervisedTrainingResult],
    metric: str = "test_accuracy",
    figsize=(10, 6),
) -> plt.Figure:
    """Plot one scalar metric over epochs for several supervised runs."""
    fig, ax = plt.subplots(figsize=figsize)
    for result in results:
        if not result.history:
            continue
        epochs = [entry.epoch for entry in result.history]
        values = [getattr(entry, metric) for entry in result.history]
        ax.plot(epochs, values, marker="o", label=_label_for_result(result))

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(metric.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_supervised_comparison_matrix(
    results: Sequence[SupervisedTrainingResult],
    dataset: str,
    architecture: str,
    metric: str = "best_test_accuracy",
    figsize=(8, 6),
) -> plt.Figure:
    """Plot the requested dataset/architecture comparison as a heatmap."""
    filtered = [
        result
        for result in results
        if result.training_config["dataset"] == dataset
        and result.classifier_config["architecture"] == architecture
    ]
    if not filtered:
        raise ValueError(f"No results found for dataset={dataset}, architecture={architecture}")

    row_labels = sorted({str(result.classifier_config["depth"]) for result in filtered}, key=int)
    col_labels = sorted({_label_for_result(result) for result in filtered})
    matrix = np.full((len(row_labels), len(col_labels)), np.nan, dtype=float)

    row_index = {label: idx for idx, label in enumerate(row_labels)}
    col_index = {label: idx for idx, label in enumerate(col_labels)}
    for result in filtered:
        r = row_index[str(result.classifier_config["depth"])]
        c = col_index[_label_for_result(result)]
        matrix[r, c] = getattr(result, metric)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"{depth}L" for depth in row_labels])
    ax.set_xlabel("Initializer / Normalization")
    ax.set_ylabel("Depth")
    ax.set_title(f"{dataset} / {architecture} / {metric}")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isfinite(matrix[r, c]):
                ax.text(c, r, f"{matrix[r, c]:.3f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
