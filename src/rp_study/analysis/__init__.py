"""Analytical functions for theoretical analysis."""

from .kernel import k_alpha, compute_output_angle, compute_kernel_values
from .geometry_metrics import (
    knn_accuracy,
    pairwise_distance_correlation,
    effective_dimensionality,
    evaluate_geometry,
)
