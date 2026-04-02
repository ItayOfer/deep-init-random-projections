"""Analytical functions for theoretical analysis."""

from .kernel import k_alpha, compute_output_angle, compute_kernel_values
from .geometry_metrics import (
    knn_accuracy,
    pairwise_distance_correlation,
    effective_dimensionality,
    evaluate_geometry,
)
"""Analysis helpers for geometry, kernels, and angle maps."""

from .geometry_metrics import (
    knn_accuracy,
    pairwise_distance_correlation,
    effective_dimensionality,
    evaluate_geometry,
)
from .kernel import k_alpha
from .angle_map import AngleMapResult, empirical_angle_map, run_angle_map_comparison
