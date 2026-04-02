"""Empirical angle-map utilities extracted from notebook 07."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..models.initializers import initialize_layer


@dataclass
class AngleMapResult:
    """Empirical input-angle to output-angle statistics for one initializer."""

    init_strategy: str
    alpha_grid: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray


def generate_pair_at_angle(alpha: float, dim: int, rng: np.random.RandomState):
    """Generate a random unit-vector pair with a prescribed angle."""
    v1 = rng.randn(dim)
    v1 = v1 / np.linalg.norm(v1)

    u = rng.randn(dim)
    u = u - np.dot(u, v1) * v1
    u = u / np.linalg.norm(u)

    v2 = np.cos(alpha) * v1 + np.sin(alpha) * u
    return v1, v2


def empirical_angle_map(
    init_strategy: str,
    dim: int = 784,
    n_angles: int = 30,
    n_pairs: int = 200,
    n_seeds: int = 5,
    base_seed: int = 42,
    init_kwargs: Optional[dict] = None,
) -> AngleMapResult:
    """Compute the empirical one-layer angle map alpha -> alpha'."""
    init_kwargs = init_kwargs or {}
    alpha_grid = np.linspace(0.05, np.pi - 0.05, n_angles)
    all_output_angles = np.zeros((n_seeds, n_angles, n_pairs), dtype=np.float64)

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        torch.manual_seed(seed)
        layer = nn.Linear(dim, dim, bias=False)
        initialize_layer(layer, strategy=init_strategy, **init_kwargs)
        weight = layer.weight.detach().cpu().numpy()

        rng = np.random.RandomState(seed + 1000)
        for angle_idx, alpha in enumerate(alpha_grid):
            for pair_idx in range(n_pairs):
                v1, v2 = generate_pair_at_angle(alpha, dim, rng)
                a1 = np.maximum(0, weight @ v1)
                a2 = np.maximum(0, weight @ v2)

                norm1 = np.linalg.norm(a1)
                norm2 = np.linalg.norm(a2)
                if norm1 < 1e-10 or norm2 < 1e-10:
                    all_output_angles[seed_idx, angle_idx, pair_idx] = np.pi / 2
                else:
                    cos_sim = np.clip(np.dot(a1, a2) / (norm1 * norm2), -1.0, 1.0)
                    all_output_angles[seed_idx, angle_idx, pair_idx] = np.arccos(cos_sim)

    per_seed_means = np.mean(all_output_angles, axis=2)
    output_mean = np.mean(per_seed_means, axis=0)
    output_std = np.std(per_seed_means, axis=0)
    return AngleMapResult(
        init_strategy=init_strategy,
        alpha_grid=alpha_grid,
        output_mean=output_mean,
        output_std=output_std,
    )


def run_angle_map_comparison(
    init_strategies,
    dim: int = 784,
    n_angles: int = 40,
    n_pairs: int = 300,
    n_seeds: int = 5,
    base_seed: int = 42,
    init_kwargs_by_strategy: Optional[Dict[str, dict]] = None,
) -> Dict[str, AngleMapResult]:
    """Run the empirical angle map for several initializers."""
    init_kwargs_by_strategy = init_kwargs_by_strategy or {}
    results = {}
    for init_strategy in init_strategies:
        results[init_strategy] = empirical_angle_map(
            init_strategy=init_strategy,
            dim=dim,
            n_angles=n_angles,
            n_pairs=n_pairs,
            n_seeds=n_seeds,
            base_seed=base_seed,
            init_kwargs=init_kwargs_by_strategy.get(init_strategy, {}),
        )
    return results
