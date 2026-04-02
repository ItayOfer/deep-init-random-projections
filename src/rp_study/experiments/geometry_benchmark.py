"""Reusable geometry benchmarking for deep random projection experiments."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..analysis.geometry_metrics import evaluate_geometry
from ..config import ExperimentConfig, GeometryBenchmarkConfig
from ..data.loaders import get_data_loader
from ..projections.random_projections import multi_layer_rp_with_init


@dataclass
class GeometryBenchmarkResult:
    """Geometry metrics for one initializer-depth pair."""

    init_strategy: str
    depth: int
    metrics: Dict[str, float]
    overflow: bool = False


def run_geometry_benchmark(
    exp_config: ExperimentConfig,
    benchmark_config: GeometryBenchmarkConfig,
    init_kwargs_by_strategy: Optional[Dict[str, dict]] = None,
) -> Dict[Tuple[str, int], GeometryBenchmarkResult]:
    """Run the notebook-07 style geometry benchmark as package code."""
    init_kwargs_by_strategy = init_kwargs_by_strategy or {}

    exp_config.setup_seeds()
    device = exp_config.get_device()
    x_all, y_all = get_data_loader(
        dataset_name=benchmark_config.dataset,
        data_dir=exp_config.data_dir,
        train=True,
        num_samples=benchmark_config.num_samples,
        flatten=benchmark_config.flatten,
        as_numpy=True,
        normalize=benchmark_config.normalize_inputs,
        seed=exp_config.seed,
    )
    x_data = x_all
    y_data = y_all

    results: Dict[Tuple[str, int], GeometryBenchmarkResult] = {}
    for init_strategy in benchmark_config.init_strategies:
        strategy_kwargs = init_kwargs_by_strategy.get(init_strategy, {})
        for depth in benchmark_config.depths:
            np.random.seed(exp_config.seed)
            torch.manual_seed(exp_config.seed)

            x_projected = multi_layer_rp_with_init(
                x_data,
                depth,
                init_strategy=init_strategy,
                seed=exp_config.seed,
                device=device,
                **strategy_kwargs,
            )

            if not np.all(np.isfinite(x_projected)):
                metrics = {
                    "knn_accuracy": float("nan"),
                    "distance_correlation": float("nan"),
                    "effective_dim": float("nan"),
                    "effective_dim_original": float("nan"),
                }
                results[(init_strategy, depth)] = GeometryBenchmarkResult(
                    init_strategy=init_strategy,
                    depth=depth,
                    metrics=metrics,
                    overflow=True,
                )
                continue

            metrics = evaluate_geometry(
                x_data,
                x_projected,
                y_data,
                k=benchmark_config.knn_k,
                n_pairs=benchmark_config.n_pairs,
                seed=exp_config.seed,
            )
            results[(init_strategy, depth)] = GeometryBenchmarkResult(
                init_strategy=init_strategy,
                depth=depth,
                metrics=metrics,
                overflow=False,
            )

    return results


def geometry_results_to_rows(
    results: Dict[Tuple[str, int], GeometryBenchmarkResult],
) -> List[dict]:
    """Flatten geometry results into JSON-serializable rows."""
    rows = []
    for (_, _), result in sorted(results.items(), key=lambda item: (item[0][0], item[0][1])):
        row = {
            "init_strategy": result.init_strategy,
            "depth": result.depth,
            "overflow": result.overflow,
        }
        row.update(result.metrics)
        rows.append(row)
    return rows


def geometry_results_by_metric(
    results: Dict[Tuple[str, int], GeometryBenchmarkResult],
) -> Dict[str, Dict[str, List[float]]]:
    """Group results by metric for plotting convenience."""
    metric_names = [
        "knn_accuracy",
        "distance_correlation",
        "effective_dim",
        "effective_dim_original",
    ]
    init_strategies = sorted({result.init_strategy for result in results.values()})
    grouped: Dict[str, Dict[str, List[float]]] = {metric: {} for metric in metric_names}
    for metric_name in metric_names:
        for init_strategy in init_strategies:
            rows = [
                result
                for result in results.values()
                if result.init_strategy == init_strategy
            ]
            rows.sort(key=lambda item: item.depth)
            grouped[metric_name][init_strategy] = [row.metrics[metric_name] for row in rows]
    return grouped


def geometry_result_to_dict(result: GeometryBenchmarkResult) -> dict:
    """Serialize one result dataclass."""
    payload = asdict(result)
    payload["metrics"] = dict(result.metrics)
    return payload
