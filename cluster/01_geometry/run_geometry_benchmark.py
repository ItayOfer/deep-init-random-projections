#!/usr/bin/env python3
"""Run the dataset geometry benchmark from the command line."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ExperimentConfig, GeometryBenchmarkConfig
from rp_study.experiments.geometry_benchmark import geometry_results_to_rows, run_geometry_benchmark


def _parse_csv_list(raw: str, cast=str):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--depths", default="5,10,15,20")
    parser.add_argument("--init-strategies", default="he,orthogonal_he,row_centered_he_var_adj")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "reports" / "results" / "geometry_benchmark.json"))
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))
    benchmark_config = GeometryBenchmarkConfig(
        dataset=args.dataset,
        num_samples=args.num_samples,
        depths=_parse_csv_list(args.depths, int),
        init_strategies=_parse_csv_list(args.init_strategies, str),
        normalize_inputs=args.normalize_inputs,
        flatten=True,
    )

    results = run_geometry_benchmark(exp_config, benchmark_config)
    rows = geometry_results_to_rows(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))

    print(f"Saved {len(rows)} geometry rows to {output_path}")
    for row in rows:
        print(
            f'{row["init_strategy"]:>30s} depth={row["depth"]:>3d} '
            f'kNN={row["knn_accuracy"]:.3f} '
            f'dist={row["distance_correlation"]:.3f} '
            f'eff_dim={row["effective_dim"]:.1f}'
        )


if __name__ == "__main__":
    main()
