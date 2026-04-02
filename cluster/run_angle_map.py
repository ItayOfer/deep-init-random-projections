#!/usr/bin/env python3
"""Run the empirical angle-map comparison from notebook 07 section 4."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.analysis.angle_map import run_angle_map_comparison


def _parse_csv_list(raw: str, cast=str):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-strategies", default="he,orthogonal_he,row_centered_he_var_adj")
    parser.add_argument("--dim", type=int, default=784)
    parser.add_argument("--n-angles", type=int, default=40)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "reports" / "results" / "angle_map.json"))
    args = parser.parse_args()

    results = run_angle_map_comparison(
        init_strategies=_parse_csv_list(args.init_strategies, str),
        dim=args.dim,
        n_angles=args.n_angles,
        n_pairs=args.n_pairs,
        n_seeds=args.n_seeds,
        base_seed=args.seed,
    )

    payload = {}
    for init_strategy, result in results.items():
        payload[init_strategy] = {
            "alpha_grid": result.alpha_grid.tolist(),
            "output_mean": result.output_mean.tolist(),
            "output_std": result.output_std.tolist(),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved angle-map comparison to {output_path}")


if __name__ == "__main__":
    main()
