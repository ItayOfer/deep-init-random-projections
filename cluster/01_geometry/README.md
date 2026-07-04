# 01 — Geometry Benchmarks (Apr 2026)

**Question.** Does row-centered initialization actually preserve data geometry through depth — measured on real data with quantitative metrics (k-NN accuracy, distance correlation, effective dimension), rather than 2-D PCA pictures?

**Builds on.** The theory notebooks: the arc-cosine kernel analysis (`notebooks/04`, `07` — `run_angle_map.py` operationalizes notebook 07 §4 at scale) and the row-centering proposal from `CONTEXT.md`. Both runners use the registry initializers from `src/rp_study` — no local re-implementations.

## What ran

Forward-pass probes only — no training, no optimizer.

| Job | Probe | Setup |
|---|---|---|
| `geometry_benchmark.sub` | k-NN / distance-correlation / effective-dim after L layers of RP+ReLU | CIFAR-10, 2000 samples, depths {5, 10, 15, 20}, inits {`he`, `orthogonal_he`, `row_centered_he_var_adj`}, seed 42 |
| `angle_map.sub` | Empirical single-layer angle map α → E[α_out] | dim 784, 40 angles × 300 pairs × 5 seeds, same 3 inits |
| `geometry_product_balanced.sub` | Same benchmark + V1/forward-balanced inits, both datasets | **outputs never retrieved — see gaps** |

## Findings

**Row-centering "spreads" the data without preserving its structure.** On CIFAR-10 (chance = 0.10), from `geometry_benchmark_cifar10.json`:

| Init | k-NN @ depth 5→20 | Effective dim @ depth 5→20 |
|---|---|---|
| `he` | 0.247 → 0.221 | 50.9 → 14.4 (compresses) |
| `orthogonal_he` | 0.239 → 0.228 | 49.3 → 14.5 |
| `row_centered_he_var_adj` | 0.219 → **0.098 (chance)** | 650 → **956 (explodes)** |

Row-centered representations reach ~40× the effective dimension of He by depth 20, yet class structure is *gone* — high-dimensional noise, not preserved geometry. He compresses hard but keeps class-relevant directions. This is the first quantitative appearance of the **"spread ≠ structure"** finding (later confirmed with k-NN across datasets in `notebooks/09_meeting_comparison_executed.ipynb`).

**Single layers are not where the difference lives.** The empirical angle maps (`angle_map.json`) of all three inits are essentially identical (output means agree to ~4 decimal places, max std ≤ 0.003) — one layer of any of these behaves like the arc-cosine kernel map. The divergence above is purely an effect of *composition through depth*.

## Reproduce

```bash
bash cluster/sync_to_cluster.sh                          # local
cd ~/thesis && sbatch cluster/01_geometry/geometry_benchmark.sub   # → reports/results/geometry_benchmark_cifar10.json
cd ~/thesis && sbatch cluster/01_geometry/angle_map.sub            # → reports/results/angle_map.json
```

Expect: JSON rows per (init, depth) with `knn_accuracy`, `distance_correlation`, `effective_dim`, `overflow`; minutes-scale runtime (forward passes only). Smoke variants (`geometry_smoke.json`, `angle_smoke.json`) exist for pipeline checks.

## Evidence & gaps

- Results: `reports/results/{geometry_benchmark_cifar10, angle_map, geometry_smoke, angle_smoke}.json` (see `reports/results/INDEX.md`).
- No figures for this campaign; the geometry story is best seen in `notebooks/07` and `09`.
- **Gaps:** `geometry_product_balanced.sub` writes `geometry_product_balanced_{fmnist,cifar10}.json` but neither exists locally (job never ran or results never pulled) — the V1/forward-balanced geometry numbers are unverified. No Fashion-MNIST full benchmark exists for the same reason.
