# Experiment Notebooks

Analysis and reporting notebooks. Numbered in the order they entered the research (see `docs/RESEARCH_LOG.md` for the narrative). Imports come from `src/rp_study` — no local re-implementations. Configuration parameters live at the top of each notebook.

## Index

| # | Notebook | Purpose | Reads |
|---|---|---|---|
| 01 | `01_shape_experiments.ipynb` | Multi-layer RP + ReLU on 2D geometric shapes; PCA vs RP | — (synthetic) |
| 02 | `02_mnist_projections.ipynb` | RP geometry on MNIST/Fashion-MNIST; initializer comparison via `multi_layer_rp_with_init()`; Johnson-Lindenstrauss | datasets |
| 03 | `03_gradient_analysis.ipynb` | Gradient flow across init strategies; dead-neuron / zero-gradient stats; variance sweeps | datasets |
| 04 | `04_kernel_analysis.ipynb` | The arc-cosine kernel K(α): theory, angle preservation, multi-layer composition | — (theory) |
| 05 | `05_initializer_dashboard.ipynb` | **One-stop-shop**: geometry + gradient + summary stats for every registry initializer. Add new initializers here first. | datasets |
| 06 | `06_gradient_diagnostics.ipynb` | Forward–backward gain asymmetry of row centering — the root cause of the gradient trap | datasets |
| 07 | `07_kernel_geometry_analysis.ipynb` | Kernel + geometry: deriving geometry-preserving initialization from theory | `geometry_*.json` |
| 08 | `08_results_dashboard.ipynb` | Reporting only — aggregates audit JSONs into scoreboards | `reports/results/*.json` |
| 09 | `09_depth_geometry_comparison.ipynb` | He vs row-centered variants vs V2 at multiple depths, incl. the k-NN geometry revision ("spread ≠ structure") | `geometry_product_balanced.json` |
| 10 | `10_fnn_training_curves.ipynb` | Epoch-by-epoch He training trajectories from the tuning sweep | `fnn_he_bn_training.json` |
| 11 | `11_eta_sweep_analysis.ipynb` | V2 layer-balanced η sweep; per-architecture η\* selection | `eta_sweep_research.json`, `eta_star_recommended.json` |
| 13 | `13_final_results.ipynb` | **The latest audit narrative** (May 30): all 12 architectures, He + recovery + V2 rounds, 17 sections | `final_audit_merged.json` + recovery/V2 JSONs |

### archive/

Superseded versions kept for the record:

- `Random_Projections.ipynb` — the original Jan exploration that seeded the project (pre-`rp_study` package).
- `09_depth_geometry_comparison_original.ipynb` — un-executed original; `09_depth_geometry_comparison.ipynb` (May 22) is canonical.
- `13_final_results_executed.ipynb` — May 22 snapshot; `13_final_results.ipynb` (May 30) is canonical.

## Running

```bash
pip install -r ../requirements.txt
jupyter lab    # from this directory
```

Notebooks auto-detect GPU where relevant (02, 03). All set seeds for reproducibility; seeds are reset between strategies when comparing initializers. For deep networks (100+ layers) reduce `NUM_SAMPLES` if memory-bound.

## Adding a new initializer

1. Register it in `src/rp_study/models/initializers.py` (`@register_initializer("name")`).
2. Document it in `INITIALIZERS.md`.
3. Add to `INIT_STRATEGIES` in `05_initializer_dashboard.ipynb` and run all cells.
