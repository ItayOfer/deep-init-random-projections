# Simulations / Runs Summary

This summary captures the current worktree differences against the repository, excluding changes under `reports/` and `cluster/` as requested.

## Scope

- Included:
  - `src/`
  - `notebooks/`
  - root-level docs / metadata files
- Excluded:
  - `reports/`
  - `cluster/`

## Core Source Changes

### [src/rp_study/config.py](/Users/itayofer/Thesis/Thesis/src/rp_study/config.py)

- Extended classifier config with BatchNorm evaluation-related knobs:
  - `bn_momentum`
  - `bn_eps`
- Extended training config with:
  - early-stop / train-fit targeting fields
  - `target_metric`
  - `log_every_epoch`
  - `onecycle` scheduler support and its schedule parameters

### [src/rp_study/experiments/supervised_training.py](/Users/itayofer/Thesis/Thesis/src/rp_study/experiments/supervised_training.py)

- Added clean eval-mode train-set metrics:
  - `eval_train_accuracy`
  - `eval_train_loss`
- Added:
  - `stop_reason`
  - `epochs_ran`
  - `final_eval_train_accuracy`
  - `final_eval_train_loss`
- Added deterministic eval loader for the train set.
- Added live run / epoch logging.
- Added target-based stopping using either train-mode or eval-train metrics.
- Added `onecycle` scheduler stepping per batch.
- Enriched exported summary rows with training hyperparameters and BN parameters.

### [src/rp_study/models/classifiers.py](/Users/itayofer/Thesis/Thesis/src/rp_study/models/classifiers.py)

- Updated both FC and CNN classifiers so BatchNorm layers can be configured with:
  - `bn_momentum`
  - `bn_eps`
- Passed those knobs through `build_classifier(...)`.

### [src/rp_study/models/initializers.py](/Users/itayofer/Thesis/Thesis/src/rp_study/models/initializers.py)

- Added two product-balanced row-centered initializers:
  - `row_centered_product_balanced`
  - `row_centered_layer_balanced_product_base`
- These encode the product-balanced forward/backward-gain idea discussed in the meeting-prep analysis.

### [INITIALIZERS.md](/Users/itayofer/Thesis/Thesis/INITIALIZERS.md)

- Documented the two new product-balanced initializers.
- Added motivation, formulas, gain properties, and recommended-use notes.

## Notebook Changes

### Modified notebooks

#### [notebooks/05_initializer_dashboard.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/05_initializer_dashboard.ipynb)

- Updated initializer dashboard content.
- Reflects newer initializer variants and analysis workflow.

#### [notebooks/06_gradient_diagnostics.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/06_gradient_diagnostics.ipynb)

- Substantial edits around gradient diagnostics.
- Now aligned with forward/backward-gain analysis and newer metric logging.

#### [notebooks/07_kernel_geometry_analysis.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/07_kernel_geometry_analysis.ipynb)

- Substantial edits around geometry-preservation analysis.
- Expanded or revised kernel/geometry comparison workflow.

#### [notebooks/08_results_dashboard.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/08_results_dashboard.ipynb)

- Results dashboard updates for parsing experiment JSONs and viewing comparison outputs.

### New notebooks

#### [notebooks/09_depth_geometry_comparison_original.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/09_depth_geometry_comparison_original.ipynb)

- Meeting-prep comparison notebook focused on product-balanced initializers.

#### [notebooks/09_depth_geometry_comparison.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/09_depth_geometry_comparison.ipynb)

- Executed/output-preserved version of the meeting comparison notebook.

#### [notebooks/10_fnn_training_curves.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/10_fnn_training_curves.ipynb)

- Dedicated notebook for FC training curves.
- Updated to:
  - prefer the eval-train rerun JSONs
  - show eval-train metrics alongside train-mode/test metrics
  - generate a compact advisor-facing criterion table
  - use final-test values in the final compact summary table

#### [notebooks/11_eta_sweep_analysis.ipynb](/Users/itayofer/Thesis/Thesis/notebooks/11_eta_sweep_analysis.ipynb)

- New notebook focused on `η`-sweep analysis for the layer-balanced initialization family.

## Repository / Metadata Additions

### [AGENTS.md](/Users/itayofer/Thesis/Thesis/AGENTS.md)

- Added project-local agent instructions/context file.

### [src/rp_study.egg-info/PKG-INFO](/Users/itayofer/Thesis/Thesis/src/rp_study.egg-info/PKG-INFO)
### [src/rp_study.egg-info/SOURCES.txt](/Users/itayofer/Thesis/Thesis/src/rp_study.egg-info/SOURCES.txt)
### [src/rp_study.egg-info/dependency_links.txt](/Users/itayofer/Thesis/Thesis/src/rp_study.egg-info/dependency_links.txt)
### [src/rp_study.egg-info/requires.txt](/Users/itayofer/Thesis/Thesis/src/rp_study.egg-info/requires.txt)
### [src/rp_study.egg-info/top_level.txt](/Users/itayofer/Thesis/Thesis/src/rp_study.egg-info/top_level.txt)

- Generated packaging metadata currently present in the worktree.

### [.DS_Store](/Users/itayofer/Thesis/Thesis/.DS_Store)
### [src/.DS_Store](/Users/itayofer/Thesis/Thesis/src/.DS_Store)

- macOS Finder metadata files currently present in the worktree.

## Practical Meaning of the Non-Cluster / Non-Reports Diff

- The repo now contains:
  - richer supervised-training instrumentation
  - configurable BN evaluation behavior
  - product-balanced initializer support
  - expanded notebook analysis flow for initializer, geometry, and training studies
- The notebook side has grown into a fuller experiment-analysis surface:
  - initializer dashboards
  - geometry diagnostics
  - meeting comparison notebooks
  - FC training-curve and criterion reporting
  - eta-sweep analysis
