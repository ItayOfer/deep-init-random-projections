#!/usr/bin/env python3
"""100L row-centered, train only 3 layers -- last3 (fc98,fc99,fc100) vs
first3 (fc1,fc2,fc3) -- everything else, INCLUDING THE HEAD, frozen in both
conditions. Where can trainable signal enter a deep row-centered network?

Symmetric by design: the head (the final Linear(hidden_dim, num_classes)
readout) is never part of either trainable window -- it is always the fixed
lens the result is read through, in both conditions. This makes last3 and
first3 a like-for-like comparison of "a 3-layer trainable window at one end
of a 100-layer frozen stack" vs "at the other end", with nothing else
differing. (An earlier draft of this runner trained fc99+fc100+head for
last3 -- an asymmetric definition, since first3 never trained the head --
corrected here; see cluster/10_rc_frozen_ends/README.md for the note on
what changed and why.)

Motivation: campaign 09 (rcfwd) showed row-centered representation content
dies by layer ~25 (probe chain) while the backward gain is ~unit scale
(g_bwd~1.0, uncorrected here -- this is plain row_centered_he, no GradRescale).
Forward gain g_fwd~0.826 per layer means the tail of the network sits on a
signal shrunk by ~0.826^97 ~ 1e-8. This campaign surgically tests the two
entry points: does gradient reach a trainable window sitting on dead/
vanishing content (last3), or a trainable window whose output must still
survive 97 frozen scrambling layers forward before reaching the frozen head
(first3)?

Freezing = requires_grad=False on the frozen Linear layers' parameters
(ClassifierConfig.trainable_layers, see src/rp_study/config.py). Backprop
still runs through frozen layers unchanged -- only their own parameter
gradients are suppressed. Verified locally (CPU, 2 epochs) before this
runner was ever synced to the cluster.

Two RECIPES (added after the "raw" pass to discriminate two hypotheses for
*why* rc-frozen-ends fails -- see cluster/10_rc_frozen_ends/README.md
"H1 vs H2" section):
  * "raw"   -- row_centered_he, no grad_rescale (the original pass). Backward
              gain here is ~unit scale already (g_bwd~1.0); the failure modes
              observed are forward-scale decay (last3) and content
              absorption (first3), NOT gradient magnitude/direction.
  * "rcfwd" -- row_centered_forward_balanced + grad_rescale=r=sqrt((pi-1)/pi)
              (campaign 09's exact corrected recipe: flat forward, backward
              gain compounding at 1/r~1.21/layer cancelled by the rescale).
              If H1 (the rescale itself was masking a recoverable gradient)
              were right, this corrected recipe should let last3/first3
              train where the raw recipe didn't. If H2 (representation
              content death, independent of gradient conditioning, is the
              real bottleneck -- per campaign 09's own probe evidence) is
              right, this should fail the same way, just without the
              float32-underflow signature (gradients here are well-scaled
              by construction).

Fixed knobs (minimal, so freezing-location and recipe are the only variables):
  * NoBN, width 500, depth 100
  * optimizer = SGD, momentum 0, weight_decay 0, scheduler none (fixed LR)
  * lr = 1e-2, bs = 256, seed = 42
  * NO gradient clipping (assert-enforced)

2 conditions x 2 datasets x 2 recipes x {smoke 20 ep, audit 200 ep} = 16 jobs.
Output: reports/results/rcfrozen_<condition>_<mode>_<dataset>_100L[_rcfwd].json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from rp_study.models.classifiers import build_classifier
from run_diagnostic import _result_to_payload, print_diagnostic_summary

WIDTH = 500
DEPTH = 100
LEARNING_RATE = 1e-2
R = math.sqrt((math.pi - 1.0) / math.pi)   # ~0.826 = 1/1.21, campaign-09's rescale

RECIPES = {
    "raw": {"init_strategy": "row_centered_he", "grad_rescale": None},
    "rcfwd": {"init_strategy": "row_centered_forward_balanced", "grad_rescale": R},
}

TRAINABLE_LAYERS = {
    "first3": ["fc1", "fc2", "fc3"],
    "last3": ["fc98", "fc99", "fc100"],
}
DATASETS = {
    "fmnist": "fashion_mnist",
    "cifar10": "cifar10",
}
CONDITION_KEYS = list(TRAINABLE_LAYERS.keys())
DATASET_KEYS = list(DATASETS.keys())
RECIPE_SUFFIX = {"raw": "", "rcfwd": "_rcfwd"}
EXPERIMENT_LABELS = [
    f"rcfrozen_{cond}_{mode}_{ds}_100L{RECIPE_SUFFIX[recipe]}"
    for recipe in RECIPES
    for cond in CONDITION_KEYS
    for mode in ("smoke", "audit")
    for ds in DATASET_KEYS
]


def _parse_label(label: str) -> Tuple[str, str, str, str]:
    # rcfrozen_<condition>_<mode>_<dataset>_100L[_rcfwd]
    recipe = "rcfwd" if label.endswith("_rcfwd") else "raw"
    core = label[: -len("_rcfwd")] if recipe == "rcfwd" else label
    parts = core.split("_")
    assert parts[0] == "rcfrozen" and parts[-1] == "100L", label
    condition, mode, dataset = parts[1], parts[2], parts[3]
    return condition, mode, dataset, recipe


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float) -> Tuple[ClassifierConfig, TrainingConfig]:
    condition, mode, dataset_key, recipe = _parse_label(label)
    dataset = DATASETS[dataset_key]
    trainable_layers = TRAINABLE_LAYERS[condition]
    recipe_cfg = RECIPES[recipe]

    cc = ClassifierConfig(
        architecture="fc", depth=DEPTH, init_strategy=recipe_cfg["init_strategy"],
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        trainable_layers=trainable_layers,
        grad_rescale=recipe_cfg["grad_rescale"],
    )
    tc = TrainingConfig(
        dataset=dataset, batch_size=256,
        optimizer="sgd", learning_rate=learning_rate,
        momentum=0.0, weight_decay=0.0,
        scheduler="none",
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=1,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        log_grad_per_layer=log_grad_per_layer,
        normalize_inputs=True,
        target_train_accuracy=None,
    )
    return cc, tc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    label = args.experiment
    condition, mode, dataset_key, recipe = _parse_label(label)
    epochs = args.epochs if args.epochs is not None else (20 if mode == "smoke" else 200)
    log_per_batch = (mode == "smoke")
    log_grad_per_layer = (mode == "smoke")

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(
        label, epochs=epochs,
        log_per_batch_first_epoch=log_per_batch,
        log_grad_per_layer=log_grad_per_layer,
        learning_rate=args.lr,
    )

    # Build a throwaway model up front just to assert + print the trainable-
    # tensor banner (run_supervised_experiment builds its own, identically
    # seeded, copy for the actual training run).
    banner_model = build_classifier(classifier_config)
    n_trainable = banner_model.trainable_tensor_count()
    expected_layers = len(classifier_config.trainable_layers)
    expected_tensors = 2 * expected_layers  # weight + bias per trainable Linear
    del banner_model

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode}, condition={condition}, recipe={recipe})")
    print(f"# init={classifier_config.init_strategy}  grad_rescale={classifier_config.grad_rescale}")
    print(f"# trainable_layers={classifier_config.trainable_layers}")
    print(f"# trainable tensors={n_trainable} (expect {expected_tensors} = "
          f"{expected_layers} layers x 2 [weight,bias])")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (fixed LR, no patience)")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None)")
    print(f"# epochs={epochs} diag_every=1 (per-layer grads every epoch)")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    assert n_trainable == expected_tensors, (
        f"trainable tensor count mismatch: got {n_trainable}, expected {expected_tensors} "
        f"for trainable_layers={classifier_config.trainable_layers}"
    )
    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for this experiment"
    )

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    if getattr(result, "abort_reason", None):
        print(f"\nABORT_REASON: {result.abort_reason}")

    payload = [_result_to_payload(label, result)]
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved 1 run to {output_path}")

    hist = payload[0].get("history", [])
    if hist:
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_test = hist[-1].get("test_accuracy", 0) or 0
        passes = final_acc >= 0.995 and final_loss <= 0.10
        flag = "PASS" if passes else "fail"
        print(f"\nSUMMARY {label} | {flag} | best_train={best_acc:.4f} "
              f"final_train={final_acc:.4f} final_loss={final_loss:.4f} "
              f"final_test={final_test:.4f} epochs_ran={len(hist)}")
    else:
        print(f"\nSUMMARY {label} | NO_HISTORY (aborted before first epoch completed)")


if __name__ == "__main__":
    main()
