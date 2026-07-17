#!/usr/bin/env python3
"""100L row_centered_he, train only 3 layers -- last3 (fc99,fc100,head) vs
first3 (fc1,fc2,fc3) -- everything else frozen. Where can trainable signal
enter a deep row-centered network?

Motivation: campaign 09 (rcfwd) showed row-centered representation content
dies by layer ~25 (probe chain) while the backward gain is ~unit scale
(g_bwd~1.0, uncorrected here -- this is plain row_centered_he, no GradRescale).
Forward gain g_fwd~0.826 per layer means the head sits on a signal shrunk by
~0.826^97 ~ 1e-8. This campaign surgically tests the two entry points:
does gradient reach a trainable head sitting on dead/vanishing content
(last3), or trainable early layers whose output must still survive 97
frozen scrambling layers forward (first3)?

Freezing = requires_grad=False on the frozen Linear layers' parameters
(ClassifierConfig.trainable_layers, see src/rp_study/config.py). Backprop
still runs through frozen layers unchanged -- only their own parameter
gradients are suppressed. Verified locally (CPU, 2 epochs) before this
runner was ever synced to the cluster.

Fixed knobs (minimal, so freezing-location is the only variable):
  * init = row_centered_he (plain: He then row-center, no variance re-adjustment)
  * NoBN, width 500, depth 100
  * optimizer = SGD, momentum 0, weight_decay 0, scheduler none (fixed LR)
  * lr = 1e-2, bs = 256, seed = 42
  * NO gradient clipping (assert-enforced)

2 conditions x 2 datasets x {smoke 20 ep, audit 200 ep} = 8 jobs.
Output: reports/results/rcfrozen_<condition>_<mode>_<dataset>_100L.json
"""

import argparse
import json
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

INIT_STRATEGY = "row_centered_he"
WIDTH = 500
DEPTH = 100
LEARNING_RATE = 1e-2

TRAINABLE_LAYERS = {
    "first3": ["fc1", "fc2", "fc3"],
    "last3": ["fc99", "fc100", "head"],
}
DATASETS = {
    "fmnist": "fashion_mnist",
    "cifar10": "cifar10",
}
CONDITION_KEYS = list(TRAINABLE_LAYERS.keys())
DATASET_KEYS = list(DATASETS.keys())
EXPERIMENT_LABELS = [
    f"rcfrozen_{cond}_{mode}_{ds}_100L"
    for cond in CONDITION_KEYS
    for mode in ("smoke", "audit")
    for ds in DATASET_KEYS
]


def _parse_label(label: str) -> Tuple[str, str, str]:
    # rcfrozen_<condition>_<mode>_<dataset>_100L
    parts = label.split("_")
    assert parts[0] == "rcfrozen" and parts[-1] == "100L", label
    condition, mode, dataset = parts[1], parts[2], parts[3]
    return condition, mode, dataset


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float) -> Tuple[ClassifierConfig, TrainingConfig]:
    condition, mode, dataset_key = _parse_label(label)
    dataset = DATASETS[dataset_key]
    trainable_layers = TRAINABLE_LAYERS[condition]

    cc = ClassifierConfig(
        architecture="fc", depth=DEPTH, init_strategy=INIT_STRATEGY,
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        trainable_layers=trainable_layers,
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
    condition, mode, dataset_key = _parse_label(label)
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
    print(f"# {label}  (mode={mode}, condition={condition})")
    print(f"# init={INIT_STRATEGY} (plain row-centered He, no variance re-adjustment)")
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
