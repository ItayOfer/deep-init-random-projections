#!/usr/bin/env python3
"""V2 NoBN + plain SGD, per-architecture gradient-ratio-minimising eta.

This is the controlled NoBN experiment: take the V2 initializer
(row_centered_layer_balanced_product_base) and, for each architecture,
use the eta that MINIMISES the empirical per-layer gradient row-norm ratio
(found by scripts/eta_sweep_research.py + eta_sweep_pick.py, stored in
reports/results/eta_star_recommended.json).

NO safety filtering on eta -- the minima are used as-is. For L=50/100 the
ratio-minimising eta sits in the regime where the forward activation bulge
overflows at init (proxy measurement). Those runs are EXPECTED to NaN early;
that overflow is a real result, not a config to be avoided.

Fixed experimental knobs (deliberately minimal so eta is the only variable):
  * NoBN
  * optimizer = SGD, momentum = 0.0, weight_decay = 0.0
  * scheduler = none (fixed LR, no patience)
  * NO gradient clipping (assert-enforced; see feedback-no-grad-clipping)
  * width 500, normalize_inputs=True

eta* per architecture (the gradient-ratio MINIMA, not safe values):
  fmnist_30L  0.60 | fmnist_50L  0.80 | fmnist_100L 0.36
  cifar10_30L 0.85 | cifar10_50L 0.90 | cifar10_100L 0.36

Modes:
  smoke (default): 20 epochs, log_per_batch_first_epoch=True, diag every 2.
  audit:           200 epochs, log_per_batch_first_epoch=False, diag every 10.

Output: reports/results/v2_nobn_sgd_<mode>_<arch>.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary

INIT_STRATEGY = "row_centered_layer_balanced_product_base"
WIDTH = 500
LEARNING_RATE = 1e-2

# Gradient-ratio MINIMA from the eta sweep (no safety filter).
ETA_STAR = {
    "fashion_mnist_30L": 0.60,
    "fashion_mnist_50L": 0.80,
    "fashion_mnist_100L": 0.36,
    "cifar10_30L": 0.85,
    "cifar10_50L": 0.90,
    "cifar10_100L": 0.36,
}

# arch_key -> (dataset, depth)
ARCH = {
    "fmnist_30L": ("fashion_mnist", 30),
    "fmnist_50L": ("fashion_mnist", 50),
    "fmnist_100L": ("fashion_mnist", 100),
    "cifar10_30L": ("cifar10", 30),
    "cifar10_50L": ("cifar10", 50),
    "cifar10_100L": ("cifar10", 100),
}

ARCHITECTURE_KEYS = list(ARCH.keys())
EXPERIMENT_LABELS = (
    [f"v2_nobn_sgd_smoke_{k}" for k in ARCHITECTURE_KEYS]
    + [f"v2_nobn_sgd_audit_{k}" for k in ARCHITECTURE_KEYS]
)


def _eta_for(dataset: str, depth: int) -> float:
    return ETA_STAR[f"{dataset}_{depth}L"]


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float) -> Tuple[ClassifierConfig, TrainingConfig]:
    arch_key = label.split("_", 4)[-1]   # e.g. "fmnist_100L"
    dataset, depth = ARCH[arch_key]
    eta = _eta_for(dataset, depth)

    cc = ClassifierConfig(
        architecture="fc", depth=depth, init_strategy=INIT_STRATEGY,
        init_kwargs={"eta": eta},
        use_batch_norm=False, fc_hidden_dim=WIDTH,
    )
    tc = TrainingConfig(
        dataset=dataset, batch_size=256,
        optimizer="sgd", learning_rate=learning_rate,
        momentum=0.0, weight_decay=0.0,
        scheduler="none",
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=1,                 # record per-layer grads EVERY epoch
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        log_grad_per_layer=log_grad_per_layer,  # print full per-layer vector each epoch
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
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override; default 20 (smoke) / 200 (audit).")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (fixed, no scheduler). Default {LEARNING_RATE}.")
    args = parser.parse_args()

    label = args.experiment
    mode = "smoke" if "_smoke_" in label else "audit"
    epochs = args.epochs if args.epochs is not None else (20 if mode == "smoke" else 200)
    log_per_batch = (mode == "smoke")
    # Per-layer grad vector every epoch is printed for smoke (live visibility);
    # the JSON history records it every epoch for BOTH modes (diagnostics_every=1).
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

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode}, V2 NoBN + plain SGD, eta=ratio-MINIMUM)")
    print(f"# init={INIT_STRATEGY} eta={classifier_config.init_kwargs['eta']}")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (fixed LR, no patience)")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} "
          f"(must be None -- clipping forbidden for V2)")
    print(f"# epochs={epochs} diag_every=1 (per-layer grads logged EVERY epoch) "
          f"log_per_batch_first_epoch={log_per_batch} log_grad_per_layer={log_grad_per_layer}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for V2 experiments"
    )

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    if getattr(result, "abort_reason", None):
        print(f"\nABORT_REASON: {result.abort_reason}")
    if getattr(result, "initial_batch_loss", None) is not None:
        print(f"INITIAL_BATCH_LOSS: {result.initial_batch_loss:.4f}")

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
