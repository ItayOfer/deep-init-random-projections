#!/usr/bin/env python3
"""row_centered_forward_balanced + per-layer BACKWARD gradient rescale by
r = sqrt((pi-1)/pi) ~ 0.826, trained with plain SGD, NoBN.

Motivation: fwd-balanced row-centering keeps the forward flat (g_fwd=1) but the
backward gain is g_bwd = 1/r ~ 1.21, which compounds to ~1e8 over 100 layers.
A GradRescale op (identity forward, multiply gradient by r in backward) inserted
after each hidden ReLU cancels this exactly -- at init it flattens rms(delta)
from 1e8 to ~1.2x and the gradient ratio from 5e7 to ~6x (He-like). This is a
closed-form, non-adaptive per-layer-LR (r^(L-l)). Question: does it TRAIN?

Fixed knobs (minimal, so the rescale is the only new variable):
  * init = row_centered_forward_balanced
  * grad_rescale = r ~ 0.826
  * NoBN, width 500
  * optimizer = SGD, momentum 0, weight_decay 0, scheduler none (fixed LR)
  * NO gradient clipping (assert-enforced)

6 architectures: {fmnist, cifar10} x {30, 50, 100}L.
smoke = 20 epochs (per-layer grads logged every epoch); audit = 200 epochs.
Output: reports/results/rcfwd_rescale_<mode>_<arch>.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary

INIT_STRATEGY = "row_centered_forward_balanced"
WIDTH = 500
LEARNING_RATE = 1e-2
R = math.sqrt((math.pi - 1.0) / math.pi)     # ~0.826 = 1/1.21

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
    [f"rcfwd_rescale_smoke_{k}" for k in ARCHITECTURE_KEYS]
    + [f"rcfwd_rescale_audit_{k}" for k in ARCHITECTURE_KEYS]
)


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float) -> Tuple[ClassifierConfig, TrainingConfig]:
    arch_key = label.split("_", 3)[-1]   # e.g. "fmnist_100L"
    dataset, depth = ARCH[arch_key]

    cc = ClassifierConfig(
        architecture="fc", depth=depth, init_strategy=INIT_STRATEGY,
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        grad_rescale=R,
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
    mode = "smoke" if "_smoke_" in label else "audit"
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

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode}, rcfwd + backward GradRescale)")
    print(f"# init={INIT_STRATEGY}  grad_rescale=r={R:.4f} (cancels g_bwd=1/r={1/R:.3f})")
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
