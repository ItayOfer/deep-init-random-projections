#!/usr/bin/env python3
"""V2 row-centered audit -- Round 4 -- V2 + BN + plain SGD.

Background: rounds 1-3 of the V2 audit used Adam for all BN architectures
(carrying over the He-audit pattern "SGD for NoBN, Adam for BN"). This is
the audit gap that round 4 fills.

Theoretical motivation. V2's eta-scheme creates a per-layer weight std
differential by design (about 13,000x layer-1 to layer-L at L=100,
eta=0.5). Under SGD, per-layer absolute step (lr * grad) scales naturally
with the local weight magnitude, so the relative step (delta_w / w) is
approximately uniform across layers. Under Adam, the adaptive per-parameter
scaling normalises step magnitude to about lr regardless of gradient, so
delta_w / w varies by V2's 13,000x weight differential. This is the
"double-preconditioning" effect that left V2+BN at L=100 stuck at chance
in smoke3.

The smoke3 measurement makes the case concrete: cifar10/100L/BN had a
per-layer gradient ratio of 5.7e4 at ep1 b0, and a per-layer weight ratio
of 1.3e4. Under SGD those nearly cancel: delta_w / w varies by only about
4x across layers, vs Adam's about 13,000x.

Architectures tested:
  * 30L BN sanity checks (V2+Adam already PASSed here in round 1; see if
    V2+SGD also passes -- confirms SGD is viable at the easy depth).
  * 50L BN (V2+Adam stuck at 28-33 percent in rounds 1-2; try V2+SGD).
  * 100L BN (V2+Adam stuck at chance in smoke3; try V2+SGD -- could close
    the L=100 BN open problem from the He audit too).

CRITICAL: no gradient clipping (see feedback-no-grad-clipping memory).
All recipe modifications use LR / warmup / momentum / scheduler only.

Modes:
  smoke (default): 20 epochs, log_per_batch_first_epoch=True, diag every 2.
  audit:           200 epochs, log_per_batch_first_epoch=False, diag every 10.

Output: reports/results/row_centered_<mode>4_<arch>.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


INIT_STRATEGY = "row_centered_layer_balanced_product_base"
WIDTH = 500
ETA = 0.5


def _build(label: str, epochs: int, diagnostics_every: int,
           log_per_batch_first_epoch: bool,
           abort_on_explosion: bool, explosion_loss_factor: float,
           ) -> Tuple[ClassifierConfig, TrainingConfig]:
    arch_key = label.split("_", 3)[-1]

    common_tc = dict(
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=diagnostics_every,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        abort_on_explosion=abort_on_explosion,
        explosion_loss_factor=explosion_loss_factor,
        normalize_inputs=True,
        target_train_accuracy=None,
    )

    def cc(depth: int, bn_momentum: float = 0.1) -> ClassifierConfig:
        return ClassifierConfig(
            architecture="fc", depth=depth, init_strategy=INIT_STRATEGY,
            init_kwargs={"eta": ETA},
            use_batch_norm=True, fc_hidden_dim=WIDTH,
            bn_momentum=bn_momentum,
        )

    # 30L BN sanity checks
    # V2+Adam audit was PASS; does V2+SGD also pass?

    if arch_key == "cifar10_30L_bn":
        return cc(30, bn_momentum=0.1), TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle",
            **common_tc,
        )

    if arch_key == "fmnist_30L_bn":
        return cc(30, bn_momentum=0.1), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle",
            **common_tc,
        )

    # 50L BN -- V2+Adam stuck at 28-33%, try V2+SGD plateau

    if arch_key == "cifar10_50L_bn":
        return cc(50, bn_momentum=0.01), TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="sgd", learning_rate=3e-3, momentum=0.9,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc,
        )

    if arch_key == "fmnist_50L_bn":
        return cc(50, bn_momentum=0.01), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="sgd", learning_rate=3e-3, momentum=0.9,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc,
        )

    # 100L BN -- V2+Adam stuck at chance, try plain SGD (recovery-style)

    if arch_key == "cifar10_100L_bn":
        return cc(100, bn_momentum=0.01), TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=5, lr_warmup_start_factor=0.1,
            **common_tc,
        )

    if arch_key == "fmnist_100L_bn":
        return cc(100, bn_momentum=0.01), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=5, lr_warmup_start_factor=0.1,
            **common_tc,
        )

    raise ValueError(f"Unknown arch_key '{arch_key}' from label '{label}'. "
                     f"Valid: {ARCHITECTURE_KEYS}")


ARCHITECTURE_KEYS = [
    "cifar10_30L_bn",
    "fmnist_30L_bn",
    "cifar10_50L_bn",
    "fmnist_50L_bn",
    "cifar10_100L_bn",
    "fmnist_100L_bn",
]

EXPERIMENT_LABELS = (
    [f"row_centered_smoke4_{k}" for k in ARCHITECTURE_KEYS]
    + [f"row_centered_audit4_{k}" for k in ARCHITECTURE_KEYS]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override; default 20 (smoke) / 200 (audit).")
    parser.add_argument("--explosion-factor", type=float, default=5.0)
    args = parser.parse_args()

    label = args.experiment
    mode = "smoke" if "_smoke4_" in label else "audit"
    epochs = args.epochs if args.epochs is not None else (20 if mode == "smoke" else 200)
    diag = 2 if mode == "smoke" else 10
    log_per_batch = (mode == "smoke")

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(
        label, epochs=epochs, diagnostics_every=diag,
        log_per_batch_first_epoch=log_per_batch,
        abort_on_explosion=True, explosion_loss_factor=args.explosion_factor,
    )

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode}, round 4 = V2+BN+SGD)")
    print(f"# init={INIT_STRATEGY} eta={ETA}")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm} "
          f"bn_momentum={classifier_config.bn_momentum}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler}", end="")
    if training_config.scheduler == "plateau":
        print(f" (p={training_config.plateau_patience} f={training_config.plateau_factor} "
              f"min_lr={training_config.plateau_min_lr})")
    else:
        print()
    print(f"# lr_warmup_epochs={training_config.lr_warmup_epochs} "
          f"start_factor={training_config.lr_warmup_start_factor}")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None -- "
          f"clipping is forbidden for V2)")
    print(f"# epochs={epochs} diag_every={diag} log_per_batch_first_epoch={log_per_batch}")
    print(f"# abort_on_explosion={training_config.abort_on_explosion} "
          f"factor={training_config.explosion_loss_factor}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for V2 experiments"
    )

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    if result.abort_reason:
        print(f"\nABORT_REASON: {result.abort_reason}")
    if result.initial_batch_loss is not None:
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
