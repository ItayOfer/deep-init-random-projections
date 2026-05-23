#!/usr/bin/env python3
"""V2 row-centered audit — Round 2 recipes for the 4 architectures that
failed/struggled in Round 1.

Round 1 triage (2026-05-23):
  | arch                  | round-1 verdict | round-1 recipe                                                  |
  | cifar10_100L_nobn     | ABORTED (NaN ep1b0) | SGD lr=1e-3 mom=0 plateau p=5 min_lr=1e-7 bs=128, eta=0.5 |
  | fmnist_100L_nobn      | ABORTED (NaN ep1b0) | SGD lr=1e-3 mom=0 plateau p=10 bs=128, eta=0.5            |
  | fmnist_50L_nobn       | ABORTED (explosion ep1 b77, loss=16) | SGD lr=3e-3 mom=0.9 plateau p=5 bs=128, eta=0.5 |
  | fmnist_50L_bn         | STUCK (loss 2.29→3.66, ill-conditioned) | Adam lr=5e-4 plateau p=5 bnm=0.01 bs=256, eta=0.5 |

Analysis (see triage report 2026-05-23):
  * L=100 NoBN: V2 with eta=0.5 overflows float32 in the forward pass at
    layer ~9–11 (layer-1 std ≈ 6.4 amplifies per-layer until inf). This is
    a *forward-pass* issue, not a training-dynamics issue: no LR/warmup
    schedule can rescue a network whose first forward already produces NaN.
    User decision (Q-T1 → A): drop to eta=0.1 for the two L=100 NoBN
    architectures. At eta=0.1 the V2 forward survives (peak A_RMS ≈ 4.6e8,
    final logit RMS ≈ 1e-4, initial CE loss ≈ 2.30). The eta=0.1 setting
    keeps a moderate V2 per-layer correction (5x weight differential
    layer1/layer100, vs He\'s 1x) — meaningful but safe.
  * L=50 NoBN: explosion mid-training at batch 77. Mid-batch loss exceeded
    5x initial because per-batch SGD updates compounded V2\'s already-large
    per-layer activations. Fix: halve LR (3e-3 → 1.5e-3) + add LR warmup
    (5 epochs, start_factor=0.1 → LR ramps 1.5e-4 → 1.5e-3 over 5 ep).
  * L=50 BN: stuck-and-deteriorating. Adam\'s adaptive step in V2\'s
    ill-conditioned (1.3e3 max/min ratio) gradient profile is itself the
    problem — preconditioning of an already-preconditioned init.
    Fix: halve LR (5e-4 → 2.5e-4) + add LR warmup (5 epochs).

CRITICAL: gradient clipping is FORBIDDEN for V2 experiments
([[feedback-no-grad-clipping]]). All recipe modifications use LR /
warmup / optimizer changes only.

Modes:
  --mode smoke   : 20 epochs, log_per_batch_first_epoch=True, diagnostics_every=2.
  --mode audit   : 200 epochs, log_per_batch_first_epoch=False, diagnostics_every=10.

Output: reports/results/row_centered_<mode>2_<arch>.json (with the "2" suffix
to keep round-2 results distinct from the round-1 smoke JSONs).
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


# label format: row_centered_<mode>2_<arch>, e.g. row_centered_smoke2_cifar10_100L_nobn
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

    def cc(depth: int, use_bn: bool, eta: float, bn_momentum: float = 0.1) -> ClassifierConfig:
        return ClassifierConfig(
            architecture="fc", depth=depth, init_strategy=INIT_STRATEGY,
            init_kwargs={"eta": eta},
            use_batch_norm=use_bn, fc_hidden_dim=WIDTH,
            bn_momentum=bn_momentum,
        )

    # cifar10_100L_nobn — recovery-2 recipe + eta=0.1 (round-1 eta=0.5 overflowed forward at L9)
    if arch_key == "cifar10_100L_nobn":
        return cc(100, False, eta=0.1), TrainingConfig(
            dataset="cifar10", batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            **common_tc,
        )

    # fmnist_100L_nobn — recovery-1 recipe + eta=0.1 (round-1 eta=0.5 overflowed forward at L9)
    if arch_key == "fmnist_100L_nobn":
        return cc(100, False, eta=0.1), TrainingConfig(
            dataset="fashion_mnist", batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=10, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc,
        )

    # fmnist_50L_nobn — eta=0.5 retained; halved LR (3e-3 → 1.5e-3) + 5-epoch warmup.
    if arch_key == "fmnist_50L_nobn":
        return cc(50, False, eta=0.5), TrainingConfig(
            dataset="fashion_mnist", batch_size=128,
            optimizer="sgd", learning_rate=1.5e-3, momentum=0.9,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=5, lr_warmup_start_factor=0.1,
            **common_tc,
        )

    # fmnist_50L_bn — eta=0.5 retained; halved LR (5e-4 → 2.5e-4) + 5-epoch warmup.
    if arch_key == "fmnist_50L_bn":
        return cc(50, True, eta=0.5, bn_momentum=0.01), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="adam", learning_rate=2.5e-4,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=5, lr_warmup_start_factor=0.1,
            **common_tc,
        )

    raise ValueError(f"Unknown arch_key '{arch_key}' from label '{label}'. "
                     f"Valid: {sorted(ARCHITECTURE_KEYS)}")


ARCHITECTURE_KEYS = [
    "cifar10_100L_nobn",
    "fmnist_100L_nobn",
    "fmnist_50L_nobn",
    "fmnist_50L_bn",
]

EXPERIMENT_LABELS = (
    [f"row_centered_smoke2_{k}" for k in ARCHITECTURE_KEYS]
    + [f"row_centered_audit2_{k}" for k in ARCHITECTURE_KEYS]
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
    mode = "smoke" if "_smoke2_" in label else "audit"
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
    print(f"# {label}  (mode={mode}, round 2)")
    print(f"# init={INIT_STRATEGY} eta={classifier_config.init_kwargs.get('eta')}")
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
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None — "
          f"clipping is forbidden for V2)")
    print(f"# epochs={epochs} diag_every={diag} log_per_batch_first_epoch={log_per_batch}")
    print(f"# abort_on_explosion={training_config.abort_on_explosion} "
          f"factor={training_config.explosion_loss_factor}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for V2 experiments (see feedback memory)"
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
