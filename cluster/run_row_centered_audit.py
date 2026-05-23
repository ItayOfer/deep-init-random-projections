#!/usr/bin/env python3
"""V2 row-centered audit — same 12-architecture grid as run_final.py, but with
the row_centered_layer_balanced_product_base initializer (V2) instead of He.

Background: this is Item 3 of the May 22, 2026 post-meeting plan. The
advisor's ask was to repeat the architecture audit with the final
row-centered initializer (V2 with η=0.5) to test whether the "right variance
in each layer" can train the same set of networks that He init handled.

This run only audits the 10 architectures that PASSED in the He audit
(per the post-meeting state). The 2 still-pending 100L/BN architectures
(cifar10 ⑥, fmnist ⑫) are deliberately skipped — they're under recovery3
with He, and we'll add them to the V2 audit once recovery3 produces a
confirmed-passing He recipe to start from.

Modes:
  --mode smoke   : 20 epochs, log_per_batch_first_epoch=True, diagnostics_every=2,
                   abort_on_explosion=True (factor 5x).
  --mode audit   : 200 epochs, log_per_batch_first_epoch=False, diagnostics_every=10,
                   abort_on_explosion=True (factor 5x).

Per-architecture recipe = the He-passing recipe for that architecture
(see plan §2 item 3). Override at run time with --override-* if needed
after smoke triage.

Output: reports/results/row_centered_<mode>_<arch>.json (matches the
naming convention of recovery_plain_sgd_*.json etc.).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


# V2 initializer settings (Q3a: η=0.5 confirmed; Q3d: width=500 confirmed).
INIT_STRATEGY = "row_centered_layer_balanced_product_base"
INIT_KWARGS = {"eta": 0.5}
WIDTH = 500


# Per-architecture He-passing recipe (the starting point for V2 smoke).
# Skip cifar10_100L_bn (⑥) and fmnist_100L_bn (⑫) — those still under recovery3.
def _build(label: str, epochs: int, diagnostics_every: int,
           log_per_batch_first_epoch: bool,
           abort_on_explosion: bool, explosion_loss_factor: float,
           ) -> Tuple[ClassifierConfig, TrainingConfig]:
    """Return (ClassifierConfig, TrainingConfig) for `label`.

    label format: row_centered_<mode>_<arch>, e.g. row_centered_smoke_cifar10_30L_nobn.
    """
    arch_key = label.split("_", 3)[-1]   # e.g. "cifar10_30L_nobn"

    common_tc_kwargs = dict(
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=diagnostics_every,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        abort_on_explosion=abort_on_explosion,
        explosion_loss_factor=explosion_loss_factor,
        normalize_inputs=True,
        target_train_accuracy=None,   # no early stop in V2 audit; we want full trajectory
    )

    def cc(depth: int, use_bn: bool, bn_momentum: float = 0.1) -> ClassifierConfig:
        return ClassifierConfig(
            architecture="fc", depth=depth, init_strategy=INIT_STRATEGY,
            init_kwargs=dict(INIT_KWARGS),
            use_batch_norm=use_bn, fc_hidden_dim=WIDTH,
            bn_momentum=bn_momentum,
        )

    # ① cifar10/30L/NoBN — SGD onecycle lr=0.01 bs=128
    if arch_key == "cifar10_30L_nobn":
        return cc(30, False), TrainingConfig(
            dataset="cifar10", batch_size=128,
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle",
            **common_tc_kwargs,
        )

    # ② cifar10/30L/BN — Adam onecycle lr=1e-3 bnm=0.1 bs=256
    if arch_key == "cifar10_30L_bn":
        return cc(30, True, bn_momentum=0.1), TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="adam", learning_rate=1e-3,
            scheduler="onecycle",
            **common_tc_kwargs,
        )

    # ③ cifar10/50L/NoBN — SGD onecycle lr=0.01 bs=128
    if arch_key == "cifar10_50L_nobn":
        return cc(50, False), TrainingConfig(
            dataset="cifar10", batch_size=128,
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle",
            **common_tc_kwargs,
        )

    # ④ cifar10/50L/BN — Adam plateau lr=1e-3 bnm=0.01 bs=256 (Phase 3 γ)
    if arch_key == "cifar10_50L_bn":
        return cc(50, True, bn_momentum=0.01), TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="adam", learning_rate=1e-3,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc_kwargs,
        )

    # ⑤ cifar10/100L/NoBN — SGD plain plateau lr=1e-3 p=5 min_lr=1e-7 bs=128 (recovery 2)
    if arch_key == "cifar10_100L_nobn":
        return cc(100, False), TrainingConfig(
            dataset="cifar10", batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            **common_tc_kwargs,
        )

    # ⑦ fmnist/30L/NoBN — SGD onecycle lr=0.01 bs=128
    if arch_key == "fmnist_30L_nobn":
        return cc(30, False), TrainingConfig(
            dataset="fashion_mnist", batch_size=128,
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle",
            **common_tc_kwargs,
        )

    # ⑧ fmnist/30L/BN — Adam cosine lr=1e-3 bnm=0.1 bs=256
    if arch_key == "fmnist_30L_bn":
        return cc(30, True, bn_momentum=0.1), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="adam", learning_rate=1e-3,
            scheduler="cosine",
            **common_tc_kwargs,
        )

    # ⑨ fmnist/50L/NoBN — SGD plateau lr=3e-3 mom=0.9 bs=128 (Phase 3 α)
    if arch_key == "fmnist_50L_nobn":
        return cc(50, False), TrainingConfig(
            dataset="fashion_mnist", batch_size=128,
            optimizer="sgd", learning_rate=3e-3, momentum=0.9,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc_kwargs,
        )

    # ⑩ fmnist/50L/BN — Adam plateau lr=5e-4 bnm=0.01 bs=256 (Phase 5 ε')
    if arch_key == "fmnist_50L_bn":
        return cc(50, True, bn_momentum=0.01), TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="adam", learning_rate=5e-4,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc_kwargs,
        )

    # ⑪ fmnist/100L/NoBN — SGD plain plateau lr=1e-3 p=10 bs=128 (recovery 1)
    if arch_key == "fmnist_100L_nobn":
        return cc(100, False), TrainingConfig(
            dataset="fashion_mnist", batch_size=128,
            optimizer="sgd", learning_rate=1e-3, momentum=0.0, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=10, plateau_factor=0.5, plateau_min_lr=1e-6,
            plateau_metric="eval_train_loss",
            **common_tc_kwargs,
        )

    raise ValueError(
        f"Unknown architecture key '{arch_key}' (from label '{label}'). "
        f"Valid: {sorted(ARCHITECTURE_KEYS)}"
    )


ARCHITECTURE_KEYS = [
    "cifar10_30L_nobn",
    "cifar10_30L_bn",
    "cifar10_50L_nobn",
    "cifar10_50L_bn",
    "cifar10_100L_nobn",
    "fmnist_30L_nobn",
    "fmnist_30L_bn",
    "fmnist_50L_nobn",
    "fmnist_50L_bn",
    "fmnist_100L_nobn",
]

EXPERIMENT_LABELS = (
    [f"row_centered_smoke_{k}" for k in ARCHITECTURE_KEYS]
    + [f"row_centered_audit_{k}" for k in ARCHITECTURE_KEYS]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_LABELS,
                        help="Which architecture to run (one job runs one arch).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None,
                        help="JSON output path. Default: reports/results/<experiment>.json")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch budget; default 20 (smoke) / 200 (audit).")
    parser.add_argument("--explosion-factor", type=float, default=5.0,
                        help="Loss-explosion abort multiplier. Default 5.0 (per plan).")
    args = parser.parse_args()

    label = args.experiment
    mode = "smoke" if "_smoke_" in label else "audit"

    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = 20 if mode == "smoke" else 200

    diagnostics_every = 2 if mode == "smoke" else 10
    log_per_batch_first_epoch = (mode == "smoke")

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(
        label,
        epochs=epochs,
        diagnostics_every=diagnostics_every,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        abort_on_explosion=True,
        explosion_loss_factor=args.explosion_factor,
    )

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode})")
    print(f"# init={INIT_STRATEGY} eta={INIT_KWARGS['eta']}")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm} "
          f"bn_momentum={classifier_config.bn_momentum}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler}", end="")
    if training_config.scheduler == "plateau":
        print(f" (patience={training_config.plateau_patience} "
              f"factor={training_config.plateau_factor} "
              f"min_lr={training_config.plateau_min_lr})")
    else:
        print()
    print(f"# epochs={training_config.epochs} "
          f"diagnostics_every={training_config.diagnostics_every} "
          f"log_per_batch_first_epoch={training_config.log_per_batch_first_epoch}")
    print(f"# abort_on_explosion={training_config.abort_on_explosion} "
          f"factor={training_config.explosion_loss_factor}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

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
