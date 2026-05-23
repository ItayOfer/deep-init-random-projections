#!/usr/bin/env python3
"""V2 row-centered audit — Round 3 — V2 + BN at L=100 (option B).

Background: recovery3 (He+BN+Adam+plateau+clip) just confirmed (2026-05-23)
that the two 100L/BN architectures (⑥, ⑫) cannot cross the thesis bar under
He. Both reached a poor local optimum (cifar10 21 %, fmnist 47 %) then
*degraded*. So the He scoreboard sits at 10/12 with no known recipe path
for those two.

Hypothesis being tested here: **BN normalizes per-layer activations, so it
should cancel V2\'s forward-pass amplification that overflows at L=100 NoBN.**
If BN does that work, then V2 (with its more uniform per-layer gradient
profile design) might train at L=100/BN where He+BN couldn\'t. If V2+BN at
L=100 also fails, the 100L/BN regime is a fundamental wall for both
initializers and the thesis statement is clean either way.

Recipe (per architecture, derived from recovery3 minus clipping):
  * cifar10/100L/BN: Adam lr=1e-3, plateau (p=5, f=0.5, min_lr=1e-7),
    bn_momentum=0.01, LR warmup 20 ep (start_factor=0.01),
    plateau_warmup_epochs=20, bs=256. eta=0.5 (V2 default).
    NO grad_clip (forbidden for V2 — feedback-no-grad-clipping).
  * fmnist/100L/BN:  Adam lr=1e-3, plateau (p=5, f=0.5, min_lr=1e-7),
    bn_momentum=0.01, LR warmup 5 ep (precaution; recovery3 used 0 here),
    bs=256. eta=0.5. NO grad_clip.

The fmnist warmup is added even though He+fmnist+recovery3 didn\'t need it,
because V2\'s layer-1 std is ~126× He\'s at L=100, and recovery3\'s
no-warmup recipe is a more aggressive starting point for V2.

Modes:
  --mode smoke   : 20 epochs, log_per_batch_first_epoch=True, diagnostics_every=2.
  --mode audit   : 200 epochs, log_per_batch_first_epoch=False, diagnostics_every=10.

Output: reports/results/row_centered_<mode>3_<arch>.json
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

    cc = ClassifierConfig(
        architecture="fc", depth=100, init_strategy=INIT_STRATEGY,
        init_kwargs={"eta": ETA},
        use_batch_norm=True, fc_hidden_dim=WIDTH,
        bn_momentum=0.01,
    )

    # cifar10_100L_bn: recovery3 cifar10 recipe minus clipping
    if arch_key == "cifar10_100L_bn":
        tc = TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="adam", learning_rate=1e-3, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=20, lr_warmup_start_factor=0.01,
            plateau_warmup_epochs=20,
            **common_tc,
        )
        return cc, tc

    # fmnist_100L_bn: recovery3 fmnist recipe + precautionary 5-ep warmup
    if arch_key == "fmnist_100L_bn":
        tc = TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="adam", learning_rate=1e-3, weight_decay=0.0,
            scheduler="plateau",
            plateau_patience=5, plateau_factor=0.5, plateau_min_lr=1e-7,
            plateau_metric="eval_train_loss",
            lr_warmup_epochs=5, lr_warmup_start_factor=0.1,
            **common_tc,
        )
        return cc, tc

    raise ValueError(f"Unknown arch_key '{arch_key}' from label '{label}'. "
                     f"Valid: {ARCHITECTURE_KEYS}")


ARCHITECTURE_KEYS = ["cifar10_100L_bn", "fmnist_100L_bn"]

EXPERIMENT_LABELS = (
    [f"row_centered_smoke3_{k}" for k in ARCHITECTURE_KEYS]
    + [f"row_centered_audit3_{k}" for k in ARCHITECTURE_KEYS]
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
    mode = "smoke" if "_smoke3_" in label else "audit"
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
    print(f"# {label}  (mode={mode}, round 3)")
    print(f"# init={INIT_STRATEGY} eta={ETA}")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm} "
          f"bn_momentum={classifier_config.bn_momentum}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (p={training_config.plateau_patience} "
          f"f={training_config.plateau_factor} min_lr={training_config.plateau_min_lr})")
    print(f"# lr_warmup_epochs={training_config.lr_warmup_epochs} "
          f"start_factor={training_config.lr_warmup_start_factor}  "
          f"plateau_warmup_epochs={training_config.plateau_warmup_epochs}")
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
