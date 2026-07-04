#!/usr/bin/env python3
"""He + BN + plain SGD + lr=1e-9 smoke test on 100L architectures.

Follow-up to lowlr (lr=1e-6) smoke: at lr=1e-6 the max-gradient layer
(~1e5 grad norm) still receives a step of ~0.1 per weight -- comparable
to He init weight std (~0.063). No learning was observed in 20 epochs.

lr=1e-9 brings the max-layer step down to ~1e-4 per weight (~0.16% of
weight std per epoch). This tests whether any stable loss decrease is
possible when the largest-gradient layer is no longer blown around each
step. The minimum-gradient layer (~1.75 grad norm) gets a step of ~1.75e-9
per weight -- essentially frozen. If nothing learns here either, the
problem is not LR magnitude but the conditioning ratio itself.

Architectures: cifar10/100L/BN and fmnist/100L/BN.
Init: He (standard Kaiming), NOT row-centered.
Optimizer: plain SGD, no momentum, no weight decay.
Scheduler: none -- fixed lr=1e-9 for the entire run (no plateau, no
decay) so the update rule is purely w <- w - 1e-9 * grad with nothing
adaptive layered on top.

Output: reports/results/he_sgd_lowlr2_smoke_{arch}.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


ARCHITECTURE_KEYS = [
    "cifar10_100L_bn",
    "fmnist_100L_bn",
]

EXPERIMENT_LABELS = [f"he_sgd_lowlr2_smoke_{k}" for k in ARCHITECTURE_KEYS]


def _build(label: str):
    arch_key = label[len("he_sgd_lowlr2_smoke_"):]

    common_tc = dict(
        epochs=20,
        log_every_epoch=True,
        diagnostics_every=2,
        log_per_batch_first_epoch=True,
        normalize_inputs=True,
        target_train_accuracy=None,
    )

    if arch_key == "cifar10_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,
        )
        tc = TrainingConfig(
            dataset="cifar10", batch_size=256,
            optimizer="sgd", learning_rate=1e-9, momentum=0.0, weight_decay=0.0,
            scheduler="none",
            **common_tc,
        )
        return cc, tc

    if arch_key == "fmnist_100L_bn":
        cc = ClassifierConfig(
            architecture="fc", depth=100, init_strategy="he",
            use_batch_norm=True, fc_hidden_dim=500,
            bn_momentum=0.01,
        )
        tc = TrainingConfig(
            dataset="fashion_mnist", batch_size=256,
            optimizer="sgd", learning_rate=1e-9, momentum=0.0, weight_decay=0.0,
            scheduler="none",
            **common_tc,
        )
        return cc, tc

    raise ValueError(f"Unknown arch_key '{arch_key}' from label '{label}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    label = args.experiment
    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(label)

    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# init=he (Kaiming) -- NOT row-centered")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm} "
          f"bn_momentum={classifier_config.bn_momentum}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} "
          f"(p={training_config.plateau_patience} f={training_config.plateau_factor} "
          f"min_lr={training_config.plateau_min_lr})")
    print(f"# epochs=20 diag_every=2 log_per_batch_first_epoch=True")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

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
        print(f"\nSUMMARY {label} | NO_HISTORY")


if __name__ == "__main__":
    main()
