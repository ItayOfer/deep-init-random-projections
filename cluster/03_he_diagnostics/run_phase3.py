#!/usr/bin/env python3
"""Phase 3: apply the plateau-LR lesson from Phase 2 across the 5 still-failing
architectures, and extend the cifar10/50L/BN winner (D) past the 0.10 loss threshold.

Run plan (all use ReduceLROnPlateau, patience=5, factor=0.5, monitor eval_train_loss):
  α: fmnist /50L /NoBN  — SGD  lr=0.003   bnm=—          100ep   push Phase 2 A past 0.995
  β: fmnist /50L /BN    — Adam lr=0.001   bnm=0.01       100ep   apply D recipe to replace collapsed B
  γ: cifar10/50L /BN    — Adam lr=0.001   bnm=0.01       200ep   extend D to drive loss ≤ 0.10
  δ1: cifar10/100L/BN   — Adam lr=0.001   bnm=0.01       100ep   does plateau unstick the 100L stall?
  δ2: fmnist /100L/BN   — Adam lr=0.001   bnm=0.01       100ep   same on fmnist
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def build_phase3_configs(
    epochs_cap: int = 200,
    diagnostics_every: int = 10,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    """5 plateau runs; γ uses full 200ep, the rest cap at 100ep."""
    configs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    short_ep = min(epochs_cap, 100)

    # ---------- α: fmnist/50L/NoBN SGD + plateau ----------
    configs.append((
        "alpha_fmnist_50L_nobn_sgd_plateau",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=short_ep,
                       optimizer="sgd", learning_rate=0.003, momentum=0.9,
                       scheduler="plateau",
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- β: fmnist/50L/BN Adam + plateau ----------
    configs.append((
        "beta_fmnist_50L_bn_adam_plateau_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=short_ep,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- γ: cifar10/50L/BN Adam + plateau, 200ep (D extended) ----------
    configs.append((
        "gamma_cifar10_50L_bn_adam_plateau_bnm001_200ep",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs_cap,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- δ1: cifar10/100L/BN Adam + plateau ----------
    configs.append((
        "delta1_cifar10_100L_bn_adam_plateau_bnm001",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=short_ep,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- δ2: fmnist/100L/BN Adam + plateau ----------
    configs.append((
        "delta2_fmnist_100L_bn_adam_plateau_bnm001",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=short_ep,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs (γ uses this; others cap at 100).")
    parser.add_argument("--diagnostics-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated list of experiment labels to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "diagnostic_phase3.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_phase3_configs(
        epochs_cap=args.epochs,
        diagnostics_every=args.diagnostics_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    if args.experiments:
        selected = {s.strip() for s in args.experiments.split(",")}
        all_configs = [(l, cc, tc) for l, cc, tc in all_configs if l in selected]
        if not all_configs:
            print(f"No experiments matched: {args.experiments}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_payload: List[dict] = []
    total = len(all_configs)

    for run_idx, (label, classifier_config, training_config) in enumerate(all_configs, 1):
        print(f"\n{'#'*60}")
        print(f"# [{run_idx}/{total}] {label}")
        print(f"{'#'*60}\n")

        exp_config.setup_seeds()
        result = run_supervised_experiment(exp_config, classifier_config, training_config)
        print_diagnostic_summary(label, result)

        all_payload.append(_result_to_payload(label, result))
        output_path.write_text(json.dumps(all_payload, indent=2))

    print(f"\n\nSaved {len(all_payload)} Phase-3 runs to {output_path}")

    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    for entry in all_payload:
        hist = entry.get("history", [])
        if not hist:
            continue
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_lr = hist[-1].get("learning_rate", 0) or 0
        label = entry["hypothesis_label"]
        print(
            f"  {label:50s} "
            f"best={best_acc:.4f} "
            f"final={final_acc:.4f} "
            f"loss={final_loss:.4f} "
            f"final_lr={final_lr:.2e}"
        )


if __name__ == "__main__":
    main()
