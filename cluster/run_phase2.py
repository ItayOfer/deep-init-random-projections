#!/usr/bin/env python3
"""Phase 2: longer (100-epoch) runs to confirm Phase 1 winners and resolve
the cifar10/50L/BN ambiguity, plus a fmnist/100L/NoBN gradient-death check
for symmetric coverage with Phase 1 run #7.

Run plan:
  A: fmnist /50L /NoBN  — SGD constant lr=0.003                 (winner, extended to 100ep)
  B: fmnist /50L /BN    — Adam cosine T_max=100 bnm=0.01        (verify no collapse past ep43)
  C: cifar10/50L /BN    — Adam constant lr=0.001 bnm=0.01       (control: does collapse happen w/o scheduler influence?)
  D: cifar10/50L /BN    — Adam plateau patience=5 factor=0.5    (adaptive: does LR-drop correlate with stall/collapse?)
  E: cifar10/100L/BN    — Adam cosine T_max=100 bnm=0.01        (best rescue candidate)
  F: fmnist /100L/BN    — Adam cosine T_max=100 bnm=0.01        (same recipe, fmnist side)
  G: fmnist /100L/NoBN  — Adam cosine 10ep, diagnostics_every=1 (gradient-death symmetry with cifar10/100L/NoBN)
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import (
    SupervisedTrainingResult,
    run_supervised_experiment,
)
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def build_phase2_configs(
    epochs: int = 100,
    diagnostics_every: int = 10,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    """Phase 2: 7 longer runs (6 × 100ep + 1 × 10ep death-check)."""
    configs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    # ---------- A: fmnist/50L/NoBN, constant LR winner extended ----------
    configs.append((
        "A_fmnist_50L_nobn_sgd_const_lr003",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="sgd", learning_rate=0.003, momentum=0.9,
                       scheduler="none",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- B: fmnist/50L/BN, cosine matched to run length ----------
    configs.append((
        "B_fmnist_50L_bn_adam_cosine_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- C: cifar10/50L/BN, constant LR control ----------
    configs.append((
        "C_cifar10_50L_bn_adam_const_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="none", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- D: cifar10/50L/BN, ReduceLROnPlateau (eval_train_loss) ----------
    configs.append((
        "D_cifar10_50L_bn_adam_plateau_bnm001",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- E: cifar10/100L/BN, cosine matched ----------
    configs.append((
        "E_cifar10_100L_bn_adam_cosine_bnm001",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- F: fmnist/100L/BN, cosine matched ----------
    configs.append((
        "F_fmnist_100L_bn_adam_cosine_bnm001",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine", batch_size=256,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- G: fmnist/100L/NoBN, gradient-death symmetry check ----------
    # Mirror of Phase 1 run #7. Capped at 10 epochs; longer is wasted compute.
    configs.append((
        "G_fmnist_100L_nobn_gradient_death_check",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        TrainingConfig(dataset="fashion_mnist", epochs=min(epochs, 10),
                       optimizer="adam", learning_rate=0.001,
                       scheduler="cosine",
                       log_every_epoch=True, diagnostics_every=1,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
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
        default=str(ROOT / "reports" / "results" / "diagnostic_phase2.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_phase2_configs(
        epochs=args.epochs,
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

    print(f"\n\nSaved {len(all_payload)} Phase-2 runs to {output_path}")

    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    for entry in all_payload:
        hist = entry.get("history", [])
        if not hist:
            continue
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_lr = hist[-1].get("learning_rate", 0) or 0
        label = entry["hypothesis_label"]
        print(
            f"  {label:46s} "
            f"best_eval_train_acc={best_acc:.4f} "
            f"final={final_acc:.4f} "
            f"final_lr={final_lr:.2e}"
        )


if __name__ == "__main__":
    main()
