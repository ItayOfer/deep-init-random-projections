#!/usr/bin/env python3
"""Phase 4: chase the remaining stable-but-not-passing architectures.

Run plan:
  ε  (epsilon): fmnist/50L/BN   — gentler plateau (patience=10, factor=0.7), 200 ep
                                  Goal: push β's loss from 0.17 down to ≤ 0.10
  ζ₁ (zeta1):   cifar10/100L/BN — warmup=20 + plateau (patience=5, factor=0.5), 200 ep
                                  Goal: prevent the premature LR-drop that killed δ₁
  ζ₂ (zeta2):   fmnist/100L/BN  — warmup=20 + plateau (patience=5, factor=0.5), 200 ep
                                  Goal: same on fmnist side; let the late breakthrough start sooner
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


def build_phase4_configs(
    epochs: int = 200,
    diagnostics_every: int = 10,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    """3 phase-4 runs — all 200 epochs, all plateau-based."""
    configs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    # ---------- ε: fmnist/50L/BN, gentler plateau ----------
    configs.append((
        "epsilon_fmnist_50L_bn_adam_plateau_p10_f07_200ep",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=10, plateau_factor=0.7,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       plateau_warmup_epochs=0,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- ζ₁: cifar10/100L/BN, warmup + plateau ----------
    configs.append((
        "zeta1_cifar10_100L_bn_adam_warmup20_plateau_200ep",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="cifar10", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       plateau_warmup_epochs=20,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    # ---------- ζ₂: fmnist/100L/BN, warmup + plateau ----------
    configs.append((
        "zeta2_fmnist_100L_bn_adam_warmup20_plateau_200ep",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=0.001,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       plateau_warmup_epochs=20,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
                       checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every),
    ))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
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
        default=str(ROOT / "reports" / "results" / "diagnostic_phase4.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_phase4_configs(
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

    print(f"\n\nSaved {len(all_payload)} Phase-4 runs to {output_path}")

    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
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
            f"  {label:55s} "
            f"best={best_acc:.4f} "
            f"final={final_acc:.4f} "
            f"loss={final_loss:.4f} "
            f"final_lr={final_lr:.2e}"
        )


if __name__ == "__main__":
    main()
