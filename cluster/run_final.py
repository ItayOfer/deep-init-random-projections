#!/usr/bin/env python3
"""Unified final audit run.

Reruns each of the 12 (dataset, depth, bn) architectures with its best-known
recipe — the configuration that produced the best stable result across the
original sweep + diagnostic phases 1–5. All runs use:

  - seed = 42
  - 200 epochs (uniform budget, no early stopping)
  - log_every_epoch=True
  - diagnostics_every=10  (grad norms + BN stats every 10 ep, LR every ep)

Output: reports/results/final_audit.json with 12 entries — one per architecture,
each with full per-epoch history suitable for unified trajectory plotting.

Source of each recipe:
  ① cifar10 / 30L /NoBN  — original sweep        (SGD onecycle, PASS)
  ② cifar10 / 30L / BN   — original sweep        (Adam onecycle, PASS)
  ③ cifar10 / 50L /NoBN  — original sweep        (SGD onecycle, PASS)
  ④ cifar10 / 50L / BN   — Phase 3 γ              (Adam plateau bnm=0.01, PASS)
  ⑤ cifar10 /100L /NoBN  — Phase 1 #7 / dead      (Adam cosine, gradient death)
  ⑥ cifar10 /100L / BN   — Phase 2 E (best 0.21)  (Adam cosine bnm=0.01)
  ⑦ fmnist  / 30L /NoBN  — original sweep        (SGD onecycle, PASS)
  ⑧ fmnist  / 30L / BN   — original sweep        (Adam cosine, PASS)
  ⑨ fmnist  / 50L /NoBN  — Phase 3 α              (SGD plateau, PASS)
  ⑩ fmnist  / 50L / BN   — Phase 5 ε'             (Adam plateau lr=5e-4, PASS)
  ⑪ fmnist  /100L /NoBN  — Phase 2 G / dead       (Adam cosine, gradient death)
  ⑫ fmnist  /100L / BN   — Phase 3 δ₂ (best 0.34) (Adam plateau bnm=0.01)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from run_diagnostic import _result_to_payload, print_diagnostic_summary


def build_final_configs(
    epochs: int = 200,
    diagnostics_every: int = 10,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    """Best-known recipe for each of the 12 architectures."""
    cfgs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    def _tc(**kw):
        """TrainingConfig with shared defaults."""
        defaults = dict(
            epochs=epochs,
            log_every_epoch=True,
            diagnostics_every=diagnostics_every,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            target_train_accuracy=None,   # no early stopping — we want full trajectories
        )
        defaults.update(kw)
        return TrainingConfig(**defaults)

    # =========================================================
    #  cifar10
    # =========================================================

    # ① cifar10/30L/NoBN — original sweep (SGD onecycle lr=0.01 bs=128)
    cfgs.append((
        "01_cifar10_30L_nobn",
        ClassifierConfig(architecture="fc", depth=30, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="cifar10",
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle", batch_size=128),
    ))

    # ② cifar10/30L/BN — original sweep (Adam onecycle lr=0.001 bnm=0.1 bs=256)
    cfgs.append((
        "02_cifar10_30L_bn",
        ClassifierConfig(architecture="fc", depth=30, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.1),
        _tc(dataset="cifar10",
            optimizer="adam", learning_rate=0.001,
            scheduler="onecycle", batch_size=256),
    ))

    # ③ cifar10/50L/NoBN — original sweep (SGD onecycle lr=0.01 bs=128)
    cfgs.append((
        "03_cifar10_50L_nobn",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="cifar10",
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle", batch_size=128),
    ))

    # ④ cifar10/50L/BN — Phase 3 γ (Adam plateau p=5 f=0.5, lr=1e-3, bnm=0.01, bs=256)
    cfgs.append((
        "04_cifar10_50L_bn",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        _tc(dataset="cifar10",
            optimizer="adam", learning_rate=0.001,
            scheduler="plateau", batch_size=256,
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-6, plateau_metric="eval_train_loss"),
    ))

    # ⑤ cifar10/100L/NoBN — gradient death (Adam cosine lr=0.001 bs=128)
    cfgs.append((
        "05_cifar10_100L_nobn",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="cifar10",
            optimizer="adam", learning_rate=0.001,
            scheduler="cosine", batch_size=128),
    ))

    # ⑥ cifar10/100L/BN — Phase 2 E best-known (Adam cosine lr=0.001, bnm=0.01, bs=256)
    cfgs.append((
        "06_cifar10_100L_bn",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        _tc(dataset="cifar10",
            optimizer="adam", learning_rate=0.001,
            scheduler="cosine", batch_size=256),
    ))

    # =========================================================
    #  fashion_mnist
    # =========================================================

    # ⑦ fmnist/30L/NoBN — original sweep (SGD onecycle lr=0.01 bs=128)
    cfgs.append((
        "07_fmnist_30L_nobn",
        ClassifierConfig(architecture="fc", depth=30, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="fashion_mnist",
            optimizer="sgd", learning_rate=0.01, momentum=0.9,
            scheduler="onecycle", batch_size=128),
    ))

    # ⑧ fmnist/30L/BN — original sweep (Adam cosine lr=0.001 bnm=0.1 bs=256)
    cfgs.append((
        "08_fmnist_30L_bn",
        ClassifierConfig(architecture="fc", depth=30, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.1),
        _tc(dataset="fashion_mnist",
            optimizer="adam", learning_rate=0.001,
            scheduler="cosine", batch_size=256),
    ))

    # ⑨ fmnist/50L/NoBN — Phase 3 α (SGD lr=0.003 + plateau p=5 f=0.5, bs=128)
    cfgs.append((
        "09_fmnist_50L_nobn",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="fashion_mnist",
            optimizer="sgd", learning_rate=0.003, momentum=0.9,
            scheduler="plateau", batch_size=128,
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-6, plateau_metric="eval_train_loss"),
    ))

    # ⑩ fmnist/50L/BN — Phase 5 ε' (Adam lr=5e-4 plateau p=5 f=0.5, bnm=0.01, bs=256)
    cfgs.append((
        "10_fmnist_50L_bn",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        _tc(dataset="fashion_mnist",
            optimizer="adam", learning_rate=5e-4,
            scheduler="plateau", batch_size=256,
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-6, plateau_metric="eval_train_loss"),
    ))

    # ⑪ fmnist/100L/NoBN — gradient death (Adam cosine lr=0.001 bs=128)
    cfgs.append((
        "11_fmnist_100L_nobn",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=False, fc_hidden_dim=500),
        _tc(dataset="fashion_mnist",
            optimizer="adam", learning_rate=0.001,
            scheduler="cosine", batch_size=128),
    ))

    # ⑫ fmnist/100L/BN — Phase 3 δ₂ best-known (Adam plateau lr=0.001 bnm=0.01 bs=256)
    cfgs.append((
        "12_fmnist_100L_bn",
        ClassifierConfig(architecture="fc", depth=100, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        _tc(dataset="fashion_mnist",
            optimizer="adam", learning_rate=0.001,
            scheduler="plateau", batch_size=256,
            plateau_patience=5, plateau_factor=0.5,
            plateau_min_lr=1e-6, plateau_metric="eval_train_loss"),
    ))

    return cfgs


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
        help="Comma-separated list of experiment labels to run (default: all 12)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "results" / "final_audit.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_final_configs(
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

    print(f"\n\nSaved {len(all_payload)} final-audit runs to {output_path}")

    print("\n" + "=" * 60)
    print("FINAL AUDIT SUMMARY")
    print("=" * 60)
    for entry in all_payload:
        hist = entry.get("history", [])
        if not hist:
            continue
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_test = hist[-1].get("test_accuracy", 0) or 0
        passes = final_acc >= 0.995 and final_loss <= 0.10
        flag = "PASS" if passes else "fail"
        label = entry["hypothesis_label"]
        print(
            f"  [{flag}] {label:30s} "
            f"best_eval_train_acc={best_acc:.4f} "
            f"final={final_acc:.4f} "
            f"loss={final_loss:.4f} "
            f"test={final_test:.4f}"
        )


if __name__ == "__main__":
    main()
