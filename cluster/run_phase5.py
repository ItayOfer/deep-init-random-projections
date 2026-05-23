#!/usr/bin/env python3
"""Phase 5: one final shot at fmnist/50L/BN.

Hypothesis: Phase 3 β (Adam lr=1e-3, plateau p=5 f=0.5, bnm=0.01) plateaued at
acc=0.9487 / loss=0.17 with loss spikes at ep 17 (1.73) and ep 41 (1.16). The
spikes suggest the starting LR was just slightly too aggressive for BN
stability. Halving it should suppress the spikes without losing learning speed
(plateau-driven LR drops will pick up the slack).

  ε': fmnist/50L/BN  Adam lr=5e-4  bnm=0.01  plateau(p=5, f=0.5)  100 ep
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


def build_phase5_configs(
    epochs: int = 100,
    diagnostics_every: int = 10,
    checkpoint_dir: str = "",
    checkpoint_every: int = 0,
) -> List[Tuple[str, ClassifierConfig, TrainingConfig]]:
    configs: List[Tuple[str, ClassifierConfig, TrainingConfig]] = []

    # ---------- ε': fmnist/50L/BN, halved LR ----------
    configs.append((
        "epsilon_prime_fmnist_50L_bn_adam_lr5e4_plateau_100ep",
        ClassifierConfig(architecture="fc", depth=50, init_strategy="he",
                         use_batch_norm=True, fc_hidden_dim=500, bn_momentum=0.01),
        TrainingConfig(dataset="fashion_mnist", epochs=epochs,
                       optimizer="adam", learning_rate=5e-4,
                       scheduler="plateau", batch_size=256,
                       plateau_patience=5, plateau_factor=0.5,
                       plateau_min_lr=1e-6, plateau_metric="eval_train_loss",
                       plateau_warmup_epochs=0,
                       log_every_epoch=True, diagnostics_every=diagnostics_every,
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
        "--output",
        default=str(ROOT / "reports" / "results" / "diagnostic_phase5.json"),
    )
    args = parser.parse_args()

    exp_config = ExperimentConfig(seed=args.seed, device=args.device, data_dir=str(ROOT / "data"))

    all_configs = build_phase5_configs(
        epochs=args.epochs,
        diagnostics_every=args.diagnostics_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

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

    print(f"\n\nSaved {len(all_payload)} Phase-5 run(s) to {output_path}")


if __name__ == "__main__":
    main()
