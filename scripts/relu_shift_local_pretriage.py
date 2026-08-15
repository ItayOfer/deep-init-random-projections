#!/usr/bin/env python3
"""Campaign 11: local CPU pre-triage -- a few real training epochs per arm,
before any cluster time is requested.

The brief's budget rule is "exhaust the local screen before requesting cluster
time". The init-time screen (scripts/relu_shift_geometry_screen.py) and the
funnel (scripts/relu_shift_funnel_fwd_bwd.py) cover forward, backward and
geometry at initialization, but they cannot tell you whether an arm actually
descends. At 30L that is cheap enough to answer on a laptop, so this runs the
exact cluster recipe (NoBN, width 500, plain SGD lr 1e-2 mom 0 wd 0, bs 256,
seed 42, no clipping, normalize_inputs) for a couple of epochs per arm and
records what moved.

This is a GO/NO-GO screen for the cluster grid, not a result: 2 epochs cannot
separate "slow" from "stuck", and the numbers here are CPU, not the CUDA
numbers the campaign will report. Its only job is to catch arms that are dead
or diverging before 16 SLURM jobs are queued behind them.

Usage:  python scripts/relu_shift_local_pretriage.py [--epochs 2] [--depth 30]
Output: reports/results/relushift_local_pretriage_<depth>L_<dataset>.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig  # noqa: E402
from rp_study.experiments.supervised_training import run_supervised_experiment  # noqa: E402

DC = 1.0 / math.sqrt(math.pi)
ARMS = {
    "he":   {"init_strategy": "he", "relu_shift": None, "detach": True},
    "rc":   {"init_strategy": "row_centered_he", "relu_shift": None, "detach": True},
    "c010": {"init_strategy": "he", "relu_shift": 0.10, "detach": True},
    "c025": {"init_strategy": "he", "relu_shift": 0.25, "detach": True},
    "c070": {"init_strategy": "he", "relu_shift": 0.70, "detach": True},
    "c025_diff": {"init_strategy": "he", "relu_shift": 0.25, "detach": False},
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--depth", type=int, default=30)
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    a = p.parse_args()

    out = {"description": "Local CPU pre-triage of campaign-11 arms (GO/NO-GO for the "
                          "cluster grid, not a reported result)",
           "config": vars(a) | {"device": "cpu", "optimizer": "sgd", "momentum": 0.0,
                                "weight_decay": 0.0, "batch_size": 256,
                                "exact_dc_constant": DC},
           "arms": {}}

    print(f"{a.dataset} {a.depth}L width {a.width} lr {a.lr} seed {a.seed} "
          f"{a.epochs} epoch(s), CPU")
    print(f"{'arm':<12}{'init':<26}{'c':>8}{'detach':>8}"
          f"{'ep1 acc':>10}{'final acc':>11}{'final loss':>12}{'grad min':>11}{'grad max':>11}")
    for arm in a.arms:
        spec = ARMS[arm]
        cc = ClassifierConfig(architecture="fc", depth=a.depth,
                              init_strategy=spec["init_strategy"], use_batch_norm=False,
                              fc_hidden_dim=a.width, relu_shift=spec["relu_shift"],
                              relu_shift_detach=spec["detach"])
        tc = TrainingConfig(dataset=a.dataset, batch_size=256, eval_batch_size=256,
                            optimizer="sgd", learning_rate=a.lr, momentum=0.0,
                            weight_decay=0.0, scheduler="none", epochs=a.epochs,
                            diagnostics_every=1, log_grad_per_layer=True,
                            normalize_inputs=True, abort_on_explosion=True)
        ec = ExperimentConfig(seed=a.seed, device="cpu", data_dir=str(ROOT / "data"))
        ec.setup_seeds()
        res = run_supervised_experiment(ec, cc, tc)
        hist = [dict(h.__dict__) if hasattr(h, "__dict__") else dict(h) for h in res.history]
        per_layer = hist[-1].get("grad_norm_per_layer") or []
        rec = {
            "init_strategy": spec["init_strategy"], "shift_c": spec["relu_shift"],
            "relu_shift_detach": spec["detach"],
            "eval_train_accuracy": [h.get("eval_train_accuracy") for h in hist],
            "eval_train_loss": [h.get("eval_train_loss") for h in hist],
            "test_accuracy": [h.get("test_accuracy") for h in hist],
            "final_grad_norm_per_layer": per_layer,
            "grad_norm_min": min(per_layer) if per_layer else None,
            "grad_norm_max": max(per_layer) if per_layer else None,
            "status": getattr(res, "status", None),
            "abort_reason": getattr(res, "abort_reason", None),
        }
        out["arms"][arm] = rec
        gmin = rec["grad_norm_min"] or float("nan")
        gmax = rec["grad_norm_max"] or float("nan")
        print(f"{arm:<12}{spec['init_strategy']:<26}"
              f"{(spec['relu_shift'] if spec['relu_shift'] is not None else float('nan')):>8.2f}"
              f"{str(spec['detach']):>8}{rec['eval_train_accuracy'][0]:>10.4f}"
              f"{rec['eval_train_accuracy'][-1]:>11.4f}{rec['eval_train_loss'][-1]:>12.4f}"
              f"{gmin:>11.2e}{gmax:>11.2e}", flush=True)

    ds_tag = "fmnist" if a.dataset == "fashion_mnist" else a.dataset
    path = ROOT / "reports" / "results" / f"relushift_local_pretriage_{a.depth}L_{ds_tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
