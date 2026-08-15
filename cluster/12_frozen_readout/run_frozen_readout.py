#!/usr/bin/env python3
"""Rank initializations by how much class content survives to layer 98 of a
100-layer net, measured by a TRAINED 2-layer readout rather than a probe.

Why this campaign exists
------------------------
Campaign 09 concluded that row-centered representation *content* dies with
depth, on the evidence of cosine-kNN and linear probes going to chance by layer
~25. Campaign 10 then trained ONLY fc98-fc100 on top of a frozen 100-layer
row-centered stack (rcfwd recipe) and reached eval_train_accuracy 0.8335
(fmnist) / 0.8170 (cifar10) at epoch ~243, still climbing -- on a
representation the probes called dead.

So the probes understate badly: information can be present without being
linearly or metrically decodable. A trained readout is the honest instrument.
The same comparison also shows end-to-end training is the *worse* protocol at
this depth -- same init, same rescale, same data:

    all 100 layers trainable, 200 ep   ->  0.1746 (fmnist) / 0.1301 (cifar10)
    only fc98-100 trainable,  243 ep   ->  0.8335 (fmnist) / 0.8170 (cifar10)

(reports/results/rcfwd_rescale_audit_{fmnist,cifar10}_100L.json vs
 reports/results/rcfrozen_last3_audit_{fmnist,cifar10}_100L_rcfwd.json)

Freezing the bulk protects the representation; gradient updates to the early
layers destroy it (campaign 10's `first3` cell drove the loss *past* ln 10 at
chance accuracy). So a frozen stack + trained readout isolates the one thing we
actually want to compare across initializations: how much usable class
structure a random deep map leaves at its output.

What this runner does
---------------------
Fixes the readout (fc99, fc100 trainable; EVERYTHING else including the head
frozen -- the exact protocol of campaign 10's `last2` cells) and varies only
the initialization. Higher final eval_train_accuracy = more recoverable content
at depth 100. Arms:

    he     plain He                                   -- the baseline
    rc     row_centered_he                            -- the weight-space DC removal
    rcfwd  row_centered_forward_balanced + grad_rescale -- campaign 09's corrected recipe
    c010   he + post-ReLU shift c=0.10                -- activation-space DC removal
    c025   he + post-ReLU shift c=0.25                --   "  (best init-time geometry)
    c070   he + post-ReLU shift c=0.70                --   "  (best cosine, chance content)

NOTE the `rcfwd` arm is numerically identical to campaign 10's
`rcfrozen_last2_audit_<ds>_100L_rcfwd` (same init, rescale, window, optimizer,
seed, epochs). It is defined here so this campaign is self-contained, but if
those runs are already in flight do NOT resubmit -- cite them instead.

Fixed knobs, matched to campaign 10's last2 audits so the numbers are directly
comparable: NoBN, width 500, depth 100, SGD lr 1e-2 (momentum 0, wd 0,
scheduler none), bs 256, seed 42, NO gradient clipping (assert-enforced).

Pass criterion is the advisor's 2026-08-15 rule: eval_train_accuracy >= 0.99,
loss condition dropped (see CLAUDE.md).

Output: reports/results/frozenro_<arm>_<mode>_<dataset>_100L.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from rp_study.models.classifiers import build_classifier
from run_diagnostic import _result_to_payload, print_diagnostic_summary

WIDTH = 500
DEPTHS = (30, 100)
LEARNING_RATE = 1e-2
PASS_ACCURACY = 0.99
R = math.sqrt((math.pi - 1.0) / math.pi)      # 0.8256


def trainable_layers(depth: int) -> list:
    """The readout: last two hidden layers only, head always frozen.

    Depth-relative so the protocol is identical at every depth -- at 100L this
    is ["fc99","fc100"], byte-identical to the original 100L-only version, so
    the ten committed frozenro_*_100L results remain reproducible.

    Campaign 11 (30L, end-to-end) and campaign 12 (100L, frozen readout)
    disagree about whether DC removal beats He, but they differ in BOTH depth
    and protocol. Running this protocol at 30L too separates the two.
    """
    return [f"fc{depth - 1}", f"fc{depth}"]

ARMS = {
    "he":    {"init_strategy": "he",                            "relu_shift": None, "grad_rescale": None},
    "rc":    {"init_strategy": "row_centered_he",               "relu_shift": None, "grad_rescale": None},
    "rcfwd": {"init_strategy": "row_centered_forward_balanced", "relu_shift": None, "grad_rescale": R},
    "c010":  {"init_strategy": "he",                            "relu_shift": 0.10, "grad_rescale": None},
    "c025":  {"init_strategy": "he",                            "relu_shift": 0.25, "grad_rescale": None},
    "c070":  {"init_strategy": "he",                            "relu_shift": 0.70, "grad_rescale": None},
}
DATASETS = {"fmnist": "fashion_mnist", "cifar10": "cifar10"}

EXPERIMENT_LABELS = [
    f"frozenro_{arm}_{mode}_{ds}_{depth}L"
    for arm in ARMS for mode in ("smoke", "audit")
    for ds in DATASETS for depth in DEPTHS
]


def _parse_label(label: str) -> Tuple[str, str, str, int]:
    """frozenro_<arm>_<mode>_<dataset>_<depth>L -> (arm, mode, dataset, depth)."""
    parts = label.split("_")
    assert parts[0] == "frozenro" and len(parts) == 5 and parts[-1].endswith("L"), (
        f"unexpected label shape: {label!r} parts={parts}"
    )
    arm, mode, dataset = parts[1], parts[2], parts[3]
    depth = int(parts[4][:-1])
    assert arm in ARMS, f"unknown arm {arm!r} in {label!r}"
    assert mode in ("smoke", "audit"), f"unknown mode {mode!r} in {label!r}"
    assert dataset in DATASETS, f"unknown dataset {dataset!r} in {label!r}"
    assert depth in DEPTHS, f"unknown depth {depth!r} in {label!r}"
    return arm, mode, dataset, depth


def _check_label_roundtrips() -> None:
    """Every label must decode to its own components. A silent mis-parse would
    run the wrong ARM under the right filename -- the worst failure available
    here, since the arm IS the independent variable."""
    for label in EXPERIMENT_LABELS:
        arm, mode, dataset, depth = _parse_label(label)
        rebuilt = f"frozenro_{arm}_{mode}_{dataset}_{depth}L"
        assert rebuilt == label, f"round-trip FAILED: {label!r} -> {rebuilt!r}"


_check_label_roundtrips()


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float,
           target_train_accuracy: Optional[float]) -> Tuple[ClassifierConfig, TrainingConfig]:
    arm, _mode, dataset_key, depth = _parse_label(label)
    cfg = ARMS[arm]

    cc = ClassifierConfig(
        architecture="fc", depth=depth, init_strategy=cfg["init_strategy"],
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        trainable_layers=trainable_layers(depth),
        relu_shift=cfg["relu_shift"],
        relu_shift_detach=True,          # the exact dual; see campaign 11 README §6
        grad_rescale=cfg["grad_rescale"],
    )
    tc = TrainingConfig(
        dataset=DATASETS[dataset_key], batch_size=256, eval_batch_size=256,
        optimizer="sgd", learning_rate=learning_rate,
        momentum=0.0, weight_decay=0.0,
        scheduler="none",
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=1,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        log_grad_per_layer=log_grad_per_layer,
        normalize_inputs=True,
        target_train_accuracy=target_train_accuracy,
        abort_on_explosion=True,
    )
    return cc, tc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--target-train-accuracy", type=float, default=None,
                        help=f"early-stop target (advisor's criterion: {PASS_ACCURACY})")
    args = parser.parse_args()

    label = args.experiment
    arm, mode, dataset_key, depth = _parse_label(label)
    epochs = args.epochs if args.epochs is not None else (20 if mode == "smoke" else 400)
    log_per_batch = (mode == "smoke")
    log_grad_per_layer = (mode == "smoke")

    output_path = Path(args.output) if args.output else (
        ROOT / "reports" / "results" / f"{label}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_config = ExperimentConfig(seed=args.seed, device=args.device,
                                  data_dir=str(ROOT / "data"))
    classifier_config, training_config = _build(
        label, epochs=epochs,
        log_per_batch_first_epoch=log_per_batch,
        log_grad_per_layer=log_grad_per_layer,
        learning_rate=args.lr,
        target_train_accuracy=args.target_train_accuracy,
    )

    # Throwaway model purely to assert the freeze + shift actually took effect
    # before burning cluster hours. A stale .pyc that predates the relu_shift
    # field would silently drop the shift and run a plain-He arm under a c=0.25
    # filename -- so check the LIVE attribute, not the config.
    banner_model = build_classifier(classifier_config)
    n_trainable = banner_model.trainable_tensor_count()
    shift_live = getattr(banner_model, "relu_shift", "MISSING")
    expected_tensors = 2 * len(trainable_layers(depth))
    del banner_model

    print(f"\n{'#'*64}")
    print(f"# {label}  (mode={mode}, arm={arm}, dataset={dataset_key}, depth={depth}L)")
    print(f"# init={classifier_config.init_strategy}  relu_shift={classifier_config.relu_shift}  "
          f"grad_rescale={classifier_config.grad_rescale}")
    print(f"# model.relu_shift={shift_live!r} (must equal config; MISSING => stale bytecode)")
    print(f"# trainable_layers={classifier_config.trainable_layers} (head FROZEN)")
    print(f"# trainable tensors={n_trainable} (expect {expected_tensors})")
    print(f"# depth={classifier_config.depth} width={WIDTH} bn={classifier_config.use_batch_norm}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"bs={training_config.batch_size} scheduler={training_config.scheduler}")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None)")
    print(f"# epochs={epochs} target_train_accuracy={training_config.target_train_accuracy}")
    print(f"# output={output_path}")
    print(f"{'#'*64}\n", flush=True)

    assert n_trainable == expected_tensors, (
        f"trainable tensor count mismatch: got {n_trainable}, expected {expected_tensors}"
    )
    assert shift_live == classifier_config.relu_shift, (
        f"model.relu_shift={shift_live!r} != config {classifier_config.relu_shift!r} -- the "
        f"forward-pass flag did not take (stale __pycache__?). Refusing to run."
    )
    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for this experiment"
    )

    exp_config.setup_seeds()
    result = run_supervised_experiment(exp_config, classifier_config, training_config)
    print_diagnostic_summary(label, result)

    if getattr(result, "abort_reason", None):
        print(f"\nABORT_REASON: {result.abort_reason}")

    payload = [_result_to_payload(label, result)]
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved 1 run to {output_path}")

    hist = payload[0].get("history", [])
    if hist:
        best_acc = max(h.get("eval_train_accuracy", 0) or 0 for h in hist)
        final_acc = hist[-1].get("eval_train_accuracy", 0) or 0
        final_loss = hist[-1].get("eval_train_loss", 0) or 0
        final_test = hist[-1].get("test_accuracy", 0) or 0
        flag = "PASS" if final_acc >= PASS_ACCURACY else "fail"
        print(f"\nSUMMARY {label} | {flag} | best_train={best_acc:.4f} "
              f"final_train={final_acc:.4f} final_loss={final_loss:.4f} "
              f"final_test={final_test:.4f} epochs_ran={len(hist)}")
    else:
        print(f"\nSUMMARY {label} | NO_HISTORY (aborted before first epoch completed)")


if __name__ == "__main__":
    main()
