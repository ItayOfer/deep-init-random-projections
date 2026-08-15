#!/usr/bin/env python3
"""Campaign 11: post-ReLU DC removal -- a = relu(Wx) - c * rms(relu(Wx)) -- on
top of UNCONSTRAINED He weights, versus he and row_centered_he baselines.

The idea. ReLU output is non-negative, so E[a] = rms(a)/sqrt(pi) > 0. That
positive DC component is shared by every sample and is the engine of the
arc-cosine kernel's rho -> 1 collapse. Subtracting it after the ReLU attacks
the mechanism directly, and -- unlike row-centering -- leaves the weights at
full He, paying neither the (1-1/d) variance penalty nor the rank-one row
constraint.

This is the ACTIVATION-SPACE DUAL of row-centering, and the identity is exact:
the next layer computes W(a - s*1) = W a - s*(W 1), so on a row-centered weight
(W 1 = 0) the shift is precisely a no-op. Verified numerically in
scripts/relu_shift_duality_check.py (row_centered_he: relative loss difference
0.0; he: O(1)).

Closed form behind the c grid (derived in cluster/11_relu_shift/README.md,
screened in scripts/relu_shift_geometry_screen.py). With A(c) = c^2 - 2c/sqrt(pi):
  * per-layer forward gain  G(c) = sqrt(1 + A(c)), minimised at c = 1/sqrt(pi)
    where G = sqrt((pi-1)/pi) = r ~ 0.8256 -- EXACTLY row-centering's forward
    gain, which is the duality showing up as a number;
  * cosine recursion rho -> (g(rho) + A)/(1 + A) with g the arc-cosine kernel,
    whose attracting fixed point rho*(c) is 0 exactly at c = 1/sqrt(pi).
G(c) < 1 for every c in (0, 2/sqrt(pi)), so ANY DC removal costs forward scale;
that tension is what this campaign measures under real training.

Arms (`he` and `row_centered_he` are run at these exact settings, not cited
from older campaigns with different recipes):
  * he    -- plain He, no shift (the dying-neurons baseline: 34-48% dataset-dead)
  * rc    -- row_centered_he, no shift (the weight-space dual)
  * c010  -- He + shift c = 0.10  (best distance-correlation-to-input at 30L)
  * c025  -- He + shift c = 0.25  (healthy forward scale, beats He on dist-corr)
  * c070  -- He + shift c = 0.70  (best geometry at 30L/60L; cos 0.089-0.137)

THE DETACHED / DIFFERENTIABLE FORK. rms(a) carries gradient, so the shift's
Jacobian is I - (c/(N*rms)) * 1 a^T -- a rank-one term that couples every unit
AND every sample in the batch. Detached, the shift is a pure per-layer additive
bias and the backward pass is bit-identical to plain He backprop. The default
here is DETACHED (`relu_shift_detach=True`), because that is the exact dual of
row-centering (a forward-side-only intervention) and keeps the comparison
clean; labels suffixed `_diff` run the differentiable variant as a control.

BATCH DEPENDENCE (watch item). rms is one scalar over the whole (batch x units)
tensor, so like BatchNorm the forward pass depends on the batch -- but unlike
BatchNorm there are no running statistics, so eval uses the eval batch's own
rms. batch_size == eval_batch_size == 256 here, which keeps train and eval
like-for-like; do not change one without the other.

Fixed knobs (minimal, so the arm is the only variable), matching campaign 10:
  * NoBN, width 500, depth 30 or 100
  * optimizer = SGD, momentum 0, weight_decay 0, scheduler none (fixed LR)
  * lr = 1e-2, bs = 256, seed = 42
  * NO gradient clipping (assert-enforced)

Pass criterion (advisor, 2026-08-15, campaign-10-onward): eval_train_accuracy
>= 0.99; the loss condition is dropped. Both are logged regardless.

Output: reports/results/relushift_<arm>_<mode>_<dataset>_<depth>L[_diff].json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from rp_study.models.classifiers import build_classifier
from run_diagnostic import _result_to_payload, print_diagnostic_summary

WIDTH = 500
LEARNING_RATE = 1e-2
DC = 1.0 / math.sqrt(math.pi)          # 0.5642, the exact DC constant
R = math.sqrt((math.pi - 1.0) / math.pi)  # 0.8256 = G(1/sqrt(pi)), campaign-09's r

ARMS = {
    "he":   {"init_strategy": "he", "relu_shift": None},
    "rc":   {"init_strategy": "row_centered_he", "relu_shift": None},
    "c010": {"init_strategy": "he", "relu_shift": 0.10},
    "c025": {"init_strategy": "he", "relu_shift": 0.25},
    "c070": {"init_strategy": "he", "relu_shift": 0.70},
}
DATASETS = {"fmnist": "fashion_mnist", "cifar10": "cifar10"}
DEPTHS = (30, 100)
PASS_ACCURACY = 0.99   # advisor 2026-08-15; loss condition dropped

# Smoke grid: the arms the local screen justifies carrying to training, sized to
# campaign 10 (16 smoke jobs) plus 2 fork controls. See README.md "What ran".
#
# 2026-08-15: the 100L row was widened from ("he","rc","c025") to ALL arms.
# 100 layers is the thesis's canonical depth -- it is where geometric collapse
# actually bites -- and the 30L smokes showed the shift arms BEAT He there, with
# c=0.10 the winner. c010 had no 100L cell at all, so the one arm that won at 30L
# could not be tested at the depth the thesis is about.
#
# An earlier decision skipped the 100L cells on the premise that end-to-end
# training at 100L fails for every initialization. That premise is false: He
# trains at 100L NoBN (reports/results/recovery_plain_sgd_fmnist_100L_nobn.json,
# train 0.9953 / test 0.8633 at 152 ep). What fails at 100L is the ROW-CENTERED
# family (rcfwd_rescale_audit_fmnist_100L: 0.1679 / 0.1672). So He is a real
# baseline to beat at 100L, not a floor of chance, and the comparison is sharp.
#
# rcfwd is not an arm here; its 100L end-to-end result already exists in
# campaign 09 (rcfwd_rescale_audit_*_100L) -- cite it rather than re-running.
SMOKE_CELLS = (
    [("30L", arm, ds) for arm in ARMS for ds in DATASETS]
    + [("100L", arm, ds) for arm in ARMS for ds in DATASETS]
)
EXPERIMENT_LABELS = (
    [f"relushift_{arm}_{mode}_{ds}_{depth}"
     for depth, arm, ds in SMOKE_CELLS for mode in ("smoke", "audit")]
    + [f"relushift_c025_{mode}_{ds}_30L_diff"
       for ds in DATASETS for mode in ("smoke", "audit")]
)


def _parse_label(label: str) -> Tuple[str, str, str, int, bool]:
    """relushift_<arm>_<mode>_<dataset>_<depth>L[_diff]"""
    differentiable = label.endswith("_diff")
    core = label[: -len("_diff")] if differentiable else label
    parts = core.split("_")
    assert parts[0] == "relushift", label
    arm, mode, dataset, depth_token = parts[1], parts[2], parts[3], parts[4]
    assert depth_token.endswith("L"), label
    depth = int(depth_token[:-1])
    assert arm in ARMS and mode in ("smoke", "audit"), label
    assert dataset in DATASETS and depth in DEPTHS, label
    return arm, mode, dataset, depth, differentiable


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float
           ) -> Tuple[ClassifierConfig, TrainingConfig]:
    arm, mode, dataset_key, depth, differentiable = _parse_label(label)
    arm_cfg = ARMS[arm]

    cc = ClassifierConfig(
        architecture="fc", depth=depth, init_strategy=arm_cfg["init_strategy"],
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        relu_shift=arm_cfg["relu_shift"],
        relu_shift_detach=not differentiable,
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
        target_train_accuracy=None,
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
    args = parser.parse_args()

    label = args.experiment
    arm, mode, dataset_key, depth, differentiable = _parse_label(label)
    epochs = args.epochs if args.epochs is not None else (20 if mode == "smoke" else 200)
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
    )

    c = classifier_config.relu_shift
    gain = math.sqrt(1.0 - 2.0 * c / math.sqrt(math.pi) + c * c) if c is not None else None

    # Throwaway model for the banner; run_supervised_experiment builds its own,
    # identically seeded, copy for the actual run.
    banner_model = build_classifier(classifier_config)
    n_params = sum(p.numel() for p in banner_model.parameters())
    shift_live = getattr(banner_model, "relu_shift", "MISSING")
    del banner_model

    print(f"\n{'#'*66}")
    print(f"# {label}  (mode={mode}, arm={arm}, depth={depth}L)")
    print(f"# init={classifier_config.init_strategy}  relu_shift={c}  "
          f"detach={classifier_config.relu_shift_detach}")
    if c is not None:
        print(f"# analytic per-layer forward gain G(c)={gain:.4f}  "
              f"=> G(c)^{depth} = {gain**depth:.3e}  (r={R:.4f}, 1/sqrt(pi)={DC:.4f})")
    print(f"# model.relu_shift={shift_live} (must equal relu_shift; MISSING => stale bytecode)")
    print(f"# dataset={training_config.dataset} width={classifier_config.fc_hidden_dim} "
          f"bn={classifier_config.use_batch_norm} params={n_params:,}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size} eval_bs={training_config.eval_batch_size}")
    print(f"# scheduler={training_config.scheduler} (fixed LR, no patience)")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None)")
    print(f"# epochs={epochs} diag_every=1  pass: eval_train_accuracy >= {PASS_ACCURACY}")
    print(f"# output={output_path}")
    print(f"{'#'*66}\n", flush=True)

    assert training_config.grad_clip_max_norm is None, (
        "grad_clip_max_norm must be None for this experiment"
    )
    assert shift_live == c, (
        f"model.relu_shift={shift_live!r} != config {c!r} -- the forward-pass flag did not "
        f"reach the model. Clear __pycache__ under src/ and cluster/ and resubmit."
    )
    assert training_config.batch_size == training_config.eval_batch_size, (
        "the shift's rms is a batch statistic; train and eval batch sizes must match"
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
