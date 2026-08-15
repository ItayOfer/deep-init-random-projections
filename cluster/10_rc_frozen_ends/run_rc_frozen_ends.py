#!/usr/bin/env python3
"""100L row-centered, train only a small window of layers -- at the tail
(last3 = fc98-fc100, or last2 = fc99-fc100) or at the head (first3 =
fc1-fc3, or first2 = fc1-fc2) -- everything else, INCLUDING THE HEAD, frozen
in every condition. Where can trainable signal enter a deep row-centered
network, and does the answer survive isolating the rcfwd recipe's two
independent interventions?

Symmetric by design: the head (the final Linear(hidden_dim, num_classes)
readout) is never part of any trainable window -- it is always the fixed
lens the result is read through, in every condition. This makes e.g. last2
and first2 a like-for-like comparison of "a trainable window at one end of
a 100-layer frozen stack" vs "at the other end", with nothing else
differing. (An earlier draft of this runner trained fc99+fc100+head for
last3 -- an asymmetric definition, since first3 never trained the head --
corrected before the original 3-layer pass; see
cluster/10_rc_frozen_ends/README.md for the note on what changed and why.)

Motivation: campaign 09 (rcfwd) showed row-centered representation content
dies by layer ~25 (probe chain) while the backward gain is ~unit scale
(g_bwd~1.0, uncorrected here -- this is plain row_centered_he, no GradRescale).
Forward gain g_fwd~0.826 per layer means the tail of the network sits on a
signal shrunk by ~0.826^97 ~ 1e-8. This campaign surgically tests the two
entry points: does gradient reach a trainable window sitting on dead/
vanishing content (last2/last3), or a trainable window whose output must
still survive ~97-98 frozen scrambling layers forward before reaching the
frozen head (first2/first3)?

Freezing = requires_grad=False on the frozen Linear layers' parameters
(ClassifierConfig.trainable_layers, see src/rp_study/config.py). Backprop
still runs through frozen layers unchanged -- only their own parameter
gradients are suppressed. Verified locally (CPU, small-sample smoke) before
every runner change here was ever synced to the cluster.

Four RECIPES -- the rcfwd recipe decomposed into its two independent
interventions, (A) the init change and (B) the backward grad_rescale (see
reports/results/recipe_decomposition_funnel.json and
cluster/10_rc_frozen_ends/README.md "H1 vs H2" / the 2x2 ablation section):
  * "raw"        -- row_centered_he, no grad_rescale (neither A nor B; the
                    original pass). Backward gain here is ~unit scale
                    already (g_bwd~1.0); the failure modes observed are
                    forward-scale decay (last3) and content absorption
                    (first3), NOT gradient magnitude/direction.
  * "rawrescale" -- row_centered_he + grad_rescale=r (B only -- "correction
                    for the backward only", advisor follow-up 2026-08-15).
                    _GradRescale is identity forward, so activations here
                    are BIT-IDENTICAL to "raw"; at init this is the flattest
                    gradient profile of all four corners (1.13x across 100
                    layers) but pinned at ~9e-9 ~ r^100 -- the standing
                    lr=1e-2 cannot move it, hence the LR ladder below.
  * "fwdbal"     -- row_centered_forward_balanced, no grad_rescale (A only).
                    Forward-flat by construction (activation RMS ~1.26 at
                    L100) but backward gain compounds at 1/r~1.21/layer
                    uncancelled -- grad_norm_per_layer hits ~4.6e8 at layer 1
                    (fc1) at init, decaying monotonically to ~3.4 by layer
                    100 (fc100): the blow-up is concentrated at the FRONT,
                    where the most 1/r factors have compounded. NOTE this is
                    layer-POSITION dependent, which matters once layers are
                    frozen: autograd never computes a gradient for a frozen
                    tensor's frozen input (requires_grad=False stops
                    backprop there), so a first-window run (first2/first3)
                    is exposed to that ~4.6e8 and is expected to abort
                    (NaN/Inf loss) almost immediately -- verified locally:
                    first2/fwdbal NaNs at epoch 1 batch 4. A last-window run
                    (last2/last3) only ever computes fc99/fc100's OWN
                    gradient (~3-4 at init, order-1 like He) and never
                    touches the frozen fc1-fc98 chain at all -- verified
                    locally: last2/fwdbal does NOT abort over 5 tiny CPU
                    epochs (loss falls steadily, no explosion). The campaign
                    grid still runs last2/fwdbal per the brief (a documented
                    non-abort is itself informative for the 2x2 story); see
                    README for the full writeup of this brief-vs-measurement
                    discrepancy.
  * "rcfwd"      -- row_centered_forward_balanced + grad_rescale=r (A and B;
                    campaign 09's exact corrected recipe: flat forward,
                    backward gain compounding at 1/r~1.21/layer cancelled by
                    the rescale).
  r = sqrt((pi-1)/pi) ~ 0.826 throughout.

rawrescale LR ladder: raw+rescale's gradients are pinned at ~9e-9 by
construction, not as a training-instability symptom (see
recipe_decomposition_funnel.json) -- so the standing lr=1e-2 is a control
point expected to be inert here, not "the" LR for this corner.
RAWRESCALE_LR_LADDER below defines the additional rungs (analytically-matched
LR ~ 1e-2/r^100 ~ 2e6); each rung is its own fully-qualified label (see
RAWRESCALE_LADDER_LABELS and _strip_lr_ladder_tag). This is a documented free
knob to reach a construction-pinned gradient scale, NOT the forbidden
"tune away instability" pattern -- see cluster/10_rc_frozen_ends/README.md.

Four CONDITIONS: last3/first3 (the original 3-layer windows, kept exactly as
they were so the 8 committed JSONs stay reproducible) and last2={fc99,fc100}
/first2={fc1,fc2} (2026-08-15 advisor follow-up: same symmetric design, one
layer narrower, head frozen in both).

--target-train-accuracy wires TrainingConfig.target_train_accuracy (default
None -- unchanged existing behavior: run the full --epochs budget). When set,
training stops once eval_train_accuracy (target_metric's default) holds at or
above it for --target-patience epoch(s). Used for the target-accuracy runs on
last2/last3 under rcfwd -- the campaign's actual deliverable (see README).

Fixed knobs (minimal, so freezing-location and recipe are the only variables
besides the documented rawrescale LR ladder):
  * NoBN, width 500, depth 100
  * optimizer = SGD, momentum 0, weight_decay 0, scheduler none (fixed LR)
  * lr = 1e-2 by default (overridable via --lr; see rawrescale LR ladder)
  * bs = 256, seed = 42
  * NO gradient clipping (assert-enforced)

Base grid: 4 conditions x 2 datasets x 4 recipes x {smoke 20 ep, audit 200 ep}
= 64 valid --experiment labels, plus 16 rawrescale-LR-ladder smoke labels
(4 non-control rungs x 2 conditions[first2,last2] x 2 datasets) = 80 total.
Only a curated subset is ever actually submitted -- see
cluster/10_rc_frozen_ends/README.md for exactly which, and in what order.
Output: reports/results/<label>.json (label = SLURM job name = JSON stem,
always -- see _parse_label / _strip_lr_ladder_tag for how a label decodes).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cluster" / "03_he_diagnostics"))

from rp_study.config import ClassifierConfig, ExperimentConfig, TrainingConfig
from rp_study.experiments.supervised_training import run_supervised_experiment
from rp_study.models.classifiers import build_classifier
from run_diagnostic import _result_to_payload, print_diagnostic_summary

WIDTH = 500
DEPTH = 100
LEARNING_RATE = 1e-2
R = math.sqrt((math.pi - 1.0) / math.pi)   # ~0.826 = 1/1.21, campaign-09's rescale

RECIPES = {
    "raw": {"init_strategy": "row_centered_he", "grad_rescale": None},
    "rcfwd": {"init_strategy": "row_centered_forward_balanced", "grad_rescale": R},
    # (B)-only: correction for the backward only, forward left at its natural
    # 0.826^l decay. Bit-identical activations to "raw" (_GradRescale is
    # identity forward) -- see module docstring + README "rawrescale" section.
    "rawrescale": {"init_strategy": "row_centered_he", "grad_rescale": R},
    # (A)-only: forward-balanced init, no backward correction. Expected to
    # abort on explosion -- see module docstring.
    "fwdbal": {"init_strategy": "row_centered_forward_balanced", "grad_rescale": None},
}

TRAINABLE_LAYERS = {
    "first3": ["fc1", "fc2", "fc3"],
    "last3": ["fc98", "fc99", "fc100"],
    # 2026-08-15 advisor follow-up: same symmetric design, one layer
    # narrower. Head stays frozen in both, exactly as for first3/last3.
    "first2": ["fc1", "fc2"],
    "last2": ["fc99", "fc100"],
}
DATASETS = {
    "fmnist": "fashion_mnist",
    "cifar10": "cifar10",
}
CONDITION_KEYS = list(TRAINABLE_LAYERS.keys())   # first3, last3, first2, last2
DATASET_KEYS = list(DATASETS.keys())             # fmnist, cifar10
RECIPE_SUFFIX = {
    "raw": "",
    "rcfwd": "_rcfwd",
    "rawrescale": "_rawrescale",
    "fwdbal": "_fwdbal",
}
# _parse_label must try these LONGEST-SUFFIX-FIRST. "raw"'s suffix is "",
# which label.endswith("") matches for every possible label -- trying it
# first (or in RECIPE_SUFFIX's plain insertion order) would silently
# classify every "..._rcfwd" / "..._rawrescale" / "..._fwdbal" label as
# "raw" too. Sorting by suffix length descending puts "raw" last, always.
_RECIPE_SUFFIX_ORDER = sorted(
    RECIPE_SUFFIX, key=lambda name: len(RECIPE_SUFFIX[name]), reverse=True
)

EXPERIMENT_LABELS = [
    f"rcfrozen_{cond}_{mode}_{ds}_100L{RECIPE_SUFFIX[recipe]}"
    for recipe in RECIPES
    for cond in CONDITION_KEYS
    for mode in ("smoke", "audit")
    for ds in DATASET_KEYS
]

# --- rawrescale LR ladder (brief 2026-08-15, "Constraints") -----------------
# raw+rescale pins grad_norm_per_layer at ~9e-9 ~ r^100 by construction
# (reports/results/recipe_decomposition_funnel.json, corners["raw+rescale"];
# flat to 1.13x across all 100 layers), so the campaign's standing lr=1e-2
# cannot move it. The control rung (lr=1e-2) IS the bare "..._rawrescale"
# smoke label already generated above -- same default LR as every other
# recipe, no separate entry needed. These are the additional ladder rungs.
# Each rung gets its own fully-qualified label (job name = JSON stem, same
# convention as every other label here); the .sub file passes the matching
# --lr explicitly rather than asking _parse_label to recover a number from
# text (see _strip_lr_ladder_tag).
RAWRESCALE_LR_LADDER: Dict[str, float] = {
    "lr1e2": 1e2,
    "lr1e4": 1e4,
    "lr1e6": 1e6,
    "lr1e7": 1e7,
}
RAWRESCALE_LADDER_LABELS: Dict[str, float] = {
    f"rcfrozen_{cond}_smoke_{ds}_100L_rawrescale_{tag}": lr
    for cond in ("first2", "last2")
    for ds in DATASET_KEYS
    for tag, lr in RAWRESCALE_LR_LADDER.items()
}
EXPERIMENT_LABELS = EXPERIMENT_LABELS + sorted(RAWRESCALE_LADDER_LABELS)


def _strip_lr_ladder_tag(label: str) -> str:
    """Strip a trailing rawrescale LR-ladder tag ('_lr1e2' etc.), if present.

    The numeric LR is never recovered FROM this tag -- the .sub file passes
    it explicitly via --lr, exactly like every other numeric knob in this
    runner (see RAWRESCALE_LR_LADDER). The tag exists only so each ladder
    rung gets its own job-name/JSON-stem; stripping it here lets the rest of
    the label parse through _parse_label completely unchanged.
    """
    for tag in RAWRESCALE_LR_LADDER:
        suffix = f"_{tag}"
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _parse_label(label: str) -> Tuple[str, str, str, str]:
    """rcfrozen_<condition>_<mode>_<dataset>_100L[<recipe_suffix>]

    Recipe suffixes are tried LONGEST-FIRST via _RECIPE_SUFFIX_ORDER -- see
    the comment there for why "raw" (suffix "") must be tried last. A silent
    mis-parse here would run the wrong recipe under the right filename with
    no error: the single worst failure mode available in this campaign
    (brief 2026-08-15). _check_label_roundtrips() below exercises every
    EXPERIMENT_LABELS entry through this function at import time as a
    standing guard against that regressing.

    Callers with a possibly LR-ladder-tagged label must call
    _strip_lr_ladder_tag() first -- this function does not know about LR
    tags at all, by design (see _strip_lr_ladder_tag docstring).
    """
    for recipe in _RECIPE_SUFFIX_ORDER:
        suffix = RECIPE_SUFFIX[recipe]
        if suffix and label.endswith(suffix):
            core = label[: -len(suffix)]
            break
    else:
        recipe = "raw"
        core = label

    parts = core.split("_")
    assert parts[0] == "rcfrozen" and parts[-1] == "100L" and len(parts) == 5, (
        f"unexpected label shape: {label!r} -> core={core!r} parts={parts!r}"
    )
    condition, mode, dataset = parts[1], parts[2], parts[3]
    assert condition in TRAINABLE_LAYERS, f"{label!r}: unknown condition {condition!r}"
    assert mode in ("smoke", "audit"), f"{label!r}: unknown mode {mode!r}"
    assert dataset in DATASETS, f"{label!r}: unknown dataset {dataset!r}"
    assert recipe in RECIPES, f"{label!r}: unknown recipe {recipe!r}"
    return condition, mode, dataset, recipe


def _check_label_roundtrips() -> None:
    """Self-test, run at import time (see call at module scope below): every
    label in EXPERIMENT_LABELS must parse -- after stripping any rawrescale
    LR-ladder tag -- back to exactly the components used to build it.

    This is the campaign's highest-risk edit (brief 2026-08-15): running it
    unconditionally at import means a mis-parse fails loudly the moment this
    module is imported (including on --help), before any cluster time is
    ever spent on it.
    """
    for label in EXPERIMENT_LABELS:
        core = _strip_lr_ladder_tag(label)
        condition, mode, dataset, recipe = _parse_label(core)
        rebuilt = f"rcfrozen_{condition}_{mode}_{dataset}_100L{RECIPE_SUFFIX[recipe]}"
        assert rebuilt == core, (
            f"_parse_label round-trip FAILED: label={label!r} core={core!r} "
            f"parsed=(condition={condition!r}, mode={mode!r}, dataset={dataset!r}, "
            f"recipe={recipe!r}) rebuilt={rebuilt!r} != core"
        )


_check_label_roundtrips()


def _build(label: str, epochs: int, log_per_batch_first_epoch: bool,
           log_grad_per_layer: bool, learning_rate: float,
           target_train_accuracy: Optional[float] = None,
           target_patience: int = 1) -> Tuple[ClassifierConfig, TrainingConfig]:
    """`label` must already be LR-tag-stripped (see _strip_lr_ladder_tag) --
    this calls _parse_label directly and does not know about LR tags."""
    condition, mode, dataset_key, recipe = _parse_label(label)
    dataset = DATASETS[dataset_key]
    trainable_layers = TRAINABLE_LAYERS[condition]
    recipe_cfg = RECIPES[recipe]

    cc = ClassifierConfig(
        architecture="fc", depth=DEPTH, init_strategy=recipe_cfg["init_strategy"],
        use_batch_norm=False, fc_hidden_dim=WIDTH,
        trainable_layers=trainable_layers,
        grad_rescale=recipe_cfg["grad_rescale"],
    )
    tc = TrainingConfig(
        dataset=dataset, batch_size=256,
        optimizer="sgd", learning_rate=learning_rate,
        momentum=0.0, weight_decay=0.0,
        scheduler="none",
        epochs=epochs,
        log_every_epoch=True,
        diagnostics_every=1,
        log_per_batch_first_epoch=log_per_batch_first_epoch,
        log_grad_per_layer=log_grad_per_layer,
        normalize_inputs=True,
        # None (default) is unchanged existing behavior: run the full epochs
        # budget. Set (via --target-train-accuracy) for the target-accuracy
        # runs -- tracks eval_train_accuracy specifically (target_metric),
        # per the brief's "track eval_train_accuracy, not eval_train_loss".
        target_train_accuracy=target_train_accuracy,
        target_patience=target_patience,
        target_metric="eval_train_accuracy",
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
    parser.add_argument(
        "--target-train-accuracy", type=float, default=None,
        help="Stop once eval_train_accuracy holds at/above this value for "
             "--target-patience epoch(s) (TrainingConfig.target_train_accuracy). "
             "Default None preserves existing behavior: run the full --epochs "
             "budget. Used for the target-accuracy runs (see README).",
    )
    parser.add_argument(
        "--target-patience", type=int, default=1,
        help="Epochs the target must hold before stopping "
             "(TrainingConfig.target_patience, default 1).",
    )
    args = parser.parse_args()

    label = args.experiment
    # Strip a rawrescale LR-ladder tag (if any) BEFORE parsing -- _parse_label
    # itself never sees LR tags (see _strip_lr_ladder_tag / _parse_label
    # docstrings). `label` (untouched) still drives the output filename and
    # every printed banner, so the LR tag survives into the JSON stem.
    core_label = _strip_lr_ladder_tag(label)
    condition, mode, dataset_key, recipe = _parse_label(core_label)
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
        core_label, epochs=epochs,
        log_per_batch_first_epoch=log_per_batch,
        log_grad_per_layer=log_grad_per_layer,
        learning_rate=args.lr,
        target_train_accuracy=args.target_train_accuracy,
        target_patience=args.target_patience,
    )

    # Build a throwaway model up front just to assert + print the trainable-
    # tensor banner (run_supervised_experiment builds its own, identically
    # seeded, copy for the actual training run).
    banner_model = build_classifier(classifier_config)
    n_trainable = banner_model.trainable_tensor_count()
    expected_layers = len(classifier_config.trainable_layers)
    expected_tensors = 2 * expected_layers  # weight + bias per trainable Linear
    del banner_model

    print(f"\n{'#'*60}")
    print(f"# {label}  (mode={mode}, condition={condition}, recipe={recipe})")
    print(f"# init={classifier_config.init_strategy}  grad_rescale={classifier_config.grad_rescale}")
    print(f"# trainable_layers={classifier_config.trainable_layers}")
    print(f"# trainable tensors={n_trainable} (expect {expected_tensors} = "
          f"{expected_layers} layers x 2 [weight,bias])")
    print(f"# dataset={training_config.dataset} depth={classifier_config.depth} "
          f"width={classifier_config.fc_hidden_dim} bn={classifier_config.use_batch_norm}")
    print(f"# optimizer={training_config.optimizer} lr={training_config.learning_rate} "
          f"momentum={training_config.momentum} wd={training_config.weight_decay} "
          f"bs={training_config.batch_size}")
    print(f"# scheduler={training_config.scheduler} (fixed LR, no patience)")
    print(f"# grad_clip_max_norm={training_config.grad_clip_max_norm} (must be None)")
    print(f"# epochs={epochs} diag_every=1 (per-layer grads every epoch)")
    print(f"# target_train_accuracy={training_config.target_train_accuracy} "
          f"target_patience={training_config.target_patience} "
          f"target_metric={training_config.target_metric}")
    print(f"# output={output_path}")
    print(f"{'#'*60}\n", flush=True)

    assert n_trainable == expected_tensors, (
        f"trainable tensor count mismatch: got {n_trainable}, expected {expected_tensors} "
        f"for trainable_layers={classifier_config.trainable_layers}"
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
        # Pass criterion for campaign 10 onward (brief 2026-08-15, W5):
        # eval_train_accuracy >= 0.99, loss condition dropped. Supersedes
        # CLAUDE.md's >=0.995 AND loss<=0.10 rule for this campaign only --
        # does not touch the historical 12-architecture He/V2 PASS counts,
        # which stay reported on the old criterion (see brief Constraints).
        passes = final_acc >= 0.99
        flag = "PASS" if passes else "fail"
        print(f"\nSUMMARY {label} | {flag} (acc>=0.99, no loss condition) | "
              f"best_train={best_acc:.4f} "
              f"final_train={final_acc:.4f} final_loss={final_loss:.4f} "
              f"final_test={final_test:.4f} epochs_ran={len(hist)}")
    else:
        print(f"\nSUMMARY {label} | NO_HISTORY (aborted before first epoch completed)")


if __name__ == "__main__":
    main()
