#!/usr/bin/env python3
"""Campaign 11 gate: prove that `relu_shift=None` is a bit-exact no-op.

`ClassifierConfig.relu_shift` / `relu_shift_detach` and the corresponding branch
in `DeepFCClassifier.forward` are the only src changes campaign 11 makes, and
they sit inside the hot path of every FC experiment the project has ever run.
Before any campaign-11 result is allowed to mean anything, this has to hold:
with `relu_shift=None` the new code must be BITWISE identical to the code on
`main`, not merely close.

Two levels, both run here:

  PART A (tensor level) -- build DeepFCClassifier from the reference revision
    and from the working tree with the same seed, run one forward + backward +
    SGD step on identical input, and require torch.equal on the output, the
    loss, every parameter gradient and every parameter after the step. A control
    with relu_shift=0.25 must CHANGE the output, otherwise "identical" would be
    vacuous (e.g. if the flag were silently ignored).

  PART B (metric level) -- run one real campaign-10 configuration end to end for
    one epoch under both revisions and require every field of the logged history
    record to match exactly.

Cross-device caveat, stated because it limits what Part B can claim: the
committed campaign-10 JSONs were produced on CUDA and this runs on CPU. A 100L
stack does not reproduce bitwise across devices, so Part B compares OLD vs NEW
*on the same machine* -- which is the actual no-op claim -- and separately
reports the committed CUDA values for reference.

Usage:  python scripts/relu_shift_noop_check.py [--ref main] [--part both]
Output: reports/results/relu_shift_noop_verification.json
"""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# The campaign-10 label Part B replays, and the committed CUDA values it is
# compared against (reports/results/<label>.json, [0]["history"][0]).
REF_LABEL = "rcfrozen_first3_smoke_fmnist_100L"
REF_RECIPE = {"init_strategy": "row_centered_he", "grad_rescale": None,
              "trainable_layers": ["fc1", "fc2", "fc3"], "depth": 100,
              "fc_hidden_dim": 500}


def export_reference_src(ref: str) -> Path:
    """git archive <ref> src -> a temp dir, so the reference code can be imported."""
    tmp = Path(tempfile.mkdtemp(prefix="relu_shift_noop_"))
    archive = subprocess.run(["git", "archive", ref, "src"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=archive, check=True)
    return tmp / "src"


def load_modules(src_path: Path):
    for mod in [m for m in list(sys.modules) if m.startswith("rp_study")]:
        del sys.modules[mod]
    sys.path.insert(0, str(src_path))
    cfg = importlib.import_module("rp_study.config")
    cls = importlib.import_module("rp_study.models.classifiers")
    sup = importlib.import_module("rp_study.experiments.supervised_training")
    sys.path.pop(0)
    return cfg, cls, sup


def part_a(src_path, extra, depth, width, seed=42):
    cfg, cls, _ = load_modules(src_path)
    kw = dict(architecture="fc", depth=depth, init_strategy=REF_RECIPE["init_strategy"],
              use_batch_norm=False, fc_hidden_dim=width, fc_input_dim=784,
              trainable_layers=REF_RECIPE["trainable_layers"])
    kw.update(extra)
    torch.manual_seed(seed)
    model = cls.build_classifier(cfg.ClassifierConfig(**kw))
    g = torch.Generator().manual_seed(7)
    x = torch.randn(64, 784, generator=g)
    y = torch.randint(0, 10, (64,), generator=g)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    return {"out": out.detach(), "loss": loss.detach(), "grads": grads, "params": params}


def part_b(src_path, extra, epochs, depth, width, data_dir, seed=42):
    cfg, _, sup = load_modules(src_path)
    kw = dict(architecture="fc", depth=depth, init_strategy=REF_RECIPE["init_strategy"],
              use_batch_norm=False, fc_hidden_dim=width,
              trainable_layers=REF_RECIPE["trainable_layers"])
    kw.update(extra)
    cc = cfg.ClassifierConfig(**kw)
    tc = cfg.TrainingConfig(
        dataset="fashion_mnist", batch_size=256, optimizer="sgd", learning_rate=1e-2,
        momentum=0.0, weight_decay=0.0, scheduler="none", epochs=epochs,
        diagnostics_every=1, log_grad_per_layer=True, normalize_inputs=True,
    )
    ec = cfg.ExperimentConfig(seed=seed, device="cpu", data_dir=data_dir)
    ec.setup_seeds()
    res = sup.run_supervised_experiment(ec, cc, tc)
    return [dict(h.__dict__) if hasattr(h, "__dict__") else dict(h) for h in res.history]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default="main", help="git revision to compare against")
    p.add_argument("--part", default="both", choices=["a", "b", "both"])
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    a = p.parse_args()

    torch.use_deterministic_algorithms(True)
    ref_src = export_reference_src(a.ref)
    new_src = ROOT / "src"
    rev = subprocess.run(["git", "rev-parse", "--short", a.ref], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    payload = {"description": "Bit-exact no-op verification for ClassifierConfig.relu_shift",
               "config": {"reference_revision": f"{a.ref} ({rev})", "depth": a.depth,
                          "width": a.width, "device": "cpu",
                          "reference_label": REF_LABEL}}
    ok = True

    if a.part in ("a", "both"):
        old = part_a(ref_src, {}, a.depth, a.width)
        new = part_a(new_src, {"relu_shift": None}, a.depth, a.width)
        ctl = part_a(new_src, {"relu_shift": 0.25}, a.depth, a.width)
        identical = (torch.equal(old["out"], new["out"])
                     and torch.equal(old["loss"], new["loss"])
                     and set(old["grads"]) == set(new["grads"])
                     and all(torch.equal(old["grads"][k], new["grads"][k]) for k in old["grads"])
                     and all(torch.equal(old["params"][k], new["params"][k]) for k in old["params"]))
        control_changes = not torch.equal(old["out"], ctl["out"])
        ok = ok and identical and control_changes
        payload["part_a_tensor_level"] = {
            "bitwise_identical": bool(identical),
            "control_relu_shift_0.25_changes_output": bool(control_changes),
            "n_grad_tensors_compared": len(old["grads"]),
            "n_param_tensors_compared": len(old["params"]),
            "loss_reference": old["loss"].item(),
            "loss_new_relu_shift_none": new["loss"].item(),
            "loss_control_relu_shift_025": ctl["loss"].item(),
        }
        print(f"PART A  bitwise identical = {identical}   control changes output = {control_changes}")
        print(f"        {len(old['grads'])} grad tensors, {len(old['params'])} param tensors, "
              f"depth {a.depth} width {a.width}")

    if a.part in ("b", "both"):
        old_h = part_b(ref_src, {}, a.epochs, a.depth, a.width, a.data_dir)
        new_h = part_b(new_src, {"relu_shift": None}, a.epochs, a.depth, a.width, a.data_dir)
        mismatches = []
        for i, (ho, hn) in enumerate(zip(old_h, new_h)):
            for k, vo in ho.items():
                if hn.get(k) != vo:
                    mismatches.append(f"epoch{i+1}.{k}")
        identical_b = not mismatches
        ok = ok and identical_b
        committed = None
        ref_json = ROOT / "reports" / "results" / f"{REF_LABEL}.json"
        if ref_json.exists():
            committed = {k: v for k, v in json.loads(ref_json.read_text())[0]["history"][0].items()
                         if not isinstance(v, list)}
        scalar = {k: v for k, v in new_h[0].items() if not isinstance(v, list)}
        payload["part_b_metric_level"] = {
            "history_bitwise_identical": bool(identical_b),
            "mismatched_fields": mismatches,
            "epochs_compared": len(old_h),
            "new_code_first_epoch_cpu": scalar,
            "committed_first_epoch_cuda": committed,
            "note": "OLD vs NEW is the no-op claim and is bitwise. The committed JSON was "
                    "produced on CUDA; a 100L stack whose logits are O(1e-6) does not "
                    "reproduce accuracy bitwise across devices, so the committed row is "
                    "reference, not an equality assertion.",
        }
        print(f"PART B  history bitwise identical = {identical_b}"
              + (f"   mismatches: {mismatches}" if mismatches else ""))
        print(f"        new (cpu)      : {scalar}")
        if committed:
            print(f"        committed (cuda): {committed}")

    payload["verdict"] = "PASS" if ok else "FAIL"
    out = ROOT / "reports" / "results" / "relu_shift_noop_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    shutil.rmtree(ref_src.parent, ignore_errors=True)
    print(f"\nVERDICT {payload['verdict']}   saved {out.relative_to(ROOT)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
