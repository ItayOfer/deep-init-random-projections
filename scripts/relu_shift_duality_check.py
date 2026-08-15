#!/usr/bin/env python3
"""Campaign 11: numerically verify the row-centering <-> post-ReLU-shift duality.

The claim (brief, 2026-08-15): the next layer computes
    W (a - s*1) = W a - s * (W 1)
so if W is ROW-CENTERED (every row sums to zero, W 1 = 0) the post-ReLU shift is
EXACTLY a no-op on the pre-activations -- and therefore on the whole forward
pass, since every weight layer including the classifier head is row-centered
under `row_centered_he`.

This script checks that identity against the real DeepFCClassifier, for a grid
of shift coefficients, and contrasts it with `he` (where W 1 != 0 and the shift
must change the output). It also records what the identity does NOT cover: the
weight GRADIENT is grad_W = delta^T a_prev, and a_prev IS shifted, so
    grad_W(shift) = grad_W(no shift) - s * delta^T 1
which is a rank-one difference that does not vanish under row centering.

Output: reports/results/relu_shift_duality_check.json
"""

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig  # noqa: E402
from rp_study.data.loaders import load_fashion_mnist  # noqa: E402
from rp_study.models.classifiers import build_classifier  # noqa: E402

DC = 1.0 / math.sqrt(math.pi)
DEPTH, WIDTH, SEED, N = 20, 500, 42, 256


def run(init_strategy, relu_shift, detach, X, y):
    cfg = ClassifierConfig(architecture="fc", depth=DEPTH, init_strategy=init_strategy,
                           use_batch_norm=False, fc_hidden_dim=WIDTH,
                           fc_input_dim=X.shape[1], num_classes=10,
                           relu_shift=relu_shift, relu_shift_detach=detach)
    torch.manual_seed(SEED)
    model = build_classifier(cfg)
    model.zero_grad()
    out = model(X)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = [l.weight.grad.detach().clone() for l in model.hidden_layers]
    return out.detach(), loss.detach(), grads


def main():
    X, y = load_fashion_mnist(data_dir=str(ROOT / "data"), num_samples=N,
                              flatten=True, normalize=True, seed=0)
    X, y = X.float(), y.long()

    # Confirm the premise: row_centered_he really does give W 1 = 0.
    cfg = ClassifierConfig(architecture="fc", depth=DEPTH,
                           init_strategy="row_centered_he", use_batch_norm=False,
                           fc_hidden_dim=WIDTH, fc_input_dim=X.shape[1], num_classes=10)
    torch.manual_seed(SEED)
    rc_model = build_classifier(cfg)
    row_sums = [l.weight.sum(dim=1).abs().max().item() for l in rc_model.hidden_layers]
    head_row_sum = rc_model.classifier.weight.sum(dim=1).abs().max().item()
    rc_scale = max(l.weight.abs().max().item() for l in rc_model.hidden_layers)

    payload = {
        "description": "Numerical check of the row-centering / post-ReLU-shift duality",
        "config": {"depth": DEPTH, "width": WIDTH, "seed": SEED, "samples": N,
                   "dataset": "fashion_mnist", "exact_dc_constant": DC},
        "row_centering_premise": {
            "max_abs_row_sum_hidden": max(row_sums),
            "max_abs_row_sum_head": head_row_sum,
            "max_abs_weight_entry": rc_scale,
        },
        "cases": {},
    }
    print(f"row_centered_he premise: max |sum_j W_ij| = {max(row_sums):.3e} (hidden), "
          f"{head_row_sum:.3e} (head); max |W_ij| = {rc_scale:.3e}")
    print(f"\n{'init':<20}{'c':>9}{'detach':>8}{'max|dout|':>13}{'rel dout':>12}"
          f"{'rel dloss':>12}{'max rel dgrad':>15}")

    for init in ("row_centered_he", "he"):
        base_out, base_loss, base_grads = run(init, None, False, X, y)
        scale = base_out.abs().max().item()
        for c in (0.25, DC, 0.7, 1.0):
            for detach in (True, False):
                out, loss, grads = run(init, c, detach, X, y)
                dout = (out - base_out).abs().max().item()
                rel_out = dout / max(scale, 1e-300)
                rel_loss = abs(loss.item() - base_loss.item()) / max(abs(base_loss.item()), 1e-300)
                rel_grad = max(
                    (g - b).norm().item() / max(b.norm().item(), 1e-300)
                    for g, b in zip(grads, base_grads))
                key = f"{init}_c{c:.4f}_{'detach' if detach else 'diff'}"
                payload["cases"][key] = {
                    "init_strategy": init, "shift_c": c, "relu_shift_detach": detach,
                    "max_abs_output_diff": dout,
                    "relative_output_diff": rel_out,
                    "relative_loss_diff": rel_loss,
                    "max_relative_grad_diff": rel_grad,
                    "baseline_output_max_abs": scale,
                }
                print(f"{init:<20}{c:>9.4f}{str(detach):>8}{dout:>13.3e}{rel_out:>12.3e}"
                      f"{rel_loss:>12.3e}{rel_grad:>15.3e}")

    out_path = ROOT / "reports" / "results" / "relu_shift_duality_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)}")
    print("\nExpected: row_centered_he -> relative output diff at float32 round-off "
          "(the shift is an exact forward no-op); he -> O(1) diff.")
    print("Expected: gradients differ even under row centering, by the rank-one "
          "term s * delta^T 1 (the duality is a FORWARD identity only).")


if __name__ == "__main__":
    main()
