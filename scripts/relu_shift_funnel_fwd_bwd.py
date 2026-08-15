#!/usr/bin/env python3
"""Campaign 11, deliverable 3: FORWARD + BACKWARD + GEOMETRY at initialization
for the post-ReLU DC-removal family, and the detached-vs-differentiable fork.

The screen (scripts/relu_shift_geometry_screen.py) covers forward and geometry.
This covers the backward pass, in the house style of scripts/he_funnel_fwd_bwd.py
and scripts/rcfwd_gradrescale_funnel.py: per-layer rms of the forward activation,
rms of the back-propagated error delta, and the mean gradient row-norm, through
the real DeepFCClassifier (not a re-implementation), so what is measured is the
code that will actually train.

THE FORK. a = u - c*rms(u), rms(u) = u.pow(2).mean().sqrt() over the whole
(batch x units) tensor. rms(u) is a function of u, so it carries gradient:

    da_k/du_j = delta_kj - (c/(N*rms)) * u_j          N = u.numel()

i.e. the Jacobian is  I - (c/(N*rms)) * 1 u^T,  a rank-one correction. So:
  * DETACHED       -> the shift is a pure per-layer additive constant. The
                      backward pass is bit-identical to plain He backprop; only
                      the forward pass (and hence the activations the gradient
                      is evaluated at) changes.
  * DIFFERENTIABLE -> every unit's gradient picks up -(c/(N*rms)) * u_j * sum_k g_k.
                      Because the sum runs over the WHOLE tensor, this couples
                      units AND SAMPLES: sample i's gradient depends on sample j.
                      That is a BatchNorm-like batch coupling, and it is the
                      reason the fork is a research question rather than taste.

What this script measures, per candidate and per fork arm:
  * fwd  : rms of the post-shift activation per layer          (forward health)
  * bwd  : rms of the back-propagated error per layer          (backward health)
  * grad : mean gradient row-norm of each hidden Linear        (what SGD sees)
  * max/min funnel ratios over depth for each of the three     (conditioning)
  * cos  : mean pairwise cosine per layer                      (geometry)
  * the RELATIVE SIZE of the rank-one coupling term, measured directly as
    ||g_detached - g_differentiable|| / ||g_detached|| per layer -- this is the
    only honest way to say whether the fork matters numerically.

Usage:  python scripts/relu_shift_funnel_fwd_bwd.py [--depth 100] [--c 0.7 0.5642]
Output: reports/results/relu_shift_funnel_fwd_bwd.json
        reports/figures/relu_shift/relu_shift_funnel_fwd_bwd.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.config import ClassifierConfig  # noqa: E402
from rp_study.data.loaders import load_fashion_mnist, load_cifar10  # noqa: E402
from rp_study.models.classifiers import build_classifier  # noqa: E402

DC = 1.0 / math.sqrt(math.pi)


def rms(t):
    return t.norm().item() / math.sqrt(t.numel())


def mean_pairwise_cosine(h):
    hn = h / h.norm(dim=1, keepdim=True).clamp(min=1e-30)
    gram = hn @ hn.T
    iu = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1)
    return gram[iu[0], iu[1]].mean().item()


def measure(init_strategy, relu_shift, detach, X, y, depth, width, seed, in_dim):
    """One forward+backward through the real DeepFCClassifier at init."""
    cfg = ClassifierConfig(
        architecture="fc", depth=depth, init_strategy=init_strategy,
        use_batch_norm=False, fc_hidden_dim=width, fc_input_dim=in_dim,
        num_classes=10, relu_shift=relu_shift, relu_shift_detach=detach,
    )
    torch.manual_seed(seed)
    model = build_classifier(cfg)

    acts, deltas = {}, {}
    hooks = []
    for idx, linear in enumerate(model.hidden_layers):
        def mk_fwd(j):
            def hook(mod, inp, out):
                acts[j] = out.detach().clone()
            return hook

        def mk_bwd(j):
            def hook(mod, gin, gout):
                deltas[j] = gout[0].detach().clone()
            return hook
        hooks.append(linear.register_forward_hook(mk_fwd(idx)))
        hooks.append(linear.register_full_backward_hook(mk_bwd(idx)))

    # Post-shift activations (what the NEXT layer actually sees) need their own
    # capture, because the shift happens outside any submodule.
    post = []
    with torch.no_grad():
        h = X
        for idx, linear in enumerate(model.hidden_layers):
            h = torch.relu(linear(h))
            if relu_shift is not None:
                h = h - relu_shift * h.pow(2).mean().sqrt()
            post.append(h.clone())

    model.zero_grad()
    out = model(X)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    for hk in hooks:
        hk.remove()

    n = depth
    fwd = np.array([rms(post[i]) for i in range(n)])
    bwd = np.array([rms(deltas[i]) for i in range(n)])
    grad = np.array([model.hidden_layers[i].weight.grad.norm(dim=1).mean().item()
                     for i in range(n)])
    grad_full = [model.hidden_layers[i].weight.grad.detach().clone() for i in range(n)]
    cos = np.array([mean_pairwise_cosine(post[i]) for i in range(n)])
    return {"fwd": fwd, "bwd": bwd, "grad": grad, "cos": cos,
            "loss": loss.item(), "grad_full": grad_full}


def ratio(x):
    finite = x[np.isfinite(x) & (x > 0)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max() / max(finite.min(), 1e-300))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--samples", type=int, default=256,
                   help="batch size the shift's rms is computed over")
    p.add_argument("--datasets", nargs="*", default=["fashion_mnist", "cifar10"])
    p.add_argument("--c", type=float, nargs="*", default=[0.25, DC, 0.7, 0.75])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    # Repo-relative data_dir so committed JSONs carry no machine-local paths.
    logged = dict(vars(a))
    try:
        logged["data_dir"] = str(Path(a.data_dir).resolve().relative_to(ROOT))
    except ValueError:
        pass
    payload = {"description": "Init-time forward/backward/geometry funnel for the "
                              "post-ReLU DC-removal family, incl. the "
                              "detached-vs-differentiable fork",
               "config": logged | {"exact_dc_constant": DC}, "datasets": {}}

    for dataset in a.datasets:
        loader = load_fashion_mnist if dataset == "fashion_mnist" else load_cifar10
        X, y = loader(data_dir=a.data_dir, num_samples=a.samples,
                      flatten=True, normalize=True, seed=0)
        X, y = X.float(), y.long()
        in_dim = X.shape[1]

        arms = [("he", "he", None, False), ("row_centered_he", "row_centered_he", None, False)]
        for c in a.c:
            arms.append((f"he_shift_c{c:.4f}_diff", "he", c, False))
            arms.append((f"he_shift_c{c:.4f}_detach", "he", c, True))

        ds_out = {}
        print(f"\n{'='*100}\n{dataset}  depth={a.depth} width={a.width} "
              f"batch={a.samples} seed={a.seed}\n{'='*100}")
        print(f"{'candidate':<26}{'fwd@L':>10}{'bwd@L':>10}{'grad@L':>10}"
              f"{'fwd ratio':>12}{'bwd ratio':>12}{'grad ratio':>12}{'cos@L':>9}{'loss':>9}")
        cache = {}
        for name, init, c, detach in arms:
            m = measure(init, c, detach, X, y, a.depth, a.width, a.seed, in_dim)
            cache[name] = m
            ds_out[name] = {
                "init_strategy": init, "shift_c": c, "relu_shift_detach": detach,
                "forward_rms": m["fwd"].tolist(),
                "backward_delta_rms": m["bwd"].tolist(),
                "grad_row_norm_mean": m["grad"].tolist(),
                "mean_pairwise_cosine": m["cos"].tolist(),
                "forward_rms_ratio": ratio(m["fwd"]),
                "backward_delta_rms_ratio": ratio(m["bwd"]),
                "grad_row_norm_ratio": ratio(m["grad"]),
                "initial_loss": m["loss"],
            }
            print(f"{name:<26}{m['fwd'][-1]:>10.2e}{m['bwd'][-1]:>10.2e}"
                  f"{m['grad'][-1]:>10.2e}{ratio(m['fwd']):>12.3g}"
                  f"{ratio(m['bwd']):>12.3g}{ratio(m['grad']):>12.3g}"
                  f"{m['cos'][-1]:>9.4f}{m['loss']:>9.4f}")

        # --- the fork, measured directly -------------------------------------
        print(f"\nFORK: relative gradient difference ||g_diff - g_detach|| / ||g_detach||, "
              f"per hidden layer ({dataset})")
        print(f"{'c':<10}{'max over layers':>18}{'median':>12}{'layer 1':>12}"
              f"{f'layer {a.depth}':>12}{'fwd identical?':>16}")
        for c in a.c:
            gd = cache[f"he_shift_c{c:.4f}_detach"]
            gf = cache[f"he_shift_c{c:.4f}_diff"]
            rel = np.array([
                (gf["grad_full"][i] - gd["grad_full"][i]).norm().item()
                / max(gd["grad_full"][i].norm().item(), 1e-300)
                for i in range(a.depth)])
            fwd_same = bool(np.array_equal(gd["fwd"], gf["fwd"]))
            ds_out[f"he_shift_c{c:.4f}_diff"]["fork_relative_grad_diff"] = rel.tolist()
            print(f"{c:<10.4f}{rel.max():>18.4g}{float(np.median(rel)):>12.4g}"
                  f"{rel[0]:>12.4g}{rel[-1]:>12.4g}{str(fwd_same):>16}")

        for v in ds_out.values():
            v.pop("grad_full", None)
        payload["datasets"][dataset] = ds_out

        if not a.no_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
            ell = range(1, a.depth + 1)
            for name in ds_out:
                if name.endswith("_detach"):
                    continue
                axes[0].semilogy(ell, np.maximum(ds_out[name]["forward_rms"], 1e-30), label=name)
                axes[1].semilogy(ell, np.maximum(ds_out[name]["backward_delta_rms"], 1e-30),
                                 label=name)
                axes[2].semilogy(ell, np.maximum(ds_out[name]["grad_row_norm_mean"], 1e-30),
                                 label=name)
            axes[0].set(xlabel="hidden layer", ylabel="rms(a) post-shift", title="forward")
            axes[1].set(xlabel="hidden layer", ylabel="rms(delta)", title="backward error")
            axes[2].set(xlabel="hidden layer", ylabel="mean grad row norm", title="gradient")
            for ax in axes:
                ax.grid(alpha=0.3, which="both")
                ax.legend(fontsize=6)
            fig.suptitle(f"Post-ReLU DC removal: forward / backward / gradient funnel, "
                         f"{a.depth}L width {a.width}, {dataset}, at init")
            fig.tight_layout()
            fp = ROOT / "reports" / "figures" / "relu_shift" / f"relu_shift_funnel_{dataset}.png"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fp, dpi=140)
            print(f"Saved {fp.relative_to(ROOT)}")

    out = ROOT / "reports" / "results" / "relu_shift_funnel_fwd_bwd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
