#!/usr/bin/env python3
"""Pre-screen for the post-ReLU DC-removal family (campaign 11), at initialization.

Idea under test: ReLU output is non-negative, so E[a] = rms(a)/sqrt(pi) > 0.
That positive DC component -- shared by every sample -- is the engine of the
arc-cosine kernel's rho -> 1 collapse. Subtracting it AFTER the ReLU attacks
the mechanism directly:

    a = relu(W x) - c * rms(relu(W x))

Relation to the row-centered family: the next layer computes
    W (a - c 1) = W a - c (W 1)
so if W is row-centered (W 1 = 0) the subtraction is exactly a no-op. The
shift family is therefore the ACTIVATION-SPACE DUAL of row-centering, applied
to unconstrained He weights: kill the DC on the way out instead of on the way
in. c = 1/sqrt(pi) ~ 0.5642 removes exactly E[a]; c is swept because the
constant is not arbitrary -- overshooting replaces a positive shared DC with a
negative one, which drives cosine back toward 1 just as hard.

Screens each candidate on the three thesis requirements at init, before any
cluster time is spent:
  (i)   no geometric collapse   -> mean pairwise cosine; distance correlation
                                   vs the INPUT (the repo's own metric --
                                   "cosine far from 1" is not the goal,
                                   matching the input geometry is)
  (ii)  gradient/forward health -> activation RMS per layer, implied per-layer
                                   forward gain, dataset-dead unit fraction
  (iii) preserved class content -> cosine k-NN accuracy per layer
                                   (scale-invariant, as in campaign 09)

CLOSED-FORM REFERENCE (derived in cluster/11_relu_shift/README.md, checked by
this script against the measurement). Write A(c) = c^2 - 2c/sqrt(pi). For a
half-Gaussian pre-activation and sample-homogeneous norms:
  * per-layer forward gain     G(c) = sqrt(1 + A(c)) = sqrt(1 - 2c/sqrt(pi) + c^2)
    -- minimised at c = 1/sqrt(pi), where G = sqrt(1 - 1/pi) = sqrt((pi-1)/pi) = r
  * cosine recursion           rho_{l+1} = (g(rho_l) + A) / (1 + A),
    g(rho) = (1/pi)[sqrt(1-rho^2) + rho(pi - arccos rho)]   (arc-cosine kernel)
    -- rho = 1 stays a fixed point but becomes REPELLING (multiplier 1/(1+A) > 1);
       the attracting fixed point rho* solves Phi(rho*) = -A with
       Phi(rho) = (g(rho) - rho)/(1 - rho), and since max Phi = Phi(0) = 1/pi
       equals max|A| = 1/pi attained exactly at c = 1/sqrt(pi), the theory
       predicts rho* = 0 there and rho* > 0 on both sides.
Both predictions are emitted per candidate so the measurement can be compared
to them directly; where they diverge, `norm_heterogeneity_kappa` is the
diagnostic that explains it (see --shift-scope).

MECHANISM CONTROL (--shift-scope): rms(a) is by default one scalar over the
WHOLE (samples x units) tensor, matching DeepFCClassifier. That makes the
subtracted constant a batch statistic, so a sample whose own RMS differs from
the batch RMS is effectively shifted by c_eff = c * rms_batch / rms_sample,
i.e. it sits at a DIFFERENT point of the U-shaped A(c) curve. --shift-scope
per_sample recomputes the RMS per row and removes that effect; the difference
between the two isolates how much of the observed behaviour is the DC theory
and how much is batch-statistic heterogeneity. per_sample is a DIAGNOSTIC
only -- it is not a training candidate.

Usage:  python scripts/relu_shift_geometry_screen.py [--depth 60] [--width 500]
Output: reports/results/relu_shift_geometry_screen[_<tag>].json
        reports/figures/relu_shift/relu_shift_geometry_screen[_<tag>].png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rp_study.analysis.geometry_metrics import pairwise_distance_correlation  # noqa: E402
from rp_study.data.loaders import load_fashion_mnist, load_cifar10  # noqa: E402
from rp_study.models.initializers import initialize_layer  # noqa: E402

DC = 1.0 / math.sqrt(math.pi)  # E[a]/rms(a) for a half-Gaussian: the exact DC
R_GAIN = math.sqrt((math.pi - 1.0) / math.pi)  # r ~ 0.8256, the campaign-09 constant
# 0.1..0.9 step 0.1 (the brief's minimum grid), plus the two extra points the
# committed 60L screen already used (0.25, 1.0) so the wide grid strictly
# CONTAINS the prior screen and reproduces it, plus the exact DC constant.
# plus 0.65/0.75 to bracket the homogenisation threshold found at 60L.
DEFAULT_C_GRID = sorted([round(0.1 * k, 4) for k in range(1, 11)]
                        + [0.25, 0.65, 0.75, DC])


def _relative_to_root(path: str) -> str:
    """Repo-relative when inside the repo, so committed JSONs carry no local paths."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- closed form

def arccos_kernel_ratio(rho: float) -> float:
    """g(rho) = E[u_i u_j] / E[u^2] for u = relu(z), z jointly Gaussian corr rho."""
    rho = min(max(rho, -1.0), 1.0)
    return (math.sqrt(max(0.0, 1.0 - rho * rho)) + rho * (math.pi - math.acos(rho))) / math.pi


def A_of_c(c: float) -> float:
    return c * c - 2.0 * c / math.sqrt(math.pi)


def analytic_gain(c: float) -> float:
    """G(c) = sqrt(1 + A(c)): the sample-homogeneous per-layer forward gain."""
    return math.sqrt(max(0.0, 1.0 + A_of_c(c)))


def analytic_cosine_fixed_point(c: float) -> float:
    """Attracting fixed point of rho -> (g(rho) + A)/(1 + A), by bisection.

    Phi(rho) = (g(rho) - rho)/(1 - rho) is decreasing on [0, 1) with
    Phi(0) = 1/pi and Phi(1-) = 0, so Phi(rho*) = -A has a unique root
    whenever 0 <= -A <= 1/pi -- which is exactly the attainable range of -A.
    """
    target = -A_of_c(c)
    if target <= 0.0:
        return 1.0
    if target >= 1.0 / math.pi:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        phi = (arccos_kernel_ratio(mid) - mid) / (1.0 - mid)
        if phi > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- forward pass

def build(depth, width, in_dim, init_strategy, seed):
    torch.manual_seed(seed)
    layers = []
    for i in range(depth):
        linear = nn.Linear(in_dim if i == 0 else width, width, bias=True)
        initialize_layer(linear, strategy=init_strategy, layer_index=i, n_layers=depth + 1)
        layers.append(linear)
    return layers


def forward_trace(layers, X, shift_c, scope="global"):
    """Return [(pre_activation, activation)] per layer. shift_c=None -> no shift."""
    trace = []
    with torch.no_grad():
        h = X
        for linear in layers:
            pre = linear(h)
            h = torch.relu(pre)
            if shift_c is not None:
                if scope == "per_sample":
                    rms = h.pow(2).mean(dim=1, keepdim=True).sqrt()
                else:
                    rms = h.pow(2).mean().sqrt()
                h = h - shift_c * rms
            trace.append((pre, h.clone()))
    return trace


# ------------------------------------------------------------------- metrics

def mean_pairwise_cosine(h):
    hn = h / h.norm(dim=1, keepdim=True).clamp(min=1e-30)
    gram = hn @ hn.T
    iu = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1)
    return gram[iu[0], iu[1]].mean().item()


def cosine_knn_accuracy(h, y, k=10, n_classes=10):
    hn = (h / h.norm(dim=1, keepdim=True).clamp(min=1e-30)).numpy()
    sim = hn @ hn.T
    np.fill_diagonal(sim, -np.inf)
    neighbours = np.argsort(-sim, axis=1)[:, :k]
    pred = np.array([np.bincount(y[row], minlength=n_classes).argmax() for row in neighbours])
    return float((pred == y).mean())


def norm_heterogeneity_kappa(h):
    """kappa = mean_s(rms_s) / rms_global, in (0, 1]. 1 = every sample has the
    same norm; smaller = a heavy-tailed norm distribution, in which case the
    GLOBAL rms over-shifts the typical sample (see the module docstring)."""
    per_sample = h.pow(2).mean(dim=1).sqrt()
    glob = h.pow(2).mean().sqrt()
    if glob.item() <= 0:
        return float("nan")
    return (per_sample.mean() / glob).item()


def shared_dc_energy_fraction(h):
    """||mean_s a_s||^2 / mean_s ||a_s||^2 -- the share of the representation's
    energy that sits in the component COMMON to every sample. This is the
    quantity the whole campaign is trying to drive to zero; mean pairwise
    cosine is essentially a monotone function of it."""
    mu = h.mean(dim=0)
    denom = h.pow(2).sum(dim=1).mean()
    if denom.item() <= 0:
        return float("nan")
    return (mu.pow(2).sum() / denom).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "cifar10"])
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", type=float, nargs="*", default=DEFAULT_C_GRID)
    parser.add_argument("--shift-scope", default="global", choices=["global", "per_sample"],
                        help="global = one rms per layer (matches DeepFCClassifier); "
                             "per_sample = one rms per row (mechanism control only)")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--tag", default=None,
                        help="suffix for the output filenames, e.g. 100L_cifar10")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    loader = load_fashion_mnist if args.dataset == "fashion_mnist" else load_cifar10
    X, y = loader(data_dir=args.data_dir, num_samples=args.samples,
                  flatten=True, normalize=True, seed=0)
    X, y = X.float(), y.numpy()
    input_rms = X.pow(2).mean().sqrt().item()

    candidates = [("he", "he", None), ("row_centered_he", "row_centered_he", None)]
    candidates += [(f"he_shift_c{c:.4f}", "he", c) for c in args.c_grid]

    probe = sorted({1, 2, 5, 10, 20, args.depth // 2, args.depth})
    probe = [p for p in probe if 1 <= p <= args.depth]
    gain_read_depths = [d for d in (10, 20, 30, 60, 100) if d <= args.depth]

    layers_cache = {}
    results = {}
    Xn = X.numpy()
    for name, init_strategy, c in candidates:
        if init_strategy not in layers_cache:
            layers_cache[init_strategy] = build(args.depth, args.width, X.shape[1],
                                                init_strategy, args.seed)
        trace = forward_trace(layers_cache[init_strategy], X, c, scope=args.shift_scope)
        act_rms = [h.pow(2).mean().sqrt().item() for _, h in trace]
        prev = [input_rms] + act_rms[:-1]
        results[name] = {
            "init_strategy": init_strategy,
            "shift_c": c,
            "mean_pairwise_cosine": [mean_pairwise_cosine(h) for _, h in trace],
            "cosine_knn_accuracy": [cosine_knn_accuracy(h, y, args.knn_k) for _, h in trace],
            "activation_rms": act_rms,
            "dataset_dead_fraction": [
                (pre.max(dim=0).values <= 0).float().mean().item() for pre, _ in trace
            ],
            "implied_forward_gain": act_rms[-1] ** (1.0 / args.depth) if act_rms[-1] > 0 else 0.0,
            # --- campaign-11 additions -------------------------------------
            "per_layer_gain": [a / p if p > 0 else float("nan")
                               for a, p in zip(act_rms, prev)],
            "implied_forward_gain_at_depth": {
                str(d): (act_rms[d - 1] ** (1.0 / d) if act_rms[d - 1] > 0 else 0.0)
                for d in gain_read_depths
            },
            "norm_heterogeneity_kappa": [norm_heterogeneity_kappa(h) for _, h in trace],
            "shared_dc_energy_fraction": [shared_dc_energy_fraction(h) for _, h in trace],
            "mean_activation": [h.mean().item() for _, h in trace],
            "analytic_gain_G": analytic_gain(c) if c is not None else None,
            "analytic_cosine_fixed_point": analytic_cosine_fixed_point(c) if c is not None else None,
        }
        row = [pairwise_distance_correlation(Xn, trace[p - 1][1].numpy(), n_pairs=4000, seed=0)
               for p in probe]
        results[name]["distance_correlation_probe_layers"] = dict(zip(map(str, probe), row))

    # Record data_dir repo-relative when it lives inside the repo, so committed
    # JSONs carry no machine-local absolute paths.
    logged = dict(vars(args))
    logged["data_dir"] = _relative_to_root(args.data_dir)
    payload = {
        "description": "Init-time screen of the post-ReLU DC-removal family on the three requirements",
        "config": logged | {"exact_dc_constant": DC, "r_gain_constant": R_GAIN,
                            "input_rms": input_rms},
        "input_baseline": {
            "mean_pairwise_cosine": mean_pairwise_cosine(X),
            "cosine_knn_accuracy": cosine_knn_accuracy(X, y, args.knn_k),
            "norm_heterogeneity_kappa": norm_heterogeneity_kappa(X),
            "shared_dc_energy_fraction": shared_dc_energy_fraction(X),
        },
        "candidates": results,
    }
    tag = f"_{args.tag}" if args.tag else ""
    out = ROOT / "reports" / "results" / f"relu_shift_geometry_screen{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    base = payload["input_baseline"]
    print(f"{args.dataset} {args.depth}L width {args.width} N={args.samples} "
          f"scope={args.shift_scope} seed={args.seed}")
    print(f"input: mean pairwise cos = {base['mean_pairwise_cosine']:.4f}   "
          f"cosine-{args.knn_k}NN = {base['cosine_knn_accuracy']:.4f}   "
          f"kappa = {base['norm_heterogeneity_kappa']:.4f}\n")
    panels = [
        ("MEAN PAIRWISE COSINE (1.0 = collapse)", "mean_pairwise_cosine", "{:>9.4f}"),
        (f"COSINE {args.knn_k}-NN ACCURACY (0.10 = chance)", "cosine_knn_accuracy", "{:>9.4f}"),
        (f"DATASET-DEAD UNIT FRACTION (N={args.samples})", "dataset_dead_fraction", "{:>9.3f}"),
        ("ACTIVATION RMS", "activation_rms", "{:>9.2e}"),
        ("SHARED-DC ENERGY FRACTION (collapse driver)", "shared_dc_energy_fraction", "{:>9.4f}"),
        ("NORM HETEROGENEITY kappa (1.0 = homogeneous)", "norm_heterogeneity_kappa", "{:>9.4f}"),
        ("PER-LAYER FORWARD GAIN", "per_layer_gain", "{:>9.4f}"),
    ]
    for title, key, fmt in panels:
        print(title)
        print(f"{'candidate':<22}" + "".join(f"{'L' + str(p):>9}" for p in probe))
        for name, res in results.items():
            print(f"{name:<22}" + "".join(fmt.format(res[key][p - 1]) for p in probe))
        print()

    print("IMPLIED FORWARD GAIN, read at several depths (constant <=> depth-independent)")
    head = "".join(f"{'L' + str(d):>9}" for d in gain_read_depths)
    print(f"{'candidate':<22}{head}{'G(c)':>9}{'rho*(c)':>9}{'cos@L':>9}")
    for name, res in results.items():
        row = "".join(f"{res['implied_forward_gain_at_depth'][str(d)]:>9.4f}"
                      for d in gain_read_depths)
        g = res["analytic_gain_G"]
        rho = res["analytic_cosine_fixed_point"]
        gs = f"{g:>9.4f}" if g is not None else f"{'--':>9}"
        rs = f"{rho:>9.4f}" if rho is not None else f"{'--':>9}"
        print(f"{name:<22}{row}{gs}{rs}{res['mean_pairwise_cosine'][-1]:>9.4f}")
    print(f"\n  reference: r = {R_GAIN:.5f}   sqrt(r) = {math.sqrt(R_GAIN):.5f}   "
          f"1/sqrt(pi) = {DC:.5f}")

    print("\nDISTANCE CORRELATION vs INPUT (Spearman; 1.0 = input geometry preserved)")
    print(f"{'candidate':<22}" + "".join(f"{'L' + str(p):>9}" for p in probe))
    for name, res in results.items():
        print(f"{name:<22}" + "".join(
            f"{res['distance_correlation_probe_layers'][str(p)]:>9.4f}" for p in probe))
    print(f"\nSaved {out.relative_to(ROOT)}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ell = range(1, args.depth + 1)
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.4))
    for name, res in results.items():
        axes[0].plot(ell, res["mean_pairwise_cosine"], label=name)
        axes[1].plot(ell, res["cosine_knn_accuracy"], label=name)
        axes[2].plot(ell, res["dataset_dead_fraction"], label=name)
        axes[3].semilogy(ell, [max(v, 1e-30) for v in res["activation_rms"]], label=name)
        axes[4].plot(ell, res["shared_dc_energy_fraction"], label=name)
    axes[0].axhline(base["mean_pairwise_cosine"], ls=":", c="k", lw=1, label="input")
    axes[0].set(xlabel="layer $\\ell$", ylabel="mean pairwise cosine", title="(i) geometry")
    axes[1].axhline(0.1, ls=":", c="k", lw=1, label="chance")
    axes[1].set(xlabel="layer $\\ell$", ylabel=f"cosine {args.knn_k}-NN accuracy",
                title="(iii) class content")
    axes[2].axhline(0.5, ls=":", c="k", lw=1, label="1/2")
    axes[2].set(xlabel="layer $\\ell$", ylabel="dataset-dead fraction",
                title=f"dead units (N={args.samples})")
    axes[3].set(xlabel="layer $\\ell$", ylabel="activation RMS", title="(ii) forward scale")
    axes[4].set(xlabel="layer $\\ell$", ylabel="$\\|\\mu\\|^2/\\langle\\|a\\|^2\\rangle$",
                title="shared-DC energy fraction")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle(f"Post-ReLU DC removal, {args.depth}L width {args.width}, "
                 f"{args.dataset}, scope={args.shift_scope}, at initialization")
    fig.tight_layout()
    fig_path = (ROOT / "reports" / "figures" / "relu_shift"
                / f"relu_shift_geometry_screen{tag}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
