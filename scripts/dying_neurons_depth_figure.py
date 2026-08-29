#!/usr/bin/env python3
"""Empirical anchor figure for the dying-neurons proof: measured dataset-dead
fraction vs depth (He init) against the theorem's universal Slepian lower
bound and the 1/2 asymptote.

Measured points come from the depth-tagged geometry screens
`reports/results/relu_shift_geometry_screen_<depth>L_<dataset>[...].json`
(discovered by glob at run time -- nothing hardcoded; the legacy untagged
`relu_shift_geometry_screen.json` is ignored). Per file:
    dead fraction   = candidates["he"]["dataset_dead_fraction"][-1]
    input cosine    = input_baseline["mean_pairwise_cosine"]  (rho_0)
    N (samples), width, dataset, seed from config.

Common machinery: the arc-cosine correlation map
    chi(rho) = (sin a + (pi - a) cos a) / pi,  a = arccos rho,
and the equicorrelated sign-agreement probability
    P[all N signs agree] = 2 * Int_0^inf phi(g) * exp(N * Phi_logcdf(lam g)) dg,
    eps = 1 - rho,  lam = sqrt((1 - eps) / eps)
(numerically stable form matching docs/scratch/proofs/oracle_spotcheck.py's
p_all_agree integrand); the bound on the dead fraction is P[all agree] / 2.

PRIMARY curve -- universal lower bound (theorem): dataset-free. At depth L use
eps = 1 - chi^{L-1}(0): after the first layer every pairwise activation
correlation is >= 0, chi is monotone, so L-1 further iterations from rho = 0
lower-bound the depth-L correlation for ANY input geometry. One solid INK
curve.

SECONDARY curves -- mean-anchored heuristic: iterate chi for L steps from each
dataset's measured mean input cosine rho_0. Not a theorem (it anchors on the
mean pairwise cosine, not the worst pair), drawn as thin dashed lines in the
dataset colors. A measured point may legitimately sit below these (cifar10 at
30L does) while still clearing the universal bound.

Verification gates (asserted below, tolerance 0.002):
    universal bound at L = 60, N = 512            ~= 0.3851
    mean-anchored, fmnist rho_0, L = 60, N = 512  ~= 0.3881
(Universal values at L = 30/60/100 are 0.2982 / 0.3851 / 0.4270.)

Output: reports/figures/dying_neurons/dead_fraction_vs_depth.png.
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import integrate
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # convention parity with the other scripts/*.py; unused directly here

# Palette: same hex values as scripts/campaign10_followup_figures.py
BLUE = "#2a78d6"    # fmnist measured points
VIOLET = "#4a3aa7"  # cifar10 measured points
GRAY = "#7a7a7a"    # 1/2 asymptote
INK = "#222222"     # theory curves

DATASET_STYLE = {
    # config's dataset field -> (display name, color, marker)
    "fashion_mnist": ("Fashion-MNIST", BLUE, "o"),
    "cifar10": ("CIFAR-10", VIOLET, "s"),
}

UNIVERSAL_L60_EXPECTED = 0.3851  # verification gate: theorem bound (see docstring)
FMNIST_L60_EXPECTED = 0.3881     # verification gate: fmnist mean-anchored heuristic


def chi(rho):
    """Arc-cosine correlation map for ReLU: rho -> (sin a + (pi-a) cos a)/pi."""
    a = math.acos(max(-1.0, min(1.0, rho)))
    return (math.sin(a) + (math.pi - a) * math.cos(a)) / math.pi


def _slepian_dead_fraction(rho, n_samples):
    """P[all N signs agree]/2 for the equicorrelated model at correlation rho.

    Log-cdf integrand (stable for large N; same form as oracle_spotcheck.py).
    """
    eps = 1.0 - rho
    lam = math.sqrt((1.0 - eps) / eps)
    f = lambda g: norm.pdf(g) * math.exp(n_samples * norm.logcdf(lam * g))
    val, _ = integrate.quad(f, 0.0, 40.0, limit=400)
    return 2.0 * val / 2.0  # P(all agree) / 2


def universal_dead_fraction(ell, n_samples):
    """The theorem's universal (dataset-free) lower bound at depth ell.

    eps = 1 - chi^{ell-1}(0): post-ReLU correlations are nonnegative after
    layer 1, and chi is monotone, so iterating from rho = 0 for the remaining
    ell - 1 layers lower-bounds the depth-ell correlation for any input.
    """
    r = 0.0
    for _ in range(ell - 1):
        r = chi(r)
    return _slepian_dead_fraction(r, n_samples)


def mean_anchored_dead_fraction(rho0, ell, n_samples):
    """Mean-anchored heuristic: iterate chi for ell steps from the dataset's
    measured mean input cosine rho0. Not a theorem -- see module docstring."""
    r = rho0
    for _ in range(ell):
        r = chi(r)
    return _slepian_dead_fraction(r, n_samples)


def load_measured(res_dir):
    """Discover depth-tagged geometry screens and pull the He dead fractions.

    Returns {dataset: {"rho0": float, "n": int, "width": int, "seed": int,
                       "points": {depth: dead_fraction}, "files": [...]}}.
    Duplicate (dataset, depth) files (e.g. *_persample forks -- the He arm has
    no shift, so their He numbers coincide) are skipped with a note.
    """
    paths = sorted(glob.glob(str(res_dir / "relu_shift_geometry_screen_*L_*.json")))
    by_ds = {}
    for p in paths:
        d = json.load(open(p))
        if isinstance(d, list):  # training payloads are lists; screens are dicts
            d = d[0]
        cfg = d["config"]
        ds, depth = cfg["dataset"], int(cfg["depth"])
        dead = d["candidates"]["he"]["dataset_dead_fraction"][-1]
        rho0 = d["input_baseline"]["mean_pairwise_cosine"]
        entry = by_ds.setdefault(ds, {"rho0": rho0, "n": int(cfg["samples"]),
                                      "width": int(cfg["width"]), "seed": cfg.get("seed"),
                                      "points": {}, "files": []})
        if depth in entry["points"]:
            print(f"  note: {Path(p).name} duplicates ({ds}, {depth}L) -- skipped")
            continue
        assert abs(entry["rho0"] - rho0) < 1e-9, f"rho0 mismatch within {ds}"
        assert entry["n"] == int(cfg["samples"]), f"samples mismatch within {ds}"
        entry["points"][depth] = dead
        entry["files"].append(Path(p).name)
        print(f"  {Path(p).name}: dataset={ds}, depth={depth}, "
              f"dead_fraction={dead:.4f}, rho0={rho0:.5f}, N={cfg['samples']}")
    return by_ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "reports" / "results",
                    help="directory containing the source result JSONs")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures" / "dying_neurons",
                    help="directory to write the PNG to")
    ap.add_argument("--dpi", type=int, default=140, help="figure DPI")
    ap.add_argument("--depth-max", type=int, default=120, help="right edge of the theory-curve depth grid")
    args = ap.parse_args()

    print("=== measured points (from depth-tagged geometry screens) ===")
    measured = load_measured(args.results_dir)
    if not measured:
        raise SystemExit("no relu_shift_geometry_screen_*L_*.json files found -- nothing to plot")

    # --- verification gates ------------------------------------------------
    n_gate = next(iter(measured.values()))["n"]
    u60 = universal_dead_fraction(60, n_gate)
    print(f"\nverification gate: universal bound(L=60, N={n_gate}) = {u60:.4f} "
          f"(expected {UNIVERSAL_L60_EXPECTED} +/- 0.002)")
    assert abs(u60 - UNIVERSAL_L60_EXPECTED) <= 0.002, (
        f"universal bound at L=60 is {u60:.4f}, expected {UNIVERSAL_L60_EXPECTED} +/- 0.002 -- "
        "the chi iteration or the Slepian integrand has drifted from the theorem's validation numbers"
    )
    fm = measured.get("fashion_mnist")
    if fm is not None:
        v60 = mean_anchored_dead_fraction(fm["rho0"], 60, fm["n"])
        print(f"verification gate: mean-anchored(fmnist rho0={fm['rho0']:.5f}, l=60, N={fm['n']}) = {v60:.4f} "
              f"(expected {FMNIST_L60_EXPECTED} +/- 0.002)")
        assert abs(v60 - FMNIST_L60_EXPECTED) <= 0.002, (
            f"mean-anchored value at l=60 is {v60:.4f}, expected {FMNIST_L60_EXPECTED} +/- 0.002 -- "
            "the chi iteration or the Slepian integrand has drifted from the proof's validation numbers"
        )
    else:
        print("WARNING: no fashion_mnist screen found -- the mean-anchored l=60 gate could not run")
    print("verification gates PASSED")

    # --- figure -----------------------------------------------------------
    n_ref = next(iter(measured.values()))["n"]
    width_ref = next(iter(measured.values()))["width"]
    depth_grid = list(range(5, args.depth_max + 1))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # PRIMARY: the theorem's universal, dataset-free lower bound (solid INK).
    universal_curve = [universal_dead_fraction(l, n_ref) for l in depth_grid]
    ax.plot(depth_grid, universal_curve, "-", color=INK, lw=1.8,
            label="universal lower bound (theorem)")

    # SECONDARY: per-dataset mean-anchored heuristic curves (thin dashed,
    # dataset colors) + the measured points.
    all_above_universal = True
    for ds, entry in measured.items():
        name, color, marker = DATASET_STYLE.get(ds, (ds, GRAY, "d"))
        heur = [mean_anchored_dead_fraction(entry["rho0"], l, entry["n"]) for l in depth_grid]
        ax.plot(depth_grid, heur, "--", color=color, lw=1.0, alpha=0.8,
                label=f"mean-anchored heuristic ({name}, " + r"$\rho_0$" + f"={entry['rho0']:.3f})")
        depths = sorted(entry["points"])
        ax.plot(depths, [entry["points"][l] for l in depths], marker, color=color,
                ms=8, mec="black", mew=0.6, ls="none", label=f"measured ({name})", zorder=5)
        for l in depths:
            uni = universal_dead_fraction(l, entry["n"])
            above = entry["points"][l] > uni
            all_above_universal &= above
            print(f"  plotted: {ds} depth={l} measured={entry['points'][l]:.4f} "
                  f"universal={uni:.4f} ({'above' if above else 'BELOW'}) "
                  f"mean_anchored={mean_anchored_dead_fraction(entry['rho0'], l, entry['n']):.4f}")
    print("all measured points above the universal bound:", all_above_universal)
    ax.axhline(0.5, color=GRAY, ls=":", lw=1.3, zorder=1)
    ax.text(depth_grid[0] + 1, 0.503, "1/2 asymptote", fontsize=8.5, color=GRAY)

    ax.set_xlabel("depth (layers)")
    ax.set_ylabel("dataset-dead fraction of neurons")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 0.56)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.set_title(f"Dead fraction vs depth — He, width {width_ref}, N={n_ref}", fontsize=11)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "dead_fraction_vs_depth.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    size = out_path.stat().st_size
    print(f"\nsaved {out_path} ({size:,} bytes)")
    if size <= 10_000:
        raise SystemExit("figure failed the >10KB non-empty check")


if __name__ == "__main__":
    main()
