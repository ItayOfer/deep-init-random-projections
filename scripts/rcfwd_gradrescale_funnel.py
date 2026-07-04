#!/usr/bin/env python3
"""TEST (init only, no training): row_centered_forward_balanced + a per-layer
BACKWARD gradient rescale by r = sqrt((pi-1)/pi) ~ 0.826, to cancel the
g_bwd = 1/r ~ 1.21 backward amplification.

Mechanism: a GradRescale op that is IDENTITY in the forward pass and multiplies
the back-flowing gradient by r. Inserted after each hidden layer's ReLU, it
touches ONLY the backward pass -- so the forward stays balanced (g_fwd=1) while
the backward per-layer gain becomes 1.21 * r ~ 1.0, i.e. flat. This decouples
forward and backward (impossible via weight scaling, which scales both).

Same plot format as rcfwd_funnel100_fwd_bwd_100L.png (constant 500 vs funnel
500->100, fmnist + cifar10; lines: rms(A) forward, rms(delta) backward, grad
row norm). If the rescale works, rms(delta) and grad-row-norm should go FLAT.
float64. Ratios .2f.
"""
import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import sys; sys.path.insert(0, "src")
import math
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rp_study.config import ExperimentConfig
from rp_study.data.loaders import get_data_loader
from rp_study.models.initializers import initialize_layer

INIT = "row_centered_forward_balanced"
L = 100
R = math.sqrt((math.pi - 1.0) / math.pi)        # ~0.826 = 1/1.21
IDIM = {"fashion_mnist": 784, "cifar10": 3072}
CONST = [500] * L
TAPER = [max(100, 500 - 4 * k) for k in range(L)]


class GradRescale(torch.autograd.Function):
    """Identity forward; multiply gradient by r in backward."""
    @staticmethod
    def forward(ctx, x, r):
        ctx.r = r
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx.r, None


def rms(t):
    return t.norm().item() / math.sqrt(t.numel())


def decompose(dataset, hidden_widths, use_rescale):
    ec = ExperimentConfig(seed=42, data_dir="data"); ec.setup_seeds()
    X, y = get_data_loader(dataset, "data", train=True, num_samples=512,
                           flatten=True, device="cpu")
    X = X.double(); targets = y.double().view(-1, 1)

    dims = [IDIM[dataset]] + hidden_widths + [1]
    n_weight_layers = len(dims) - 1
    linears = []
    for i, (fi, fo) in enumerate(zip(dims[:-1], dims[1:])):
        lin = nn.Linear(fi, fo).double()
        initialize_layer(lin, strategy=INIT, layer_index=i, n_layers=n_weight_layers)
        linears.append(lin)

    # backward hooks capture delta^l = grad wrt each Linear's output
    deltas = {}
    hooks = []
    for idx, lin in enumerate(linears):
        def mk(j):
            def hook(mod, gin, gout):
                deltas[j] = gout[0].detach().clone()
            return hook
        hooks.append(lin.register_full_backward_hook(mk(idx)))

    # forward
    acts = []
    x = X
    for i, lin in enumerate(linears):
        x = lin(x)
        if i < len(linears) - 1:           # hidden layer
            x = torch.relu(x)
            if use_rescale:
                x = GradRescale.apply(x, R)
            acts.append(x)                  # post-ReLU (rescale is identity fwd)
    loss = torch.sum((targets - x) ** 2)
    for lin in linears:
        lin.zero_grad()
    loss.backward()
    for h in hooks:
        h.remove()

    nL = len(hidden_widths)
    a = np.array([rms(acts[i]) for i in range(nL)])
    d = np.array([rms(deltas[i]) for i in range(nL)])      # hidden layers only
    g = np.array([np.linalg.norm(linears[i].weight.grad.detach().numpy(), axis=1).mean()
                  for i in range(nL)])
    return a, d, g


def ratio(x):
    return x.max() / max(x.min(), 1e-300)


print(f"init={INIT} + backward GradRescale by r={R:.4f} (cancels g_bwd=1/r={1/R:.3f})")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for col, dataset in enumerate(["fashion_mnist", "cifar10"]):
    for row, (widths, name) in enumerate(
            [(CONST, "constant width 500"), (TAPER, "funnel 500->100")]):
        a, d, g = decompose(dataset, widths, use_rescale=True)
        ax = axes[row][col]
        ax.plot(range(len(a)), a, "-", color="tab:blue", lw=1.5, label="rms(A) forward activation")
        ax.plot(range(len(d)), d, "-", color="tab:red", lw=1.5, label="rms(delta) backward error")
        ax.plot(range(len(g)), g, "-", color="black", lw=1.3, label="grad row norm (~A*delta)")
        ax.set_yscale("log")
        ax.set_title(f"{dataset} | rcfwd + backward-rescale(r) {name}\n"
                     f"rms(A) ratio {ratio(a):.2f}x | rms(delta) ratio {ratio(d):.2f}x | "
                     f"grad ratio {ratio(g):.2f}x", loc="left", fontsize=10)
        ax.set_xlabel("hidden layer index (0=input side, 99=output side)")
        ax.set_ylabel("RMS / norm (log)")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
        print(f"  {dataset:>14s} {name:<20s}: rms(A) ratio {ratio(a):.2f}x  "
              f"rms(delta) ratio {ratio(d):.2f}x  grad ratio {ratio(g):.2f}x")
fig.suptitle(f"row_centered_forward_balanced + per-layer BACKWARD rescale by r={R:.3f}, 100L at init: "
             "does rms(delta) go flat?", fontsize=13, y=1.01)
fig.tight_layout()
out = "reports/figures/rcfwd_gradrescale_funnel100_100L.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
