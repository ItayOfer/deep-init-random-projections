"""
Extensible weight initialization strategies.

This module uses a registry pattern to make adding new initialization
strategies trivial. To add a new initializer:

1. Define a function that takes (layer: nn.Linear, **kwargs)
2. Decorate it with @register_initializer("your_name")

Example:
    @register_initializer("lecun")
    def lecun_init(layer: nn.Linear, **kwargs):
        fan_in = layer.weight.shape[1]
        std = math.sqrt(1.0 / fan_in)
        with torch.no_grad():
            layer.weight.normal_(0.0, std)
            if layer.bias is not None:
                layer.bias.zero_()
"""

import math
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn


# Registry for initialization strategies
INITIALIZERS: Dict[str, Callable] = {}


def register_initializer(name: str):
    """Decorator to register a new initialization strategy.

    Args:
        name: Name to register the initializer under.

    Example:
        @register_initializer("my_init")
        def my_init(layer: nn.Linear, **kwargs):
            # Initialize layer.weight and layer.bias
            pass
    """
    def decorator(fn: Callable) -> Callable:
        INITIALIZERS[name] = fn
        return fn
    return decorator


def initialize_layer(
    layer: nn.Linear,
    strategy: str = "he",
    **kwargs,
) -> None:
    """Initialize a linear layer using the specified strategy.

    Args:
        layer: The nn.Linear layer to initialize.
        strategy: Name of the initialization strategy.
        **kwargs: Additional arguments passed to the initializer.

    Raises:
        ValueError: If the strategy is not registered.
    """
    if strategy not in INITIALIZERS:
        available = list(INITIALIZERS.keys())
        raise ValueError(f"Unknown initialization strategy: {strategy}. Available: {available}")

    INITIALIZERS[strategy](layer, **kwargs)


def list_initializers() -> list:
    """Return a list of available initialization strategies."""
    return list(INITIALIZERS.keys())


# =============================================================================
# Built-in initialization strategies
# =============================================================================


@register_initializer("he")
def he_init(layer: nn.Linear, **kwargs) -> None:
    """He (Kaiming) initialization for ReLU networks.

    Uses fan_in mode: Var(W) = 2/fan_in

    This is the standard initialization for ReLU networks from:
    "Delving Deep into Rectifiers" (He et al., 2015)
    """
    nn.init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("row_centered_he")
def row_centered_he_init(layer: nn.Linear, **kwargs) -> None:
    """Row-centered He initialization.

    Draws from N(0, 2/fan_in) then subtracts the mean of each row,
    ensuring each row sums to zero. Biases are set to zero.

    This variant may have different gradient flow properties.

    Note: Row-centering reduces variance by a factor of (1 - 1/d), which can
    cause gradient vanishing in deep networks. See row_centered_he_var_adj
    for a variance-adjusted version.
    """
    fan_in = layer.weight.shape[1]
    std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=std)
        # Subtract row mean to center each row
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("row_centered_he_var_adj")
def row_centered_he_var_adj_init(layer: nn.Linear, **kwargs) -> None:
    """Row-centered He initialization with variance adjustment.

    This initialization combines the geometric benefits of row-centering
    (preventing geometric collapse) with proper variance scaling to maintain
    healthy gradient flow in deep networks.

    The procedure:
    1. Sample weights from N(0, sigma^2) where sigma^2 = 2/fan_in (He init)
    2. Center each row by subtracting its mean (for geometric properties)
    3. Rescale each row to restore He variance: W_row *= sqrt(2/d) / std(W_row)

    Mathematical background:
    - Row-centering reduces variance by factor (1 - 1/d) because:
      Var(w_j - mean(w)) = Var(w_j) * (1 - 1/d)
    - This causes gradients to decay as (1 - 1/d)^L in L-layer networks
    - Rescaling restores the target variance of 2/d per element

    This ensures:
    - Each row sums to zero (geometric collapse prevention)
    - Each row has the correct He variance (healthy gradient flow)
    """
    fan_in = layer.weight.shape[1]
    target_std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        # Step 1: Sample from He distribution
        layer.weight.normal_(mean=0.0, std=target_std)

        # Step 2: Center each row (subtract row mean)
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)

        # Step 3: Rescale each row to restore He variance
        # Compute std of each row and rescale to target_std
        # Use unbiased=False to match numpy's default (population std)
        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        # Avoid division by zero for edge cases
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("partial_centered_he")
def partial_centered_he_init(layer: nn.Linear, alpha: float = 0.5, **kwargs) -> None:
    """Partial row-centered He initialization.

    Instead of forcing rows to sum to exactly zero (which creates a hard
    constraint that may trap gradients), this uses a "soft" centering with
    parameter alpha < 1.

    The procedure:
    1. Sample weights from N(0, sigma^2) where sigma^2 = 2/fan_in (He init)
    2. Partially center: W = W - alpha * mean(W, dim=1)
    3. Rescale each row to restore He variance

    Args:
        layer: Layer to initialize.
        alpha: Centering strength (0 = no centering, 1 = full centering).
               Default 0.5 provides partial geometric benefit without full constraint.

    This allows tuning the trade-off between geometric collapse prevention
    (higher alpha) and gradient flow (lower alpha).
    """
    fan_in = layer.weight.shape[1]
    target_std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=target_std)

        # Partial centering
        layer.weight -= alpha * layer.weight.mean(dim=1, keepdim=True)

        # Rescale to restore He variance
        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("orthogonal_he")
def orthogonal_he_init(layer: nn.Linear, **kwargs) -> None:
    """Orthogonal initialization with He-like scaling.

    Uses QR decomposition to create orthogonal rows (for square matrices)
    or semi-orthogonal rows (for non-square). The rows are orthogonal to
    each other but do NOT have the zero-sum constraint.

    This provides geometric benefits (orthogonal rows prevent collapse)
    without the "gradient trap" caused by zero-sum rows.

    The procedure:
    1. Sample random matrix from N(0, 1)
    2. Apply QR decomposition to get orthogonal matrix Q
    3. Scale by sqrt(2) to approximate He variance for ReLU

    Note: For ReLU networks, gain=sqrt(2) is recommended.
    """
    # Use PyTorch's orthogonal init with gain=sqrt(2) for ReLU
    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("orthogonal_tuned")
def orthogonal_tuned_init(layer: nn.Linear, **kwargs) -> None:
    """Orthogonal initialization with tuned gain for row-centered-like networks.

    Same as orthogonal_he but with gain=sqrt(1.65) instead of sqrt(2).
    This accounts for the higher neuron survival rate (~60% vs 50%)
    that we observed empirically with structured initializations.

    Use this if orthogonal_he (gain=sqrt(2)) still shows mild explosion.
    """
    nn.init.orthogonal_(layer.weight, gain=math.sqrt(1.65))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("centered_with_dc_he")
def centered_with_dc_he_init(layer: nn.Linear, dc_scale: float = 0.1, **kwargs) -> None:
    """Row-centered He initialization with DC component restored.

    After row-centering (which forces row sums to zero), this adds back
    a small constant (DC component) to break the zero-sum constraint
    while preserving most of the centering benefit.

    The procedure:
    1. Sample weights from N(0, sigma^2) where sigma^2 = 2/fan_in (He init)
    2. Center each row by subtracting its mean
    3. Rescale to restore He variance
    4. Add DC component: W = W + dc_scale * sqrt(2/fan_in)

    Args:
        layer: Layer to initialize.
        dc_scale: Scale of the DC component relative to He std.
                  Default 0.1 adds a small constant to break zero-sum.

    The DC component breaks the structural constraint that traps gradients,
    while the centering still provides geometric benefits.
    """
    fan_in = layer.weight.shape[1]
    target_std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        # Standard He init
        layer.weight.normal_(mean=0.0, std=target_std)

        # Center each row
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)

        # Rescale to restore variance
        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        # Add DC component to break zero-sum constraint
        layer.weight += dc_scale * target_std

        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("custom_variance")
def custom_variance_init(
    layer: nn.Linear,
    variance: Optional[float] = None,
    mean: float = 0.0,
    **kwargs,
) -> None:
    """Custom variance initialization.

    Args:
        layer: Layer to initialize.
        variance: Variance of the weight distribution. If None, uses 2/fan_in.
        mean: Mean of the weight distribution.

    This allows experiments with different variance scales (e.g., 2/d, 2.5/d, 4/d).
    """
    fan_in = layer.weight.shape[1]
    if variance is None:
        variance = 2.0 / fan_in

    std = math.sqrt(variance)

    with torch.no_grad():
        layer.weight.normal_(mean=mean, std=std)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("xavier")
def xavier_init(layer: nn.Linear, **kwargs) -> None:
    """Xavier (Glorot) initialization.

    Uses the average of fan_in and fan_out: Var(W) = 2/(fan_in + fan_out)

    Suitable for sigmoid/tanh activations.
    """
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("uniform_he")
def uniform_he_init(layer: nn.Linear, **kwargs) -> None:
    """Uniform distribution with He-like variance.

    Uses U(-a, a) where a = sqrt(3 * 2/fan_in) so that Var = 2/fan_in.

    This matches the original notebook's uniform initialization style.
    """
    fan_in = layer.weight.shape[1]
    a = math.sqrt(3 * 2 / fan_in)  # Var(U(-a,a)) = a^2/3

    with torch.no_grad():
        layer.weight.uniform_(-a, a)
        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("orthogonal")
def orthogonal_init(layer: nn.Linear, gain: float = 1.0, **kwargs) -> None:
    """Orthogonal initialization.

    Initializes the weight matrix to be orthogonal (or semi-orthogonal
    if not square). Can help with gradient flow in deep networks.

    Args:
        layer: Layer to initialize.
        gain: Multiplicative factor for the weights.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

@register_initializer("kernel_preserving")
def kernel_preserving_init(
    layer: nn.Linear,
    n_samples: int = 500,
    n_pairs: int = 1000,
    lr: float = 0.01,
    n_steps: int = 200,
    norm_weight: float = 1.0,
    **kwargs,
) -> None:
    """Kernel-preserving initialization via optimization.

    Finds W that minimizes the discrepancy between input and output inner
    products after ReLU, i.e.:

        min_W  sum_{i,j} ( <ReLU(Wx_i), ReLU(Wx_j)> - <x_i, x_j> )^2

    Subject to norm preservation: E[||Wx||^2] = ||x||^2 for unit vectors.

    This is motivated by the ReLU kernel K(alpha) which always contracts
    angles. We seek W such that the contraction is minimized.

    The optimization uses synthetic unit vectors as proxies (data-independent
    in distribution, but the solution generalizes because the objective is
    over random directions).

    Args:
        layer: Layer to initialize.
        n_samples: Number of random unit vectors to use.
        n_pairs: Number of random pairs to sample for the objective.
        lr: Learning rate for Adam optimizer.
        n_steps: Number of optimization steps.
        norm_weight: Weight for norm preservation constraint in loss.
    """
    fan_in = layer.weight.shape[1]
    fan_out = layer.weight.shape[0]
    device = layer.weight.device

    # Start from He initialization as a warm start
    nn.init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")

    # Generate random unit vectors as proxy data (on same device as layer)
    X = torch.randn(n_samples, fan_in, device=device)
    X = X / X.norm(dim=1, keepdim=True)

    # Random pairs for kernel objective
    idx_i = torch.randint(0, n_samples, (n_pairs,), device=device)
    idx_j = torch.randint(0, n_samples, (n_pairs,), device=device)

    # Precompute input inner products for the sampled pairs
    input_ips = (X[idx_i] * X[idx_j]).sum(dim=1)  # <x_i, x_j>

    # Temporarily enable gradients for optimization
    layer.weight.requires_grad_(True)

    optimizer = torch.optim.Adam([layer.weight], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward: compute ReLU(Wx^T) for all samples
        # layer.weight is (fan_out, fan_in), X is (n_samples, fan_in)
        # output: (n_samples, fan_out)
        out = torch.relu(X @ layer.weight.t())

        # Kernel preservation loss: match inner products
        out_i = out[idx_i]  # (n_pairs, fan_out)
        out_j = out[idx_j]  # (n_pairs, fan_out)
        output_ips = (out_i * out_j).sum(dim=1) / fan_out  # normalize by d_out
        kernel_loss = ((output_ips - input_ips) ** 2).mean()

        # Norm preservation loss: E[||ReLU(Wx)||^2] should scale properly
        norms_sq = (out ** 2).sum(dim=1) / fan_out  # should be ~1 for unit inputs
        norm_loss = ((norms_sq - 1.0) ** 2).mean()

        loss = kernel_loss + norm_weight * norm_loss
        loss.backward()
        optimizer.step()

    # Disable gradients and detach
    layer.weight.requires_grad_(False)

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("angle_preserving")
def angle_preserving_init(
    layer: nn.Linear,
    n_samples: int = 500,
    n_pairs: int = 2000,
    lr: float = 0.01,
    n_steps: int = 200,
    norm_weight: float = 0.2,
    pair_bins: int = 8,
    eps: float = 1e-8,
    **kwargs,
) -> None:
    """Angle-preserving initialization via cosine-based optimization.

    This optimizer targets angle preservation more directly than
    ``kernel_preserving_init`` by matching pairwise cosine similarities after
    ReLU, with an objective that better reflects what happens in deep ReLU
    stacks.

    Differences vs ``kernel_preserving``:
    1. Optimizes cosine similarity instead of raw inner products.
    2. Uses target ``max(cos(x_i, x_j), 0)`` because ReLU outputs are
       non-negative and cannot represent strongly negative cosine values well.
    3. Reweights sampled pairs across cosine bins so near-orthogonal pairs
       do not dominate the objective.
    4. Uses a norm target of ``fan_out / fan_in`` for unit inputs, which
       matches He-style scaling under ReLU.
    5. Leaves ``layer.weight.requires_grad = True`` after initialization.

    Args:
        layer: Layer to initialize.
        n_samples: Number of random unit vectors for the proxy objective.
        n_pairs: Number of sampled vector pairs for angle matching.
        lr: Learning rate for Adam.
        n_steps: Number of optimizer steps.
        norm_weight: Weight of the norm-preservation term.
        pair_bins: Number of cosine bins used for pair reweighting.
        eps: Small constant for numerical stability.
    """
    fan_in = layer.weight.shape[1]
    fan_out = layer.weight.shape[0]
    device = layer.weight.device
    dtype = layer.weight.dtype
    target_norm_sq = fan_out / fan_in

    # He warm start.
    nn.init.kaiming_normal_(layer.weight, a=0.0, mode="fan_in", nonlinearity="relu")

    # Data-independent proxy: random directions on the unit sphere.
    X = torch.randn(n_samples, fan_in, device=device, dtype=dtype)
    X = X / X.norm(dim=1, keepdim=True).clamp(min=eps)

    # Sample random pairs once; keep objective fixed during optimization.
    idx_i = torch.randint(0, n_samples, (n_pairs,), device=device)
    idx_j = torch.randint(0, n_samples, (n_pairs,), device=device)
    same = idx_i == idx_j
    if same.any():
        idx_j[same] = (idx_j[same] + 1) % n_samples

    input_cos = (X[idx_i] * X[idx_j]).sum(dim=1).clamp(-1.0, 1.0)
    target_cos = input_cos.clamp(min=0.0)

    # Reweight pairs so rare angle regions contribute comparably.
    pair_bins = max(2, int(pair_bins))
    bin_edges = torch.linspace(-1.0, 1.0, steps=pair_bins + 1, device=device, dtype=dtype)
    bin_ids = torch.bucketize(input_cos.detach(), bin_edges[1:-1])
    bin_counts = torch.bincount(bin_ids, minlength=pair_bins).clamp(min=1).to(dtype=dtype)
    pair_weights = (1.0 / bin_counts[bin_ids]).to(dtype=dtype)
    pair_weights = pair_weights / pair_weights.mean().clamp(min=eps)

    layer.weight.requires_grad_(True)
    optimizer = torch.optim.Adam([layer.weight], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()

        out = torch.relu(X @ layer.weight.t())  # (n_samples, fan_out)
        out_norm = out.norm(dim=1, keepdim=True).clamp(min=eps)
        out_unit = out / out_norm

        out_cos = (out_unit[idx_i] * out_unit[idx_j]).sum(dim=1).clamp(-1.0, 1.0)
        angle_loss = (pair_weights * (out_cos - target_cos) ** 2).mean()

        norms_sq = (out ** 2).sum(dim=1)
        norm_loss = ((norms_sq / target_norm_sq - 1.0) ** 2).mean()

        loss = angle_loss + norm_weight * norm_loss
        loss.backward()
        optimizer.step()

    # Clear temporary optimizer gradient; keep weight trainable for later use.
    layer.weight.grad = None

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@register_initializer("row_centered_final")
def row_centered_final_init(layer: nn.Linear, **kwargs) -> None:
    """Row-centered initialization with Tuned Variance (Factor ~1.65).

    Data-driven tuning:
    - Factor 1.0 (Xavier) -> Vanishing (Gain ~0.77)
    - Factor 1.8 (Optimized) -> Mild Explosion (Gain ~1.04)
    - Factor 2.0 (He) -> Strong Explosion (Gain ~1.05+)
    
    Implied Active Fraction is ~60%.
    To achieve Gain=1.0, we need Factor = 1 / 0.6 = 1.66.
    """
    fan_in = layer.weight.shape[1]
    
    # MAGIC NUMBER: 1.65
    target_std = math.sqrt(1.65 / fan_in)

    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=target_std)
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)

        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("row_centered_layer_balanced")
def row_centered_layer_balanced_init(
    layer: nn.Linear,
    layer_index: int = 0,
    n_layers: int = 1,
    eta: float = 1.0,
    **kwargs,
) -> None:
    """Row-centered He with per-layer variance scaling for gradient uniformity.

    Mathematical derivation:
        From BP4: ||∂C/∂W^l|| ∝ ||a^{l-1}|| · ||δ^l||

        With row centering, the forward gain per layer is:
            g_f = √((π-1)/π) ≈ 0.826  (structural, from centering)

        With per-layer std s_l, after l-1 forward layers:
            ||a^{l-1}|| ∝ ∏_{k=1}^{l-1} (s_k · g_f)

        And backward from layer L to l:
            ||δ^l|| ∝ ∏_{k=l+1}^{L} s_k   (backward gain ≈ 1 per layer)

        Gradient norm at layer l:
            ||∇W^l|| ∝ g_f^{l-1} · ∏_{k≠l} s_k

        For uniform gradients across layers, we need:
            g_f^{l-1} · s_l^{-1} = const  ⟹  s_l ∝ g_f^{l-1}

        Centering at middle layer (l = (L+1)/2):
            s_l = r^{η·(l - (L+1)/2)}

        where r = √((π-1)/π) ≈ 0.826.

    Effect:
        - Early layers (l < L/2): s_l > 1, boosted variance compensates
          future forward decay
        - Late layers (l > L/2): s_l < 1, reduced variance prevents
          backward explosion
        - All layers keep full row centering → geometry preserved

    Args:
        layer: nn.Linear layer to initialize.
        layer_index: 0-based index of this layer in the network.
        n_layers: Total number of layers in the network.
        eta: Correction strength. 0 = standard row-centered He (var_adj),
             1 = full gradient balance. Default 1.0.
    """
    fan_in = layer.weight.shape[1]

    # Per-layer scaling factor
    r = math.sqrt((math.pi - 1.0) / math.pi)  # ≈ 0.826
    l = layer_index + 1  # 1-indexed layer number
    L = n_layers
    s_l = r ** (eta * (l - (L + 1) / 2))

    # Base std: variance-adjusted He (compensates average forward gain loss)
    # This ensures the geometric mean of forward gains ≈ 1
    var_adj_factor = math.sqrt(math.pi / (math.pi - 1.0))  # ≈ 1.211
    base_std = math.sqrt(2.0 / fan_in) * var_adj_factor
    target_std = base_std * s_l

    with torch.no_grad():
        # Sample Gaussian weights
        layer.weight.normal_(mean=0.0, std=target_std)

        # Row centering: each row sums to zero
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)

        # Renormalize rows to target_std (centering changes row norms)
        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        if layer.bias is not None:
            layer.bias.zero_()


@register_initializer("row_centered_forward_balanced")
def row_centered_forward_balanced_init(layer: nn.Linear, **kwargs) -> None:
    """Row-centered with variance targeting forward gain = 1.0.

    THIS IS A DIAGNOSTIC INITIALIZER — not intended as a solution.
    Its purpose is to validate the forward-backward gain asymmetry theory.

    Theory (from BP equations):
        Row-centered W operates on centered activations:
            z_j = Σ_k W_{jk}(a_k - ā)  since Σ_k W_{jk} = 0

        Forward variance is proportional to Var(a), not E[a²]:
            Var(z_j) ≈ Var(W) · d · Var(a)

        For post-ReLU: Var(a)/E[a²] = (π-1)/π ≈ 0.682

        To make forward gain = 1:
            Var(W) = 2π / ((π-1) · d) ≈ 2.93/d

    Expected behavior:
        - Forward gain ≈ 1.0 (activations preserved)
        - Backward gain ≈ 1.47 (error signals explode backward)
        - This proves the asymmetry: no single variance fixes both directions

    After confirming the asymmetry, use the alpha sweep (partial centering)
    to find a structural solution.
    """
    fan_in = layer.weight.shape[1]

    # Var(W) = 2π / ((π-1) · d) ≈ 2.934 / d
    factor = 2.0 * math.pi / (math.pi - 1.0)
    target_std = math.sqrt(factor / fan_in)

    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=target_std)
        layer.weight -= layer.weight.mean(dim=1, keepdim=True)

        row_stds = layer.weight.std(dim=1, keepdim=True, unbiased=False)
        row_stds = torch.clamp(row_stds, min=1e-8)
        layer.weight *= target_std / row_stds

        if layer.bias is not None:
            layer.bias.zero_()
