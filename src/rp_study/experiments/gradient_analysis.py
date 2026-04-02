"""
Gradient analysis experiment framework.

Provides a unified class for running gradient analysis experiments
on neural networks with various initialization strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from ..config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from ..data.loaders import get_data_loader
from ..models.networks import FeedForward


@dataclass
class LayerGradientStats:
    """Statistics about gradients for a single layer."""

    layer_idx: int
    matrix: np.ndarray  # The full gradient matrix
    row_norms: np.ndarray  # L2 norm of each row
    zero_count: int  # Number of exactly zero gradient entries
    total_entries: int  # Total entries in gradient matrix
    zero_row_count: int  # Number of rows that are entirely zero
    total_rows: int  # Total number of rows

    @property
    def zero_proportion(self) -> float:
        """Proportion of zero gradient entries."""
        return self.zero_count / self.total_entries if self.total_entries > 0 else 0.0

    @property
    def zero_row_proportion(self) -> float:
        """Proportion of entirely zero rows."""
        return self.zero_row_count / self.total_rows if self.total_rows > 0 else 0.0

    @property
    def mean_row_norm(self) -> float:
        """Mean L2 norm across rows."""
        return float(np.mean(self.row_norms))


@dataclass
class ActivationStats:
    """Statistics about activations for a single layer."""

    layer_idx: int
    values: np.ndarray  # The activation values
    zero_count: int
    total_entries: int
    truly_inactive_neurons: int  # Neurons that are zero for ALL samples
    total_neurons: int

    @property
    def zero_proportion(self) -> float:
        """Proportion of zero activations."""
        return self.zero_count / self.total_entries if self.total_entries > 0 else 0.0

    @property
    def truly_inactive_proportion(self) -> float:
        """Proportion of neurons that are inactive for all samples."""
        return self.truly_inactive_neurons / self.total_neurons if self.total_neurons > 0 else 0.0


@dataclass
class LayerSignalStats:
    """Per-layer forward/backward signal statistics.

    These statistics decompose gradient behavior into its forward (activation)
    and backward (error signal) components, enabling diagnosis of the
    forward-backward gain asymmetry caused by row-centered initialization.

    Mathematical context:
        ∂C/∂W^l = δ^l · (a^{l-1})^T  (BP4)

    So ||∂C/∂W^l|| ∝ ||δ^l|| · ||a^{l-1}||. By tracking these separately,
    we can determine whether gradient issues come from forward signal decay
    (activation norms shrinking) or backward signal growth (error norms growing).
    """

    layer_idx: int
    activation_rms: float          # rms(a^l) = ||a^l||_F / sqrt(n_samples * d_l)
                                   # Per-neuron signal level (dimension-normalized)
    preactivation_rms: float       # rms(z^l) = ||z^l||_F / sqrt(n_samples * d_l)
    error_signal_rms: float        # rms(δ^l) = ||δ^l||_F / sqrt(n_samples * d_l)
    forward_gain: float            # rms(a^l) / rms(a^{l-1})
                                   # Per-neuron signal gain. Should be ~1.0 for He+ReLU
                                   # regardless of layer dimensions.
    backward_gain: float           # rms(δ^l) / rms(δ^{l+1})
                                   # Per-neuron backward signal gain.
    gain_product: float            # forward_gain * backward_gain
                                   # Useful for diagnosing coupled gain trade-offs.
    relu_survival_rate: float      # fraction of z^l entries > 0
    centering_ratio: float         # Var(a^{l-1})/E[(a^{l-1})²] per sample, averaged
                                   # This is the factor by which row-centering reduces
                                   # the effective forward gain. Should be ~0.68 for
                                   # half-Gaussian post-ReLU, and ~1.0 for standard He.
    gradient_scale_proxy: float    # rms(a^{l-1}) * rms(δ^l)
                                   # A scale proxy for ||∂C/∂W^l|| from BP4.


@dataclass
class ExperimentResults:
    """Results from a gradient analysis experiment."""

    loss_value: float
    grad_stats: Dict[str, LayerGradientStats]  # Keyed by "L1", "L2", etc.
    activation_stats: List[ActivationStats]
    network_config: NetworkConfig
    experiment_config: GradientExperimentConfig
    signal_stats: List[LayerSignalStats] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_zero_gradient_counts(self) -> Dict[str, int]:
        """Get zero gradient counts per layer."""
        return {k: v.zero_count for k, v in self.grad_stats.items()}

    def get_zero_gradient_proportions(self) -> Dict[str, float]:
        """Get zero gradient proportions per layer."""
        return {k: v.zero_proportion for k, v in self.grad_stats.items()}

    def get_mean_row_norms(self) -> Dict[str, float]:
        """Get mean row norms per layer."""
        return {k: v.mean_row_norm for k, v in self.grad_stats.items()}

    def get_activation_zero_proportions(self) -> Dict[str, float]:
        """Get zero activation proportions per layer.

        This measures the proportion of activation values that are exactly zero
        (due to ReLU). This should be ~50% for all initializations.
        """
        return {f"L{s.layer_idx}": s.zero_proportion for s in self.activation_stats}

    def get_truly_inactive_proportions(self) -> Dict[str, float]:
        """Get proportion of truly inactive neurons per layer.

        A neuron is "truly inactive" if it outputs zero for ALL samples in the batch.
        This is the neuron death phenomenon from the thesis.
        """
        return {f"L{s.layer_idx}": s.truly_inactive_proportion for s in self.activation_stats}

    # --- Signal statistics accessors ---

    def get_forward_gains(self) -> Dict[str, float]:
        """Get per-layer forward gain: ||a^l||_F / ||a^{l-1}||_F."""
        return {f"L{s.layer_idx}": s.forward_gain for s in self.signal_stats}

    def get_backward_gains(self) -> Dict[str, float]:
        """Get per-layer backward gain: ||δ^l||_F / ||δ^{l+1}||_F."""
        return {f"L{s.layer_idx}": s.backward_gain for s in self.signal_stats}

    def get_gain_products(self) -> Dict[str, float]:
        """Get per-layer forward/backward gain product."""
        return {f"L{s.layer_idx}": s.gain_product for s in self.signal_stats}

    def get_relu_survival_rates(self) -> Dict[str, float]:
        """Get per-layer ReLU survival rate (fraction of z^l > 0)."""
        return {f"L{s.layer_idx}": s.relu_survival_rate for s in self.signal_stats}

    def get_centering_ratios(self) -> Dict[str, float]:
        """Get per-layer centering ratio Var(a)/E[a²].

        This ratio measures how much row-centering reduces the effective forward
        gain compared to standard (non-centered) initialization. A ratio of 1.0
        means no reduction (standard He), while ~0.68 means the centering removes
        about 32% of the effective input variance.
        """
        return {f"L{s.layer_idx}": s.centering_ratio for s in self.signal_stats}

    def get_activation_rms(self) -> Dict[str, float]:
        """Get per-layer activation RMS: ||a^l||_F / sqrt(n_samples * d_l)."""
        return {f"L{s.layer_idx}": s.activation_rms for s in self.signal_stats}

    def get_error_signal_rms(self) -> Dict[str, float]:
        """Get per-layer error signal RMS: ||δ^l||_F / sqrt(n_samples * d_l)."""
        return {f"L{s.layer_idx}": s.error_signal_rms for s in self.signal_stats}

    def get_gradient_scale_proxies(self) -> Dict[str, float]:
        """Get BP4 proxy rms(a^{l-1}) * rms(δ^l) for each hidden layer."""
        return {f"L{s.layer_idx}": s.gradient_scale_proxy for s in self.signal_stats}


class GradientExperiment:
    """Experiment class for analyzing gradient flow in neural networks.

    This class handles:
    - Loading data
    - Building the network with specified initialization
    - Running forward/backward pass
    - Collecting gradient and activation statistics

    Example:
        exp_config = ExperimentConfig(seed=42)
        net_config = NetworkConfig(layer_sizes=[784, 784, 512, 256, 1], init_strategy="he")
        grad_config = GradientExperimentConfig(num_samples=1000, dataset="fashion_mnist")

        experiment = GradientExperiment(exp_config, net_config, grad_config)
        results = experiment.run()

        # Access results
        print(f"Loss: {results.loss_value}")
        print(f"Zero gradients: {results.get_zero_gradient_proportions()}")
    """

    def __init__(
        self,
        exp_config: ExperimentConfig,
        network_config: NetworkConfig,
        grad_config: GradientExperimentConfig,
    ):
        self.exp_config = exp_config
        self.network_config = network_config
        self.grad_config = grad_config

        # Setup
        self.exp_config.setup_seeds()
        self.device = self.exp_config.get_device()

        # Load data
        self.inputs, self.targets = get_data_loader(
            dataset_name=grad_config.dataset,
            data_dir=exp_config.data_dir,
            train=True,
            num_samples=grad_config.num_samples,
            flatten=True,
            device=self.device,
        )
        # Convert targets to float for regression loss
        self.targets = self.targets.float().view(-1, 1)

        # Build network
        self.net = FeedForward(
            layer_sizes=network_config.layer_sizes,
            init_strategy=network_config.init_strategy,
            use_bias=network_config.use_bias,
            variance=network_config.weight_variance,
            mean=network_config.weight_mean,
            **network_config.init_kwargs,
        ).to(self.device)

    def run(self) -> ExperimentResults:
        """Run the experiment: forward pass, compute loss, backward pass, collect stats.

        Collects three types of statistics:
        1. Gradient stats: per-layer weight gradient norms, zero counts
        2. Activation stats: per-layer activation zeros, dead neurons
        3. Signal stats: forward/backward gain decomposition, centering ratios

        The signal stats use backward hooks to capture δ^l (error signals) at
        each layer. This enables decomposing ||∂C/∂W^l|| into ||δ^l|| · ||a^{l-1}||
        to diagnose forward-backward gain asymmetry.

        Returns:
            ExperimentResults containing all statistics.
        """
        self.net.train()

        # --- Register backward hooks to capture δ^l ---
        # For nn.Linear layer l, grad_output[0] = ∂C/∂z^l = δ^l
        # because ReLU backward has already been applied before reaching
        # the Linear module's backward.
        error_signals = {}  # layer_idx -> δ^l tensor
        hooks = []

        for idx, layer in enumerate(self.net.layers, start=1):
            def make_hook(layer_idx):
                def hook_fn(module, grad_input, grad_output):
                    # grad_output[0] shape: (n_samples, fan_out)
                    # This IS δ^l: the gradient of C w.r.t. the output of
                    # this linear layer (after ReLU backward for hidden layers)
                    error_signals[layer_idx] = grad_output[0].detach().clone()
                return hook_fn
            h = layer.register_full_backward_hook(make_hook(idx))
            hooks.append(h)

        # --- Forward pass with both activations and pre-activations ---
        output, activations, preactivations = self.net(
            self.inputs, return_activations=True, return_preactivations=True
        )

        # Compute loss (squared error)
        loss = torch.sum((self.targets - output) ** 2)

        # Backward pass (hooks capture δ^l at each layer)
        self.net.zero_grad()
        loss.backward()

        # Remove hooks
        for h in hooks:
            h.remove()

        # --- Collect all statistics ---
        grad_stats = self._collect_gradient_stats()
        activation_stats = self._collect_activation_stats(activations)
        signal_stats = self._collect_signal_stats(
            activations, preactivations, error_signals
        )

        return ExperimentResults(
            loss_value=loss.item(),
            grad_stats=grad_stats,
            activation_stats=activation_stats,
            network_config=self.network_config,
            experiment_config=self.grad_config,
            signal_stats=signal_stats,
            metadata={"device": str(self.device)},
        )

    def _collect_gradient_stats(self) -> Dict[str, LayerGradientStats]:
        """Collect gradient statistics for all layers."""
        stats = {}

        for idx, layer in enumerate(self.net.layers, start=1):
            if layer.weight.grad is None:
                continue

            g = layer.weight.grad.detach().cpu().numpy()

            stats[f"L{idx}"] = LayerGradientStats(
                layer_idx=idx,
                matrix=g,
                row_norms=np.linalg.norm(g, axis=1),
                zero_count=int((g == 0).sum()),
                total_entries=g.size,
                zero_row_count=int(np.sum(np.all(g == 0, axis=1))),
                total_rows=g.shape[0],
            )

        return stats

    def _collect_signal_stats(
        self,
        activations: List[torch.Tensor],
        preactivations: List[torch.Tensor],
        error_signals: Dict[int, torch.Tensor],
    ) -> List[LayerSignalStats]:
        """Collect per-layer signal statistics for gain decomposition.

        Math:
            Forward gain at layer l:
                gain_fwd_l = ||a^l||_F / ||a^{l-1}||_F
                where a^0 = X (input), a^l = ReLU(z^l) for hidden layers

            Backward gain at layer l (hidden layers only):
                gain_bwd_l = ||δ^l||_F / ||δ^{l+1}||_F
                where δ^l = (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l)

            Centering ratio for layer l's INPUT a^{l-1}:
                For each sample i: ratio_i = Var_k(a^{l-1}_{ik}) / E_k[(a^{l-1}_{ik})²]
                Then centering_ratio = mean_i(ratio_i)
                This measures how much row-centering reduces the effective
                forward gain. Equals 1 - (E[a])²/E[a²] per sample.

            ReLU survival at layer l:
                fraction of z^l entries that are > 0

        Args:
            activations: List of post-ReLU activations a^l for l=1..L
                         (last entry is linear output for output layer)
            preactivations: List of pre-ReLU values z^l for l=1..L-1
                            (hidden layers only)
            error_signals: Dict mapping layer_idx -> δ^l tensor
        """
        stats = []
        n_hidden = len(preactivations)  # Number of hidden layers
        n_samples = self.inputs.shape[0]

        # Helper: compute RMS = ||x||_F / sqrt(n_samples * d)
        # This is the per-neuron signal level, independent of layer width.
        def rms(tensor: torch.Tensor) -> float:
            return torch.norm(tensor).item() / (tensor.numel() ** 0.5)

        # Compute RMS for all activations: a^0 = input, a^1..a^{L-1} = hidden
        # activations list: [a^1, a^2, ..., a^L] where a^L is output (no ReLU)
        act_rms = [rms(self.inputs)]  # Index 0 = input a^0
        for act in activations:
            act_rms.append(rms(act))

        # Compute RMS for all error signals: δ^1..δ^L
        err_rms = {}
        for idx in sorted(error_signals.keys()):
            err_rms[idx] = rms(error_signals[idx])

        # Build signal stats for each hidden layer (l = 1, ..., L-1)
        for i in range(n_hidden):
            layer_idx = i + 1  # 1-indexed

            # Pre-activation z^l (shape: n_samples x d_l)
            z = preactivations[i]

            # RMS values
            a_rms_val = act_rms[layer_idx]      # rms(a^l)
            z_rms_val = rms(z)                   # rms(z^l)
            delta_rms_val = err_rms.get(layer_idx, 0.0)  # rms(δ^l)

            # Forward gain: rms(a^l) / rms(a^{l-1})
            prev_rms = act_rms[layer_idx - 1]
            fwd_gain = a_rms_val / prev_rms if prev_rms > 0 else float('nan')

            # Backward gain: rms(δ^l) / rms(δ^{l+1})
            next_delta_rms = err_rms.get(layer_idx + 1, 0.0)
            bwd_gain = delta_rms_val / next_delta_rms if next_delta_rms > 0 else float('nan')
            gain_product = fwd_gain * bwd_gain if np.isfinite(fwd_gain) and np.isfinite(bwd_gain) else float("nan")

            # ReLU survival rate: fraction of z^l > 0
            survival = (z > 0).float().mean().item()

            # Centering ratio: Var(a^{l-1}) / E[(a^{l-1})²] computed per sample
            # a^{l-1} is the input to layer l
            if layer_idx == 1:
                a_prev = self.inputs  # a^0 = X
            else:
                a_prev = activations[layer_idx - 2]  # a^{l-1} from activations list

            # Per-sample computation across the feature dimension:
            # For sample i: E_k[a²] = mean(a_i²), Var_k(a) = E_k[a²] - (E_k[a])²
            # ratio_i = Var_k(a_i) / E_k[a_i²] = 1 - (mean(a_i))² / mean(a_i²)
            a_prev_detached = a_prev.detach()
            per_sample_mean_sq = (a_prev_detached ** 2).mean(dim=1)  # E_k[a²] per sample
            per_sample_mean = a_prev_detached.mean(dim=1)            # E_k[a] per sample
            # Avoid division by zero for samples where all activations are 0
            safe_mean_sq = torch.clamp(per_sample_mean_sq, min=1e-12)
            per_sample_ratio = 1.0 - (per_sample_mean ** 2) / safe_mean_sq
            centering_ratio = per_sample_ratio.mean().item()
            gradient_scale_proxy = prev_rms * delta_rms_val

            stats.append(LayerSignalStats(
                layer_idx=layer_idx,
                activation_rms=a_rms_val,
                preactivation_rms=z_rms_val,
                error_signal_rms=delta_rms_val,
                forward_gain=fwd_gain,
                backward_gain=bwd_gain,
                gain_product=gain_product,
                relu_survival_rate=survival,
                centering_ratio=centering_ratio,
                gradient_scale_proxy=gradient_scale_proxy,
            ))

        return stats

    def _collect_activation_stats(self, activations: List[torch.Tensor]) -> List[ActivationStats]:
        """Collect activation statistics for all layers."""
        stats = []

        for idx, act in enumerate(activations, start=1):
            a = act.detach().cpu().numpy()

            # Truly inactive = neurons that are zero for ALL samples
            truly_inactive = int(np.sum(np.all(a == 0, axis=0)))

            stats.append(ActivationStats(
                layer_idx=idx,
                values=a,
                zero_count=int((a == 0).sum()),
                total_entries=a.size,
                truly_inactive_neurons=truly_inactive,
                total_neurons=a.shape[1],
            ))

        return stats


def run_multi_config_experiment(
    exp_config: ExperimentConfig,
    network_configs: List[NetworkConfig],
    grad_config: GradientExperimentConfig,
) -> List[ExperimentResults]:
    """Run experiments with multiple network configurations.

    Useful for comparing different initializations or architectures.

    Args:
        exp_config: Base experiment configuration.
        network_configs: List of network configurations to test.
        grad_config: Gradient experiment configuration.

    Returns:
        List of ExperimentResults, one per network configuration.
    """
    results = []

    for net_config in network_configs:
        # Reset seeds for fair comparison
        exp_config.setup_seeds()

        experiment = GradientExperiment(exp_config, net_config, grad_config)
        result = experiment.run()
        results.append(result)

    return results


def compare_initializations(
    layer_sizes: List[int],
    init_strategies: List[str],
    num_samples: int = 1000,
    dataset: str = "fashion_mnist",
    seed: int = 42,
) -> Dict[str, ExperimentResults]:
    """Compare different initialization strategies on the same architecture.

    Args:
        layer_sizes: Network architecture.
        init_strategies: List of initialization strategy names.
        num_samples: Number of data samples.
        dataset: Dataset to use.
        seed: Random seed.

    Returns:
        Dictionary mapping init strategy name to its results.
    """
    exp_config = ExperimentConfig(seed=seed)
    grad_config = GradientExperimentConfig(num_samples=num_samples, dataset=dataset)

    results = {}

    for strategy in init_strategies:
        exp_config.setup_seeds()  # Reset for fair comparison
        net_config = NetworkConfig(layer_sizes=layer_sizes, init_strategy=strategy)
        experiment = GradientExperiment(exp_config, net_config, grad_config)
        results[strategy] = experiment.run()

    return results
