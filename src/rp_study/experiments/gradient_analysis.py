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
class ExperimentResults:
    """Results from a gradient analysis experiment."""

    loss_value: float
    grad_stats: Dict[str, LayerGradientStats]  # Keyed by "L1", "L2", etc.
    activation_stats: List[ActivationStats]
    network_config: NetworkConfig
    experiment_config: GradientExperimentConfig
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
            variance=network_config.weight_variance,
            mean=network_config.weight_mean,
        ).to(self.device)

    def run(self) -> ExperimentResults:
        """Run the experiment: forward pass, compute loss, backward pass, collect stats.

        Returns:
            ExperimentResults containing all gradient and activation statistics.
        """
        self.net.train()

        # Forward pass with activations
        output, activations = self.net(self.inputs, return_activations=True)

        # Compute loss (squared error)
        loss = torch.sum((self.targets - output) ** 2)

        # Backward pass
        self.net.zero_grad()
        loss.backward()

        # Collect statistics
        grad_stats = self._collect_gradient_stats()
        activation_stats = self._collect_activation_stats(activations)

        return ExperimentResults(
            loss_value=loss.item(),
            grad_stats=grad_stats,
            activation_stats=activation_stats,
            network_config=self.network_config,
            experiment_config=self.grad_config,
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
