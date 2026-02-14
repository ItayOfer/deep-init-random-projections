# rp_study Package

Core Python package for random projections research.

## Module Overview

### config.py
Configuration dataclasses for experiments.

```python
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig

exp_config = ExperimentConfig(seed=42, device="auto")
net_config = NetworkConfig(layer_sizes=[784, 512, 1], init_strategy="he")
grad_config = GradientExperimentConfig(num_samples=1000, dataset="fashion_mnist")
```

### data/
Data loading utilities.

**loaders.py**: MNIST and Fashion-MNIST loaders
```python
from rp_study.data import load_mnist, load_fashion_mnist, get_data_loader

X, y = load_fashion_mnist(num_samples=1000, as_numpy=True)
```

**shapes.py**: Synthetic 2D shape generators
```python
from rp_study.data import generate_circle, generate_ellipse, generate_square
from rp_study.data.shapes import generate_standard_shapes

shapes = generate_standard_shapes(n_points=200)
```

### models/
Neural network models and initialization.

**initializers.py**: Extensible initialization registry
```python
from rp_study.models import list_initializers, register_initializer

print(list_initializers())  # ['he', 'row_centered_he', 'custom_variance', ...]

# Add custom initializer
@register_initializer("my_init")
def my_init(layer, **kwargs):
    ...
```

**networks.py**: FeedForward network class
```python
from rp_study.models import FeedForward, create_deep_network

net = FeedForward([784, 512, 256, 1], init_strategy="he")
deep_net = create_deep_network(num_hidden_layers=100, hidden_widths="random")
```

### projections/
Random projection utilities.

```python
from rp_study.projections import random_projection_matrix, multi_layer_projection, multi_layer_rp_with_init

R = random_projection_matrix(784, 2, variance="he")
X_proj = multi_layer_projection(X, num_layers=10, mode="square")

# Use any registry initializer for multi-layer RP + ReLU
X_proj = multi_layer_rp_with_init(X, n_layers=10, init_strategy="row_centered_he")
```

### experiments/
Experiment frameworks.

**gradient_analysis.py**: Gradient flow analysis
```python
from rp_study.experiments import GradientExperiment, compare_initializations

experiment = GradientExperiment(exp_config, net_config, grad_config)
results = experiment.run()

# Or compare multiple initializations
results = compare_initializations(
    layer_sizes=[784, 512, 1],
    init_strategies=["he", "row_centered_he"]
)
```

### analysis/
Theoretical analysis functions.

**kernel.py**: K(α) arc-cosine kernel
```python
from rp_study.analysis import k_alpha, compute_output_angle, plot_k_alpha

k_values = k_alpha(np.linspace(0, np.pi, 100))
```

### visualization/
Plotting utilities.

```python
from rp_study.visualization import gradient_plots, projection_plots

# Gradient analysis plots
gradient_plots.plot_gradient_histograms(results)
gradient_plots.plot_zero_gradient_stats(results)

# Projection plots
projection_plots.plot_pca_vs_rp(X, X_pca, X_rp, labels=y)
```

## Adding New Initialization Strategies

The initializer system uses a registry pattern for easy extension:

```python
from rp_study.models.initializers import register_initializer
import torch.nn as nn
import math

@register_initializer("lecun")
def lecun_init(layer: nn.Linear, **kwargs):
    """LeCun initialization for SELU networks."""
    fan_in = layer.weight.shape[1]
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        layer.weight.normal_(0.0, std)
        if layer.bias is not None:
            layer.bias.zero_()
```

After registering, use it like any other strategy:
```python
net = FeedForward([784, 512, 1], init_strategy="lecun")
```
