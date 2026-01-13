# Random Projections Study

A research toolkit for studying random projections, gradient flow analysis, and neural network initialization strategies.

## Installation

```bash
# Clone the repository
git clone https://github.com/itayofer/thesis.git
cd thesis

# Install dependencies
pip install -e .

# Or install dependencies directly
pip install torch torchvision matplotlib scikit-learn numpy
```

## Project Structure

```
Thesis/
├── src/rp_study/              # Main Python package
│   ├── config.py              # Configuration dataclasses
│   ├── data/                  # Data loading utilities
│   │   ├── loaders.py         # MNIST, Fashion-MNIST loaders
│   │   └── shapes.py          # Synthetic shape generators
│   ├── models/                # Neural network models
│   │   ├── initializers.py    # Extensible initialization strategies
│   │   └── networks.py        # FeedForward network class
│   ├── projections/           # Random projection utilities
│   │   └── random_projections.py
│   ├── experiments/           # Experiment frameworks
│   │   └── gradient_analysis.py
│   ├── analysis/              # Theoretical analysis
│   │   └── kernel.py          # K(α) arc-cosine kernel
│   └── visualization/         # Plotting utilities
│       ├── plots.py
│       ├── gradient_plots.py
│       └── projection_plots.py
├── notebooks/                 # Experiment notebooks
│   ├── 01_shape_experiments.ipynb
│   ├── 02_mnist_projections.ipynb
│   ├── 03_gradient_analysis.ipynb
│   └── 04_kernel_analysis.ipynb
├── Random_Projections.ipynb   # Original notebook (reference)
└── pyproject.toml             # Package configuration
```

## Quick Start

### Running Experiments via Notebooks

The easiest way to run experiments is through the Jupyter notebooks in the `notebooks/` folder. See [notebooks/README.md](notebooks/README.md) for details.

### Using the Package in Python

```python
from rp_study.config import ExperimentConfig, NetworkConfig, GradientExperimentConfig
from rp_study.experiments import GradientExperiment
from rp_study.visualization import gradient_plots

# Configure experiment
exp_config = ExperimentConfig(seed=42)
net_config = NetworkConfig(
    layer_sizes=[784, 784, 512, 256, 1],
    init_strategy="he"  # or "row_centered_he", "custom_variance", etc.
)
grad_config = GradientExperimentConfig(num_samples=1000, dataset="fashion_mnist")

# Run experiment
experiment = GradientExperiment(exp_config, net_config, grad_config)
results = experiment.run()

# Visualize results
gradient_plots.plot_gradient_histograms(results)
gradient_plots.plot_zero_gradient_stats(results)
```

## Available Initialization Strategies

The package includes an extensible initialization system. Built-in strategies:

| Strategy | Description |
|----------|-------------|
| `he` | Standard He/Kaiming initialization: N(0, sqrt(2/fan_in)) |
| `row_centered_he` | He initialization with row means subtracted |
| `custom_variance` | Custom variance: N(mean, sqrt(variance)) |
| `xavier` | Xavier/Glorot initialization |
| `uniform_he` | Uniform distribution with He-like variance |
| `orthogonal` | Orthogonal weight initialization |

### Adding New Initializers

```python
from rp_study.models.initializers import register_initializer

@register_initializer("my_custom_init")
def my_custom_init(layer, **kwargs):
    # Your initialization logic here
    with torch.no_grad():
        layer.weight.normal_(0, 0.01)
        if layer.bias is not None:
            layer.bias.zero_()
```

## Key Components

### Configuration Classes

- `ExperimentConfig`: Base config (seed, device, data directory)
- `NetworkConfig`: Network architecture and initialization
- `GradientExperimentConfig`: Gradient analysis settings

### Data Loaders

```python
from rp_study.data import load_mnist, load_fashion_mnist, get_data_loader

# Load Fashion-MNIST
X, y = load_fashion_mnist(num_samples=1000, flatten=True, as_numpy=True)

# Or use unified loader
X, y = get_data_loader("fashion_mnist", num_samples=1000)
```

### Random Projections

```python
from rp_study.projections import random_projection_matrix, multi_layer_projection

# Single projection matrix
R = random_projection_matrix(784, 2, variance="he")

# Multi-layer RP + ReLU
X_projected = multi_layer_projection(X, num_layers=10, mode="square")
```

### Gradient Analysis

```python
from rp_study.experiments import GradientExperiment, compare_initializations

# Compare different initializations
results = compare_initializations(
    layer_sizes=[784, 512, 256, 1],
    init_strategies=["he", "row_centered_he", "xavier"],
    num_samples=1000
)
```

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `01_shape_experiments` | 2D shape transformations under RP + ReLU |
| `02_mnist_projections` | MNIST/Fashion-MNIST projections, initialization comparison |
| `03_gradient_analysis` | Gradient flow analysis with different initializations |
| `04_kernel_analysis` | K(α) arc-cosine kernel theoretical analysis |

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0
- scikit-learn >= 0.24.0

## License

MIT
