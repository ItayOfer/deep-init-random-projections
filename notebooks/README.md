# Experiment Notebooks

This folder contains Jupyter notebooks for running experiments. Each notebook is self-contained and designed to be easy to use.

## Prerequisites

Make sure you have installed the required dependencies:

```bash
pip install torch torchvision matplotlib scikit-learn numpy jupyter
```

## Notebooks

### 01_shape_experiments.ipynb

**Purpose**: Explore how random projections with ReLU affect 2D geometric shapes.

**Experiments**:
- Compare PCA vs Random Projections on shapes (circle, ellipse, square, rectangle)
- Multi-layer RP + ReLU transformations (1, 2, 3, 5, 10, 20 layers)
- Effect of shape position (negative vs positive coordinates)
- Tilted shapes experiment

**Key Parameters** (modify in the Configuration cell):
```python
N_POINTS = 200           # Points per shape
LAYER_COUNTS = [1, 2, 3, 5, 10, 20]  # Layers to test
```

---

### 02_mnist_projections.ipynb

**Purpose**: Study random projections on MNIST/Fashion-MNIST image data.

**Experiments**:
- PCA vs Random Projection comparison
- Rectangle vs Square projection matrices
- Multi-layer RP + ReLU transformations
- **Initialization comparison** (Section 4b): Uses all registry initializers via `multi_layer_rp_with_init()`
- Johnson-Lindenstrauss comparison
- GPU-accelerated multi-layer projections

**Key Parameters**:
```python
DATASET = "fashion_mnist"  # or "mnist"
TARGET_DIM = 2
LAYER_COUNTS = [1, 5, 10, 20]
```

**Note**: Initialization strategies are imported from the registry. See `INITIALIZERS.md` for the full list.

---

### 03_gradient_analysis.ipynb

**Purpose**: Analyze gradient flow in neural networks with different initializations.

**Experiments**:
- Single architecture gradient analysis
- Comparison across initialization strategies
- Deep network (100+ layers) analysis
- Zero gradient and dead neuron statistics
- Custom variance experiments (2/d, 2.5/d, 3/d, 4/d)

**Key Parameters**:
```python
SEED = 42
DATASET = "fashion_mnist"
NUM_SAMPLES = 1000
LAYER_SIZES = [784, 784, 512, 256, 1]
INIT_STRATEGY = "he"
```

**Outputs**:
- Gradient entry histograms per layer
- Row norm histograms
- Activation histograms
- Zero gradient statistics
- Mean row norm per layer plots

---

### 05_initializer_dashboard.ipynb

**Purpose**: Unified one-stop-shop for evaluating initialization strategies. Configure once, get geometry + gradient + statistics analysis in one run.

**Sections**:
1. **Geometry**: PCA projections after multi-layer RP + ReLU (grid: initializers x layer counts)
2. **Gradient Flow**: Mean row norms, zero proportions across layers
3. **Summary Statistics**: Table with gradient zeros, activation zeros, dead neurons, gain estimates

**Key Parameters** (all in one configuration cell):
```python
INIT_STRATEGIES = ["he", "row_centered_he", ...]  # From registry
GEOM_LAYER_COUNTS = [1, 5, 10, 20]
GRAD_N_HIDDEN = 50
GRAD_WIDTH = 784
DATASET = "fashion_mnist"
```

**Workflow for new initializers**:
1. Register in `src/rp_study/models/initializers.py`
2. Add to `INIT_STRATEGIES` list in this notebook
3. Run all cells

---

### 04_kernel_analysis.ipynb

**Purpose**: Explore the theoretical K(α) arc-cosine kernel function.

**Topics**:
- K(α) kernel visualization
- Input-output angle relationship
- Inner product transformation under ReLU
- Angle preservation analysis
- Multi-layer kernel composition

**The K(α) Kernel**:
```
K(α) = (sin(α) + (π - α)cos(α)) / (2π)
```

This kernel describes the expected inner product between two vectors after applying a random projection followed by ReLU activation.

---

## Running the Notebooks

### Option 1: Jupyter Notebook
```bash
cd notebooks
jupyter notebook
```

### Option 2: JupyterLab
```bash
cd notebooks
jupyter lab
```

### Option 3: VS Code
Open any `.ipynb` file in VS Code with the Jupyter extension installed.

### Option 4: Google Colab
Upload the notebook to Google Colab. Add this cell at the top to install dependencies:
```python
!pip install torch torchvision matplotlib scikit-learn

# If running from Colab, clone the repo first:
!git clone https://github.com/itayofer/thesis.git
import sys
sys.path.insert(0, 'thesis/src')
```

## Tips

1. **Modify configurations at the top**: Each notebook has a "Configuration" section where you can adjust parameters before running.

2. **Run cells in order**: The notebooks are designed to be run from top to bottom.

3. **GPU acceleration**: Notebooks 02 and 03 support GPU acceleration. The device is auto-detected.

4. **Memory considerations**: For deep networks (100+ layers) or large sample sizes, consider:
   - Reducing `NUM_SAMPLES`
   - Using a GPU
   - Running fewer layer configurations

5. **Reproducibility**: All notebooks set random seeds for reproducible results.
