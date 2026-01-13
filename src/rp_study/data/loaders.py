"""
Data loading utilities for MNIST and Fashion-MNIST datasets.
"""

from typing import Tuple, Optional, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transform(flatten: bool = True) -> transforms.Compose:
    """Get the standard transform for MNIST-like datasets.

    Args:
        flatten: Whether to flatten the 28x28 image to 784 dimensions.

    Returns:
        Composed transform pipeline.
    """
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda img: img.view(-1)))
    return transforms.Compose(transform_list)


def load_mnist(
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
) -> Tuple:
    """Load MNIST dataset.

    Args:
        data_dir: Directory to download/load data from.
        train: Whether to load training or test set.
        num_samples: Number of samples to load (None for all).
        flatten: Whether to flatten images to 784 dimensions.
        as_numpy: Whether to return numpy arrays instead of torch tensors.

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    transform = get_transform(flatten=flatten)
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)

    if num_samples is not None:
        dataset = Subset(dataset, range(min(num_samples, len(dataset))))

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))

    if as_numpy:
        return X.numpy(), y.numpy()
    return X, y


def load_fashion_mnist(
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
) -> Tuple:
    """Load Fashion-MNIST dataset.

    Args:
        data_dir: Directory to download/load data from.
        train: Whether to load training or test set.
        num_samples: Number of samples to load (None for all).
        flatten: Whether to flatten images to 784 dimensions.
        as_numpy: Whether to return numpy arrays instead of torch tensors.

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    transform = get_transform(flatten=flatten)
    dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)

    if num_samples is not None:
        dataset = Subset(dataset, range(min(num_samples, len(dataset))))

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))

    if as_numpy:
        return X.numpy(), y.numpy()
    return X, y


def get_data_loader(
    dataset_name: Literal["mnist", "fashion_mnist"],
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple:
    """Unified data loading function.

    Args:
        dataset_name: Which dataset to load ("mnist" or "fashion_mnist").
        data_dir: Directory to download/load data from.
        train: Whether to load training or test set.
        num_samples: Number of samples to load (None for all).
        flatten: Whether to flatten images to 784 dimensions.
        as_numpy: Whether to return numpy arrays instead of torch tensors.
        device: Device to move tensors to (ignored if as_numpy=True).

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    if dataset_name == "mnist":
        X, y = load_mnist(data_dir, train, num_samples, flatten, as_numpy)
    elif dataset_name == "fashion_mnist":
        X, y = load_fashion_mnist(data_dir, train, num_samples, flatten, as_numpy)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not as_numpy and device is not None:
        X = X.to(device)
        y = y.to(device)

    return X, y


def load_openml_mnist(as_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST via sklearn's fetch_openml (for backward compatibility).

    This is the original loading method used in the notebook.

    Args:
        as_numpy: Whether to return numpy arrays.

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()

    return X, y


def load_openml_fashion_mnist(as_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST via sklearn's fetch_openml (for backward compatibility).

    This is the original loading method used in the notebook.

    Args:
        as_numpy: Whether to return numpy arrays.

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    from sklearn.datasets import fetch_openml

    fashion = fetch_openml("Fashion-MNIST", version=1)
    X = fashion.data.astype(np.float32)
    y = fashion.target.astype(np.int64)

    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()

    return X, y
