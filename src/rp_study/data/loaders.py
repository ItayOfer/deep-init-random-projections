"""
Data loading utilities for MNIST, Fashion-MNIST, and CIFAR-10 datasets.
"""

from typing import Dict, Tuple, Optional, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DATASET_NORMALIZATION: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def get_transform(
    dataset_name: Literal["mnist", "fashion_mnist", "cifar10"],
    flatten: bool = True,
    normalize: bool = False,
) -> transforms.Compose:
    """Get the standard transform for supported datasets.

    Args:
        dataset_name: Which dataset transform to build.
        flatten: Whether to flatten the image tensor.
        normalize: Whether to apply dataset-specific normalization.

    Returns:
        Composed transform pipeline.
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        mean, std = DATASET_NORMALIZATION[dataset_name]
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    if flatten:
        transform_list.append(transforms.Lambda(lambda img: img.view(-1)))
    return transforms.Compose(transform_list)


def _subset_dataset(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dataset:
    """Optionally subset a dataset, using a deterministic permutation when seeded."""
    if num_samples is None:
        return dataset

    n_select = min(num_samples, len(dataset))
    if seed is None:
        indices = list(range(n_select))
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(dataset))[:n_select].tolist()
    return Subset(dataset, indices)


def load_mnist(
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
    normalize: bool = False,
    seed: Optional[int] = None,
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
    transform = get_transform("mnist", flatten=flatten, normalize=normalize)
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    dataset = _subset_dataset(dataset, num_samples=num_samples, seed=seed)

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
    normalize: bool = False,
    seed: Optional[int] = None,
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
    transform = get_transform("fashion_mnist", flatten=flatten, normalize=normalize)
    dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    dataset = _subset_dataset(dataset, num_samples=num_samples, seed=seed)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))

    if as_numpy:
        return X.numpy(), y.numpy()
    return X, y


def load_cifar10(
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
    normalize: bool = False,
    seed: Optional[int] = None,
) -> Tuple:
    """Load CIFAR-10 dataset.

    Args:
        data_dir: Directory to download/load data from.
        train: Whether to load training or test set.
        num_samples: Number of samples to load (None for all).
        flatten: Whether to flatten images to 3072 dimensions.
        as_numpy: Whether to return numpy arrays instead of torch tensors.
        normalize: Whether to apply CIFAR-10 normalization.
        seed: Optional random seed for deterministic subsampling.

    Returns:
        Tuple of (X, y) where X is data and y is labels.
    """
    transform = get_transform("cifar10", flatten=flatten, normalize=normalize)
    dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    dataset = _subset_dataset(dataset, num_samples=num_samples, seed=seed)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, y = next(iter(loader))

    if as_numpy:
        return X.numpy(), y.numpy()
    return X, y


def get_dataset(
    dataset_name: Literal["mnist", "fashion_mnist", "cifar10"],
    data_dir: str = "./data",
    train: bool = True,
    flatten: bool = True,
    normalize: bool = False,
) -> Dataset:
    """Return a torchvision-style dataset object for iterative training."""
    transform = get_transform(dataset_name, flatten=flatten, normalize=normalize)
    if dataset_name == "mnist":
        return datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)
    if dataset_name == "cifar10":
        return datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def create_classification_loaders(
    dataset_name: Literal["mnist", "fashion_mnist", "cifar10"],
    data_dir: str = "./data",
    batch_size: int = 128,
    eval_batch_size: Optional[int] = None,
    num_train_samples: Optional[int] = None,
    num_test_samples: Optional[int] = None,
    flatten: bool = False,
    normalize: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test loaders for supervised classification experiments."""
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=True,
        flatten=flatten,
        normalize=normalize,
    )
    test_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=False,
        flatten=flatten,
        normalize=normalize,
    )

    train_dataset = _subset_dataset(train_dataset, num_samples=num_train_samples, seed=seed)
    test_dataset = _subset_dataset(test_dataset, num_samples=num_test_samples, seed=seed + 1)

    eval_batch_size = eval_batch_size or batch_size
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_data_loader(
    dataset_name: Literal["mnist", "fashion_mnist", "cifar10"],
    data_dir: str = "./data",
    train: bool = True,
    num_samples: Optional[int] = None,
    flatten: bool = True,
    as_numpy: bool = False,
    device: Optional[torch.device] = None,
    normalize: bool = False,
    seed: Optional[int] = None,
) -> Tuple:
    """Unified data loading function.

    Args:
        dataset_name: Which dataset to load.
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
        X, y = load_mnist(
            data_dir=data_dir,
            train=train,
            num_samples=num_samples,
            flatten=flatten,
            as_numpy=as_numpy,
            normalize=normalize,
            seed=seed,
        )
    elif dataset_name == "fashion_mnist":
        X, y = load_fashion_mnist(
            data_dir=data_dir,
            train=train,
            num_samples=num_samples,
            flatten=flatten,
            as_numpy=as_numpy,
            normalize=normalize,
            seed=seed,
        )
    elif dataset_name == "cifar10":
        X, y = load_cifar10(
            data_dir=data_dir,
            train=train,
            num_samples=num_samples,
            flatten=flatten,
            as_numpy=as_numpy,
            normalize=normalize,
            seed=seed,
        )
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
