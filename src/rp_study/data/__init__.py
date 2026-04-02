"""Data loading utilities for image datasets and synthetic shapes."""

from .loaders import (
    load_mnist,
    load_fashion_mnist,
    load_cifar10,
    get_dataset,
    get_data_loader,
    create_classification_loaders,
)
from .shapes import generate_circle, generate_ellipse, generate_square, generate_rectangle
