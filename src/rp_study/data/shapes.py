"""
Synthetic shape generation utilities.

Generates 2D geometric shapes for random projection experiments.
"""

from typing import Tuple, Optional
import numpy as np


def generate_circle(
    n_points: int = 200,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate points along a circle boundary.

    Args:
        n_points: Number of points to generate.
        radius: Radius of the circle.
        center: Center coordinates (x, y).

    Returns:
        Array of shape (n_points, 2) with circle boundary points.
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.vstack([x, y]).T


def generate_ellipse(
    n_points: int = 200,
    a: float = 2.0,
    b: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate points along an ellipse boundary.

    Args:
        n_points: Number of points to generate.
        a: Horizontal radius (semi-major axis along x).
        b: Vertical radius (semi-minor axis along y).
        center: Center coordinates (x, y).

    Returns:
        Array of shape (n_points, 2) with ellipse boundary points.
    """
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + a * np.cos(angles)
    y = center[1] + b * np.sin(angles)
    return np.vstack([x, y]).T


def generate_square(
    n_points: int = 200,
    side: float = 2.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate random points inside a square.

    Args:
        n_points: Number of points to generate.
        side: Side length of the square.
        center: Center coordinates (x, y).

    Returns:
        Array of shape (n_points, 2) with points inside the square.
    """
    low = np.array(center) - side / 2
    high = np.array(center) + side / 2
    return np.random.uniform(low, high, size=(n_points, 2))


def generate_rectangle(
    n_points: int = 200,
    width: float = 4.0,
    height: float = 2.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Generate random points inside a rectangle.

    Args:
        n_points: Number of points to generate.
        width: Width of the rectangle.
        height: Height of the rectangle.
        center: Center coordinates (x, y).

    Returns:
        Array of shape (n_points, 2) with points inside the rectangle.
    """
    low = np.array(center) - np.array([width / 2, height / 2])
    high = np.array(center) + np.array([width / 2, height / 2])
    return np.random.uniform(low, high, size=(n_points, 2))


def rotate_shape(X: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a 2D dataset by the specified angle.

    Args:
        X: Array of shape (n_points, 2).
        angle_degrees: Rotation angle in degrees.

    Returns:
        Rotated array of shape (n_points, 2).
    """
    theta = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return X @ rotation_matrix


def generate_shape(
    shape_type: str,
    n_points: int = 200,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_degrees: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Unified shape generation function.

    Args:
        shape_type: One of "circle", "ellipse", "square", "rectangle".
        n_points: Number of points to generate.
        center: Center coordinates.
        rotation_degrees: Optional rotation angle.
        **kwargs: Shape-specific parameters (radius, a, b, side, width, height).

    Returns:
        Array of shape (n_points, 2).
    """
    generators = {
        "circle": generate_circle,
        "ellipse": generate_ellipse,
        "square": generate_square,
        "rectangle": generate_rectangle,
    }

    if shape_type not in generators:
        raise ValueError(f"Unknown shape type: {shape_type}. Choose from {list(generators.keys())}")

    # Filter kwargs to only include parameters relevant to each shape
    shape_params = {"n_points": n_points, "center": center}

    if shape_type == "circle":
        if "radius" in kwargs:
            shape_params["radius"] = kwargs["radius"]
    elif shape_type == "ellipse":
        if "a" in kwargs:
            shape_params["a"] = kwargs["a"]
        if "b" in kwargs:
            shape_params["b"] = kwargs["b"]
    elif shape_type == "square":
        if "side" in kwargs:
            shape_params["side"] = kwargs["side"]
    elif shape_type == "rectangle":
        if "width" in kwargs:
            shape_params["width"] = kwargs["width"]
        if "height" in kwargs:
            shape_params["height"] = kwargs["height"]

    X = generators[shape_type](**shape_params)

    if rotation_degrees != 0.0:
        X = rotate_shape(X, rotation_degrees)

    return X


def generate_standard_shapes(
    n_points: int = 200,
    with_negative_coords: bool = True,
) -> dict:
    """Generate the standard set of shapes used in experiments.

    Args:
        n_points: Number of points per shape.
        with_negative_coords: If True, use original centers (some negative).
                             If False, shift to all positive coordinates.

    Returns:
        Dictionary mapping shape names to their point arrays.
    """
    if with_negative_coords:
        shapes = {
            "circle": generate_circle(n_points, radius=1.0, center=(-6, 0)),
            "ellipse": generate_ellipse(n_points, a=2.0, b=1.0, center=(-3, 0)),
            "square": generate_square(n_points, side=2.0, center=(0, 0)),
            "rectangle": generate_rectangle(n_points, width=4.0, height=2.0, center=(6, 0)),
        }
    else:
        shapes = {
            "circle": generate_circle(n_points, radius=1.0, center=(3, 3)),
            "ellipse": generate_ellipse(n_points, a=2.0, b=1.0, center=(6, 3)),
            "square": generate_square(n_points, side=2.0, center=(1, 1)),
            "rectangle": generate_rectangle(n_points, width=4.0, height=2.0, center=(9, 3)),
        }

    return shapes
