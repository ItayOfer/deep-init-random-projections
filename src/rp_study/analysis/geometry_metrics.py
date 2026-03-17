"""Geometry preservation metrics for evaluating initializers.

Measures how well multi-layer ReLU projections preserve data geometry.
Three complementary metrics:

1. k-NN accuracy: class structure preservation (thesis objective proxy)
2. Pairwise distance correlation: relative distance preservation
3. Effective dimensionality: representation spread (collapse detection)
"""

from typing import Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def knn_accuracy(
    X_projected: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> float:
    """Compute k-NN classification accuracy in the projected space.

    Uses leave-one-out: for each point, find k nearest neighbors among
    the remaining points and predict via majority vote.

    This directly measures the thesis objective: if inner products are
    preserved, class structure is preserved, so k-NN accuracy stays high.

    Args:
        X_projected: Projected data, shape (n, d).
        y: Labels, shape (n,).
        k: Number of neighbors.

    Returns:
        Accuracy in [0, 1].
    """
    n = X_projected.shape[0]

    # Check for degenerate data — if all points are identical, k-NN is random
    pairwise_dists = pdist(X_projected, metric="euclidean")
    if np.max(pairwise_dists) < 1e-10 * (np.mean(np.linalg.norm(X_projected, axis=1)) + 1e-15):
        # All points collapsed to same location — return chance-level accuracy
        unique_labels = np.unique(y)
        return 1.0 / len(unique_labels)

    dists = squareform(pairwise_dists)
    # Set self-distance to infinity so a point is never its own neighbor
    np.fill_diagonal(dists, np.inf)

    # For each point, find k nearest neighbors
    knn_indices = np.argpartition(dists, k, axis=1)[:, :k]

    # Majority vote
    correct = 0
    for i in range(n):
        neighbor_labels = y[knn_indices[i]]
        # Find most common label
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted = unique[np.argmax(counts)]
        if predicted == y[i]:
            correct += 1

    return correct / n


def pairwise_distance_correlation(
    X_original: np.ndarray,
    X_projected: np.ndarray,
    n_pairs: int = 2000,
    seed: Optional[int] = None,
) -> float:
    """Spearman rank correlation of pairwise distances before/after projection.

    Samples random pairs and computes whether relative distance ordering
    is preserved. A correlation of 1.0 means perfect rank preservation.

    Args:
        X_original: Original data, shape (n, d_in).
        X_projected: Projected data, shape (n, d_out).
        n_pairs: Number of random pairs to sample.
        seed: Random seed for reproducibility.

    Returns:
        Spearman correlation in [-1, 1].
    """
    rng = np.random.RandomState(seed)
    n = X_original.shape[0]

    # Sample random pairs (without replacement of pair indices)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % n

    # Compute distances
    diff_orig = X_original[idx_i] - X_original[idx_j]
    dist_orig = np.linalg.norm(diff_orig, axis=1)

    diff_proj = X_projected[idx_i] - X_projected[idx_j]
    dist_proj = np.linalg.norm(diff_proj, axis=1)

    # Handle degenerate case (all projected distances are the same = collapse)
    if np.std(dist_proj) < 1e-10:
        return 0.0

    corr, _ = spearmanr(dist_orig, dist_proj)
    return float(corr)


def effective_dimensionality(X: np.ndarray) -> float:
    """Effective dimensionality via eigenvalue entropy.

    eff_dim = exp(H(p)) where p_i = lambda_i / sum(lambda)
    and H(p) = -sum(p_i * log(p_i)).

    High value = data spread across many dimensions (good geometry).
    Low value = data collapsed into few dimensions (geometric collapse).
    Value of 1 = complete collapse to a single direction.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        Effective dimensionality (float >= 1).
    """
    # Center the data
    X_centered = X - X.mean(axis=0)

    # Detect collapse: if the variation is negligible relative to the
    # overall scale of the data, the representation has collapsed.
    # Use coefficient of variation: Frobenius(X_centered) / Frobenius(X)
    data_norm = np.linalg.norm(X)
    centered_norm = np.linalg.norm(X_centered)

    if data_norm < 1e-15:
        return 1.0  # All-zero data

    relative_variation = centered_norm / data_norm
    if relative_variation < 1e-6:
        # Data is near-constant: all points collapsed to approximately
        # the same vector. This IS geometric collapse — eff_dim = 1.
        return 1.0

    # Normalize X_centered for numerical stability in SVD
    # (prevents issues with very large or very small values)
    scale = centered_norm / np.sqrt(X.shape[0])
    X_normalized = X_centered / scale

    try:
        _, s, _ = np.linalg.svd(X_normalized, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD didn't converge even after normalization.
        # This means the data is nearly rank-deficient — return 1.0.
        return 1.0

    # s contains singular values of X_normalized; eigenvalues of
    # the normalized covariance are s^2 / (n-1)
    eigenvalues = s ** 2 / (X.shape[0] - 1)

    # Normalize to probability distribution
    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0

    p = eigenvalues / total
    # Remove negligible components
    p = p[p > 1e-12]

    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def evaluate_geometry(
    X_original: np.ndarray,
    X_projected: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    n_pairs: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """Compute all geometry preservation metrics.

    Args:
        X_original: Original data, shape (n, d_in).
        X_projected: Projected data, shape (n, d_out).
        y: Labels, shape (n,).
        k: Number of neighbors for k-NN.
        n_pairs: Number of pairs for distance correlation.
        seed: Random seed for distance correlation.

    Returns:
        Dictionary with keys:
        - knn_accuracy: k-NN classification accuracy
        - distance_correlation: Spearman rank correlation of pairwise distances
        - effective_dim: Effective dimensionality of projected data
        - effective_dim_original: Effective dimensionality of original data
    """
    return {
        "knn_accuracy": knn_accuracy(X_projected, y, k=k),
        "distance_correlation": pairwise_distance_correlation(
            X_original, X_projected, n_pairs=n_pairs, seed=seed
        ),
        "effective_dim": effective_dimensionality(X_projected),
        "effective_dim_original": effective_dimensionality(X_original),
    }
