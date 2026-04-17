"""Sampling and grid utilities for disk domains."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

__all__ = ["grid_disk", "sample_disk", "sample_disk_boundary"]


@jaxtyped(typechecker=beartype)
def sample_disk(
    n_points: int,
    radius: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Float[np.ndarray, "{n_points} 2"]:
    """Sample uniform random points inside a disk.

    Uses the rejection-free square-root method for uniform distribution.

    Args:
        n_points: Number of interior points to generate.
        radius: Radius of the disk.
        rng: Optional numpy random Generator for reproducibility.
        seed: Optional seed; ignored if *rng* is provided.

    Returns:
        (n_points, 2) array of (x1, x2) coordinates.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    u = rng.random(n_points)
    v = rng.random(n_points)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return np.column_stack([x1, x2])


@jaxtyped(typechecker=beartype)
def sample_disk_boundary(
    n_points: int,
    radius: float = 1.0,
) -> Float[np.ndarray, "{n_points} 2"]:
    """Sample equispaced points on the boundary circle.

    Args:
        n_points: Number of boundary points.
        radius: Radius of the circle.

    Returns:
        (n_points, 2) array of (x1, x2) coordinates on the circle.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x1 = radius * np.cos(theta)
    x2 = radius * np.sin(theta)
    return np.column_stack([x1, x2])


@jaxtyped(typechecker=beartype)
def grid_disk(
    resolution: int,
    radius: float = 1.0,
) -> Float[np.ndarray, "N 2"]:
    """Create a regular Cartesian grid filtered to the disk interior.

    The *resolution* parameter controls grid density as the number of grid
    points per axis (resolution x resolution grid).

    Args:
        resolution: Number of grid points per axis.
        radius: Radius of the disk.

    Returns:
        (M, 2) array of interior grid points, where M <= resolution^2.
    """
    lin = np.linspace(-radius, radius, resolution)
    x1, x2 = np.meshgrid(lin, lin)
    x1_flat = x1.ravel()
    x2_flat = x2.ravel()
    inside = x1_flat**2 + x2_flat**2 < radius**2
    return np.column_stack([x1_flat[inside], x2_flat[inside]])
