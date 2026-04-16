"""Data export and import utilities for MAE datasets."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["load_dataset", "save_dataset"]


def save_dataset(path: str | Path, points: NDArray, values_dict: dict[str, NDArray]) -> None:
    """Save a dataset to .npz format.

    Args:
        path: Output file path (.npz extension added automatically if absent).
        points: (N, D) point coordinates, stored under key ``"x"``.
        values_dict: Mapping of field names to arrays of shape (N, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"x": points, **values_dict}
    np.savez(path, **arrays)


def load_dataset(path: str | Path) -> dict[str, NDArray]:
    """Load a dataset from .npz format.

    Args:
        path: Path to the .npz file (with or without the .npz extension).

    Returns:
        Dictionary mapping field names to numpy arrays.
    """
    path = Path(path)
    data = np.load(path if path.suffix == ".npz" else path.with_suffix(".npz"))
    return dict(data)
