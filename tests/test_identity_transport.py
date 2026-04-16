"""Tests for identity transport problem and supporting utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mae_data_gen.io import load_dataset, save_dataset
from mae_data_gen.mesh import grid_disk, sample_disk, sample_disk_boundary
from mae_data_gen.problems import IdentityTransport, create_problem


@pytest.fixture
def problem() -> IdentityTransport:
    return IdentityTransport()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# IdentityTransport tests
# ---------------------------------------------------------------------------


def test_rhs_is_one(problem: IdentityTransport, rng: np.random.Generator) -> None:
    """f(x) must equal 1 at every interior point."""
    x = sample_disk(100, rng=rng)
    f = problem.rhs(x)
    assert f.shape == (100,)
    np.testing.assert_array_equal(f, 1.0)


def test_exact_solution_satisfies_pde(problem: IdentityTransport, rng: np.random.Generator) -> None:
    """det(D^2 u) must equal f = 1 at every interior point."""
    x = sample_disk(200, rng=rng)
    hess = problem.exact_hessian(x)
    det = hess[:, 0, 0] * hess[:, 1, 1] - hess[:, 0, 1] * hess[:, 1, 0]
    f = problem.rhs(x)
    np.testing.assert_allclose(det, f, atol=1e-12)


def test_boundary_condition_matches_exact(problem: IdentityTransport) -> None:
    """Boundary condition g must equal u on the boundary circle."""
    x_bnd = sample_disk_boundary(200)
    g = problem.boundary_value(x_bnd)
    u = problem.exact_solution(x_bnd)
    np.testing.assert_allclose(g, u, atol=1e-14)


def test_exact_gradient(problem: IdentityTransport, rng: np.random.Generator) -> None:
    """Gradient of u(x) = 0.5*||x||^2 must equal x."""
    x = sample_disk(50, rng=rng)
    grad = problem.exact_gradient(x)
    np.testing.assert_allclose(grad, x, atol=1e-14)


def test_exact_hessian_is_identity(problem: IdentityTransport, rng: np.random.Generator) -> None:
    """D^2 u must be the identity matrix at every point."""
    x = sample_disk(50, rng=rng)
    hess = problem.exact_hessian(x)
    expected = np.broadcast_to(np.eye(2), (len(x), 2, 2))
    np.testing.assert_allclose(hess, expected, atol=1e-14)


def test_domain_contains(problem: IdentityTransport) -> None:
    """Points inside the disk should return True, outside False."""
    inside = np.array([[0.0, 0.0], [0.5, 0.5]])
    outside = np.array([[1.0, 1.0], [2.0, 0.0]])
    assert np.all(problem.domain_contains(inside))
    assert not np.any(problem.domain_contains(outside))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_create_problem() -> None:
    """create_problem must return an IdentityTransport instance."""
    p = create_problem("identity_transport")
    assert isinstance(p, IdentityTransport)


def test_registry_unknown_raises() -> None:
    """Unknown problem names must raise KeyError."""
    with pytest.raises(KeyError, match="Unknown problem"):
        create_problem("nonexistent_problem")


# ---------------------------------------------------------------------------
# Mesh tests
# ---------------------------------------------------------------------------


def test_sample_disk_inside(rng: np.random.Generator) -> None:
    """All sampled interior points must lie strictly inside the unit disk."""
    x = sample_disk(1000, rng=rng)
    assert x.shape == (1000, 2)
    radii = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    assert np.all(radii < 1.0), f"Points outside disk: max radius = {radii.max()}"


def test_sample_disk_boundary_on_circle() -> None:
    """Boundary samples must lie exactly on the unit circle."""
    x = sample_disk_boundary(360)
    assert x.shape == (360, 2)
    radii = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    np.testing.assert_allclose(radii, 1.0, atol=1e-14)


def test_sample_disk_custom_radius(rng: np.random.Generator) -> None:
    """Interior samples must respect a custom radius."""
    r = 2.5
    x = sample_disk(500, radius=r, rng=rng)
    radii = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    assert np.all(radii < r)


def test_sample_disk_seed_reproducibility() -> None:
    """Passing the same seed must produce identical samples."""
    a = sample_disk(100, seed=123)
    b = sample_disk(100, seed=123)
    np.testing.assert_array_equal(a, b)


def test_grid_disk_inside() -> None:
    """All grid points must lie inside the disk."""
    x = grid_disk(50)
    assert x.ndim == 2 and x.shape[1] == 2
    radii = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    assert np.all(radii < 1.0)


def test_grid_disk_nonempty() -> None:
    x = grid_disk(10)
    assert len(x) > 0


# ---------------------------------------------------------------------------
# IO tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(rng: np.random.Generator) -> None:
    """Saved and loaded arrays must match the originals exactly."""
    x = sample_disk(100, rng=rng)
    u = 0.5 * np.sum(x**2, axis=1)
    grad = x.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_data"
        save_dataset(path, x, {"u": u, "grad_u": grad})

        data = load_dataset(str(path) + ".npz")

    np.testing.assert_array_equal(data["x"], x)
    np.testing.assert_array_equal(data["u"], u)
    np.testing.assert_array_equal(data["grad_u"], grad)


def test_save_creates_parent_dirs(rng: np.random.Generator) -> None:
    """save_dataset must create missing parent directories."""
    x = sample_disk(10, rng=rng)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "data"
        save_dataset(path, x, {})
        assert (Path(str(path) + ".npz")).exists() or (path.with_suffix(".npz")).exists()


def test_load_without_npz_suffix(rng: np.random.Generator) -> None:
    """load_dataset must work when path is given without the .npz extension."""
    x = sample_disk(10, rng=rng)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data"
        save_dataset(path, x, {"v": x[:, 0]})
        data = load_dataset(path)
        np.testing.assert_array_equal(data["x"], x)
