"""Identity optimal transport (T=id) on the unit disk.

PDE:  det(D^2 u) = 1  on Omega = {x : ||x|| < 1}
BC:   u = 0.5 * ||x||^2  on dOmega
Exact solution: u(x) = 0.5 * ||x||^2
"""

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

from mae_data_gen.problems.base import Problem
from mae_data_gen.problems.registry import register


@register("identity_transport")
class IdentityTransport(Problem):
    """Identity optimal transport on the unit disk.

    The optimal transport map is the identity, so the Monge-Ampere equation
    reduces to det(D^2 u) = 1 with solution u(x) = 0.5 * ||x||^2.
    """

    name: str = "identity_transport"
    dim: int = 2

    def __init__(self, radius: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> None:
        self.radius = radius
        self.center = np.asarray(center, dtype=float)

    @jaxtyped(typechecker=beartype)
    def domain_contains(self, x: Float[np.ndarray, "N dim"]) -> Bool[np.ndarray, "N"]:
        """Test whether points lie inside the disk.

        Args:
            x: (N, 2) point coordinates.

        Returns:
            (N,) boolean array.
        """
        x = np.asarray(x, dtype=float)
        return np.sum((x - self.center) ** 2, axis=1) < self.radius**2

    @jaxtyped(typechecker=beartype)
    def rhs(self, x: Float[np.ndarray, "N dim"]) -> Float[np.ndarray, "N"]:
        """Return f(x) = 1 (right-hand side of det(D^2 u) = f).

        Args:
            x: (N, 2) point coordinates.

        Returns:
            (N,) array of ones.
        """
        x = np.asarray(x, dtype=float)
        return np.ones(x.shape[0])

    @jaxtyped(typechecker=beartype)
    def boundary_value(self, x: Float[np.ndarray, "N dim"]) -> Float[np.ndarray, "N"]:
        """Return g(x) = 0.5 * ||x||^2 (Dirichlet BC).

        Args:
            x: (N, 2) boundary point coordinates.

        Returns:
            (N,) array of boundary values.
        """
        x = np.asarray(x, dtype=float)
        return 0.5 * np.sum(x**2, axis=1)

    @jaxtyped(typechecker=beartype)
    def exact_solution(self, x: Float[np.ndarray, "N dim"]) -> Float[np.ndarray, "N"]:
        """Return u(x) = 0.5 * ||x||^2.

        Args:
            x: (N, 2) point coordinates.

        Returns:
            (N,) array of solution values.
        """
        x = np.asarray(x, dtype=float)
        return 0.5 * np.sum(x**2, axis=1)

    @jaxtyped(typechecker=beartype)
    def exact_gradient(self, x: Float[np.ndarray, "N dim"]) -> Float[np.ndarray, "N dim"]:
        """Return grad u(x) = x.

        Args:
            x: (N, 2) point coordinates.

        Returns:
            (N, 2) gradient vectors.
        """
        return np.asarray(x, dtype=float).copy()

    @jaxtyped(typechecker=beartype)
    def exact_hessian(self, x: Float[np.ndarray, "N dim"]) -> Float[np.ndarray, "N dim dim"]:
        """Return D^2 u(x) = I (identity matrix at every point).

        Args:
            x: (N, 2) point coordinates.

        Returns:
            (N, 2, 2) Hessian matrices.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        hess = np.zeros((n, 2, 2))
        hess[:, 0, 0] = 1.0
        hess[:, 1, 1] = 1.0
        return hess
