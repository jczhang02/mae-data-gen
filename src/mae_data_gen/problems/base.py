"""Abstract base class for Monge-Ampere problem definitions."""

from abc import ABC, abstractmethod

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


class Problem(ABC):
    """Base class for Monge-Ampere equation problems.

    Subclasses define a specific PDE on a domain with boundary conditions
    and (optionally) analytical solutions.
    """

    name: str
    dim: int

    @abstractmethod
    def domain_contains(self, x: Float[np.ndarray, " N dim"]) -> Bool[np.ndarray, " N"]:
        """Test whether points lie inside the domain.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) boolean array, True for interior points.
        """
        ...

    @abstractmethod
    def rhs(self, x: Float[np.ndarray, " N dim"]) -> Float[np.ndarray, " N"]:
        """Right-hand side f of det(D^2 u) = f.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) array of f values.
        """
        ...

    @abstractmethod
    def boundary_value(self, x: Float[np.ndarray, " N dim"]) -> Float[np.ndarray, " N"]:
        """Dirichlet boundary condition g.

        Args:
            x: (N, dim) boundary point coordinates.

        Returns:
            (N,) array of boundary values.
        """
        ...

    def exact_solution(self, x: Float[np.ndarray, " N dim"]) -> Float[np.ndarray, " N"]:
        """Analytical solution u(x), if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) array of solution values.

        Raises:
            NotImplementedError: If no analytical solution is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical solution.")

    def exact_gradient(self, x: Float[np.ndarray, " N dim"]) -> Float[np.ndarray, " N dim"]:
        """Gradient of the analytical solution, if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N, dim) array of gradient vectors.

        Raises:
            NotImplementedError: If no analytical gradient is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical gradient.")

    def exact_hessian(self, x: Float[np.ndarray, " N dim"]) -> Float[np.ndarray, " N dim dim"]:
        """Hessian of the analytical solution, if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N, dim, dim) array of Hessian matrices.

        Raises:
            NotImplementedError: If no analytical Hessian is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical Hessian.")


# Keep beartype and jaxtyped in namespace for subclass use
__all__ = ["Problem", "beartype", "jaxtyped", "Bool", "Float"]
