"""Abstract base class for Monge-Ampere problem definitions."""

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Problem(ABC):
    """Base class for Monge-Ampere equation problems.

    Subclasses define a specific PDE on a domain with boundary conditions
    and (optionally) analytical solutions.
    """

    name: str
    dim: int

    @abstractmethod
    def domain_contains(self, x: NDArray) -> NDArray:
        """Test whether points lie inside the domain.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) boolean array, True for interior points.
        """
        ...

    @abstractmethod
    def rhs(self, x: NDArray) -> NDArray:
        """Right-hand side f of det(D^2 u) = f.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) array of f values.
        """
        ...

    @abstractmethod
    def boundary_value(self, x: NDArray) -> NDArray:
        """Dirichlet boundary condition g.

        Args:
            x: (N, dim) boundary point coordinates.

        Returns:
            (N,) array of boundary values.
        """
        ...

    def exact_solution(self, x: NDArray) -> NDArray:
        """Analytical solution u(x), if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N,) array of solution values.

        Raises:
            NotImplementedError: If no analytical solution is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical solution.")

    def exact_gradient(self, x: NDArray) -> NDArray:
        """Gradient of the analytical solution, if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N, dim) array of gradient vectors.

        Raises:
            NotImplementedError: If no analytical gradient is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical gradient.")

    def exact_hessian(self, x: NDArray) -> NDArray:
        """Hessian of the analytical solution, if available.

        Args:
            x: (N, dim) point coordinates.

        Returns:
            (N, dim, dim) array of Hessian matrices.

        Raises:
            NotImplementedError: If no analytical Hessian is known.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical Hessian.")
