"""Problem registry for discovering and instantiating problem definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mae_data_gen.problems.base import Problem

REGISTRY: dict[str, type[Problem]] = {}


def register(name: str):
    """Class decorator that registers a Problem subclass under *name*.

    Usage::

        @register("identity_transport")
        class IdentityTransport(Problem): ...
    """

    def decorator(cls: type[Problem]) -> type[Problem]:
        REGISTRY[name] = cls
        return cls

    return decorator


def create_problem(name: str, **kwargs) -> Problem:
    """Instantiate a registered problem by name.

    Args:
        name: Registry key (e.g. ``"identity_transport"``).
        **kwargs: Forwarded to the problem constructor.

    Returns:
        An instance of the requested Problem subclass.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"Unknown problem {name!r}. Available: {available}")
    return REGISTRY[name](**kwargs)
