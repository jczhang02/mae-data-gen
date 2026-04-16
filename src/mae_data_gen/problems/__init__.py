"""Problem definitions for Monge-Ampere equations."""

from mae_data_gen.problems.base import Problem
from mae_data_gen.problems.identity_transport import IdentityTransport
from mae_data_gen.problems.registry import REGISTRY, create_problem, register

__all__ = [
    "REGISTRY",
    "IdentityTransport",
    "Problem",
    "create_problem",
    "register",
]
