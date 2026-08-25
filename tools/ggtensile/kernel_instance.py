"""Passive exact kernel inputs shared by catalogs and deployment tooling."""

from dataclasses import dataclass

from .identity import GFX1151_TARGET, KernelFamily, KernelTarget
from .model import ProblemType


@dataclass(frozen=True)
class KernelInstance:
    """A family, exact problem, and complete kernel specification.

    This record is intentionally passive. Parsing, validation, identity, launch
    derivation, and emission are owned by the family registry and consumers.
    """

    family: KernelFamily
    target: KernelTarget
    problem_type: ProblemType
    problem: object
    kernel_spec: object

    @classmethod
    def for_gfx1151(
        cls,
        family: KernelFamily,
        problem_type: ProblemType,
        problem: object,
        kernel_spec: object,
    ) -> "KernelInstance":
        return cls(family, GFX1151_TARGET, problem_type, problem, kernel_spec)
