"""Format-neutral context and protocol for MMQ forward body lowerings."""

from dataclasses import dataclass
from typing import Protocol

from .mmq_fwd_spec import DerivedForwardState
from .model import SolutionKey


class ForwardKernelWriterError(RuntimeError):
    """A validated forward key cannot be lowered by its selected mechanism."""


@dataclass(frozen=True)
class ForwardLoweringContext:
    """One validated key and its single authoritative derived state."""

    solution_key: SolutionKey
    state: DerivedForwardState


class ForwardBodyLowering(Protocol):
    """A mechanism lowerer that emits only a kernel body."""

    def body(self) -> str: ...
