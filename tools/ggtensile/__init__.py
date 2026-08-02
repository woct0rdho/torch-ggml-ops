"""Exact-problem gfx1151 assembly generation for packed GGUF MMQ."""

from .inspection import ArtifactInspection, InspectionError, inspect_artifact
from .model import (
    DenseForwardSolution,
    KernelArtifact,
    ProblemSize,
    ProblemType,
    Solution,
    SolutionKey,
)
from .runtime import (
    DenseBackwardModule,
    DenseForwardModule,
    FixedHipDenseForwardModule,
    FixedQ81F16D4S4QuantizerModule,
    HIPRuntimeError,
)
from .validation import RejectReason, validate_solution

__all__ = [
    "ArtifactInspection",
    "DenseBackwardModule",
    "DenseForwardModule",
    "DenseForwardSolution",
    "FixedHipDenseForwardModule",
    "FixedQ81F16D4S4QuantizerModule",
    "HIPRuntimeError",
    "InspectionError",
    "KernelArtifact",
    "ProblemSize",
    "ProblemType",
    "RejectReason",
    "Solution",
    "SolutionKey",
    "inspect_artifact",
    "validate_solution",
]
