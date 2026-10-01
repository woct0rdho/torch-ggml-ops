"""Launchers for the deployed dense-backward HIP controls.

The deployed symbol for each exact backward key and its launch geometry both
come from `tools.ggtensile.hip_deployment`. This module only binds them to
the public backward ABI and its validation.
"""

from pathlib import Path

from .hip_deployment import select_hip_control
from .mmq_bwd_spec import BackwardKernelSpec
from .model import ProblemSize
from .runtime import (
    BackwardModule,
    _find_installed_kernel,
    _resolve_code_object,
)


class InstalledDenseBackwardModule(BackwardModule):
    """Launch the deployed dense-backward HIP control for the exact key."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: BackwardKernelSpec,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.hip_control = select_hip_control(
            "OrdinaryBackward",
            quant_type,
            problem_size.m,
            problem_size.n,
            problem_size.k,
        )
        selected = (
            _resolve_code_object(code_object, self.hip_control.symbol)
            if code_object is not None
            else _find_installed_kernel(self.hip_control.symbol)
        )
        super().__init__(
            problem_size,
            quant_type,
            kernel_spec,
            selected,
            self.hip_control.symbol,
            hip_library,
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        size = self.problem_size
        return self.hip_control.launch_configuration(size.m, size.n, size.k)


__all__ = ["InstalledDenseBackwardModule"]
