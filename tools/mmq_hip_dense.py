"""Launchers for the deployed dense-backward HIP controls.

The deployed symbol for each exact backward key and its launch geometry both
come from `tools.mmq_hip_deployment`. This module only binds them to
the public backward ABI and its validation.
"""

import ctypes
from pathlib import Path

import torch

from tools.ggtensile.mmq_bwd_spec import BackwardKernelSpec
from tools.ggtensile.model import ProblemSize
from tools.mmq_abi import DENSE_BACKWARD_SPLIT_K_ABI, SPLIT_K_REDUCE_ABI
from tools.mmq_bundle_wrapper_source import DenseBackwardConfig
from tools.mmq_hip_deployment import (
    SPLIT_K_REDUCE_SYMBOL,
    control_inventory,
    select_hip_control,
    split_k_chunk,
    split_k_reduce_configuration,
)
from tools.mmq_runtime import (
    BackwardModule,
    HIPRuntimeError,
    _find_installed_kernel,
    _HIPModule,
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


class InstalledDenseBackwardSplitKModule(_HIPModule):
    """Launch the deployed split-contraction dense-backward control."""

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
        config = control_inventory()[self.hip_control.symbol].config
        if not isinstance(config, DenseBackwardConfig):
            raise HIPRuntimeError(
                f"{self.hip_control.symbol} is not a dense backward control"
            )
        self.slices = int(config.split_k)
        if not self.slices:
            raise HIPRuntimeError(
                f"{self.hip_control.symbol} is not a split-contraction control"
            )
        self.problem_size = problem_size
        self.split_chunk = split_k_chunk(
            config.exact_out_features, self.slices, config.k_iteration
        )
        selected = (
            _resolve_code_object(code_object, self.hip_control.symbol)
            if code_object is not None
            else _find_installed_kernel(self.hip_control.symbol)
        )
        super().__init__(selected, hip_library, self.hip_control.symbol)

    def allocate(
        self, rows: int, in_features: int, device: torch.device
    ) -> torch.Tensor:
        """Return the FP32 partial workspace for one launch."""

        return torch.empty(
            (self.slices, rows, in_features), dtype=torch.float32, device=device
        )

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        partials: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        size = self.problem_size
        if grad_output.shape != (size.m, size.k):
            raise HIPRuntimeError("grad_output shape does not match ProblemSize")
        if packed_weight.dtype != torch.uint8 or partials.dtype != torch.float32:
            raise HIPRuntimeError(
                "split-contraction launch needs packed weights and FP32 partials"
            )
        if tuple(partials.shape) != (self.slices, size.m, size.n):
            raise HIPRuntimeError(
                "partial workspace does not match the control geometry"
            )
        packed_arguments = DENSE_BACKWARD_SPLIT_K_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "partials": partials.data_ptr(),
                "rows": size.m,
                "out_features": size.k,
                "in_features": size.n,
                "blocks_per_weight_row": size.n // 256,
                "split_chunk": self.split_chunk,
            }
        )
        grid, block, shared_memory = self.hip_control.launch_configuration(
            size.m, size.n, size.k
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledSplitKReduceModule(_HIPModule):
    """Sum split-contraction partial tiles in ascending slice order."""

    def __init__(
        self,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        selected = code_object or _find_installed_kernel(SPLIT_K_REDUCE_SYMBOL)
        super().__init__(selected, hip_library, SPLIT_K_REDUCE_SYMBOL)

    def launch(
        self,
        partials: torch.Tensor,
        grad_input: torch.Tensor,
        rows: int,
        in_features: int,
        slices: int,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        packed_arguments = SPLIT_K_REDUCE_ABI.pack(
            {
                "partials": partials.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "rows": rows,
                "in_features": in_features,
                "splits": slices,
            }
        )
        grid, block, shared_memory = split_k_reduce_configuration(rows, in_features)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


__all__ = [
    "InstalledDenseBackwardModule",
    "InstalledDenseBackwardSplitKModule",
    "InstalledSplitKReduceModule",
]
