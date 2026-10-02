"""Launchers for the deployed paired grouped-forward HIP controls.

Every body is chosen by `tools.mmq_hip_deployment`, which owns the routed rule
tables for all problem types. This module only binds a symbol to its launch
geometry and validation.
"""

import ctypes
from pathlib import Path
from typing import ClassVar

import torch

from tools.mmq_abi import GROUPED_FORWARD_ABI, GROUPED_FORWARD_ROW_TASK_ABI
from tools.mmq_hip_row_task import GroupedForwardPairRowTaskWorkspace
from tools.mmq_runtime import HIPRuntimeError, _find_installed_kernel, _HIPModule


class InstalledGroupedForwardPairRowTaskControl(_HIPModule):
    """Launch one installed IQ2_S N512/K2048 J64 row-task projection."""

    SYMBOL = "grouped_fwd_row_task_iq2_s_n512_k2048_j64"
    BYTES_PER_EXPERT = 335_872
    DYNAMIC_LDS_BYTES = 30_976

    def __init__(
        self,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        super().__init__(
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            self.SYMBOL,
        )

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        tasks: GroupedForwardPairRowTaskWorkspace,
        *,
        aggregate_rows: int,
        stream: int,
    ) -> None:
        tensors = (packed_weight, activations, output, tasks.storage)
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed row-task controls require contiguous HIP tensors"
            )
        if (
            packed_weight.dtype != torch.uint8
            or activations.dtype != torch.uint8
            or output.dtype != torch.bfloat16
            or tasks.storage.dtype != torch.int32
        ):
            raise HIPRuntimeError(
                "installed row-task control tensor dtypes are invalid"
            )
        packed_arguments = GROUPED_FORWARD_ROW_TASK_ABI.pack(
            {
                "weights": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst": output.data_ptr(),
                "task_count": tasks.task_count.data_ptr(),
                "task_experts": tasks.task_experts.data_ptr(),
                "task_row_starts": tasks.task_row_starts.data_ptr(),
                "task_row_ends": tasks.task_row_ends.data_ptr(),
                "nrows_activation": aggregate_rows,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                8,
                tasks.capacity,
                1,
                32,
                4,
                1,
                self.DYNAMIC_LDS_BYTES,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedForwardPairSerialControl(_HIPModule):
    """Launch the installed single-projection N512/K2048 IQ2_S control."""

    SYMBOL = "grouped_fwd_serial_iq2_s_n512_k2048_j64"
    BYTES_PER_EXPERT = 335_872
    DYNAMIC_LDS_BYTES = 30_976

    def __init__(
        self,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        super().__init__(
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            self.SYMBOL,
        )

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        aggregate_rows: int,
        stream: int,
    ) -> None:
        tensors = (packed_weight, activations, output, expert_indices, expert_offsets)
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed paired controls require contiguous HIP tensors"
            )
        if (
            packed_weight.dtype != torch.uint8
            or activations.dtype != torch.uint8
            or output.dtype != torch.bfloat16
        ):
            raise HIPRuntimeError("installed paired control tensor dtypes are invalid")
        route_entries = expert_indices.numel()
        if expert_indices.dtype != torch.int64 or expert_offsets.dtype != torch.int32:
            raise HIPRuntimeError("installed paired control route dtypes are invalid")
        packed_arguments = GROUPED_FORWARD_ABI.pack(
            {
                "weights": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst": output.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": 256,
                "nrows_weight": 512,
                "nrows_activation": aggregate_rows,
                "blocks_per_weight_row": 8,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                8,
                route_entries,
                1,
                32,
                4,
                1,
                self.DYNAMIC_LDS_BYTES,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedForwardPairIQ2XXSSerialControl(_HIPModule):
    """Launch one installed DeepSeek IQ2_XXS serial projection."""

    _CONFIGS: ClassVar[dict[int, tuple[str, int]]] = {
        64: (
            "grouped_fwd_serial_iq2_xxs_n2048_k4096_j64",
            28_928,
        ),
        80: (
            "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80",
            31_552,
        ),
    }
    BYTES_PER_EXPERT = 2_162_688

    def __init__(
        self,
        row_tile: int = 64,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        config = self._CONFIGS.get(row_tile)
        if config is None:
            raise HIPRuntimeError("installed IQ2_XXS control requires J64 or J80")
        symbol, dynamic_lds_bytes = config
        self.row_tile = row_tile
        self.dynamic_lds_bytes = dynamic_lds_bytes
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        aggregate_rows: int,
        stream: int,
    ) -> None:
        tensors = (packed_weight, activations, output, expert_indices, expert_offsets)
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed IQ2_XXS controls require contiguous HIP tensors"
            )
        if (
            packed_weight.dtype != torch.uint8
            or activations.dtype != torch.uint8
            or output.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("installed IQ2_XXS control tensor dtypes are invalid")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > 256:
            raise HIPRuntimeError("installed IQ2_XXS route count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("installed IQ2_XXS route metadata lengths differ")
        if tuple(packed_weight.shape) != (256, 2048, 1056):
            raise HIPRuntimeError("installed IQ2_XXS packed shape is invalid")
        if tuple(activations.shape) != (32, aggregate_rows, 144):
            raise HIPRuntimeError("installed IQ2_XXS activation shape is invalid")
        if tuple(output.shape) != (aggregate_rows, 2048):
            raise HIPRuntimeError("installed IQ2_XXS output shape is invalid")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("installed IQ2_XXS tensors must share one device")
        packed_arguments = GROUPED_FORWARD_ABI.pack(
            {
                "weights": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst": output.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": 256,
                "nrows_weight": 2048,
                "nrows_activation": aggregate_rows,
                "blocks_per_weight_row": 16,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                32,
                route_entries,
                1,
                32,
                4,
                1,
                self.dynamic_lds_bytes,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedForwardPairQ3RowTaskControl(
    InstalledGroupedForwardPairRowTaskControl
):
    """Launch one installed Q3_K N512/K2048 J64 row-task projection."""

    SYMBOL = "grouped_fwd_row_task_q3_k_n512_k2048_j64"
    BYTES_PER_EXPERT = 450_560
    DYNAMIC_LDS_BYTES = 30_976


class InstalledGroupedForwardPairQ3SerialControl(
    InstalledGroupedForwardPairSerialControl
):
    """Launch one installed Q3_K N512/K2048 J64 serial projection."""

    SYMBOL = "grouped_fwd_serial_q3_k_n512_k2048_j64"
    BYTES_PER_EXPERT = 450_560
    DYNAMIC_LDS_BYTES = 30_976
