"""Launchers for the deployed paired grouped-backward HIP controls.

Every body is chosen by `tools.mmq_hip_deployment`. This module only binds a
symbol to its launch geometry and validation.
"""

import ctypes
from pathlib import Path
from typing import ClassVar

import torch

from tools.mmq_abi import (
    GROUPED_BACKWARD_PAIR_ABI,
    GROUPED_BACKWARD_PAIR_ROW_TASK_ABI,
)
from tools.mmq_hip_row_task import RowTaskWorkspace
from tools.mmq_runtime import (
    HIPRuntimeError,
    _find_installed_kernel,
    _HIPModule,
)


class InstalledGroupedBackwardPairQ3KControl(_HIPModule):
    """Launch one installed specialized Qwen Q3_K backward-pair body."""

    _SYMBOLS: ClassVar[tuple[str, ...]] = (
        "grouped_bwd_pair_q3_k_n512_k2048_mt64_nt64",
        "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64",
        "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip",
    )
    PHYSICAL_EXPERTS = 256
    OUT_FEATURES = 512
    IN_FEATURES = 2048
    PACKED_ROW_BYTES = 880
    BYTES_PER_EXPERT = 450_560

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if symbol not in self._SYMBOLS:
            raise HIPRuntimeError(
                f"no installed Q3_K pair control for symbol {symbol!r}"
            )
        self.symbol = symbol
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    def launch(
        self,
        first_grad_output: torch.Tensor,
        second_grad_output: torch.Tensor,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        tensors = (
            first_grad_output,
            second_grad_output,
            first_packed_weight,
            second_packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed Q3_K pair control requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError(
                "installed Q3_K pair tensors need zero storage offsets"
            )
        if (
            first_grad_output.dtype != torch.bfloat16
            or second_grad_output.dtype != torch.bfloat16
            or first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("installed Q3_K pair tensor dtypes are invalid")
        rows = first_grad_output.shape[0]
        if tuple(first_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("first Q3_K pair gradient has the wrong shape")
        if tuple(second_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("second Q3_K pair gradient has the wrong shape")
        expected_weight_shape = (
            self.PHYSICAL_EXPERTS,
            self.OUT_FEATURES,
            self.PACKED_ROW_BYTES,
        )
        if tuple(first_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("first Q3_K pair bank has the wrong shape")
        if tuple(second_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("second Q3_K pair bank has the wrong shape")
        if tuple(grad_input.shape) != (rows, self.IN_FEATURES):
            raise HIPRuntimeError("Q3_K pair gradient input has the wrong shape")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("Q3_K pair route metadata must be one-dimensional")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > self.PHYSICAL_EXPERTS:
            raise HIPRuntimeError("Q3_K pair route entry count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("Q3_K pair route metadata lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("installed Q3_K pair tensors must share one device")

        arguments = GROUPED_BACKWARD_PAIR_ABI.pack(
            {
                "first_grad_output": first_grad_output.data_ptr(),
                "second_grad_output": second_grad_output.data_ptr(),
                "first_packed_weight": first_packed_weight.data_ptr(),
                "second_packed_weight": second_packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.PHYSICAL_EXPERTS,
                "rows": rows,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                self.IN_FEATURES // 64,
                route_entries,
                1,
                128,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedBackwardPairIQ2SControl(_HIPModule):
    """Launch one installed specialized Qwen IQ2_S backward-pair body."""

    _SYMBOLS: ClassVar[tuple[str, ...]] = (
        "grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64",
        "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64",
        "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64_s2",
        "grouped_bwd_pair_iq2_s_n512_k2048_mt256_nt64_s3_skip",
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip",
    )
    PHYSICAL_EXPERTS = 256
    OUT_FEATURES = 512
    IN_FEATURES = 2048
    PACKED_ROW_BYTES = 656
    BYTES_PER_EXPERT = 335_872

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if symbol not in self._SYMBOLS:
            raise HIPRuntimeError(
                f"no installed IQ2_S pair control for symbol {symbol!r}"
            )
        self.symbol = symbol
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    def launch(
        self,
        first_grad_output: torch.Tensor,
        second_grad_output: torch.Tensor,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        tensors = (
            first_grad_output,
            second_grad_output,
            first_packed_weight,
            second_packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed IQ2_S pair control requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError(
                "installed IQ2_S pair tensors need zero storage offsets"
            )
        if (
            first_grad_output.dtype != torch.bfloat16
            or second_grad_output.dtype != torch.bfloat16
            or first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("installed IQ2_S pair tensor dtypes are invalid")
        rows = first_grad_output.shape[0]
        if tuple(first_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("first pair gradient has the wrong shape")
        if tuple(second_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("second pair gradient has the wrong shape")
        expected_weight_shape = (
            self.PHYSICAL_EXPERTS,
            self.OUT_FEATURES,
            self.PACKED_ROW_BYTES,
        )
        if tuple(first_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("first IQ2_S pair bank has the wrong shape")
        if tuple(second_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("second IQ2_S pair bank has the wrong shape")
        if tuple(grad_input.shape) != (rows, self.IN_FEATURES):
            raise HIPRuntimeError("pair gradient input has the wrong shape")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("pair route metadata must be one-dimensional")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > self.PHYSICAL_EXPERTS:
            raise HIPRuntimeError("pair route entry count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("pair route metadata lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("installed IQ2_S pair tensors must share one device")

        arguments = GROUPED_BACKWARD_PAIR_ABI.pack(
            {
                "first_grad_output": first_grad_output.data_ptr(),
                "second_grad_output": second_grad_output.data_ptr(),
                "first_packed_weight": first_packed_weight.data_ptr(),
                "second_packed_weight": second_packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.PHYSICAL_EXPERTS,
                "rows": rows,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                self.IN_FEATURES // 64,
                route_entries,
                1,
                128,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedBackwardPairIQ2XXSControl(_HIPModule):
    """Launch one installed specialized DeepSeek IQ2_XXS backward pair."""

    _SYMBOLS: ClassVar[tuple[str, ...]] = (
        "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt64_nt64",
        "grouped_bwd_tuned_pair_iq2_xxs_n2048_k4096_mt128_nt64",
        "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt128_nt64_s2",
        "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip",
    )
    PHYSICAL_EXPERTS = 256
    OUT_FEATURES = 2048
    IN_FEATURES = 4096
    PACKED_ROW_BYTES = 1056
    BYTES_PER_EXPERT = 2_162_688

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if symbol not in self._SYMBOLS:
            raise HIPRuntimeError(
                f"no installed IQ2_XXS pair control for symbol {symbol!r}"
            )
        self.symbol = symbol
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    def launch(
        self,
        first_grad_output: torch.Tensor,
        second_grad_output: torch.Tensor,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        tensors = (
            first_grad_output,
            second_grad_output,
            first_packed_weight,
            second_packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "installed IQ2_XXS pair requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError("installed IQ2_XXS pair needs zero storage offsets")
        if (
            first_grad_output.dtype != torch.bfloat16
            or second_grad_output.dtype != torch.bfloat16
            or first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("installed IQ2_XXS pair tensor dtypes are invalid")
        rows = first_grad_output.shape[0]
        if tuple(first_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("first IQ2_XXS pair gradient shape is invalid")
        if tuple(second_grad_output.shape) != (rows, self.OUT_FEATURES):
            raise HIPRuntimeError("second IQ2_XXS pair gradient shape is invalid")
        expected_weight_shape = (
            self.PHYSICAL_EXPERTS,
            self.OUT_FEATURES,
            self.PACKED_ROW_BYTES,
        )
        if tuple(first_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("first IQ2_XXS pair bank shape is invalid")
        if tuple(second_packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("second IQ2_XXS pair bank shape is invalid")
        if tuple(grad_input.shape) != (rows, self.IN_FEATURES):
            raise HIPRuntimeError("IQ2_XXS pair destination shape is invalid")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("IQ2_XXS pair routes must be one-dimensional")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > self.PHYSICAL_EXPERTS:
            raise HIPRuntimeError("IQ2_XXS pair route count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("IQ2_XXS pair route metadata lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("installed IQ2_XXS pair tensors must share a device")

        arguments = GROUPED_BACKWARD_PAIR_ABI.pack(
            {
                "first_grad_output": first_grad_output.data_ptr(),
                "second_grad_output": second_grad_output.data_ptr(),
                "first_packed_weight": first_packed_weight.data_ptr(),
                "second_packed_weight": second_packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.PHYSICAL_EXPERTS,
                "rows": rows,
                "bytes_per_expert": self.BYTES_PER_EXPERT,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                self.IN_FEATURES // 64,
                route_entries,
                1,
                128,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


PAIR_ROW_TASK_PREFIX = "grouped_bwd_pair_task_"


def is_pair_row_task_body(symbol: str) -> bool:
    """Return whether a catalogued pair symbol names a row-task body."""

    return symbol.startswith(PAIR_ROW_TASK_PREFIX)


class InstalledGroupedBackwardPairRowTaskControl(_HIPModule):
    """Launch one installed device row-task paired-backward body.

    The body consumes the task descriptor bank built by
    `grouped_row_task_setup`: one task per 128-row tile of each active expert.
    Its grid is `(output columns / 64, task capacity, 1)`. Workgroups whose task
    index is past the decoded task count return without touching memory.
    """

    @property
    def physical_experts(self) -> int:
        return self.EXPERTS.get(self.family, self.PHYSICAL_EXPERTS)

    _SYMBOLS: ClassVar[tuple[str, ...]] = (
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip",
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip_g4",
        "grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip_g4",
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4",
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt256_nt64_s2_skip",
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt256_nt64_s2_skip_g4",
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt256_nt64_s2_skip_ki64_g4_mb3_abar",
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip_abar",
        "grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip_g4_abar",
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip_g4_abar",
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4_mb3_abar",
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt128_nt64_s2_skip_g4_mb3",
    )
    PHYSICAL_EXPERTS = 256
    EXPERTS: ClassVar[dict[str, int]] = {
        "q3_k": 256,
        "iq2_s": 256,
        "iq2_xxs": 256,
        "q2_0": 512,
    }
    # Column tile of the body behind each symbol. The deployed family shape is
    # sixteen columns times four tiles. A body built with a different tile
    # count must declare it here, because the grid is derived from it.
    # Task row tile of the body behind each symbol. A body that consumes a wider
    # task than the deployed 128 rows must declare it here, because the task
    # bank it is launched with has to match.
    ROW_TILES: ClassVar[dict[str, int]] = {
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt256_nt64_s2_skip": 256,
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt256_nt64_s2_skip_g4": 256,
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt256_nt64_s2_skip_ki64_g4_mb3_abar": 256,
    }
    THREADS: ClassVar[dict[str, int]] = {
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt256_nt64_s2_skip": 256,
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt256_nt64_s2_skip_g4": 256,
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt256_nt64_s2_skip_ki64_g4_mb3_abar": 256,
    }
    COLUMN_TILES: ClassVar[dict[str, int]] = {
        "grouped_bwd_pair_task_q3_k_n512_k2048_mt256_nt64_s2_skip": 64,
        "grouped_bwd_pair_task_iq2_s_n512_k2048_mt256_nt64_s2_skip_g4": 64,
        "grouped_bwd_pair_task_q2_0_n640_k2560_mt256_nt64_s2_skip_ki64_g4_mb3_abar": 64,
    }
    OUT_FEATURES: ClassVar[dict[str, int]] = {
        "q3_k": 512,
        "iq2_s": 512,
        "iq2_xxs": 2048,
        "q2_0": 640,
    }
    IN_FEATURES: ClassVar[dict[str, int]] = {
        "q3_k": 2048,
        "iq2_s": 2048,
        "iq2_xxs": 4096,
        "q2_0": 2560,
    }
    ROW_TASK_ROWS = 128

    @property
    def column_tile(self) -> int:
        return self.COLUMN_TILES.get(self.symbol, 64)

    @property
    def threads(self) -> int:
        return self.THREADS.get(self.symbol, 128)

    @property
    def row_tile(self) -> int:
        return self.ROW_TILES.get(self.symbol, self.ROW_TASK_ROWS)

    @classmethod
    def record_geometry(
        cls, symbol: str, out_features: int, in_features: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int]:
        """Return `(grid, block, shared_bytes, row_task_rows)` for one record.

        The device grid is derived from the same constants the launch uses, so
        the generated deployment record and the runtime agree without loading
        the artifact. The grid's Y dimension is the task capacity, which every
        launch replaces with the capacity of its own route bank.
        """

        if symbol not in cls._SYMBOLS:
            raise HIPRuntimeError(
                f"no installed paired row-task control for symbol {symbol!r}"
            )
        column_tile = cls.COLUMN_TILES.get(symbol, 64)
        if in_features % column_tile:
            raise HIPRuntimeError(
                "paired row-task output width is not a multiple of the column tile"
            )
        return (
            (in_features // column_tile, 0, 1),
            (cls.THREADS.get(symbol, 128), 1, 1),
            0,
            cls.ROW_TILES.get(symbol, cls.ROW_TASK_ROWS),
        )

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if symbol not in self._SYMBOLS:
            raise HIPRuntimeError(
                f"no installed paired row-task control for symbol {symbol!r}"
            )
        self.symbol = symbol
        family = next(
            (name for name in self.OUT_FEATURES if f"_{name}_" in symbol), None
        )
        if family is None:
            raise HIPRuntimeError(
                f"paired row-task symbol {symbol!r} has no family geometry"
            )
        self.family = family
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    BYTES_PER_EXPERT: ClassVar[dict[str, int]] = {
        "q3_k": 450_560,
        "iq2_s": 335_872,
        "iq2_xxs": 2_162_688,
        "q2_0": 460_800,
    }

    def launch(
        self,
        first_grad_output: torch.Tensor,
        second_grad_output: torch.Tensor,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        tasks: RowTaskWorkspace,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        tensors = (
            first_grad_output,
            second_grad_output,
            first_packed_weight,
            second_packed_weight,
            grad_input,
            tasks.storage,
            tasks.task_count,
            tasks.task_experts,
            tasks.task_row_starts,
            tasks.task_row_ends,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "paired row-task control requires contiguous HIP tensors"
            )
        out_features = self.OUT_FEATURES[self.family]
        in_features = self.IN_FEATURES[self.family]
        rows = int(first_grad_output.shape[0])
        if tuple(first_grad_output.shape) != (rows, out_features):
            raise HIPRuntimeError("paired row-task gradient shape is invalid")
        if tuple(second_grad_output.shape) != (rows, out_features):
            raise HIPRuntimeError("paired row-task gradient shape is invalid")
        if tuple(grad_input.shape) != (rows, in_features):
            raise HIPRuntimeError("paired row-task output shape is invalid")
        if int(tasks.row_task_rows) != self.row_tile:
            raise HIPRuntimeError("paired row-task tile does not match the body")
        arguments = GROUPED_BACKWARD_PAIR_ROW_TASK_ABI.pack(
            {
                "first_grad_output": first_grad_output.data_ptr(),
                "second_grad_output": second_grad_output.data_ptr(),
                "first_packed_weight": first_packed_weight.data_ptr(),
                "second_packed_weight": second_packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "task_count": tasks.task_count.data_ptr(),
                "task_experts": tasks.task_experts.data_ptr(),
                "task_row_starts": tasks.task_row_starts.data_ptr(),
                "task_row_ends": tasks.task_row_ends.data_ptr(),
                "num_experts": self.physical_experts,
                "rows": rows,
                "bytes_per_expert": self.BYTES_PER_EXPERT[self.family],
            }
        )
        if in_features % self.column_tile:
            raise HIPRuntimeError(
                "paired row-task output width is not a multiple of the column tile"
            )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                in_features // self.column_tile,
                tasks.capacity,
                1,
                self.threads,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )
