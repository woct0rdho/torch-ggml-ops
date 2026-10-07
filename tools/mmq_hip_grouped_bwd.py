"""Direct launchers for installed standalone grouped-backward controls."""

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from tools.mmq_abi import (
    GROUPED_BACKWARD_ABI,
    GROUPED_BACKWARD_ROW_TASK_ABI,
)
from tools.mmq_hip_row_task import RowTaskWorkspace
from tools.mmq_runtime import HIPRuntimeError, _HIPModule, find_control


@dataclass(frozen=True)
class _ControlSpec:
    symbol: str
    out_features: int
    in_features: int
    packed_row_bytes: int
    tiled_n: int = 128
    physical_experts: int = 256

    @property
    def bytes_per_expert(self) -> int:
        return self.out_features * self.packed_row_bytes


_SYMBOL_SPECS: dict[str, _ControlSpec] = {
    spec.symbol: spec
    for spec in (
        _ControlSpec("grouped_bwd_single_q2_k_n4096_k2048_mt64_nt64", 4096, 2048, 672),
        _ControlSpec("grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64", 4096, 2048, 672),
        _ControlSpec(
            "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64_u2", 4096, 2048, 672
        ),
        _ControlSpec("grouped_bwd_single_q4_k_n2048_k512_mt64_nt64", 2048, 512, 288),
        _ControlSpec("grouped_bwd_single_q4_k_n2048_k512_mt128_nt64", 2048, 512, 288),
        _ControlSpec("grouped_bwd_single_q5_k_n2048_k512_mt64_nt64", 2048, 512, 352),
        _ControlSpec("grouped_bwd_single_iq2_s_n2048_k512_mt64_nt64", 2048, 512, 164),
        _ControlSpec("grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64", 2048, 512, 164),
        _ControlSpec(
            "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128", 2048, 512, 288
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128", 2048, 512, 352
        ),
        _ControlSpec(
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128", 2048, 512, 164
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3",
            2560,
            640,
            180,
            tiled_n=64,
            physical_experts=512,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4",
            4096,
            2048,
            672,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8_g4",
            2560,
            640,
            180,
            tiled_n=64,
            physical_experts=512,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8",
            2560,
            640,
            180,
            tiled_n=64,
            physical_experts=512,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3",
            2048,
            512,
            288,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2",
            2048,
            512,
            352,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2",
            2048,
            512,
            164,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3",
            4096,
            2048,
            672,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_0_n2560_k640_mt256_nt64_s3_ki64_sw8_g4_abar",
            2560,
            640,
            180,
            tiled_n=64,
            physical_experts=512,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_0_n2560_k640_mt128_nt64_s3_sw8_g4_abar",
            2560,
            640,
            180,
            tiled_n=64,
            physical_experts=512,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2_abar",
            2048,
            512,
            164,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2_abar",
            2048,
            512,
            352,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_k_n4096_k2048_mt256_nt64_s3_g4_ki64",
            4096,
            2048,
            672,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3_g4_abar",
            4096,
            2048,
            672,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_q5_k_n2048_k512_mt256_nt64_s2",
            2048,
            512,
            352,
            tiled_n=64,
        ),
        _ControlSpec(
            "grouped_bwd_row_task_iq2_s_n2048_k512_mt256_nt64_s2",
            2048,
            512,
            164,
            tiled_n=64,
        ),
    )
}


class InstalledGroupedBackwardControl(_HIPModule):
    """Launch the old specialized grouped-backward HIP body directly."""

    PHYSICAL_EXPERTS = 256

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        spec = _SYMBOL_SPECS.get(symbol)
        if spec is None:
            raise HIPRuntimeError(
                f"no standalone grouped-backward control for symbol {symbol!r}"
            )
        self.spec = spec
        super().__init__(
            find_control(spec.symbol, code_object), hip_library, spec.symbol
        )

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        tensors = (
            grad_output,
            packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "grouped-backward control requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError(
                "grouped-backward control requires zero storage offsets"
            )
        if (
            grad_output.dtype != torch.bfloat16
            or packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("grouped-backward control tensor dtypes are invalid")
        spec = self.spec
        rows = grad_output.shape[0]
        if tuple(grad_output.shape) != (rows, spec.out_features):
            raise HIPRuntimeError("grouped-backward control gradient shape is invalid")
        if tuple(packed_weight.shape) != (
            spec.physical_experts,
            spec.out_features,
            spec.packed_row_bytes,
        ):
            raise HIPRuntimeError("grouped-backward control weight shape is invalid")
        if tuple(grad_input.shape) != (rows, spec.in_features):
            raise HIPRuntimeError("grouped-backward control output shape is invalid")
        route_entries = expert_indices.numel()
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError(
                "grouped-backward control routes must be one-dimensional"
            )
        if route_entries <= 0 or route_entries > self.PHYSICAL_EXPERTS:
            raise HIPRuntimeError("grouped-backward control route count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("grouped-backward control route lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError(
                "grouped-backward control tensors must share a device"
            )
        arguments = GROUPED_BACKWARD_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.PHYSICAL_EXPERTS,
                "rows": rows,
                "bytes_per_expert": spec.bytes_per_expert,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                spec.in_features // 64,
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


class InstalledGroupedBackwardRowTaskControl(_HIPModule):
    """Launch one M128/N128 row-task grouped-backward body.

    The body indexes lanes with `threadIdx.x` only and expects the 128-thread
    backward block, so it launches as `(128, 1, 1)`. Its task descriptors must
    cover the full 128-row M tile. Callers size the workspace with
    `ROW_TASK_ROWS`.
    """

    PHYSICAL_EXPERTS = 256
    TILED_N = 128
    ROW_TASK_ROWS = 128
    # Task descriptor height and workgroup width per symbol. A wide body tiles
    # more rows per descriptor, so the bank must be built with the same height
    # and the launch with the same width.
    ROW_TILES: ClassVar[dict[str, int]] = {
        "grouped_bwd_row_task_iq2_s_n2048_k512_mt256_nt64_s2": 256,
        "grouped_bwd_row_task_q5_k_n2048_k512_mt256_nt64_s2": 256,
        "grouped_bwd_row_task_q2_k_n4096_k2048_mt256_nt64_s3_g4_ki64": 256,
        "grouped_bwd_row_task_q2_0_n2560_k640_mt256_nt64_s3_ki64_sw8_g4_abar": 256,
    }
    THREADS: ClassVar[dict[str, int]] = {
        "grouped_bwd_row_task_iq2_s_n2048_k512_mt256_nt64_s2": 256,
        "grouped_bwd_row_task_q5_k_n2048_k512_mt256_nt64_s2": 256,
        "grouped_bwd_row_task_q2_k_n4096_k2048_mt256_nt64_s3_g4_ki64": 256,
        "grouped_bwd_row_task_q2_0_n2560_k640_mt256_nt64_s3_ki64_sw8_g4_abar": 256,
    }

    @property
    def row_tile(self) -> int:
        return self.ROW_TILES.get(self.spec.symbol, self.ROW_TASK_ROWS)

    @property
    def threads(self) -> int:
        return self.THREADS.get(self.spec.symbol, 128)

    def __init__(
        self,
        symbol: str,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        spec = _SYMBOL_SPECS.get(symbol)
        if spec is None or not symbol.startswith("grouped_bwd_row_task_"):
            raise HIPRuntimeError(
                f"no grouped-backward row-task control for symbol {symbol!r}"
            )
        self.spec = spec
        super().__init__(find_control(symbol, code_object), hip_library, symbol)

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        tasks: RowTaskWorkspace,
        *,
        stream: int,
    ) -> None:
        tensors = (
            grad_output,
            packed_weight,
            grad_input,
            tasks.storage,
            tasks.task_count,
            tasks.task_experts,
            tasks.task_row_starts,
            tasks.task_row_ends,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "grouped-backward row-task control requires contiguous HIP tensors"
            )
        spec = self.spec
        rows = int(grad_output.shape[0])
        if tuple(grad_output.shape) != (rows, spec.out_features):
            raise HIPRuntimeError("row-task gradient shape is invalid")
        if tuple(packed_weight.shape) != (
            spec.physical_experts,
            spec.out_features,
            spec.packed_row_bytes,
        ):
            raise HIPRuntimeError("row-task weight shape is invalid")
        if tuple(grad_input.shape) != (rows, spec.in_features):
            raise HIPRuntimeError("row-task output shape is invalid")
        arguments = GROUPED_BACKWARD_ROW_TASK_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "task_count": tasks.task_count.data_ptr(),
                "task_experts": tasks.task_experts.data_ptr(),
                "task_row_starts": tasks.task_row_starts.data_ptr(),
                "task_row_ends": tasks.task_row_ends.data_ptr(),
                "bytes_per_expert": spec.bytes_per_expert,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                (spec.in_features + spec.tiled_n - 1) // spec.tiled_n,
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
