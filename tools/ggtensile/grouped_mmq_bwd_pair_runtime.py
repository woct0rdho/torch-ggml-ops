"""Research-only launchers for paired grouped MMQ backward artifacts."""

import ctypes
from pathlib import Path
from typing import ClassVar

import torch

from .grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .grouped_mmq_bwd_pair_validation import validate_grouped_backward_pair_solution
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .runtime import HIPRuntimeError, _find_installed_kernel, _HIPModule


class GroupedBackwardPairModule(_HIPModule):
    """Launch one exact fused grouped backward-pair code object."""

    def __init__(
        self,
        solution_key: GroupedBackwardPairSolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        reasons = validate_grouped_backward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected backward pair: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedBackwardPairState.from_solution_key(solution_key)
        super().__init__(
            code_object,
            hip_library,
            kernel_name or solution_key.kernel_name,
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
        state = self.state
        problem = self.solution_key.problem
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
                "backward-pair launch requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError("backward-pair tensors require zero storage offsets")
        if (
            first_grad_output.dtype != torch.bfloat16
            or second_grad_output.dtype != torch.bfloat16
            or first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("backward-pair tensor dtypes are invalid")
        expected_grad = (problem.aggregate_rows, problem.out_features)
        if tuple(first_grad_output.shape) != expected_grad:
            raise HIPRuntimeError("first backward-pair gradient shape is invalid")
        if tuple(second_grad_output.shape) != expected_grad:
            raise HIPRuntimeError("second backward-pair gradient shape is invalid")
        if tuple(first_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError("first backward-pair weight shape is invalid")
        if tuple(second_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError("second backward-pair weight shape is invalid")
        if tuple(grad_input.shape) != (problem.aggregate_rows, problem.in_features):
            raise HIPRuntimeError("backward-pair destination shape is invalid")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError(
                "backward-pair route metadata must be one-dimensional"
            )
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > problem.max_route_entries:
            raise HIPRuntimeError("backward-pair route count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("backward-pair route metadata lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("backward-pair tensors must share one device")

        arguments = GROUPED_BACKWARD_PAIR_ABI.pack(
            {
                "first_grad_output": first_grad_output.data_ptr(),
                "second_grad_output": second_grad_output.data_ptr(),
                "first_packed_weight": first_packed_weight.data_ptr(),
                "second_packed_weight": second_packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": problem.physical_experts,
                "rows": problem.aggregate_rows,
                "bytes_per_expert": state.bytes_per_expert,
            }
        )
        compute = self.solution_key.solution.compute
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                problem.in_features // compute.macro_tile1,
                route_entries * self.solution_key.solution.route_ownership.split_factor,
                1,
                *compute.work_group,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedBackwardPairIQ2SControl(_HIPModule):
    """Launch one installed specialized Qwen IQ2_S backward-pair body."""

    _CONFIGS: ClassVar[dict[int, str]] = {
        64: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64"
        ),
        128: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64"
        ),
    }
    PHYSICAL_EXPERTS = 256
    OUT_FEATURES = 512
    IN_FEATURES = 2048
    PACKED_ROW_BYTES = 656
    BYTES_PER_EXPERT = 335_872

    def __init__(
        self,
        macro_tile0: int,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        symbol = self._CONFIGS.get(macro_tile0)
        if symbol is None:
            raise HIPRuntimeError("installed IQ2_S pair control requires M64 or M128")
        self.macro_tile0 = macro_tile0
        super().__init__(
            code_object or _find_installed_kernel(symbol),
            hip_library,
            symbol,
        )

    @staticmethod
    def dispatched_macro_tile(aggregate_rows: int, route_entries: int) -> int:
        if aggregate_rows <= 0 or route_entries <= 0:
            raise HIPRuntimeError("installed pair dispatch dimensions must be positive")
        return 128 if aggregate_rows >= 128 * route_entries else 64

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
        expected_tile = self.dispatched_macro_tile(rows, route_entries)
        if self.macro_tile0 != expected_tile:
            raise HIPRuntimeError(
                f"installed pair dispatch selects M{expected_tile}, not M{self.macro_tile0}"
            )

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

    _CONFIGS: ClassVar[dict[int, str]] = {
        64: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_bwd_pair_iq2_xxs_"
            "n2048_k4096_mt64_nt64"
        ),
        128: (
            "torch_ggml_ops_mmq_gfx1151_v1_grouped_bwd_tuned_pair_iq2_xxs_"
            "n2048_k4096_mt128_nt64"
        ),
    }
    QUALIFIED_M128_ROWS = frozenset({49_152, 196_608})
    PHYSICAL_EXPERTS = 256
    OUT_FEATURES = 2048
    IN_FEATURES = 4096
    PACKED_ROW_BYTES = 1056
    BYTES_PER_EXPERT = 2_162_688

    def __init__(
        self,
        macro_tile0: int,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        symbol = self._CONFIGS.get(macro_tile0)
        if symbol is None:
            raise HIPRuntimeError("installed IQ2_XXS pair control requires M64 or M128")
        self.macro_tile0 = macro_tile0
        super().__init__(
            code_object or _find_installed_kernel(symbol), hip_library, symbol
        )

    @classmethod
    def dispatched_macro_tile(cls, aggregate_rows: int) -> int:
        if aggregate_rows <= 0:
            raise HIPRuntimeError("installed IQ2_XXS pair rows must be positive")
        return 128 if aggregate_rows in cls.QUALIFIED_M128_ROWS else 64

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
        expected_tile = self.dispatched_macro_tile(rows)
        if self.macro_tile0 != expected_tile:
            raise HIPRuntimeError(
                f"installed IQ2_XXS pair dispatch selects M{expected_tile}, "
                f"not M{self.macro_tile0}"
            )

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
