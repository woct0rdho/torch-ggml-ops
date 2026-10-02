"""HIP launchers for paired grouped MMQ artifacts.

These launch the paired forward and paired backward code objects built by the
GGTensile writer path. The deployed installed-control counterparts live in
`tools.mmq_hip_grouped_pair_fwd` and `tools.mmq_hip_grouped_pair_bwd`.
"""

import ctypes
from pathlib import Path

import torch

from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from tools.ggtensile.grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
    GroupedBackwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_bwd_pair_validation import (
    validate_grouped_backward_pair_solution,
)
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_pair_validation import (
    validate_grouped_forward_pair_solution,
)
from tools.mmq_abi import (
    GROUPED_BACKWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
)
from tools.mmq_hip_row_task import GroupedForwardPairRowTaskWorkspace
from tools.mmq_runtime import HIPRuntimeError, _HIPModule
from tools.mmq_work_group_mapping import mapped_grid_extent


class GroupedForwardPairModule(_HIPModule):
    """Launch one exact paired grouped forward code object."""

    def __init__(
        self,
        problem: GroupedForwardPairProblem,
        kernel_spec: GroupedForwardPairKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_grouped_forward_pair_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedGroupedForwardPairState.from_problem_spec(
            problem, kernel_spec
        )
        if (
            self.state.kernel_spec.route_ownership
            is not GroupedPairRouteOwnership.SerialRoutes
        ):
            raise HIPRuntimeError(
                "serial paired launcher requires serial route ownership"
            )
        super().__init__(code_object, hip_library, kernel_name)

    def launch(
        self,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        activations: torch.Tensor,
        first_output: torch.Tensor,
        second_output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = self.state
        tensors = (
            first_packed_weight,
            second_packed_weight,
            activations,
            first_output,
            second_output,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all paired launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all paired launch tensors must be contiguous")
        if (
            first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
        ):
            raise HIPRuntimeError("paired packed weights must be uint8")
        if activations.dtype != torch.uint8:
            raise HIPRuntimeError("paired activations must be uint8 Q8_1 F32_D4")
        if (
            first_output.dtype != torch.bfloat16
            or second_output.dtype != torch.bfloat16
        ):
            raise HIPRuntimeError("paired outputs must be BF16")
        if expert_indices.dtype != torch.int64 or expert_offsets.dtype != torch.int32:
            raise HIPRuntimeError("paired route metadata has the wrong dtype")
        if tuple(first_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError(
                "first packed weight shape does not match paired bank"
            )
        if tuple(second_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError(
                "second packed weight shape does not match paired bank"
            )
        if tuple(activations.shape) != state.expected_activation_shape:
            raise HIPRuntimeError(
                "activation workspace shape does not match paired contract"
            )
        if (
            tuple(first_output.shape) != state.expected_output_shape
            or tuple(second_output.shape) != state.expected_output_shape
        ):
            raise HIPRuntimeError("paired output shape does not match the exact key")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("paired route metadata must be one-dimensional")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > self.problem.max_route_entries:
            raise HIPRuntimeError("paired route entry count is outside the contract")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("paired route metadata lengths must match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("all paired launch tensors must be on one device")

        packed_arguments = GROUPED_FORWARD_PAIR_ABI.pack(
            {
                "weights_first": first_packed_weight.data_ptr(),
                "weights_second": second_packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst_first": first_output.data_ptr(),
                "dst_second": second_output.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.problem.physical_experts,
                "nrows_weight": self.problem.output_features,
                "nrows_activation": self.problem.aggregate_rows,
                "blocks_per_weight_row": state.blocks_per_weight_row,
                "bytes_per_expert": state.bytes_per_expert,
            }
        )
        grid = state.grid(route_entries)
        block = state.kernel_spec.geometry.work_group
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                0,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class GroupedForwardPairRowTaskModule(_HIPModule):
    """Launch one exact paired row-task grouped code object."""

    def __init__(
        self,
        problem: GroupedForwardPairProblem,
        kernel_spec: GroupedForwardPairKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_grouped_forward_pair_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedGroupedForwardPairState.from_problem_spec(
            problem, kernel_spec
        )
        if (
            self.state.kernel_spec.route_ownership
            is not GroupedPairRouteOwnership.DeviceRowTasks
        ):
            raise HIPRuntimeError("row-task launcher requires row-task ownership")
        super().__init__(code_object, hip_library, kernel_name)

    def launch(
        self,
        first_packed_weight: torch.Tensor,
        second_packed_weight: torch.Tensor,
        activations: torch.Tensor,
        first_output: torch.Tensor,
        second_output: torch.Tensor,
        tasks: GroupedForwardPairRowTaskWorkspace,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = self.state
        tensors = (
            first_packed_weight,
            second_packed_weight,
            activations,
            first_output,
            second_output,
            tasks.storage,
            tasks.task_count,
            tasks.task_experts,
            tasks.task_row_starts,
            tasks.task_row_ends,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "paired row-task launch requires contiguous HIP tensors"
            )
        if (
            first_packed_weight.dtype != torch.uint8
            or second_packed_weight.dtype != torch.uint8
            or activations.dtype != torch.uint8
            or first_output.dtype != torch.bfloat16
            or second_output.dtype != torch.bfloat16
            or tasks.storage.dtype != torch.int32
        ):
            raise HIPRuntimeError("paired row-task tensor dtypes are invalid")
        if tuple(first_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError(
                "first packed weight shape does not match paired bank"
            )
        if tuple(second_packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError(
                "second packed weight shape does not match paired bank"
            )
        if tuple(activations.shape) != state.expected_activation_shape:
            raise HIPRuntimeError(
                "activation workspace shape does not match paired contract"
            )
        if (
            tuple(first_output.shape) != state.expected_output_shape
            or tuple(second_output.shape) != state.expected_output_shape
        ):
            raise HIPRuntimeError("paired output shape does not match the exact key")
        if tasks.capacity != state.row_task_capacity(tasks.route_entries):
            raise HIPRuntimeError(
                "paired row-task capacity does not match the exact key"
            )
        if tasks.row_task_rows != state.kernel_spec.row_task_rows:
            raise HIPRuntimeError(
                "paired row-task rows do not match the kernel specification"
            )
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("paired row-task tensors must share one device")
        packed_arguments = GROUPED_FORWARD_PAIR_ROW_TASK_ABI.pack(
            {
                "weights_first": first_packed_weight.data_ptr(),
                "weights_second": second_packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst_first": first_output.data_ptr(),
                "dst_second": second_output.data_ptr(),
                "task_count": tasks.task_count.data_ptr(),
                "task_experts": tasks.task_experts.data_ptr(),
                "task_row_starts": tasks.task_row_starts.data_ptr(),
                "task_row_ends": tasks.task_row_ends.data_ptr(),
                "num_experts": self.problem.physical_experts,
                "nrows_weight": self.problem.output_features,
                "nrows_activation": self.problem.aggregate_rows,
                "blocks_per_weight_row": state.blocks_per_weight_row,
                "bytes_per_expert": state.bytes_per_expert,
            }
        )
        grid = state.row_task_grid(tasks.route_entries)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *state.kernel_spec.geometry.work_group,
                0,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class GroupedBackwardPairModule(_HIPModule):
    """Launch one exact fused grouped backward-pair code object."""

    def __init__(
        self,
        problem: GroupedBackwardPairProblem,
        kernel_spec: GroupedBackwardPairKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_grouped_backward_pair_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedGroupedBackwardPairState.from_problem_spec(
            problem, kernel_spec
        )
        super().__init__(code_object, hip_library, kernel_name)

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
        problem = self.problem
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
        compute = self.kernel_spec.compute
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                mapped_grid_extent(
                    problem.in_features // compute.geometry.macro_tile1,
                    compute.geometry.work_group_mapping,
                ),
                route_entries * self.kernel_spec.route_ownership.split_factor,
                1,
                *compute.geometry.work_group,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )
