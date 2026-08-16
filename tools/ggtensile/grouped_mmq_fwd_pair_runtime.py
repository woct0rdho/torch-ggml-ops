"""Research-only HIP launchers for paired grouped IQ2_S artifacts."""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch

from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from .grouped_mmq_fwd_pair_validation import validate_grouped_forward_pair_solution
from .runtime import (
    HIPRuntimeError,
    _find_installed_kernel,
    _HIPModule,
)


class GroupedForwardPairModule(_HIPModule):
    """Launch one exact paired grouped IQ2_S code object."""

    def __init__(
        self,
        solution_key: GroupedForwardPairSolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        reasons = validate_grouped_forward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected paired solution: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardPairState.from_solution_key(solution_key)
        if (
            self.state.contract.route_ownership
            is not GroupedPairRouteOwnership.SerialRoutes
        ):
            raise HIPRuntimeError(
                "serial paired launcher requires serial route ownership"
            )
        super().__init__(
            code_object, hip_library, kernel_name or solution_key.kernel_name
        )

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
        if (
            route_entries <= 0
            or route_entries > self.solution_key.problem.max_route_entries
        ):
            raise HIPRuntimeError("paired route entry count is outside the contract")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("paired route metadata lengths must match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("all paired launch tensors must be on one device")

        arguments = (
            ctypes.c_uint64(first_packed_weight.data_ptr()),
            ctypes.c_uint64(second_packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(first_output.data_ptr()),
            ctypes.c_uint64(second_output.data_ptr()),
            ctypes.c_uint64(expert_indices.data_ptr()),
            ctypes.c_uint64(expert_offsets.data_ptr()),
            ctypes.c_uint32(self.solution_key.problem.physical_experts),
            ctypes.c_uint32(self.solution_key.problem.output_features),
            ctypes.c_uint32(self.solution_key.problem.aggregate_rows),
            ctypes.c_uint32(state.blocks_per_weight_row),
            ctypes.c_uint64(state.bytes_per_expert),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        grid = state.grid(route_entries)
        block = self.solution_key.solution.work_group
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                0,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


@dataclass(frozen=True)
class GroupedForwardPairRowTaskWorkspace:
    storage: torch.Tensor
    task_count: torch.Tensor
    task_experts: torch.Tensor
    task_row_starts: torch.Tensor
    task_row_ends: torch.Tensor
    capacity: int
    route_entries: int

    @classmethod
    def allocate(
        cls,
        reference: torch.Tensor,
        *,
        aggregate_rows: int,
        route_entries: int,
        row_tile: int = 64,
    ) -> "GroupedForwardPairRowTaskWorkspace":
        if route_entries <= 0 or route_entries > 256:
            raise HIPRuntimeError("row-task route entry count is outside the contract")
        if aggregate_rows <= 0 or row_tile != 64:
            raise HIPRuntimeError(
                "paired row-task workspace requires positive rows and J64"
            )
        capacity = (aggregate_rows + row_tile - 1) // row_tile + route_entries
        storage = torch.empty(
            1 + 3 * capacity,
            device=reference.device,
            dtype=torch.int32,
        )
        task_count = storage[:1]
        task_experts = storage[1 : 1 + capacity]
        task_row_starts = storage[1 + capacity : 1 + 2 * capacity]
        task_row_ends = storage[1 + 2 * capacity :]
        return cls(
            storage,
            task_count,
            task_experts,
            task_row_starts,
            task_row_ends,
            capacity,
            route_entries,
        )


class InstalledGroupedForwardRowTaskSetup(_HIPModule):
    """Launch the installed device-only cumulative-route task builder."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_row_task_setup"

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
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        workspace: GroupedForwardPairRowTaskWorkspace,
        *,
        aggregate_rows: int,
        stream: int,
    ) -> None:
        tensors = (
            expert_indices,
            expert_offsets,
            workspace.storage,
            workspace.task_count,
            workspace.task_experts,
            workspace.task_row_starts,
            workspace.task_row_ends,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("row-task setup requires contiguous HIP tensors")
        if expert_indices.dtype != torch.int64 or expert_offsets.dtype != torch.int32:
            raise HIPRuntimeError("row-task setup route dtypes are invalid")
        if workspace.storage.dtype != torch.int32:
            raise HIPRuntimeError("row-task workspace must be int32")
        route_entries = expert_indices.numel()
        if route_entries != workspace.route_entries:
            raise HIPRuntimeError("row-task workspace route count does not match")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("row-task route metadata lengths must match")
        expected_capacity = (aggregate_rows + 63) // 64 + route_entries
        if workspace.capacity != expected_capacity:
            raise HIPRuntimeError("row-task workspace capacity does not match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("row-task setup tensors must share one device")
        arguments = (
            ctypes.c_uint64(expert_indices.data_ptr()),
            ctypes.c_uint64(expert_offsets.data_ptr()),
            ctypes.c_uint64(workspace.task_count.data_ptr()),
            ctypes.c_uint64(workspace.task_experts.data_ptr()),
            ctypes.c_uint64(workspace.task_row_starts.data_ptr()),
            ctypes.c_uint64(workspace.task_row_ends.data_ptr()),
            ctypes.c_uint32(256),
            ctypes.c_uint32(route_entries),
            ctypes.c_uint32(aggregate_rows),
            ctypes.c_uint32(64),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                1,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class GroupedForwardPairRowTaskModule(_HIPModule):
    """Launch one exact paired row-task IQ2_S code object."""

    def __init__(
        self,
        solution_key: GroupedForwardPairSolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
    ) -> None:
        reasons = validate_grouped_forward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected paired solution: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardPairState.from_solution_key(solution_key)
        if (
            self.state.contract.route_ownership
            is not GroupedPairRouteOwnership.DeviceRowTasks64
        ):
            raise HIPRuntimeError("row-task launcher requires row-task ownership")
        super().__init__(code_object, hip_library, solution_key.kernel_name)

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
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("paired row-task tensors must share one device")
        arguments = (
            ctypes.c_uint64(first_packed_weight.data_ptr()),
            ctypes.c_uint64(second_packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(first_output.data_ptr()),
            ctypes.c_uint64(second_output.data_ptr()),
            ctypes.c_uint64(tasks.task_count.data_ptr()),
            ctypes.c_uint64(tasks.task_experts.data_ptr()),
            ctypes.c_uint64(tasks.task_row_starts.data_ptr()),
            ctypes.c_uint64(tasks.task_row_ends.data_ptr()),
            ctypes.c_uint32(self.solution_key.problem.physical_experts),
            ctypes.c_uint32(self.solution_key.problem.output_features),
            ctypes.c_uint32(self.solution_key.problem.aggregate_rows),
            ctypes.c_uint32(state.blocks_per_weight_row),
            ctypes.c_uint64(state.bytes_per_expert),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        grid = state.row_task_grid(tasks.route_entries)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *self.solution_key.solution.work_group,
                0,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedForwardPairRowTaskControl(_HIPModule):
    """Launch one installed IQ2_S N512/K2048 J64 row-task projection."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_row_task_iq2_s_n512_k2048_j64"

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
        arguments = (
            ctypes.c_uint64(packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_uint64(tasks.task_count.data_ptr()),
            ctypes.c_uint64(tasks.task_experts.data_ptr()),
            ctypes.c_uint64(tasks.task_row_starts.data_ptr()),
            ctypes.c_uint64(tasks.task_row_ends.data_ptr()),
            ctypes.c_uint32(aggregate_rows),
            ctypes.c_uint64(335872),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
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
                30_976,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class InstalledGroupedForwardPairSerialControl(_HIPModule):
    """Launch the installed single-projection N512/K2048 IQ2_S control."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_iq2_s_n512_k2048_j64"

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
        arguments = (
            ctypes.c_uint64(packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_uint64(expert_indices.data_ptr()),
            ctypes.c_uint64(expert_offsets.data_ptr()),
            ctypes.c_uint32(256),
            ctypes.c_uint32(512),
            ctypes.c_uint32(aggregate_rows),
            ctypes.c_uint32(8),
            ctypes.c_uint64(335872),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
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
                30_976,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )
