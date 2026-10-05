"""Device-only row-task descriptors shared by the grouped controls.

The setup kernel turns cumulative route offsets into explicit
`(expert, row_start, row_end)` tasks, one task per `row_task_rows` rows of each
active group, so the consumer bodies can be launched without host-side route
walking. The workspace is the only allocation the row-task path needs. Its
capacity covers every valid expert partition of the aggregate row count.
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import torch
from typing_extensions import Self

from tools.mmq_abi import GROUPED_ROW_TASK_SETUP_ABI
from tools.mmq_runtime import HIPRuntimeError, _find_installed_kernel, _HIPModule
from torch_ggml_ops.runtime_contract import paired_row_task_capacity


@dataclass(frozen=True)
class RowTaskWorkspace:
    """One row-task descriptor bank in the production caller's contract."""

    storage: torch.Tensor
    task_count: torch.Tensor
    task_experts: torch.Tensor
    task_row_starts: torch.Tensor
    task_row_ends: torch.Tensor
    capacity: int
    route_entries: int
    row_task_rows: int
    aggregate_rows: int

    @classmethod
    def allocate(
        cls,
        reference: torch.Tensor,
        *,
        route_entries: int,
        row_tile: int = 64,
    ) -> Self:
        """Size a task bank for the routed rows of `reference`.

        The reference is the routed activation or gradient tensor, so its row
        count is the aggregate row count the consumer bodies and the setup
        kernel both report.
        """

        aggregate_rows = int(reference.shape[0])
        if route_entries <= 0 or route_entries > 512:
            raise HIPRuntimeError("row-task route entry count is outside the contract")
        if aggregate_rows <= 0 or not 0 < row_tile <= 256:
            raise HIPRuntimeError(
                "row-task workspace requires positive rows and at most M128 tiles"
            )
        # One task per tile of each active group plus one per active group.
        capacity = paired_row_task_capacity(aggregate_rows, route_entries, row_tile)
        storage = torch.empty(
            1 + 3 * capacity,
            device=reference.device,
            dtype=torch.int32,
        )
        return cls(
            storage,
            storage[:1],
            storage[1 : 1 + capacity],
            storage[1 + capacity : 1 + 2 * capacity],
            storage[1 + 2 * capacity :],
            capacity,
            route_entries,
            row_tile,
            aggregate_rows,
        )


class InstalledGroupedRowTaskSetup(_HIPModule):
    """Launch the installed device-only cumulative-route task builder."""

    SYMBOL = "grouped_row_task_setup"

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
        workspace: RowTaskWorkspace,
        *,
        stream: int,
        num_experts: int = 256,
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
        route_entries = int(expert_indices.numel())
        if route_entries != workspace.route_entries:
            raise HIPRuntimeError("row-task workspace route count does not match")
        if int(expert_offsets.numel()) != route_entries:
            raise HIPRuntimeError("row-task route metadata lengths must match")
        expected_capacity = paired_row_task_capacity(
            workspace.aggregate_rows, route_entries, workspace.row_task_rows
        )
        if workspace.capacity != expected_capacity:
            raise HIPRuntimeError("row-task workspace capacity does not match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("row-task setup tensors must share one device")
        packed_arguments = GROUPED_ROW_TASK_SETUP_ABI.pack(
            {
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "task_count": workspace.task_count.data_ptr(),
                "task_experts": workspace.task_experts.data_ptr(),
                "task_row_starts": workspace.task_row_starts.data_ptr(),
                "task_row_ends": workspace.task_row_ends.data_ptr(),
                "num_experts": num_experts,
                "num_groups": route_entries,
                "nrows_activation": workspace.aggregate_rows,
                "row_tile": workspace.row_task_rows,
            }
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
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class GroupedForwardPairRowTaskWorkspace(RowTaskWorkspace):
    """Row-task workspace for the paired kernels, which use J64 tiles."""

    @classmethod
    def allocate(
        cls,
        reference: torch.Tensor,
        *,
        route_entries: int,
        row_tile: int = 64,
    ) -> Self:
        if not 0 < row_tile <= 64:
            raise HIPRuntimeError(
                "paired row-task workspace requires positive rows and at most J64"
            )
        return super().allocate(
            reference, route_entries=route_entries, row_tile=row_tile
        )


InstalledGroupedForwardRowTaskSetup = InstalledGroupedRowTaskSetup
