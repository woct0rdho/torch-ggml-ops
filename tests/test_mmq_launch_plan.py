"""Guards for the shared deployed-case launch plan."""

import torch

from tools.mmq_deployment_cases import public_deployment_cases
from tools.mmq_hip_row_task import (
    GroupedForwardPairRowTaskWorkspace,
    RowTaskWorkspace,
)
from tools.mmq_launch import (
    FORWARD_OPERATIONS,
    ROUTE_FREE_OPERATIONS,
    ROUTE_OPERATIONS,
    SUPPORTED_OPERATIONS,
)
from torch_ggml_ops.runtime_contract import paired_row_task_capacity


def test_supported_operations_match_the_deployment_inventory() -> None:
    operations = {case.operation for case in public_deployment_cases()}
    assert operations == SUPPORTED_OPERATIONS


def test_operation_sets_are_disjoint_and_complete() -> None:
    assert not ROUTE_FREE_OPERATIONS & ROUTE_OPERATIONS
    assert ROUTE_FREE_OPERATIONS | ROUTE_OPERATIONS == SUPPORTED_OPERATIONS
    assert FORWARD_OPERATIONS <= SUPPORTED_OPERATIONS


def test_row_task_workspaces_take_rows_and_capacity_from_the_contract() -> None:
    """Task banks are sized from the routed reference and the public contract."""

    reference = torch.empty(1152, 8)
    for factory, tile in (
        (RowTaskWorkspace, 128),
        (GroupedForwardPairRowTaskWorkspace, 64),
    ):
        workspace = factory.allocate(reference, route_entries=7, row_tile=tile)
        assert workspace.aggregate_rows == 1152
        assert workspace.row_task_rows == tile
        assert workspace.capacity == paired_row_task_capacity(1152, 7, tile)
        assert workspace.task_row_starts.numel() == workspace.capacity
