"""Prepare direct HIP and GGTensile kernel implementations.

Both routes are built from `tools.mmq_launch`, which owns the per-operation
knowledge of which module to build, which auxiliary inputs a launch needs
(activation workspace, device row-task bank), how large the outputs are and how
to launch. This module only wires those prepared launches into benchmark
`Implementation` objects and their report metadata.
"""

import contextlib
from pathlib import Path
from typing import Any, cast

from bench.benchmark_common import Implementation
from bench.benchmark_data import RouteSelection
from tools.ggtensile.family_registry import instance_name
from tools.ggtensile.kernel_instance import KernelInstance
from tools.mmq_correctness import PreparedCase
from tools.mmq_deployment_cases import DeploymentCase, public_artifact_path
from tools.mmq_launch import (
    FORWARD_OPERATIONS,
    ROUTE_FREE_OPERATIONS,
    Launcher,
    activation_workspace,
    build_launcher,
    output_tensors,
)


def _metadata(
    name: str,
    case: DeploymentCase,
    *,
    artifact: Path | None = None,
    launcher: Launcher | None = None,
) -> dict[str, object]:
    if name == "ggtensile":
        assert artifact is not None
        return {"artifact": str(artifact), "symbol": case.symbol}
    assert launcher is not None
    return {"artifacts": [str(path) for path in launcher.code_objects]}


def _distinct_entry_counts(selection: RouteSelection) -> tuple[int, ...]:
    return tuple(
        sorted({int(route.expert_indices.numel()) for route in selection.routes})
    )


def _prepare_case(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    artifact: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    """Prepare one implementation per direct name, sharing case-level state.

    Route-consuming families build one launcher per distinct route-bank width,
    because the deployed control is selected from that width, and one prepared
    launch per route vector so the timed call is a bare launch.
    """

    workspace = (
        activation_workspace(case, prepared, stack)
        if case.operation in FORWARD_OPERATIONS
        else None
    )
    device = (
        prepared.input.device
        if case.operation in FORWARD_OPERATIONS and prepared.input is not None
        else prepared.grad_outputs[0].device
    )
    launchers: dict[str, dict[int | None, Launcher]] = {}
    for name in names:
        entries = (
            (None,)
            if case.operation in ROUTE_FREE_OPERATIONS
            else _distinct_entry_counts(selection)
        )
        launchers[name] = {
            count: build_launcher(
                case,
                prepared,
                name,
                stack=stack,
                root=hip_root,
                artifact=artifact,
                instance=instance_name(cast(KernelInstance, case.instance)),
                route_entries=count,
                workspace=workspace,
            )
            for count in entries
        }

    implementations: dict[str, Implementation] = {}
    for name in names:
        by_entries = launchers[name]
        if case.operation in ROUTE_FREE_OPERATIONS:
            # Route-free families prepare one launch and reuse it for every
            # measured route slot.
            routes = [next(iter(by_entries.values())).prepare_route(None, stack)]
        else:
            routes = [
                by_entries[int(route.expert_indices.numel())].prepare_route(
                    route, stack
                )
                for route in selection.routes
            ]
        outputs = output_tensors(case, device)

        def launch(routes=routes, outputs=outputs, selection=selection) -> Any:
            routes[selection.index].call(outputs)
            return outputs[0] if len(outputs) == 1 else outputs

        launcher = next(iter(by_entries.values()))
        implementations[name] = Implementation(
            name,
            launch,
            _metadata(name, case, artifact=artifact, launcher=launcher),
        )
    return implementations


def prepare_direct_implementations(
    stack: contextlib.ExitStack,
    case: DeploymentCase,
    prepared: PreparedCase,
    selection: RouteSelection,
    names: tuple[str, ...],
    *,
    ggtensile_root: Path | None,
    hip_root: Path | None,
) -> dict[str, Implementation]:
    direct_names = tuple(name for name in names if name != "baseline")
    if not direct_names:
        raise ValueError("at least one direct implementation is required")
    artifact = (
        public_artifact_path(case, ggtensile_root)
        if "ggtensile" in direct_names
        else None
    )
    if artifact is not None and not artifact.is_file():
        raise FileNotFoundError(f"deployed GGTensile artifact not found: {artifact}")
    return _prepare_case(
        stack, case, prepared, selection, direct_names, artifact, hip_root
    )
