"""Deployed-case launch plans.

One module answers, per operation and implementation: which module to build,
which auxiliary inputs a launch needs (a quantized activation workspace, a
device row-task bank), how large the caller-owned outputs are, and how to
launch. `bench.benchmark_kernels` and `tools.mmq_deployment_runner` both consume
it, so a new control, workspace or task bank is wired once instead of once per
caller.

A launcher is case-scoped: modules, workspaces and route-independent state are
built once, and `prepare_route` returns the launch for a single route bank for
callers that time route vectors individually.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import FixedBackwardKernelSpec
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardProblem
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import FixedForwardKernelSpec
from tools.ggtensile.grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from tools.ggtensile.grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec
from tools.ggtensile.grouped_mmq_bwd_spec import GroupedBackwardKernelSpec
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardProblem
from tools.ggtensile.grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
)
from tools.ggtensile.grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_spec import GroupedForwardKernelSpec
from tools.ggtensile.mmq_bwd_spec import BackwardKernelSpec
from tools.ggtensile.mmq_fwd_spec import ForwardKernelSpec
from tools.ggtensile.model import ProblemSize
from tools.mmq_deployment_cases import DeploymentCase
from tools.mmq_hip_dense import (
    InstalledDenseBackwardModule,
    InstalledDenseBackwardSplitKModule,
    InstalledSplitKReduceModule,
)
from tools.mmq_hip_deployment import (
    control_inventory,
    select_grouped_forward_control,
    select_hip_control,
    select_routed_control,
)
from tools.mmq_hip_grouped_pair_bwd import is_pair_row_task_body
from tools.mmq_hip_launchers import (
    build_backward_control,
    build_backward_pair_control,
    build_forward_pair_control,
    is_row_task_body,
    quantizer_module,
)
from tools.mmq_hip_row_task import (
    GroupedForwardPairRowTaskWorkspace,
    InstalledGroupedRowTaskSetup,
    RowTaskWorkspace,
)
from tools.mmq_pair_runtime import (
    GroupedBackwardPairModule,
    GroupedForwardPairModule,
    GroupedForwardPairRowTaskModule,
)
from tools.mmq_runtime import (
    BackwardModule,
    FixedGroupedQ8BackwardModule,
    FixedGroupedQ8ForwardModule,
    FixedHipForwardModule,
    ForwardModule,
    GroupedBackwardModule,
    GroupedForwardModule,
    InstalledFixedGroupedQ8BackwardModule,
    InstalledFixedGroupedQ8ForwardModule,
    InstalledGroupedForwardModule,
)

Outputs = tuple[torch.Tensor, ...]

# Operations whose launch takes neither a route bank nor a task bank.
ROUTE_FREE_OPERATIONS = frozenset(
    {
        "OrdinaryForward",
        "OrdinaryBackward",
        "FixedGroupedForward",
        "FixedGroupedBackward",
    }
)
FORWARD_OPERATIONS = frozenset(
    {
        "OrdinaryForward",
        "GroupedForward",
        "GroupedForwardPair",
        "FixedGroupedForward",
    }
)
# Operations whose launch consumes a route bank or a device task bank.
ROUTE_OPERATIONS = frozenset(
    {
        "GroupedForward",
        "GroupedBackward",
        "GroupedForwardPair",
        "GroupedBackwardPair",
    }
)
# Every operation the shared plan can launch. The deployment inventory must
# select from this set, so a new family cannot bypass the plan.
SUPPORTED_OPERATIONS = ROUTE_FREE_OPERATIONS | ROUTE_OPERATIONS


@dataclass(frozen=True)
class Launch:
    """One prepared launch for a case, an implementation and a route bank."""

    call: Callable[[Outputs], None]
    tasks: Any = None


@dataclass(frozen=True)
class Launcher:
    """Case-scoped launch plan for one implementation.

    Modules are already entered in the caller's stack and stay alive for
    metadata and teardown. `workspace` is the activation workspace a forward
    launch consumes. `row_tile` is the device task tile when the selected body
    reads a task bank instead of the route bank, and `tasks_factory` is the
    workspace type that matches that body.
    """

    operation: str
    implementation: str
    modules: tuple[Any, ...]
    symbols: tuple[str, ...]
    inputs: tuple[torch.Tensor, ...]
    weights: tuple[torch.Tensor, ...]
    workspace: torch.Tensor | None
    row_tile: int | None
    tasks_factory: Any
    reference: torch.Tensor
    device: torch.device
    partials: torch.Tensor | None = None
    reduce_modules: tuple[Any, ...] = ()
    split_k_shape: tuple[int, int, int] | None = None

    @property
    def needs_route(self) -> bool:
        """Return whether the launch consumes route metadata."""

        return self.operation in ROUTE_OPERATIONS

    @property
    def code_objects(self) -> tuple[Path, ...]:
        """Return the code objects behind the modules, for reporting."""

        return tuple(
            module.code_object
            for module in self.modules
            if getattr(module, "code_object", None) is not None
        )

    def prepare_route(self, route: Any, stack: contextlib.ExitStack) -> Launch:
        """Return the launch for one route bank, building its task bank."""

        if not self.needs_route:
            return Launch(self._call(None, None))
        if self.row_tile is None:
            return Launch(self._call(route, None))
        setup = stack.enter_context(InstalledGroupedRowTaskSetup())
        tasks = self.tasks_factory.allocate(
            self.reference,
            route_entries=int(route.expert_indices.numel()),
            row_tile=self.row_tile,
        )
        setup.launch(
            route.expert_indices,
            route.expert_offsets,
            tasks,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        return Launch(self._call(route, tasks), tasks)

    def _call(self, route: Any, tasks: Any) -> Callable[[Outputs], None]:
        """Return the launch that fills caller-owned outputs."""

        stream = torch.cuda.current_stream().cuda_stream
        hip = self.implementation == "hip"
        modules = self.modules
        weights = self.weights
        inputs = self.inputs
        workspace = self.workspace
        partials = self.partials
        reduce_modules = self.reduce_modules
        operation = self.operation
        rows, in_features, slices = self.split_k_shape or (0, 0, 0)

        def dense_forward(outputs: Outputs) -> None:
            modules[0].launch(weights[0], workspace, outputs[0], stream=stream)

        def dense_backward(outputs: Outputs) -> None:
            if partials is None:
                modules[0].launch(inputs[0], weights[0], outputs[0], stream=stream)
                return
            modules[0].launch(inputs[0], weights[0], partials, stream=stream)
            reduce_modules[0].launch(
                partials, outputs[0], rows, in_features, slices, stream=stream
            )

        def grouped_forward(outputs: Outputs) -> None:
            modules[0].launch(
                weights[0],
                workspace,
                outputs[0],
                route.expert_indices,
                route.expert_offsets,
                stream=stream,
            )

        def grouped_backward(outputs: Outputs) -> None:
            if tasks is None:
                modules[0].launch(
                    inputs[0],
                    weights[0],
                    outputs[0],
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
            else:
                modules[0].launch(
                    inputs[0], weights[0], outputs[0], tasks, stream=stream
                )

        # The standalone controls launch one projection each. The exact
        # bundled module launches both projections in one call. Both derive the
        # routed row count themselves.

        def grouped_forward_pair(outputs: Outputs) -> None:
            if not hip:
                if tasks is None:
                    modules[0].launch(
                        weights[0],
                        weights[1],
                        workspace,
                        outputs[0],
                        outputs[1],
                        route.expert_indices,
                        route.expert_offsets,
                        stream=stream,
                    )
                else:
                    modules[0].launch(
                        weights[0],
                        weights[1],
                        workspace,
                        outputs[0],
                        outputs[1],
                        tasks,
                        stream=stream,
                    )
                return
            for module, weight, output in zip(modules, weights, outputs, strict=True):
                if tasks is None:
                    module.launch(
                        weight,
                        workspace,
                        output,
                        route.expert_indices,
                        route.expert_offsets,
                        stream=stream,
                    )
                else:
                    module.launch(weight, workspace, output, tasks, stream=stream)

        def grouped_backward_pair(outputs: Outputs) -> None:
            if tasks is None:
                modules[0].launch(
                    inputs[0],
                    inputs[1],
                    weights[0],
                    weights[1],
                    outputs[0],
                    route.expert_indices,
                    route.expert_offsets,
                    stream=stream,
                )
            else:
                modules[0].launch(
                    inputs[0],
                    inputs[1],
                    weights[0],
                    weights[1],
                    outputs[0],
                    tasks,
                    stream=stream,
                )

        if operation in {"OrdinaryForward", "FixedGroupedForward"}:
            return dense_forward
        if operation in {"OrdinaryBackward", "FixedGroupedBackward"}:
            return dense_backward
        if operation == "GroupedForward":
            return grouped_forward
        if operation == "GroupedBackward":
            return grouped_backward
        if operation == "GroupedForwardPair":
            return grouped_forward_pair
        if operation == "GroupedBackwardPair":
            return grouped_backward_pair
        raise ValueError(f"unsupported launch operation {operation!r}")


def output_tensors(case: DeploymentCase, device: torch.device) -> Outputs:
    """Allocate the caller-owned outputs of one deployed case."""

    if case.operation == "GroupedForwardPair":
        first = torch.empty(
            case.rows, case.out_features, device=device, dtype=torch.bfloat16
        )
        return (first, torch.empty_like(first))
    if case.operation == "FixedGroupedForward":
        return (
            torch.empty(
                case.rows, 8, case.out_features, device=device, dtype=torch.bfloat16
            ),
        )
    if case.operation == "FixedGroupedBackward":
        return (
            torch.empty(
                case.rows, 8, case.in_features, device=device, dtype=torch.bfloat16
            ),
        )
    if case.operation in FORWARD_OPERATIONS:
        return (
            torch.empty(
                case.rows, case.out_features, device=device, dtype=torch.bfloat16
            ),
        )
    return (
        torch.empty(case.rows, case.in_features, device=device, dtype=torch.bfloat16),
    )


def activation_workspace(
    case: DeploymentCase,
    prepared: Any,
    stack: contextlib.ExitStack,
) -> torch.Tensor:
    """Quantize the prepared forward input and return its activation workspace."""

    if prepared.input is None:
        raise ValueError("forward input is missing")
    source = prepared.input
    if case.operation.startswith("Fixed"):
        source = source.view(case.rows * 8, case.in_features)
    quantizer = stack.enter_context(quantizer_module(case.operation, case.quant_type)())
    workspace = quantizer.allocate(source)
    quantizer.launch(source, workspace, stream=torch.cuda.current_stream().cuda_stream)
    return workspace


def _forward_pair_tasks(
    problem: GroupedForwardPairProblem,
    spec: GroupedForwardPairKernelSpec,
) -> tuple[int, Any]:
    state = DerivedGroupedForwardPairState.from_problem_spec(problem, spec)
    row_tile = state.kernel_spec.row_task_rows
    if row_tile is None:
        raise ValueError("row-task specification is missing its task row count")
    return row_tile, GroupedForwardPairRowTaskWorkspace


def build_launcher(
    case: DeploymentCase,
    prepared: Any,
    implementation: str,
    *,
    stack: contextlib.ExitStack,
    root: Path | None = None,
    artifact: Path | None = None,
    instance: str | None = None,
    route_entries: int | None = None,
    workspace: torch.Tensor | None = None,
) -> Launcher:
    """Build the case-scoped launch plan for one implementation.

    `route_entries` must cover the widest route bank the caller will launch and
    is required for route-consuming operations. Forward operations must pass the
    quantized activation `workspace`, because callers own its lifetime.
    """

    operation = case.operation
    hip = implementation == "hip"
    if hip and root is None:
        raise ValueError("HIP root is missing")
    if not hip:
        if artifact is None:
            raise ValueError("GGTensile artifact is missing")
        if instance is None:
            raise ValueError("GGTensile launch needs the instance name")
    if operation in FORWARD_OPERATIONS and workspace is None:
        raise ValueError("forward launcher requires an activation workspace")
    entries: int | None = route_entries
    if operation not in ROUTE_FREE_OPERATIONS and entries is None:
        entries = max(int(route.expert_indices.numel()) for route in prepared.routes)
    modules: list[Any] = []
    inputs: tuple[torch.Tensor, ...] = ()
    row_tile: int | None = None
    tasks_factory: Any = RowTaskWorkspace
    # Split-contraction state: a partial workspace, the reduction module and
    # the shape the reduction needs. Only the dense backward branch fills them.
    partials: torch.Tensor | None = None
    reduce_modules: list[Any] = []
    split_slices = 0
    split_k_shape: tuple[int, int, int] | None = None
    problem = case.instance.problem
    spec = case.instance.kernel_spec
    # The routed tensor carries the aggregate row count on axis 0: forward
    # families pass the unquantized activations, backward families the routed
    # gradient. Task banks are sized from it and every launch validates the
    # other tensors against its own derivation.
    if operation in FORWARD_OPERATIONS:
        if prepared.input is None:
            raise ValueError("forward case is missing its input tensor")
        reference = prepared.input
    else:
        reference = prepared.grad_outputs[0]
    device = reference.device

    if operation == "OrdinaryForward":
        assert isinstance(problem, ProblemSize)
        assert isinstance(spec, ForwardKernelSpec)
        modules.append(
            stack.enter_context(
                ForwardModule(
                    problem,
                    case.quant_type,
                    spec,
                    _artifact(artifact),
                    _instance(instance),
                )
                if not hip
                else FixedHipForwardModule(problem, case.quant_type, spec, root)
            )
        )
    elif operation == "OrdinaryBackward":
        assert isinstance(problem, ProblemSize)
        assert isinstance(spec, BackwardKernelSpec)
        inputs = (prepared.grad_outputs[0],)
        reduce_modules = []
        partials = None
        split_slices = 0
        split_k_shape = None
        if hip:
            split_control = select_hip_control(
                "OrdinaryBackward",
                case.quant_type,
                problem.m,
                problem.n,
                problem.k,
            )
            split_config = control_inventory()[split_control.symbol].config
            split_slices = int(getattr(split_config, "split_k", 0))
            if split_slices:
                split_module = stack.enter_context(
                    InstalledDenseBackwardSplitKModule(
                        problem, case.quant_type, spec, root
                    )
                )
                modules.append(split_module)
                partials = split_module.allocate(problem.m, problem.k, device)
                split_k_shape = (problem.m, problem.k, split_slices)
                reduce_modules.append(
                    stack.enter_context(InstalledSplitKReduceModule(root))
                )
        if not split_slices:
            modules.append(
                stack.enter_context(
                    BackwardModule(
                        problem,
                        case.quant_type,
                        spec,
                        _artifact(artifact),
                        _instance(instance),
                    )
                    if not hip
                    else InstalledDenseBackwardModule(
                        problem, case.quant_type, spec, root
                    )
                )
            )
    elif operation == "GroupedForward":
        assert isinstance(problem, GroupedForwardProblem)
        assert isinstance(spec, GroupedForwardKernelSpec)
        if hip:
            control = select_grouped_forward_control(
                case.quant_type,
                case.out_features,
                case.in_features,
                case.rows,
                _entries(entries),
            )
            modules.append(
                stack.enter_context(
                    InstalledGroupedForwardModule(problem, spec, control, root)
                )
            )
        else:
            modules.append(
                stack.enter_context(
                    GroupedForwardModule(
                        problem, spec, _artifact(artifact), _instance(instance)
                    )
                )
            )
    elif operation in {"GroupedBackward", "GroupedBackwardPair"}:
        pair = operation == "GroupedBackwardPair"
        if pair:
            assert isinstance(problem, GroupedBackwardPairProblem)
            assert isinstance(spec, GroupedBackwardPairKernelSpec)
            inputs = (prepared.grad_outputs[0], prepared.grad_outputs[1])
            choice = _backward_choice(case, _entries(entries))
            if hip:
                control = stack.enter_context(build_backward_pair_control(choice, root))
                modules.append(control)
                if is_pair_row_task_body(choice.symbol):
                    row_tile = control.row_tile
            else:
                modules.append(
                    stack.enter_context(
                        GroupedBackwardPairModule(
                            problem, spec, _artifact(artifact), _instance(instance)
                        )
                    )
                )
        else:
            assert isinstance(problem, ProblemSize)
            assert isinstance(spec, GroupedBackwardKernelSpec)
            inputs = (prepared.grad_outputs[0],)
            choice = _backward_choice(case, _entries(entries))
            if hip:
                control = stack.enter_context(build_backward_control(choice, root))
                modules.append(control)
                if is_row_task_body(choice.symbol):
                    row_tile = control.row_tile
            else:
                modules.append(
                    stack.enter_context(
                        GroupedBackwardModule(
                            problem,
                            case.quant_type,
                            spec,
                            _artifact(artifact),
                            _instance(instance),
                        )
                    )
                )
    elif operation == "GroupedForwardPair":
        assert isinstance(problem, GroupedForwardPairProblem)
        assert isinstance(spec, GroupedForwardPairKernelSpec)
        if hip:
            choice = select_routed_control(
                "GroupedForwardPair",
                case.quant_type,
                case.out_features,
                case.in_features,
                case.rows,
                _entries(entries),
            )
            for _ in range(2):
                modules.append(
                    stack.enter_context(build_forward_pair_control(choice, root))
                )
            if is_row_task_body(choice.symbol):
                row_tile, tasks_factory = _forward_pair_tasks(problem, spec)
        else:
            device_row_tasks = (
                spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
            )
            module_type = (
                GroupedForwardPairRowTaskModule
                if device_row_tasks
                else GroupedForwardPairModule
            )
            modules.append(
                stack.enter_context(
                    module_type(problem, spec, _artifact(artifact), _instance(instance))
                )
            )
            if device_row_tasks:
                row_tile, tasks_factory = _forward_pair_tasks(problem, spec)
    elif operation == "FixedGroupedForward":
        assert isinstance(problem, FixedForwardProblem)
        assert isinstance(spec, FixedForwardKernelSpec)
        modules.append(
            stack.enter_context(
                InstalledFixedGroupedQ8ForwardModule(problem, spec, root)
                if hip
                else FixedGroupedQ8ForwardModule(
                    problem, spec, _artifact(artifact), _instance(instance)
                )
            )
        )
    elif operation == "FixedGroupedBackward":
        assert isinstance(problem, FixedBackwardProblem)
        assert isinstance(spec, FixedBackwardKernelSpec)
        inputs = (prepared.grad_outputs[0],)
        modules.append(
            stack.enter_context(
                InstalledFixedGroupedQ8BackwardModule(problem, spec, root)
                if hip
                else FixedGroupedQ8BackwardModule(
                    problem, spec, _artifact(artifact), _instance(instance)
                )
            )
        )
    else:
        raise ValueError(f"unsupported launch operation {operation!r}")

    return Launcher(
        operation=operation,
        implementation=implementation,
        modules=tuple(modules),
        symbols=tuple(
            str(module.symbol) for module in modules if getattr(module, "symbol", None)
        ),
        inputs=inputs,
        weights=tuple(prepared.packed_weights),
        workspace=workspace,
        row_tile=row_tile,
        tasks_factory=tasks_factory,
        reference=reference,
        device=device,
        partials=partials,
        reduce_modules=tuple(reduce_modules),
        split_k_shape=split_k_shape,
    )


def _entries(route_entries: int | None) -> int:
    """Return the route bank width, failing loudly when it is missing."""

    if route_entries is None:
        raise ValueError("route-consuming launch needs a route bank width")
    return route_entries


def _backward_choice(case: DeploymentCase, route_entries: int) -> Any:
    """Return the catalog choice for a single-projection backward family."""

    return select_routed_control(
        case.operation,
        case.quant_type,
        case.out_features,
        case.in_features,
        case.rows,
        route_entries,
    )


def _artifact(artifact: Path | None) -> Path:
    """Return the GGTensile artifact path, failing loudly when it is missing."""

    if artifact is None:
        raise ValueError("GGTensile artifact is missing")
    return artifact


def _instance(instance: str | None) -> str:
    """Return the GGTensile instance name, failing loudly when it is missing."""

    if instance is None:
        raise ValueError("GGTensile launch needs the instance name")
    return instance
