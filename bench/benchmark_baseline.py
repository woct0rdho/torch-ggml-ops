"""Prepared BF16 baselines for the benchmark kernel surfaces."""

import torch
from aiter.ops.triton.gmm import gmm

from bench.benchmark_common import Implementation
from bench.benchmark_data import RouteSelection
from tools.aiter_gmm_heuristics import gmm_config
from tools.mmq_correctness import PreparedCase, RouteData


def _selected_weights(
    prepared: PreparedCase, routes: tuple[RouteData, ...], forward: bool
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if not prepared.logical_weights:
        raise ValueError("the baseline requires materialized BF16 weights")
    result = []
    for route in routes:
        values = []
        for weight in prepared.logical_weights:
            selected = weight.index_select(0, route.expert_indices).contiguous()
            values.append(selected.transpose(1, 2) if forward else selected)
        result.append(tuple(values))
    return tuple(result)


def _gmm_config(prepared: PreparedCase, prior: str) -> dict[str, int]:
    case = prepared.case
    forward = "Forward" in case.operation
    return gmm_config(
        case.rows,
        case.in_features if forward else case.out_features,
        case.out_features if forward else case.in_features,
        forward,
        prior,
    )


def prepare_baseline(
    prepared: PreparedCase,
    selection: RouteSelection,
    prior: str | None,
    baseline_label: str,
) -> Implementation:
    case = prepared.case
    input_tensor = prepared.input
    if input_tensor is None and case.operation.endswith("Forward"):
        raise ValueError("baseline forward input is missing")
    if case.operation.startswith("Grouped") and prior is None:
        raise ValueError("grouped baseline requires an expert prior")

    if case.operation == "OrdinaryForward":
        assert input_tensor is not None
        rhs = prepared.logical_weights[0].transpose(0, 1)
        output = torch.empty(
            case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
        )

        def launch() -> torch.Tensor:
            return torch.mm(input_tensor, rhs, out=output)

    elif case.operation == "OrdinaryBackward":
        rhs = prepared.logical_weights[0]
        grad_output = prepared.grad_outputs[0]
        output = torch.empty(
            case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
        )

        def launch() -> torch.Tensor:
            return torch.mm(grad_output, rhs, out=output)

    elif case.operation in {"GroupedForward", "GroupedBackward"}:
        forward = case.operation == "GroupedForward"
        assert prior is not None
        routes = selection.routes
        route_weights = _selected_weights(prepared, routes, forward)
        config = _gmm_config(prepared, prior)
        lhs = input_tensor if forward else prepared.grad_outputs[0]
        assert lhs is not None
        output = torch.empty(
            case.rows,
            case.out_features if forward else case.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def launch() -> torch.Tensor:
            route = selection.current
            rhs = route_weights[selection.index][0]
            return gmm(
                lhs,
                rhs,
                route.group_sizes,
                preferred_element_type=lhs.dtype,
                existing_out=output,
                config=config,
            )

    elif case.operation == "GroupedForwardPair":
        assert prior is not None and input_tensor is not None
        lhs: torch.Tensor = input_tensor
        route_weights = _selected_weights(prepared, selection.routes, True)
        config = _gmm_config(prepared, prior)
        outputs = tuple(
            torch.empty(
                case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
            )
            for _ in prepared.logical_weights
        )

        def launch(lhs: torch.Tensor = lhs) -> tuple[torch.Tensor, ...]:
            route = selection.current
            return tuple(
                gmm(
                    lhs,
                    rhs,
                    route.group_sizes,
                    preferred_element_type=lhs.dtype,
                    existing_out=output,
                    config=config,
                )
                for rhs, output in zip(
                    route_weights[selection.index], outputs, strict=True
                )
            )

    elif case.operation == "GroupedBackwardPair":
        assert prior is not None
        route_weights = _selected_weights(prepared, selection.routes, False)
        config = _gmm_config(prepared, prior)
        outputs = tuple(
            torch.empty(
                case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
            )
            for _ in prepared.logical_weights
        )
        result = torch.empty_like(outputs[0])

        def launch() -> torch.Tensor:
            route = selection.current
            values = tuple(
                gmm(
                    grad_output,
                    rhs,
                    route.group_sizes,
                    preferred_element_type=grad_output.dtype,
                    existing_out=output,
                    config=config,
                )
                for grad_output, rhs, output in zip(
                    prepared.grad_outputs,
                    route_weights[selection.index],
                    outputs,
                    strict=True,
                )
            )
            return torch.add(values[0], values[1], out=result)

    elif case.operation.startswith("FixedGrouped"):
        if case.operation == "FixedGroupedForward":
            assert input_tensor is not None
            lhs = input_tensor.permute(1, 0, 2)
            rhs = prepared.logical_weights[0].transpose(1, 2)
            output = torch.empty(
                8, case.rows, case.out_features, device="cuda", dtype=torch.bfloat16
            )
        else:
            lhs = prepared.grad_outputs[0].permute(1, 0, 2)
            rhs = prepared.logical_weights[0]
            output = torch.empty(
                8, case.rows, case.in_features, device="cuda", dtype=torch.bfloat16
            )

        def launch() -> torch.Tensor:
            return torch.bmm(lhs, rhs, out=output)

    else:
        raise ValueError(f"unsupported operation {case.operation}")

    return Implementation(
        "baseline", launch, {"label": baseline_label, "prepared": "BF16"}
    )
