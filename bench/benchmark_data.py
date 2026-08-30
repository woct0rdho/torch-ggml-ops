"""Materialize exact deployment cases for both benchmark surfaces."""

from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import torch
from benchmark_routes import (
    fitted_prior_distributions_for_rows,
    make_route_tensors,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.mmq_correctness import PreparedCase, RouteData
from tools.mmq_deployment_cases import DeploymentCase


@dataclass
class BenchmarkInput:
    prepared: PreparedCase
    tensor_shapes: tuple[tuple[int, ...], ...]
    route_metadata: dict[str, object] | None
    route_bank: tuple[RouteData, ...] | None = None


def _packed_shape(case: DeploymentCase) -> tuple[int, ...]:
    fmt = BACKWARD_QUANT_FORMATS[case.quant_type]
    row_bytes = case.in_features // fmt.block_values * fmt.block_bytes
    if case.operation.startswith("Ordinary"):
        return case.out_features, row_bytes
    if case.operation.startswith("Fixed"):
        return 8, case.out_features, row_bytes
    return 256, case.out_features, row_bytes


def _find_tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    return tensor


def _random_bf16(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()


def prepare_input(
    case: DeploymentCase,
    reader: gguf.GGUFReader,
    *,
    expert_prior: str | None,
    seed: int,
    route_vectors: int = 1,
    route_seed: int | None = None,
    full_logical_weights: bool = False,
) -> BenchmarkInput:
    expected_shape = _packed_shape(case)
    packed_weights = []
    tensor_types = []
    tensor_shapes = []
    for name in case.tensor_source.names:
        tensor = _find_tensor(reader, name)
        if tensor.tensor_type.name != case.quant_type:
            raise ValueError(
                f"{name} has {tensor.tensor_type.name}, expected {case.quant_type}"
            )
        host = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
        physical_shape = tuple(int(value) for value in host.shape)
        allowed_shapes = {expected_shape}
        if case.operation.startswith("Fixed"):
            allowed_shapes.add(
                (expected_shape[0] * expected_shape[1], expected_shape[2])
            )
        if physical_shape not in allowed_shapes:
            raise ValueError(
                f"{name} physical shape {physical_shape} is not one of "
                f"{sorted(allowed_shapes)}"
            )
        packed_weights.append(
            torch.from_numpy(host.reshape(expected_shape)).to("cuda").contiguous()
        )
        tensor_types.append(tensor.tensor_type)
        tensor_shapes.append(tuple(int(value) for value in tensor.data.shape))

    route = None
    route_metadata = None
    route_bank = None
    if case.operation.startswith("Grouped"):
        if expert_prior is None:
            raise ValueError("routed benchmark cases require an expert prior")
        distributions = fitted_prior_distributions_for_rows(
            expert_prior,
            case.rows,
            route_vectors,
            seed=route_seed,
        )
        route_bank = tuple(
            RouteData(
                *make_route_tensors(distribution),
                distribution.expert_indices_cpu,
                distribution.group_sizes_cpu,
            )
            for distribution in distributions
        )
        route = route_bank[0]

        def offsets_for(distribution) -> list[int]:
            total = 0
            offsets = []
            for rows in distribution.group_sizes_cpu:
                total += rows
                offsets.append(total)
            return offsets

        first_distribution = distributions[0]
        route_metadata = {
            "vector_count": len(distributions),
            "bank_seed": first_distribution.profile.seed,
            "seed_stride": 1,
            "profile": first_distribution.profile.to_mapping(),
            "expert_indices": list(first_distribution.expert_indices_cpu),
            "expert_offsets": offsets_for(first_distribution),
            "group_sizes": list(first_distribution.group_sizes_cpu),
            "vectors": [
                {
                    "profile": distribution.profile.to_mapping(),
                    "expert_indices": list(distribution.expert_indices_cpu),
                    "expert_offsets": offsets_for(distribution),
                    "group_sizes": list(distribution.group_sizes_cpu),
                }
                for distribution in distributions
            ],
            "generated_from_fitted_prior": True,
            "captured_or_synthetic_timing_profiles": False,
        }

    logical_weights = []
    for packed, tensor_type in zip(packed_weights, tensor_types, strict=True):
        active_route = route is not None and not full_logical_weights
        source = (
            packed.index_select(0, route.expert_indices) if active_route else packed
        )
        if active_route:
            logical_shape = (
                route.expert_indices.numel(),
                case.out_features,
                case.in_features,
            )
        elif route:
            logical_shape = (256, case.out_features, case.in_features)
        elif case.operation.startswith("Fixed"):
            logical_shape = (8, case.out_features, case.in_features)
        else:
            logical_shape = (case.out_features, case.in_features)
        logical_weights.append(
            dequantize_gguf_tensor(
                source,
                tensor_type,
                dtype=torch.bfloat16,
                device="cuda",
            )
            .reshape(logical_shape)
            .contiguous()
        )

    case_seed = (seed ^ int(case.identity[-8:], 16)) & 0xFFFFFFFF
    input_shape = (
        (case.rows, 8, case.in_features)
        if case.operation.startswith("Fixed")
        else (case.rows, case.in_features)
    )
    input_tensor = _random_bf16(input_shape, case_seed)
    if case.operation == "OrdinaryBackward":
        grad_shapes = ((case.rows, case.out_features),)
    elif case.operation in {"GroupedBackward", "GroupedBackwardPair"}:
        grad_shapes = tuple((case.rows, case.out_features) for _ in packed_weights)
    elif case.operation == "FixedGroupedBackward":
        grad_shapes = ((case.rows, 8, case.out_features),)
    else:
        grad_shapes = ()
    grad_outputs = tuple(
        _random_bf16(shape, case_seed + index + 1)
        for index, shape in enumerate(grad_shapes)
    )
    prepared = PreparedCase(
        case,
        tuple(packed_weights),
        tuple(logical_weights),
        tuple(tensor_types),
        input_tensor,
        grad_outputs,
        route,
    )
    return BenchmarkInput(
        prepared,
        tuple(tensor_shapes),
        route_metadata,
        route_bank,
    )


def input_mapping(value: BenchmarkInput, model: Path) -> dict[str, object]:
    prepared = value.prepared
    return {
        "model": str(model),
        "tensor_names": list(prepared.case.tensor_source.names),
        "physical_tensor_shapes": [list(shape) for shape in value.tensor_shapes],
        "input_shape": list(prepared.input.shape)
        if prepared.input is not None
        else None,
        "grad_output_shapes": [list(item.shape) for item in prepared.grad_outputs],
        "route": value.route_metadata,
        "input_dtype": str(torch.bfloat16),
        "packed_dtype": str(torch.uint8),
    }
