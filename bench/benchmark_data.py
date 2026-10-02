"""Materialize packed inputs and optional BF16 baseline weights."""

from dataclasses import dataclass
from pathlib import Path

import gguf
import numpy as np
import torch

from bench.benchmark_routes import (
    fitted_prior_distributions_for_rows,
    make_route_tensors,
)
from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.gguf_dequant_compat import dequantize_logical
from tools.mmq_correctness import PreparedCase, RouteData
from tools.mmq_deployment_cases import DeploymentCase


@dataclass
class BenchmarkInput:
    prepared: PreparedCase
    tensor_shapes: tuple[tuple[int, ...], ...]
    route_metadata: dict[str, object] | None
    case_seed: int
    route_bank: tuple[RouteData, ...] | None = None


@dataclass
class RouteSelection:
    routes: tuple[RouteData, ...]
    index: int = 0

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.routes):
            raise IndexError(f"route vector index {index} is outside the route bank")
        self.index = index

    @property
    def current(self) -> RouteData:
        if not self.routes:
            raise RuntimeError("route selection is empty")
        return self.routes[self.index]


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
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)


def _route_metadata(distributions) -> dict[str, object]:
    def offsets_for(distribution) -> list[int]:
        total = 0
        offsets = []
        for rows in distribution.group_sizes_cpu:
            total += rows
            offsets.append(total)
        return offsets

    first = distributions[0]
    return {
        "vector_count": len(distributions),
        "bank_seed": first.profile.seed,
        "seed_stride": 1,
        "vectors": [
            {
                "profile": distribution.profile.to_mapping(),
                "active_experts": len(distribution.active_expert_indices_cpu),
                "active_expert_indices": list(distribution.active_expert_indices_cpu),
                "group_sizes": list(distribution.group_sizes_cpu),
                "expert_offsets": offsets_for(distribution),
            }
            for distribution in distributions
        ],
        "generated_from_fitted_prior": True,
        "captured_or_synthetic_timing_profiles": False,
    }


def prepare_input(
    case: DeploymentCase,
    reader: gguf.GGUFReader,
    *,
    expert_prior: str | None,
    seed: int,
    route_vectors: int = 1,
    materialize_baseline: bool = False,
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

    # One stable seed per case derives both the input tensors and the route bank,
    # so cases never share a random stream. The two consumers draw from different
    # generators (torch CUDA for inputs, numpy for routes).
    case_seed = (seed ^ int(case.identity[-8:], 16)) & 0xFFFFFFFF

    route = None
    route_bank = None
    route_metadata = None
    if case.operation.startswith("Grouped"):
        if expert_prior is None:
            raise ValueError("grouped benchmark cases require an expert prior")
        distributions = fitted_prior_distributions_for_rows(
            expert_prior, case.rows, route_vectors, seed=case_seed
        )
        route_bank = tuple(
            RouteData(
                *make_route_tensors(distribution),
                # CPU tuples are report provenance: the active support and the
                # complete production group sizes.
                distribution.active_expert_indices_cpu,
                distribution.group_sizes_cpu,
            )
            for distribution in distributions
        )
        route = route_bank[0]
        route_metadata = _route_metadata(distributions)

    logical_weights = []
    if materialize_baseline:
        for packed, tensor_type in zip(packed_weights, tensor_types, strict=True):
            if route is not None:
                logical_shape = (256, case.out_features, case.in_features)
            elif case.operation.startswith("Fixed"):
                logical_shape = (8, case.out_features, case.in_features)
            else:
                logical_shape = (case.out_features, case.in_features)
            logical_weights.append(
                dequantize_logical(packed, tensor_type, logical_shape)
            )

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
        prepared, tuple(tensor_shapes), route_metadata, case_seed, route_bank
    )


def input_mapping(value: BenchmarkInput, model: Path) -> dict[str, object]:
    prepared = value.prepared
    return {
        "model": str(model),
        "tensor_names": list(prepared.case.tensor_source.names),
        "physical_tensor_shapes": [list(shape) for shape in value.tensor_shapes],
        "case_seed": value.case_seed,
        "input_shape": list(prepared.input.shape)
        if prepared.input is not None
        else None,
        "grad_output_shapes": [list(item.shape) for item in prepared.grad_outputs],
        "route": value.route_metadata,
        "baseline_weights_materialized": bool(prepared.logical_weights),
        "input_dtype": str(torch.bfloat16),
        "packed_dtype": str(torch.uint8),
    }
