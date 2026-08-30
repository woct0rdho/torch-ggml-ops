"""Shared numerical correctness primitives for MMQ tests and benchmarks."""

import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

from tools.aiter_gmm_compat import gmm_config
from tools.ggtensile.grouped_mmq_bwd_pair_spec import GroupedBackwardPairKernelSpec
from tools.ggtensile.quant_formats import BACKWARD_QUANT_FORMATS
from tools.mmq_deployment_cases import DeploymentCase, model_path

CONTROL_NRMSE_LIMIT = 5e-4
REFERENCE_NRMSE_LIMIT = 0.04
CONTROL_MAX_ABSOLUTE_ERROR = 0.015625

_QUANT_TYPES = {
    "Q2_K": gguf.GGMLQuantizationType.Q2_K,
    "Q3_K": gguf.GGMLQuantizationType.Q3_K,
    "Q4_K": gguf.GGMLQuantizationType.Q4_K,
    "Q5_K": gguf.GGMLQuantizationType.Q5_K,
    "Q6_K": gguf.GGMLQuantizationType.Q6_K,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "IQ2_XXS": gguf.GGMLQuantizationType.IQ2_XXS,
    "IQ2_S": gguf.GGMLQuantizationType.IQ2_S,
}


class CorrectnessPrerequisite(RuntimeError):
    """Raised when a numerical case cannot be materialized in this environment."""


@dataclass(frozen=True)
class RouteData:
    expert_indices: torch.Tensor
    expert_offsets: torch.Tensor
    group_sizes: torch.Tensor
    expert_indices_cpu: tuple[int, ...]
    group_sizes_cpu: tuple[int, ...]


@dataclass
class PreparedCase:
    case: DeploymentCase
    packed_weights: tuple[torch.Tensor, ...]
    logical_weights: tuple[torch.Tensor, ...]
    tensor_types: tuple[gguf.GGMLQuantizationType, ...]
    input: torch.Tensor | None
    grad_outputs: tuple[torch.Tensor, ...]
    route: RouteData | None


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    actual_value = actual.detach().float()
    expected_value = expected.detach().float()
    difference = actual_value - expected_value
    finite = bool(
        torch.isfinite(actual_value).all() and torch.isfinite(expected_value).all()
    )
    reference_rms = float(expected_value.square().mean().sqrt())
    error_rms = float(difference.square().mean().sqrt()) if finite else None
    normalized = None
    if error_rms is not None:
        normalized = (
            0.0
            if reference_rms == 0.0 and error_rms == 0.0
            else (error_rms / reference_rms if reference_rms else float("inf"))
        )
    return {
        "different_bf16_elements": int(
            torch.count_nonzero(actual.detach() != expected.detach())
        ),
        "elements": actual.numel(),
        "finite": finite,
        "max_absolute_error": float(difference.abs().max()) if finite else None,
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_rmse": normalized,
    }


def assert_external_reference(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    maximum: float = REFERENCE_NRMSE_LIMIT,
) -> dict[str, object]:
    metrics = error_metrics(actual, expected)
    normalized = metrics["normalized_rmse"]
    if metrics["finite"] is not True or not isinstance(normalized, float):
        raise AssertionError(f"{label}: non-finite output: {metrics}")
    if normalized > maximum:
        raise AssertionError(
            f"{label}: external-reference NRMSE {normalized:.6e} exceeds "
            f"{maximum:.6e}: {metrics}"
        )
    return metrics


def assert_implementation_equivalence(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    maximum: float = CONTROL_NRMSE_LIMIT,
    maximum_absolute: float | None = None,
) -> dict[str, object]:
    metrics = error_metrics(actual, expected)
    normalized = metrics["normalized_rmse"]
    absolute = metrics["max_absolute_error"]
    if metrics["finite"] is not True or not isinstance(normalized, float):
        raise AssertionError(f"{label}: non-finite output: {metrics}")
    if normalized > maximum:
        raise AssertionError(
            f"{label}: implementation-equivalence NRMSE {normalized:.6e} "
            f"exceeds {maximum:.6e}: {metrics}"
        )
    if maximum_absolute is not None and (
        not isinstance(absolute, float) or absolute > maximum_absolute
    ):
        raise AssertionError(
            f"{label}: maximum absolute error {absolute} exceeds "
            f"{maximum_absolute}: {metrics}"
        )
    return metrics


def assert_repeat(first: torch.Tensor, second: torch.Tensor, label: str) -> None:
    if int(torch.count_nonzero(first != second)):
        raise AssertionError(f"{label}: repeated launch changed the output")


def assert_changed(before: torch.Tensor, after: torch.Tensor, label: str) -> None:
    if not int(torch.count_nonzero(before != after)):
        raise AssertionError(f"{label}: input mutation did not affect the output")


def _packed_shape(case: DeploymentCase) -> tuple[int, ...]:
    # Backward/grouped deployment tables include the IQ2 physical layouts;
    # using the superset here keeps packed-shape derivation independent of the
    # operation family.
    fmt = BACKWARD_QUANT_FORMATS[case.quant_type]
    row_bytes = case.in_features // fmt.block_values * fmt.block_bytes
    if case.operation in {"OrdinaryForward", "OrdinaryBackward"}:
        return case.out_features, row_bytes
    if case.operation.startswith("Fixed"):
        return 8, case.out_features, row_bytes
    return 256, case.out_features, row_bytes


def _data_mode(mode: str | None) -> str:
    selected = mode or os.environ.get("MMQ_NUMERICAL_DATA", "model")
    if selected not in {"model", "synthetic"}:
        raise ValueError("MMQ_NUMERICAL_DATA must be 'model' or 'synthetic'")
    return selected


@cache
def _reader(path: Path) -> gguf.GGUFReader:
    return gguf.GGUFReader(path)


def _find_tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None:
        raise CorrectnessPrerequisite(f"GGUF tensor is missing: {name}")
    return tensor


def _half_bytes(value: float) -> bytes:
    return struct.pack("<e", value)


def _synthetic_block(quant_type: str, variant: int) -> np.ndarray:
    """Build a small valid, nonzero block for each deployed GGUF format."""
    code = 1 + variant
    if quant_type == "Q8_0":
        block = bytearray(34)
        block[:2] = _half_bytes(1.0)
        block[2:] = bytes([code]) * 32
    elif quant_type == "Q2_K":
        block = bytearray(84)
        block[:16] = bytes([code]) * 16
        block[16:80] = bytes([0x55 if code == 1 else 0xAA]) * 64
        block[80:82] = _half_bytes(1.0)
    elif quant_type == "Q3_K":
        block = bytearray(110)
        block[:32] = b"\xff" * 32
        block[32:96] = bytes([0x55 if code == 1 else 0xAA]) * 64
        scales = bytearray(12)
        for group in range(16):
            scale = 32 + code
            scales[group % 8] |= (scale & 0x0F) << (4 * (group // 8))
            scales[8 + group % 4] |= (scale >> 4) << (2 * (group // 4))
        block[96:108] = scales
        block[108:110] = _half_bytes(1.0)
    elif quant_type in {"Q4_K", "Q5_K"}:
        block = bytearray(144 if quant_type == "Q4_K" else 176)
        block[:2] = _half_bytes(1.0)
        block[4:8] = bytes([code]) * 4
        block[12:16] = bytes([code]) * 4
        if quant_type == "Q4_K":
            block[16:] = bytes([0x11 if code == 1 else 0x22]) * 128
        else:
            block[16:48] = b"\x00" * 32
            block[48:] = bytes([0x11 if code == 1 else 0x22]) * 128
    elif quant_type == "Q6_K":
        block = bytearray(210)
        block[:128] = bytes([0x11 if code == 1 else 0x22]) * 128
        block[128:192] = b"\xaa" * 64
        block[192:208] = bytes([code]) * 16
        block[208:210] = _half_bytes(1.0)
    elif quant_type == "IQ2_XXS":
        block = bytearray(66)
        block[:2] = _half_bytes(1.0)
        block[2:34] = bytes([code]) * 32
    elif quant_type == "IQ2_S":
        block = bytearray(82)
        block[:2] = _half_bytes(1.0)
        block[2:34] = bytes([code]) * 32
        block[66:74] = bytes([0x00 if code == 1 else 0x11]) * 8
    else:
        raise ValueError(f"unsupported synthetic quantization type: {quant_type}")
    return np.frombuffer(bytes(block), dtype=np.uint8)


def _synthetic_packed(case: DeploymentCase, name: str) -> np.ndarray:
    shape = _packed_shape(case)
    fmt = BACKWARD_QUANT_FORMATS[case.quant_type]
    row_bytes = shape[-1]
    blocks_per_row = row_bytes // fmt.block_bytes
    salt = (int(case.identity[-8:], 16) + sum(map(ord, name))) & 1
    patterns = tuple(
        np.tile(_synthetic_block(case.quant_type, salt ^ variant), blocks_per_row)
        for variant in (0, 1)
    )
    rows = np.empty((int(np.prod(shape[:-1])), row_bytes), dtype=np.uint8)
    rows[0::2] = patterns[0]
    rows[1::2] = patterns[1]
    return rows.reshape(shape)


def load_case_weights(
    case: DeploymentCase,
    *,
    device: torch.device,
    mode: str | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[gguf.GGMLQuantizationType, ...]]:
    selected_mode = _data_mode(mode)
    expected_shape = _packed_shape(case)
    arrays: list[np.ndarray] = []
    types: list[gguf.GGMLQuantizationType] = []
    if selected_mode == "model":
        path = model_path(case.tensor_source)
        if not path.is_file():
            raise CorrectnessPrerequisite(f"GGUF model is unavailable: {path}")
        reader = _reader(path)
        for name in case.tensor_source.names:
            tensor = _find_tensor(reader, name)
            if tensor.tensor_type.name != case.quant_type:
                raise CorrectnessPrerequisite(
                    f"{name} has {tensor.tensor_type.name}, expected {case.quant_type}"
                )
            data = np.array(tensor.data, dtype=np.uint8, copy=True, order="C")
            if data.size != int(np.prod(expected_shape)):
                raise CorrectnessPrerequisite(
                    f"{name} physical shape {data.shape} does not match "
                    f"{expected_shape} for {case.identity}"
                )
            arrays.append(data.reshape(expected_shape))
            types.append(tensor.tensor_type)
    else:
        for name in case.tensor_source.names:
            arrays.append(_synthetic_packed(case, name))
            types.append(_QUANT_TYPES[case.quant_type])
    packed = tuple(
        torch.from_numpy(array).to(device=device).contiguous() for array in arrays
    )
    return packed, tuple(types)


def _route(case: DeploymentCase, device: torch.device) -> RouteData | None:
    if not case.operation.startswith("Grouped"):
        return None
    entries = 8 if case.tensor_source.model_family == "qwen" else 6
    # The installed paired-backward Q3 control dispatches M64 only when the
    # aggregate rows are below 128 rows per active route. Keep the neutral
    # route large enough to exercise that exact selected control.
    if case.operation == "GroupedBackwardPair":
        spec = case.instance.kernel_spec
        assert isinstance(spec, GroupedBackwardPairKernelSpec)
        macro_tile = spec.compute.geometry.macro_tile0
        if macro_tile == 64:
            entries = min(256, case.rows // 128 + 1)
    preferred = (0, 2, 5, 7, 11, 13, 17, 19)
    experts = tuple(preferred[:entries]) + tuple(
        expert for expert in range(256) if expert not in preferred
    )
    experts = experts[:entries]
    base, remainder = divmod(case.rows, entries)
    sizes = tuple(base + (index < remainder) for index in range(entries))
    indices = torch.tensor(experts, device=device, dtype=torch.int64).contiguous()
    group_sizes = torch.tensor(sizes, device=device, dtype=torch.int32).contiguous()
    offsets = group_sizes.cumsum(0).to(dtype=torch.int32).contiguous()
    return RouteData(indices, offsets, group_sizes, experts, sizes)


def _random_bf16(shape: Sequence[int], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        *shape, device=device, dtype=torch.bfloat16, generator=generator
    ).contiguous()


def prepare_case(
    case: DeploymentCase,
    *,
    device: torch.device,
    mode: str | None = None,
) -> PreparedCase:
    packed, types = load_case_weights(case, device=device, mode=mode)
    route = _route(case, device)
    grouped = case.operation.startswith("Grouped")
    if grouped:
        assert route is not None
        logical_shape = (
            len(route.expert_indices_cpu),
            case.out_features,
            case.in_features,
        )
    elif case.operation.startswith("Fixed"):
        logical_shape = (8, case.out_features, case.in_features)
    else:
        logical_shape = (case.out_features, case.in_features)

    # Routed references only consume active banks. Dequantizing all 256 experts
    # needlessly pins several GiB of temporary and cached TTM allocations.
    logical = tuple(
        dequantize_gguf_tensor(
            weight.index_select(0, route.expert_indices)
            if grouped and route is not None
            else weight,
            tensor_type,
            dtype=torch.bfloat16,
            device=device,
        )
        .reshape(logical_shape)
        .contiguous()
        for weight, tensor_type in zip(packed, types, strict=True)
    )
    seed = (int(case.identity[-8:], 16) ^ 0x4D4D5100) & 0xFFFFFFFF
    if case.operation in {
        "OrdinaryForward",
        "GroupedForward",
        "GroupedForwardPair",
        "FixedGroupedForward",
    }:
        shape = (
            (case.rows, case.in_features)
            if case.operation != "FixedGroupedForward"
            else (case.rows, 8, case.in_features)
        )
        input_tensor = _random_bf16(shape, seed, device)
    else:
        input_tensor = None
    if case.operation == "OrdinaryBackward":
        grad_shapes = ((case.rows, case.out_features),)
    elif case.operation in {"GroupedBackward", "GroupedBackwardPair"}:
        grad_shapes = tuple((case.rows, case.out_features) for _ in packed)
    elif case.operation == "FixedGroupedBackward":
        grad_shapes = ((case.rows, 8, case.out_features),)
    else:
        grad_shapes = ()
    grads = tuple(
        _random_bf16(shape, seed + index + 1, device)
        for index, shape in enumerate(grad_shapes)
    )
    return PreparedCase(case, packed, logical, types, input_tensor, grads, route)


def _gmm(
    lhs: torch.Tensor, rhs: torch.Tensor, groups: torch.Tensor, config: dict[str, int]
) -> torch.Tensor:
    from aiter.ops.triton.gmm import gmm

    return gmm(
        lhs,
        rhs,
        groups,
        preferred_element_type=lhs.dtype,
        config=config,
    ).contiguous()


def dequantize_active_routed_weight(
    packed_weight: torch.Tensor,
    tensor_type: gguf.GGMLQuantizationType,
    expert_indices: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize only the expert banks consumed by a routed reference."""
    selected = packed_weight.index_select(0, expert_indices)
    return (
        dequantize_gguf_tensor(
            selected,
            tensor_type,
            dtype=torch.bfloat16,
            device=packed_weight.device,
        )
        .reshape(expert_indices.numel(), out_features, in_features)
        .contiguous()
    )


def bf16_forward_reference(
    input_tensor: torch.Tensor, logical_weight: torch.Tensor
) -> torch.Tensor:
    return torch.mm(input_tensor, logical_weight.transpose(0, 1)).contiguous()


def bf16_backward_reference(
    grad_output: torch.Tensor, logical_weight: torch.Tensor
) -> torch.Tensor:
    return torch.mm(grad_output, logical_weight).contiguous()


def _active_routed_weight(
    logical_weight: torch.Tensor, expert_indices: torch.Tensor
) -> torch.Tensor:
    # Callers may provide all physical experts or the already-selected active
    # banks. Both forms occur in focused tests and deployment preparation.
    if (
        logical_weight.shape[0] != 256
        and logical_weight.shape[0] == expert_indices.numel()
    ):
        return logical_weight
    return logical_weight.index_select(0, expert_indices)


def routed_forward_reference(
    input_tensor: torch.Tensor,
    logical_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    quant_type: str | None = None,
) -> torch.Tensor:
    selected = _active_routed_weight(logical_weight, expert_indices).transpose(1, 2)
    config = gmm_config(
        input_tensor.shape[0],
        input_tensor.shape[1],
        selected.shape[-1],
        selected.stride(1) == 1,
        quant_type=quant_type,
    )
    return _gmm(input_tensor, selected, group_sizes, config)


def routed_backward_reference(
    grad_output: torch.Tensor,
    logical_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    quant_type: str | None = None,
) -> torch.Tensor:
    selected = _active_routed_weight(logical_weight, expert_indices).contiguous()
    config = gmm_config(
        grad_output.shape[0],
        grad_output.shape[1],
        selected.shape[-1],
        selected.stride(1) == 1,
        quant_type=quant_type,
    )
    return _gmm(grad_output, selected, group_sizes, config)


def routed_backward_pair_reference(
    first_grad_output: torch.Tensor,
    second_grad_output: torch.Tensor,
    first_logical_weight: torch.Tensor,
    second_logical_weight: torch.Tensor,
    expert_indices: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    quant_type: str | None = None,
) -> torch.Tensor:
    first = routed_backward_reference(
        first_grad_output,
        first_logical_weight,
        expert_indices,
        group_sizes,
        quant_type=quant_type,
    )
    second = routed_backward_reference(
        second_grad_output,
        second_logical_weight,
        expert_indices,
        group_sizes,
        quant_type=quant_type,
    )
    return torch.add(first, second).contiguous()


def fixed_forward_reference(
    input_tensor: torch.Tensor, logical_weight: torch.Tensor
) -> torch.Tensor:
    return (
        torch.bmm(input_tensor.permute(1, 0, 2), logical_weight.transpose(1, 2))
        .permute(1, 0, 2)
        .contiguous()
    )


def fixed_backward_reference(
    grad_output: torch.Tensor, logical_weight: torch.Tensor
) -> torch.Tensor:
    return (
        torch.bmm(grad_output.permute(1, 0, 2), logical_weight)
        .permute(1, 0, 2)
        .contiguous()
    )


def _routed_forward_reference(
    case: DeploymentCase,
    input_tensor: torch.Tensor,
    logical: torch.Tensor,
    route: RouteData,
) -> torch.Tensor:
    return routed_forward_reference(
        input_tensor,
        logical,
        route.expert_indices,
        route.group_sizes,
        quant_type=case.quant_type,
    )


def _routed_backward_reference(
    case: DeploymentCase,
    grad_outputs: tuple[torch.Tensor, ...],
    logical: tuple[torch.Tensor, ...],
    route: RouteData,
) -> torch.Tensor:
    if len(logical) == 1:
        return routed_backward_reference(
            grad_outputs[0],
            logical[0],
            route.expert_indices,
            route.group_sizes,
            quant_type=case.quant_type,
        )
    return routed_backward_pair_reference(
        grad_outputs[0],
        grad_outputs[1],
        logical[0],
        logical[1],
        route.expert_indices,
        route.group_sizes,
        quant_type=case.quant_type,
    )


def external_reference(
    prepared: PreparedCase,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    case = prepared.case
    if case.operation == "OrdinaryForward":
        assert prepared.input is not None
        return bf16_forward_reference(prepared.input, prepared.logical_weights[0])
    if case.operation == "OrdinaryBackward":
        return bf16_backward_reference(
            prepared.grad_outputs[0], prepared.logical_weights[0]
        )
    if case.operation in {"GroupedForward", "GroupedForwardPair"}:
        assert prepared.input is not None and prepared.route is not None
        values = tuple(
            _routed_forward_reference(case, prepared.input, weight, prepared.route)
            for weight in prepared.logical_weights
        )
        return values[0] if len(values) == 1 else values
    if case.operation == "GroupedBackward" or case.operation == "GroupedBackwardPair":
        assert prepared.route is not None
        return _routed_backward_reference(
            case, prepared.grad_outputs, prepared.logical_weights, prepared.route
        )
    logical = prepared.logical_weights[0]
    if case.operation == "FixedGroupedForward":
        assert prepared.input is not None
        return fixed_forward_reference(prepared.input, logical)
    return fixed_backward_reference(prepared.grad_outputs[0], logical)
