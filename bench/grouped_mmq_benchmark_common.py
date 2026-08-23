import argparse
from dataclasses import dataclass
from pathlib import Path

import gguf
import torch
from benchmark_routes import (
    RouteDistribution,
    distribution_summary,
)
from mmq_benchmark_common import (
    load_packed_tensor,
    make_benchmark_parser,
    select_family_cases,
    validate_benchmark_args,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor
from workload_prior import EXPERT_PRIOR_NAMES

from tools.mmq_correctness import dequantize_active_routed_weight

QUANT_BLOCK_GEOMETRY = {
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "IQ2_XXS": (256, 66),
    "IQ2_S": (256, 82),
}


@dataclass(frozen=True)
class GroupedMMQCase:
    name: str
    kind: str
    tensor_names: tuple[str, ...]
    out_features: int
    in_features: int
    model_calls: int
    description: str
    quant_type: str
    priority: str = "primary"
    top_k: int = 8
    fixed_groups: int = 0

    @property
    def projections(self) -> int:
        return self.fixed_groups or len(self.tensor_names)

    @property
    def routed(self) -> bool:
        return self.kind in {"single", "pair"}

    @property
    def packed_row_bytes(self) -> int:
        block_values, block_bytes = QUANT_BLOCK_GEOMETRY[self.quant_type]
        if self.in_features % block_values != 0:
            raise ValueError(
                f"{self.name} K={self.in_features} is not divisible by "
                f"the {self.quant_type} block size {block_values}"
            )
        return self.in_features // block_values * block_bytes


GROUPED_CASES_BY_MODEL_FAMILY = {
    "qwen": (
        GroupedMMQCase(
            "gate_up_q3_k",
            "pair",
            ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
            512,
            2048,
            20,
            "paired routed gate/up for layers 0-9 and 30-39",
            "Q3_K",
        ),
        GroupedMMQCase(
            "gate_up_iq2_s",
            "pair",
            ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight"),
            512,
            2048,
            20,
            "paired routed gate/up for layers 10-29",
            "IQ2_S",
        ),
        GroupedMMQCase(
            "down_iq2_s",
            "single",
            ("blk.10.ffn_down_exps.weight",),
            2048,
            512,
            20,
            "routed down projection for layers 10-29",
            "IQ2_S",
        ),
        GroupedMMQCase(
            "down_q4_k",
            "single",
            ("blk.2.ffn_down_exps.weight",),
            2048,
            512,
            18,
            "routed down projection for layers 2-9 and 30-39",
            "Q4_K",
        ),
        GroupedMMQCase(
            "down_q5_k",
            "single",
            ("blk.0.ffn_down_exps.weight",),
            2048,
            512,
            2,
            "routed down projection for layers 0-1",
            "Q5_K",
            priority="secondary",
        ),
    ),
    "deepseek": (
        GroupedMMQCase(
            "ds4_output_a_q8_0",
            "fixed",
            ("blk.0.attn_output_a.weight",),
            1024,
            4096,
            43,
            "eight fixed attention output-A groups",
            "Q8_0",
            top_k=1,
            fixed_groups=8,
        ),
        GroupedMMQCase(
            "ds4_gate_up_iq2_xxs",
            "pair",
            ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
            2048,
            4096,
            43,
            "paired top-six routed gate/up for all expert layers",
            "IQ2_XXS",
            top_k=6,
        ),
        GroupedMMQCase(
            "ds4_down_q2_k",
            "single",
            ("blk.0.ffn_down_exps.weight",),
            4096,
            2048,
            43,
            "top-six routed down projection for all expert layers",
            "Q2_K",
            top_k=6,
        ),
    ),
}


def parse_grouped_benchmark_args(
    description: str,
    default_output: Path,
    seed: int,
    *,
    transient_control_help: str | None = None,
) -> argparse.Namespace:
    parser = make_benchmark_parser(
        description,
        default_output=default_output,
        seed=seed,
        repeats=9,
    )
    parser.add_argument(
        "--expert-prior",
        choices=EXPERT_PRIOR_NAMES,
        required=True,
        help="the sole fitted law used for every routed performance case",
    )
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="benchmark only cases marked primary",
    )
    if transient_control_help is not None:
        parser.add_argument(
            "--transient-bf16-control",
            action="store_true",
            help=transient_control_help,
        )
    args = parser.parse_args()
    validate_benchmark_args(parser, args)
    if args.correctness_rows <= 0:
        parser.error("--correctness-rows must be positive")
    if not args.expert_prior.startswith(f"{args.model_family}-"):
        parser.error("--expert-prior must match --model-family")
    return args


@dataclass(frozen=True)
class RoutedWeights:
    packed: tuple[torch.Tensor, ...]
    logical: tuple[torch.Tensor, ...]
    physical_shapes: tuple[tuple[int, ...], ...]
    quant_type: int
    quant_name: str


@dataclass(frozen=True)
class FixedWeight:
    packed: torch.Tensor
    logical: torch.Tensor
    quant_type: int
    quant_name: str


def validate_grouped_quant_type(
    case: GroupedMMQCase,
    tensors: tuple[gguf.ReaderTensor, ...],
) -> tuple[int, str]:
    quant_types = {int(tensor.tensor_type) for tensor in tensors}
    if len(quant_types) != 1:
        raise RuntimeError(f"{case.name} paired tensors have different quant types")
    quant_type = quant_types.pop()
    quant_name = tensors[0].tensor_type.name
    if quant_name != case.quant_type:
        raise RuntimeError(
            f"{case.name} has quant type {quant_name}, expected {case.quant_type}"
        )
    return quant_type, quant_name


def load_routed_weights(
    case: GroupedMMQCase,
    tensors: tuple[gguf.ReaderTensor, ...],
) -> RoutedWeights:
    quant_type, quant_name = validate_grouped_quant_type(case, tensors)
    packed_weights = tuple(load_packed_tensor(tensor) for tensor in tensors)
    physical_shapes = []
    expected_shape = (case.out_features, case.in_features)
    for tensor, packed in zip(tensors, packed_weights, strict=True):
        logical_shape = tuple(int(value) for value in reversed(tensor.shape[:-1]))
        if logical_shape != expected_shape:
            raise RuntimeError(
                f"{tensor.name} has logical per-expert shape {logical_shape}, "
                f"expected {expected_shape}"
            )
        if packed.shape[0] != 256:
            raise RuntimeError(
                f"{tensor.name} has {packed.shape[0]} experts, expected 256"
            )
        if packed.shape[2] != case.packed_row_bytes:
            raise RuntimeError(
                f"{tensor.name} has {packed.shape[2]} packed bytes per row, "
                f"expected {case.packed_row_bytes} for {case.quant_type}"
            )
        physical_shapes.append(tuple(packed.shape))

    # Routed logical banks are materialized per profile, after the active
    # expert set is known. Keeping all 256 banks here causes large TTM peaks.
    return RoutedWeights(
        packed_weights,
        (),
        tuple(physical_shapes),
        quant_type,
        quant_name,
    )


def load_active_routed_logical_weights(
    case: GroupedMMQCase,
    weights: RoutedWeights,
    expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Materialize only the routed banks consumed by one benchmark profile."""
    tensor_type = gguf.GGMLQuantizationType(weights.quant_type)
    return tuple(
        dequantize_active_routed_weight(
            packed,
            tensor_type,
            expert_indices,
            case.out_features,
            case.in_features,
        )
        for packed in weights.packed
    )


def load_fixed_weight(
    case: GroupedMMQCase,
    tensor: gguf.ReaderTensor,
) -> FixedWeight:
    quant_type, quant_name = validate_grouped_quant_type(case, (tensor,))
    flat_packed = load_packed_tensor(tensor)
    expected_packed_shape = (
        case.fixed_groups * case.out_features,
        case.packed_row_bytes,
    )
    if tuple(flat_packed.shape) != expected_packed_shape:
        raise RuntimeError(
            f"{tensor.name} has packed shape {tuple(flat_packed.shape)}, "
            f"expected {expected_packed_shape}"
        )
    expected_logical_shape = (
        case.fixed_groups * case.out_features,
        case.in_features,
    )
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    if logical_shape != expected_logical_shape:
        raise RuntimeError(
            f"{tensor.name} has logical shape {logical_shape}, "
            f"expected {expected_logical_shape}"
        )

    packed_weight = flat_packed.view(
        case.fixed_groups,
        case.out_features,
        flat_packed.shape[-1],
    )
    logical_weight = dequantize_gguf_tensor(
        packed_weight,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(case.fixed_groups, case.out_features, case.in_features)
    return FixedWeight(
        packed_weight,
        logical_weight,
        quant_type,
        quant_name,
    )


def bf16_fixed_mmq_reference(
    input: torch.Tensor,
    logical_weight: torch.Tensor,
) -> torch.Tensor:
    group_major = torch.bmm(
        input.permute(1, 0, 2),
        logical_weight.transpose(1, 2),
    )
    return group_major.permute(1, 0, 2).contiguous()


def bf16_fixed_grad_input_reference(
    grad_output: torch.Tensor,
    logical_weight: torch.Tensor,
) -> torch.Tensor:
    group_major = torch.bmm(grad_output.permute(1, 0, 2), logical_weight)
    return group_major.permute(1, 0, 2).contiguous()


def select_cases(
    case_names: str,
    primary_only: bool,
    model_family: str,
) -> tuple[GroupedMMQCase, ...]:
    return select_family_cases(
        case_names,
        primary_only,
        model_family,
        GROUPED_CASES_BY_MODEL_FAMILY,
        description="grouped benchmark",
    )


def grouped_result_metadata(
    case: GroupedMMQCase,
    batch: int,
    rows: int,
    quant_name: str,
    quant_type: int,
    distribution: RouteDistribution,
    physical_weight_shapes: tuple[tuple[int, ...], ...],
    *,
    top_k: int | None = None,
) -> dict[str, object]:
    result = {
        "case": case.name,
        "kind": case.kind,
        "description": case.description,
        "priority": case.priority,
        "batch": batch,
        "rows": rows,
        "out_features": case.out_features,
        "in_features": case.in_features,
        "projections": case.projections,
        "quant_type": quant_name,
        "quant_type_id": quant_type,
        "distribution": distribution.name,
        "group_summary": distribution_summary(distribution),
        "expert_indices": list(distribution.expert_indices_cpu),
        "group_sizes": list(distribution.group_sizes_cpu),
        "physical_weight_shapes": [list(shape) for shape in physical_weight_shapes],
        "model_calls": case.model_calls,
    }
    if top_k is not None:
        result["top_k"] = top_k
    if distribution.profile is not None:
        result["expert_prior"] = distribution.profile.prior.value
        result["expert_prior_profile"] = distribution.profile.to_mapping()
    return result
