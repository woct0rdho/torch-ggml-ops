"""Physical ownership for the fixed-group Q8_0 forward experiment."""

from dataclasses import dataclass

from .mmq_fwd_physical import (
    SignedInt8SmallMTiledLdsPhysicalPlan,
    derive_forward_physical_plan,
)
from .mmq_fwd_spec import ForwardKernelSpec, ForwardResourceUsage


@dataclass(frozen=True)
class FixedQ8ForwardPhysicalPlan:
    """Fixed-group ABI wrapper around the compatible Q8 small-M plan."""

    ordinary: SignedInt8SmallMTiledLdsPhysicalPlan
    resources: ForwardResourceUsage
    paired_weight_scale_address_vgpr: int | None
    activation_plane_stride_sgpr: int | None
    weight_lane_offset_vgpr: int | None
    weight_payload_lds_address_vgpr: int | None
    weight_scale_lds_address_vgpr: int | None


def fixed_q8_forward_physical_plan(
    kernel_spec: ForwardKernelSpec,
    fixed_address_hoist: str,
) -> FixedQ8ForwardPhysicalPlan:
    """Derive fixed resources from the ordinary Q8 physical authority."""
    ordinary = derive_forward_physical_plan(kernel_spec)
    if not isinstance(ordinary, SignedInt8SmallMTiledLdsPhysicalPlan):
        raise TypeError("fixed Q8 forward requires the small-M LDS physical plan")
    if fixed_address_hoist == "None":
        paired_scale_address = None
        activation_stride_sgpr = None
        weight_lane_offset = None
        weight_payload_address = None
        weight_scale_address = None
    elif fixed_address_hoist in (
        "ReductionLoop",
        "ReductionLoopAndWeightStage",
    ):
        paired_scale_address = ordinary.registers.register_count
        if paired_scale_address >= ordinary.registers.declared_vgprs:
            raise ValueError("fixed Q8 address hoist requires one rounded VGPR slot")
        activation_stride_sgpr = 15
        if fixed_address_hoist == "ReductionLoopAndWeightStage":
            weight_lane_offset = paired_scale_address + 1
            weight_payload_address = paired_scale_address + 2
            weight_scale_address = paired_scale_address + 3
            if weight_scale_address >= ordinary.registers.declared_vgprs:
                raise ValueError(
                    "fixed Q8 weight-stage hoist requires four rounded VGPR slots"
                )
        else:
            weight_lane_offset = None
            weight_payload_address = None
            weight_scale_address = None
    else:
        raise ValueError(f"unsupported fixed Q8 address hoist {fixed_address_hoist!r}")
    return FixedQ8ForwardPhysicalPlan(
        ordinary,
        ordinary.resources,
        paired_scale_address,
        activation_stride_sgpr,
        weight_lane_offset,
        weight_payload_address,
        weight_scale_address,
    )
