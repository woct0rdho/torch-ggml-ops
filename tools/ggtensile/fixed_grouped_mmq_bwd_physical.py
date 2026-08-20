"""Physical ownership for fixed-group Q8_0 backward."""

from dataclasses import dataclass, replace

from .mmq_bwd_physical import BackwardPhysicalPlan, BackwardResourcePlan


@dataclass(frozen=True)
class FixedBackwardScalarPlan:
    group_byte_offset: int


@dataclass(frozen=True)
class FixedBackwardPhysicalPlan:
    ordinary: BackwardPhysicalPlan
    resources: BackwardResourcePlan
    scalar: FixedBackwardScalarPlan


def derive_fixed_backward_physical_plan(
    ordinary: BackwardPhysicalPlan,
) -> FixedBackwardPhysicalPlan:
    group_byte_offset = ordinary.registers.total_sgprs
    total_sgprs = group_byte_offset + 1
    if total_sgprs > 64:
        raise ValueError("fixed backward group offset exceeds the SGPR file")
    registers = replace(ordinary.registers, total_sgprs=total_sgprs)
    resources = replace(ordinary.resources, total_sgprs=total_sgprs)
    fixed_ordinary = replace(ordinary, registers=registers, resources=resources)
    return FixedBackwardPhysicalPlan(
        ordinary=fixed_ordinary,
        resources=resources,
        scalar=FixedBackwardScalarPlan(group_byte_offset),
    )
