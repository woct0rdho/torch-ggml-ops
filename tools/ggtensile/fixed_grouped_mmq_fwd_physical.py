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

    @property
    def layout(self):
        return self.ordinary.layout

    @property
    def registers(self):
        return self.ordinary.registers

    @property
    def policy(self):
        return self.ordinary.policy


def fixed_q8_forward_physical_plan(
    kernel_spec: ForwardKernelSpec,
) -> FixedQ8ForwardPhysicalPlan:
    """Derive fixed resources from the ordinary Q8 physical authority."""
    ordinary = derive_forward_physical_plan(kernel_spec)
    if not isinstance(ordinary, SignedInt8SmallMTiledLdsPhysicalPlan):
        raise TypeError("fixed Q8 forward requires the small-M LDS physical plan")
    return FixedQ8ForwardPhysicalPlan(ordinary, ordinary.resources)
