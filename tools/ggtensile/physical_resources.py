"""Canonical resource facts for generated kernels."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareResourceCapacity:
    """Fixed gfx1151 capacity used by every family admission check."""

    max_vgprs: int = 256
    max_sgprs: int = 106
    max_lds_bytes: int = 64 * 1024
    require_zero_spills: bool = True


GFX1151_RESOURCE_CAPACITY = HardwareResourceCapacity()


@dataclass(frozen=True)
class PhysicalResourceUsage:
    """Static resources exposed by every generated physical plan."""

    vgprs: int
    sgprs: int
    lds_bytes: int
    private_bytes: int = 0
    vgpr_spills: int = 0
    sgpr_spills: int = 0

    def admit(self, capacity: HardwareResourceCapacity) -> None:
        assert self.vgprs >= 0
        assert self.sgprs >= 0
        assert self.lds_bytes >= 0
        assert self.private_bytes >= 0
        assert self.vgpr_spills >= 0
        assert self.sgpr_spills >= 0
        assert self.vgprs <= capacity.max_vgprs
        assert self.sgprs <= capacity.max_sgprs
        assert self.lds_bytes <= capacity.max_lds_bytes
        if capacity.require_zero_spills:
            assert not (self.private_bytes or self.vgpr_spills or self.sgpr_spills)
