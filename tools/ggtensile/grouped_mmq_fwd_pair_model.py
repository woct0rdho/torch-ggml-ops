"""Strict identities for research-only paired grouped forward paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GroupedPairWeightStaging(str, Enum):
    GridHalfWeightLds = "GridHalfWeightLds"
    ParityGridHalfWeightLds = "ParityGridHalfWeightLds"
    SignedThreeBitHalfWeightLds = "SignedThreeBitHalfWeightLds"


class GroupedPairProjectionInterleave(str, Enum):
    Interleaved = "Interleaved"


@dataclass(frozen=True)
class GroupedPairFixedDecodePolicy:
    pass


@dataclass(frozen=True)
class GroupedPairIQ2XXSDecodePolicy:
    fused_grid_selector: bool


@dataclass(frozen=True)
class GroupedPairQ3DecodePolicy:
    variable_bitfield_extraction: bool


GroupedPairDecodePolicy = (
    GroupedPairFixedDecodePolicy
    | GroupedPairIQ2XXSDecodePolicy
    | GroupedPairQ3DecodePolicy
)


class GroupedPairRouteOwnership(str, Enum):
    SerialRoutes = "SerialRoutes"
    DeviceRowTasks = "DeviceRowTasks"


@dataclass(frozen=True)
class GroupedForwardPairProblem:
    """One exact two-projection aggregate routed problem."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int
    projection_count: int
