"""Strict identities for isolated grouped MMQ forward kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GroupedWeightStaging(str, Enum):
    GroupedDirectGlobal = "GroupedDirectGlobal"
    GroupedDecodedWeightLds = "GroupedDecodedWeightLds"
    GroupedIQ2SFullWeightLds = "GroupedIQ2SFullWeightLds"


class GroupedActivationStaging(str, Enum):
    AggregateRows = "AggregateRows"
    AggregateRowsTiled = "AggregateRowsTiled"
    AggregateRowsTiledLinear = "AggregateRowsTiledLinear"


@dataclass(frozen=True)
class GroupedDecodedPolicy:
    metadata_conversion: str
    independent_metadata_extraction: bool
    defer_metadata_reads: bool


@dataclass(frozen=True, kw_only=True)
class GroupedQ2DecodePolicy:
    metadata_conversion: str
    unrolled_groups: bool
    hip_association: bool
    partial_lds: bool
    pre_negated_dm: bool
    paired_payload_writes: bool
    paired_metadata_writes: bool
    distributed_producer: bool

    def __post_init__(self) -> None:
        if self.hip_association:
            assert self.unrolled_groups
        if self.partial_lds:
            assert self.hip_association
        if self.pre_negated_dm:
            assert self.partial_lds
        if self.paired_payload_writes:
            assert self.pre_negated_dm
        if self.paired_metadata_writes:
            assert self.paired_payload_writes
        if self.distributed_producer:
            assert self.paired_metadata_writes

    @property
    def independent_metadata_extraction(self) -> bool:
        return False

    @property
    def defer_metadata_reads(self) -> bool:
        return False


@dataclass(frozen=True)
class GroupedIQ2SDecodePolicy:
    metadata_conversion: str
    payload_prefetch: bool

    @property
    def independent_metadata_extraction(self) -> bool:
        return False

    @property
    def defer_metadata_reads(self) -> bool:
        return False


GroupedDecodePolicy = (
    GroupedDecodedPolicy | GroupedQ2DecodePolicy | GroupedIQ2SDecodePolicy
)


@dataclass(frozen=True)
class GroupedForwardProblem:
    """One aggregate routed problem over the fixed packed expert bank."""

    quant_data_type: str
    aggregate_rows: int
    output_features: int
    input_features: int
    physical_experts: int
    max_route_entries: int
