"""Strict identities for research-only paired grouped backward kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .mmq_bwd_spec import BackwardKernelSpec


class PairCodebookStorage(str, Enum):
    Lds = "Lds"
    Global = "Global"


class PairPointerMode(str, Enum):
    Swapped = "Swapped"
    Direct = "Direct"


class PairReadConcurrency(str, Enum):
    Serial = "Serial"
    OverlapSecond = "OverlapSecond"
    Concurrent = "Concurrent"


@dataclass(frozen=True)
class GroupedBackwardPairProjectionPolicy:
    dual_lds: bool
    codebook_storage: PairCodebookStorage
    split_full_tiles: bool
    pointer_mode: PairPointerMode
    read_concurrency: PairReadConcurrency
    activation_prefetch: bool
    pipeline_k: bool

    @property
    def concurrent_reads(self) -> bool:
        return self.read_concurrency is PairReadConcurrency.Concurrent

    @property
    def direct_second_pointers(self) -> bool:
        return self.pointer_mode is PairPointerMode.Direct

    @property
    def uses_global_codebook(self) -> bool:
        return self.codebook_storage is PairCodebookStorage.Global

    @property
    def overlaps_second_read(self) -> bool:
        return self.read_concurrency is PairReadConcurrency.OverlapSecond

    def validate(self, quant_type: str, compute: BackwardKernelSpec) -> None:
        needs_second_lds = (
            self.pointer_mode is PairPointerMode.Direct
            or self.read_concurrency is not PairReadConcurrency.Serial
            or self.pipeline_k
        )
        if needs_second_lds:
            assert self.dual_lds
        if not self.dual_lds:
            assert not self.activation_prefetch
            assert self.codebook_storage is PairCodebookStorage.Lds
        assert (
            self.activation_prefetch == compute.pipeline.iteration.prefetch_activation
        )
        if self.overlaps_second_read:
            assert self.pointer_mode is PairPointerMode.Direct
            assert self.activation_prefetch
        if self.pipeline_k:
            assert self.dual_lds
            assert self.pointer_mode is PairPointerMode.Direct
            assert self.read_concurrency is PairReadConcurrency.Concurrent
            assert self.activation_prefetch
            assert compute.pipeline.global_read_prefetch > 1
            assert compute.pipeline.prefetch_next_packed_weight
        if self.codebook_storage is PairCodebookStorage.Global:
            assert quant_type == "IQ2_S"


@dataclass(frozen=True)
class GroupedBackwardPairRouteOwnership:
    kind: str
    split_factor: int

    @classmethod
    def serial(cls) -> GroupedBackwardPairRouteOwnership:
        return cls("Serial", 1)

    @classmethod
    def packed_split(cls, factor: int) -> GroupedBackwardPairRouteOwnership:
        assert factor > 1 and not factor & (factor - 1)
        return cls("PackedSplit", factor)

    def validate(self, macro_tile0: int) -> None:
        assert self.split_factor > 0
        assert not (self.split_factor & (self.split_factor - 1))
        assert (self.kind == "Serial") == (self.split_factor == 1)
        assert self.kind in {"Serial", "PackedSplit"}
        assert self.split_factor * macro_tile0 <= 0xFFFFFFFF


@dataclass(frozen=True)
class GroupedBackwardPairProblem:
    """One exact two-projection routed backward sum."""

    quant_data_type: str
    aggregate_rows: int
    out_features: int
    in_features: int
    physical_experts: int
    max_route_entries: int
    projection_count: int
