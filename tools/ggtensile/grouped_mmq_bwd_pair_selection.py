"""Research-only selection policy for qualified Qwen Q3_K pair artifacts."""

from typing import ClassVar

from .grouped_mmq_bwd_pair_model import (
    GroupedBackwardPairProblem,
    GroupedBackwardPairSolution,
    GroupedBackwardPairSolutionKey,
)


class ResearchGroupedBackwardPairQ3KSelector:
    """Select a measured Q3_K key without changing public dispatch."""

    QUALIFIED_ROWS: ClassVar[frozenset[int]] = frozenset({16_384, 65_536, 262_144})
    MAX_ROUTE_ENTRIES = 256
    M128_THRESHOLD_FACTOR = 128

    @classmethod
    def _validate_dimensions(cls, aggregate_rows: int, route_entries: int) -> None:
        if type(aggregate_rows) is not int or aggregate_rows not in cls.QUALIFIED_ROWS:
            raise ValueError(
                "research Q3_K selection requires a qualified production row count"
            )
        if (
            type(route_entries) is not int
            or not 0 < route_entries <= cls.MAX_ROUTE_ENTRIES
        ):
            raise ValueError("research Q3_K selection requires 1..256 route entries")

    @classmethod
    def selected_macro_tile(cls, aggregate_rows: int, route_entries: int) -> int:
        """Return the measured M64/M128 geometry for the route workload."""

        cls._validate_dimensions(aggregate_rows, route_entries)
        return (
            128 if aggregate_rows >= cls.M128_THRESHOLD_FACTOR * route_entries else 64
        )

    @classmethod
    def select_solution(
        cls, aggregate_rows: int, route_entries: int
    ) -> GroupedBackwardPairSolution:
        """Return the retained schedule for one qualified workload."""

        macro_tile = cls.selected_macro_tile(aggregate_rows, route_entries)
        if macro_tile == 64:
            return GroupedBackwardPairSolution.q3_k_m64_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()
        if aggregate_rows == 262_144:
            return GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_overlap_second_read_prefetch_a()
        return GroupedBackwardPairSolution.q3_k_m128_n64_dual_lds_full_tile_split_direct_pointers_prefetch_a()

    @classmethod
    def select_solution_key(
        cls, aggregate_rows: int, route_entries: int
    ) -> GroupedBackwardPairSolutionKey:
        """Return the exact identity consumed by the research launcher."""

        return GroupedBackwardPairSolutionKey(
            GroupedBackwardPairProblem.q3_k(aggregate_rows),
            cls.select_solution(aggregate_rows, route_entries),
        )
