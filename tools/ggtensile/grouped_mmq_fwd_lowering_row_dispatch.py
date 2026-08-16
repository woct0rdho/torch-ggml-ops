"""Policy-driven row-body dispatch for grouped decoded lowerings."""

from collections.abc import Callable
from dataclasses import dataclass

from .grouped_mmq_fwd_spec import GroupedRowTileDispatchPolicy
from .kernel_writer_assembly import Assembly, RegisterAssignment


@dataclass(frozen=True)
class GroupedRowDispatchLabels:
    branch_labels: tuple[tuple[int, str], ...]
    done_label: str | None
    body_suffixes: tuple[str, ...]

    @classmethod
    def activation(
        cls,
        policy: GroupedRowTileDispatchPolicy,
        stage: int,
    ) -> "GroupedRowDispatchLabels":
        if len(policy.body_rows) == 1:
            return cls((), None, ("Macro",))
        if len(policy.body_rows) == 2:
            return cls(
                ((policy.body_rows[-1], f".LGroupedQ4KActivationTailDispatch{stage}"),),
                f".LGroupedQ4KActivationDispatchDone{stage}",
                ("Macro", "Tail"),
            )
        cls._require_three_way(policy)
        return cls(
            (
                (32, f".LGroupedQ5KActivationRows32Dispatch{stage}"),
                (64, f".LGroupedQ5KActivationRows64Dispatch{stage}"),
            ),
            f".LGroupedQ5KActivationThreeWayDone{stage}",
            ("Rows128", "Rows64", "Rows32"),
        )

    @classmethod
    def decoded_mma(
        cls,
        policy: GroupedRowTileDispatchPolicy,
        group_base: int,
    ) -> "GroupedRowDispatchLabels":
        if len(policy.body_rows) == 1:
            return cls((), None, ("",))
        if len(policy.body_rows) == 2:
            return cls(
                ((policy.body_rows[-1], f".LGroupedQ4KMmaTailDispatch{group_base}"),),
                f".LGroupedQ4KMmaDispatchDone{group_base}",
                ("Macro", "Tail"),
            )
        cls._require_three_way(policy)
        return cls(
            (
                (32, f".LGroupedQ5KMmaRows32Dispatch{group_base}"),
                (64, f".LGroupedQ5KMmaRows64Dispatch{group_base}"),
            ),
            f".LGroupedQ5KMmaThreeWayDone{group_base}",
            ("Rows128", "Rows64", "Rows32"),
        )

    @classmethod
    def q2_mma(
        cls,
        policy: GroupedRowTileDispatchPolicy,
        group_base: int,
    ) -> "GroupedRowDispatchLabels":
        if len(policy.body_rows) == 1:
            return cls((), None, ("",))
        if len(policy.body_rows) != 2:
            raise ValueError("grouped Q2 row dispatch implements one or two bodies")
        return cls(
            ((policy.body_rows[-1], f".LGroupedQ2KMmaTailDispatch{group_base}"),),
            f".LGroupedQ2KMmaDispatchDone{group_base}",
            ("Macro", "Tail"),
        )

    @classmethod
    def epilogue(
        cls,
        policy: GroupedRowTileDispatchPolicy,
    ) -> "GroupedRowDispatchLabels":
        if len(policy.body_rows) == 1:
            return cls((), None, ("",))
        if len(policy.body_rows) == 2:
            return cls(
                ((policy.body_rows[-1], ".LGroupedQ4KEpilogueTailDispatch"),),
                ".LGroupedQ4KEpilogueDispatchDone",
                ("", ""),
            )
        cls._require_three_way(policy)
        return cls(
            (
                (32, ".LGroupedQ5KEpilogueRows32Dispatch"),
                (64, ".LGroupedQ5KEpilogueRows64Dispatch"),
            ),
            ".LGroupedQ5KEpilogueThreeWayDone",
            ("", "", ""),
        )

    @staticmethod
    def _require_three_way(policy: GroupedRowTileDispatchPolicy) -> None:
        if policy.body_rows != (128, 64, 32):
            raise ValueError("grouped three-way row dispatch requires 128/64/32 bodies")


GroupedRowBodyEmitter = Callable[[Assembly, int, str], None]
GroupedRowDispatchFinalizer = Callable[[Assembly], None]


@dataclass(frozen=True)
class GroupedRowTileDispatchEmitter:
    policy: GroupedRowTileDispatchPolicy
    row_tile_rows: RegisterAssignment

    def emit(
        self,
        asm: Assembly,
        labels: GroupedRowDispatchLabels,
        emit_body: GroupedRowBodyEmitter,
        finalize: GroupedRowDispatchFinalizer | None = None,
    ) -> None:
        body_rows = self.policy.body_rows
        if len(labels.body_suffixes) != len(body_rows):
            raise ValueError("grouped row dispatch labels do not cover every body")
        if len(body_rows) == 1:
            emit_body(asm, body_rows[0], labels.body_suffixes[0])
            if finalize is not None:
                finalize(asm)
            return
        if labels.done_label is None or len(labels.branch_labels) != len(body_rows) - 1:
            raise ValueError("grouped multi-body dispatch requires complete labels")

        branch_labels = dict(labels.branch_labels)
        if set(branch_labels) != set(body_rows[1:]):
            raise ValueError("grouped row dispatch thresholds do not match tail bodies")
        row_count = self.row_tile_rows.first_register
        for threshold, label in labels.branch_labels:
            asm.inst(f"s_cmp_le_u32 s{row_count}, {threshold}")
            asm.inst(f"s_cbranch_scc1 {label}")

        emit_body(asm, body_rows[0], labels.body_suffixes[0])
        asm.inst(f"s_branch {labels.done_label}")
        for index, rows in enumerate(body_rows[1:], start=1):
            asm.label(branch_labels[rows])
            emit_body(asm, rows, labels.body_suffixes[index])
            if index != len(body_rows) - 1:
                asm.inst(f"s_branch {labels.done_label}")
        asm.label(labels.done_label)
        if finalize is not None:
            finalize(asm)
