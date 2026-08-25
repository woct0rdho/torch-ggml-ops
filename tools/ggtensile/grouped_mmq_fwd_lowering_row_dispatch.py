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
    def _multi_way(
        cls,
        policy: GroupedRowTileDispatchPolicy,
        *,
        branch_label: Callable[[int], str],
        done_label: str,
        suffixes: tuple[str, ...],
    ) -> "GroupedRowDispatchLabels":
        body_rows = policy.body_rows
        return cls(
            tuple((rows, branch_label(rows)) for rows in reversed(body_rows[1:])),
            done_label,
            suffixes,
        )

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
        return cls._multi_way(
            policy,
            branch_label=lambda rows: (
                f".LGroupedQ5KActivationRows{rows}Dispatch{stage}"
            ),
            done_label=(
                f".LGroupedQ5KActivationThreeWayDone{stage}"
                if len(policy.body_rows) == 3
                else f".LGroupedQ5KActivationDispatchDone{stage}"
            ),
            suffixes=tuple(f"Rows{rows}" for rows in policy.body_rows),
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
        return cls._multi_way(
            policy,
            branch_label=lambda rows: f".LGroupedQ5KMmaRows{rows}Dispatch{group_base}",
            done_label=(
                f".LGroupedQ5KMmaThreeWayDone{group_base}"
                if len(policy.body_rows) == 3
                else f".LGroupedQ5KMmaDispatchDone{group_base}"
            ),
            suffixes=tuple(f"Rows{rows}" for rows in policy.body_rows),
        )

    @classmethod
    def q2_mma(
        cls,
        policy: GroupedRowTileDispatchPolicy,
        group_base: int,
    ) -> "GroupedRowDispatchLabels":
        if len(policy.body_rows) == 1:
            return cls((), None, ("",))
        if len(policy.body_rows) == 2:
            return cls(
                ((policy.body_rows[-1], f".LGroupedQ2KMmaTailDispatch{group_base}"),),
                f".LGroupedQ2KMmaDispatchDone{group_base}",
                ("Macro", "Tail"),
            )
        return cls._multi_way(
            policy,
            branch_label=lambda rows: f".LGroupedQ2KMmaRows{rows}Dispatch{group_base}",
            done_label=f".LGroupedQ2KMmaDispatchDone{group_base}",
            suffixes=tuple(f"Rows{rows}" for rows in policy.body_rows),
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
        return cls._multi_way(
            policy,
            branch_label=lambda rows: f".LGroupedQ5KEpilogueRows{rows}Dispatch",
            done_label=(
                ".LGroupedQ5KEpilogueThreeWayDone"
                if len(policy.body_rows) == 3
                else ".LGroupedQ5KEpilogueDispatchDone"
            ),
            suffixes=("",) * len(policy.body_rows),
        )


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
        assert len(labels.body_suffixes) == len(body_rows)
        if len(body_rows) == 1:
            emit_body(asm, body_rows[0], labels.body_suffixes[0])
            if finalize is not None:
                finalize(asm)
            return
        assert not (
            labels.done_label is None or len(labels.branch_labels) != len(body_rows) - 1
        )

        branch_labels = dict(labels.branch_labels)
        assert set(branch_labels) == set(body_rows[1:])
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
