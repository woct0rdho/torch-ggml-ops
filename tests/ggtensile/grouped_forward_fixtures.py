from dataclasses import replace
from pathlib import Path

from tools.ggtensile.campaign import DeploymentCatalog, load_catalog
from tools.ggtensile.grouped_mmq_fwd_model import (
    GroupedActivationAddressing,
    GroupedDecodedPolicy,
    GroupedForwardProblem,
    GroupedOperandSource,
)
from tools.ggtensile.grouped_mmq_fwd_spec import GroupedForwardKernelSpec

_CONFIG = Path(__file__).resolve().parents[2] / "tools/ggtensile/configs"


def _catalog(quant_type: str) -> DeploymentCatalog:
    return load_catalog(_CONFIG / f"mmq_grouped_fwd_{quant_type.lower()}_catalog.json")


def _specs(catalog: DeploymentCatalog) -> tuple[GroupedForwardKernelSpec, ...]:
    result = []
    seen = set()
    for entry in catalog.entries:
        instance = entry.instance
        assert isinstance(instance.problem, GroupedForwardProblem)
        assert isinstance(instance.kernel_spec, GroupedForwardKernelSpec)
        if instance.kernel_spec in seen:
            continue
        seen.add(instance.kernel_spec)
        result.append(instance.kernel_spec)
    return tuple(result)


_CATALOGS = {
    quant_type: _catalog(quant_type) for quant_type in ("Q2_K", "Q4_K", "Q5_K", "IQ2_S")
}

_Q2_J32, _Q2_J64 = _specs(_CATALOGS["Q2_K"])
_Q4_J64, _Q4_J128 = _specs(_CATALOGS["Q4_K"])
_Q5_J64, _Q5_J128 = _specs(_CATALOGS["Q5_K"])
(_IQ2_S_SELECTED,) = _specs(_CATALOGS["IQ2_S"])


def grouped_forward_problem(
    quant_type: str, aggregate_rows: int
) -> GroupedForwardProblem:
    problem = _CATALOGS[quant_type].entries[0].instance.problem
    assert isinstance(problem, GroupedForwardProblem)
    return replace(problem, aggregate_rows=aggregate_rows)


def _geometry(
    spec: GroupedForwardKernelSpec, **changes: object
) -> GroupedForwardKernelSpec:
    geometry = replace(spec.geometry, **changes)
    return replace(spec, geometry=geometry)


def _decode(
    spec: GroupedForwardKernelSpec, **changes: object
) -> GroupedForwardKernelSpec:
    return replace(spec, decode=replace(spec.decode, **changes))


def _epilogue(
    spec: GroupedForwardKernelSpec, **changes: int
) -> GroupedForwardKernelSpec:
    return replace(spec, epilogue=replace(spec.epilogue, **changes))


class GroupedForwardTestSolutions:
    @classmethod
    def q4_k_serial_decoded_lds(cls) -> GroupedForwardKernelSpec:
        return _epilogue(
            replace(
                _Q4_J128,
                decode=GroupedDecodedPolicy("DirectFloat16Unsigned16", False, False),
            ),
            tiles_ahead=8,
            dependency_width=1,
            priority=0,
        )

    @classmethod
    def q4_k_serial_direct(cls) -> GroupedForwardKernelSpec:
        spec = cls.q4_k_serial_decoded_lds()
        return replace(
            _geometry(
                spec,
                work_group=(32, 1, 1),
                matrix_instruction=(16, 16, 16, 1, 1, 1, 1, 1, 1),
                macro_tile=(16, 16),
                tail_macro_tile0=16,
            ),
            operand_source=GroupedOperandSource.GroupedDirectGlobal,
            activation=replace(
                spec.activation,
                addressing=GroupedActivationAddressing.AggregateRows,
            ),
            decode=GroupedDecodedPolicy("Float32ThenFloat16", False, False),
            epilogue=replace(spec.epilogue, tiles_ahead=1),
        )

    @classmethod
    def q4_k_serial_decoded_lds_64(cls) -> GroupedForwardKernelSpec:
        return _epilogue(
            _geometry(
                cls.q4_k_serial_decoded_lds(), macro_tile=(64, 64), tail_macro_tile0=64
            ),
            tiles_ahead=4,
        )

    @staticmethod
    def q4_k_serial_decoded_lds_scheduled_a1d2p2() -> GroupedForwardKernelSpec:
        return _Q4_J128

    @staticmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32() -> GroupedForwardKernelSpec:
        return _epilogue(_Q4_J64, tiles_ahead=4, dependency_width=1, priority=0)

    @staticmethod
    def q4_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2() -> (
        GroupedForwardKernelSpec
    ):
        return _Q4_J64

    @staticmethod
    def iq2_s_serial_full_weight_lds_64_linear_payload_prefetch() -> (
        GroupedForwardKernelSpec
    ):
        return _IQ2_S_SELECTED

    @classmethod
    def iq2_s_serial_full_weight_lds_64_linear_activation(
        cls,
    ) -> GroupedForwardKernelSpec:
        return _decode(_IQ2_S_SELECTED, payload_prefetch=False)

    @classmethod
    def iq2_s_serial_full_weight_lds_64(cls) -> GroupedForwardKernelSpec:
        return replace(
            cls.iq2_s_serial_full_weight_lds_64_linear_activation(),
            activation=replace(
                _IQ2_S_SELECTED.activation,
                addressing=GroupedActivationAddressing.AggregateRowsTiled,
            ),
        )

    @staticmethod
    def q2_k_serial_decoded_lds_32_hip_pre_negated_dm_write2_meta2_distributed_mixed16() -> (
        GroupedForwardKernelSpec
    ):
        return _Q2_J32

    @staticmethod
    def q2_k_serial_decoded_lds_64_hip_distributed() -> GroupedForwardKernelSpec:
        return _Q2_J64

    @classmethod
    def q2_k_serial_decoded_lds_32(cls) -> GroupedForwardKernelSpec:
        return _decode(
            _geometry(
                _Q2_J32,
                tail_macro_tile0=32,
            ),
            unrolled_groups=False,
            hip_association=False,
            partial_lds=False,
            pre_negated_dm=False,
            paired_payload_writes=False,
            paired_metadata_writes=False,
            distributed_producer=False,
        )

    @classmethod
    def q2_k_serial_decoded_lds_32_unrolled(cls) -> GroupedForwardKernelSpec:
        return _decode(cls.q2_k_serial_decoded_lds_32(), unrolled_groups=True)

    @classmethod
    def q2_k_serial_decoded_lds_64_unrolled(cls) -> GroupedForwardKernelSpec:
        spec = _geometry(
            cls.q2_k_serial_decoded_lds_32(), macro_tile=(64, 64), tail_macro_tile0=64
        )
        return _decode(_epilogue(spec, tiles_ahead=4), unrolled_groups=True)

    @staticmethod
    def q5_k_serial_decoded_lds_64_scheduled_mixed32_a1d4p2() -> (
        GroupedForwardKernelSpec
    ):
        return _Q5_J64

    @staticmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_mixed32_a1d4p2() -> (
        GroupedForwardKernelSpec
    ):
        return _Q5_J128

    @staticmethod
    def q5_k_serial_decoded_lds() -> GroupedForwardKernelSpec:
        return _epilogue(
            _geometry(
                replace(
                    _Q5_J128,
                    decode=GroupedDecodedPolicy(
                        "DirectFloat16Unsigned16", False, False
                    ),
                ),
                tail_macro_tile0=128,
            ),
            tiles_ahead=8,
            dependency_width=1,
            priority=0,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_a1d2p2(cls) -> GroupedForwardKernelSpec:
        return _epilogue(
            _decode(
                cls.q5_k_serial_decoded_lds(),
                independent_metadata_extraction=True,
                defer_metadata_reads=True,
            ),
            tiles_ahead=1,
            dependency_width=2,
            priority=2,
        )

    @classmethod
    def q5_k_serial_decoded_lds_scheduled_mixed64_a1d2p2(
        cls,
    ) -> GroupedForwardKernelSpec:
        return _geometry(
            cls.q5_k_serial_decoded_lds_scheduled_a1d2p2(), tail_macro_tile0=64
        )

    @staticmethod
    def q5_k_serial_decoded_lds_64_scheduled_a1d4p2() -> GroupedForwardKernelSpec:
        return _geometry(_Q5_J64, tail_macro_tile0=64)
