"""Typed contract and derived state for fixed-group Q8_0 forward."""

from dataclasses import dataclass
from typing import cast

from .fixed_grouped_mmq_fwd_model import (
    FixedForwardProblem,
)
from .fixed_grouped_mmq_fwd_physical import (
    FixedQ8ForwardPhysicalPlan,
    fixed_q8_forward_physical_plan,
)
from .mmq_fwd_spec import (
    DecodeSpec,
    DerivedForwardState,
    EpilogueSpec,
    FixedForwardDecodePolicy,
    ForwardActivationStaging,
    ForwardDataMovementPolicy,
    ForwardKernelSpec,
    ForwardPipelinePolicy,
    ForwardProblemContract,
    ForwardWeightStaging,
    GeometrySpec,
    GlobalMemorySpec,
    InstructionPolicy,
    LdsSpec,
    OwnershipSpec,
    SemanticSchedulePolicy,
    forward_mechanism_contract,
)
from .model import ProblemSize
from .quant_formats import Q8_1_D4_BLOCK_VALUES, QUANT_FORMATS
from .schema import enum_value as _enum
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _mapping
from .schema import string as _string

_U32_MAX = 0xFFFFFFFF


def validate_fixed_forward_problem(problem: FixedForwardProblem) -> None:
    quant = QUANT_FORMATS.get(problem.quant_data_type)
    assert quant is not None
    assert problem.quant_data_type == "Q8_0"
    for value in (
        problem.tokens,
        problem.output_features,
        problem.input_features,
    ):
        assert 0 < value <= _U32_MAX
    assert problem.groups == 8
    assert problem.tokens % 64 == 0
    assert problem.output_features % 64 == 0
    assert problem.input_features % quant.block_values == 0
    assert problem.input_features % Q8_1_D4_BLOCK_VALUES == 0

    activation_blocks = problem.input_features // Q8_1_D4_BLOCK_VALUES
    activation_bytes = (
        activation_blocks * problem.total_activation_rows * quant.activation_block_bytes
    )
    output_bytes = problem.tokens * problem.groups * problem.output_features * 2
    assert problem.bytes_per_group <= _U32_MAX
    assert activation_bytes <= _U32_MAX
    assert output_bytes <= _U32_MAX


@dataclass(frozen=True)
class FixedForwardProblemContract:
    """Non-tunable fixed-group data, arithmetic, and ABI contract."""

    quant_type: str
    output_features: int
    input_features: int
    groups: int
    block_values: int
    packed_weight_block_bytes: int
    activation_layout: str
    activation_block_bytes: int
    arithmetic_contract: str
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    weight_decode: str
    activation_staging: ForwardActivationStaging
    scale_arithmetic: str
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    destination_type: str = "BFloat16"
    bf16_rounding: str = "RNEPreserveNaN"
    abi: str = "FixedGroupedQ8OutputV1"

    @classmethod
    def for_problem(cls, problem: FixedForwardProblem) -> "FixedForwardProblemContract":
        quant = QUANT_FORMATS.get(problem.quant_data_type)
        assert quant is not None
        validate_fixed_forward_problem(problem)
        return cls(
            quant_type=problem.quant_data_type,
            output_features=problem.output_features,
            input_features=problem.input_features,
            groups=problem.groups,
            block_values=quant.block_values,
            packed_weight_block_bytes=quant.block_bytes,
            activation_layout=quant.activation_layout,
            activation_block_bytes=quant.activation_block_bytes,
            arithmetic_contract=quant.arithmetic_contract,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            weight_decode="DirectSignedInt8",
            activation_staging=ForwardActivationStaging.FixedGroupRows,
            scale_arithmetic="Int32ScaleF32",
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=False,
        )

    def problem(self, tokens: int) -> FixedForwardProblem:
        return FixedForwardProblem(
            self.quant_type,
            tokens,
            self.output_features,
            self.input_features,
            self.groups,
        )

    def ordinary(self) -> ForwardProblemContract:
        return ForwardProblemContract(
            quant_type=self.quant_type,
            block_values=self.block_values,
            packed_weight_block_bytes=self.packed_weight_block_bytes,
            activation_layout=self.activation_layout,
            activation_block_bytes=self.activation_block_bytes,
            kernel_language=self.kernel_language,
            isa=self.isa,
            wavefront_size=self.wavefront_size,
            signed_weight=self.signed_weight,
            signed_activation=self.signed_activation,
            wmma_clamp=self.wmma_clamp,
            weight_decode=self.weight_decode,
            scale_arithmetic=self.scale_arithmetic,
            arithmetic_contract=self.arithmetic_contract,
            destination_type=self.destination_type,
            bf16_rounding=self.bf16_rounding,
        )


@dataclass(frozen=True)
class FixedForwardKernelSpec:
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile_tokens: int
    macro_tile_features: int
    depth_u: int
    weight_staging: ForwardWeightStaging
    lds_address_hoist: str
    fixed_address_hoist: str

    @classmethod
    def from_mapping(cls, value: object) -> "FixedForwardKernelSpec":
        item = _mapping(
            value,
            "FixedForwardKernelSpec",
            frozenset({"Geometry", "Lowering"}),
        )
        geometry = _mapping(
            item["Geometry"],
            "FixedForwardKernelSpec.Geometry",
            frozenset(
                {
                    "WorkGroup",
                    "MatrixInstruction",
                    "MacroTileTokens",
                    "MacroTileFeatures",
                    "DepthU",
                }
            ),
        )
        lowering = _mapping(
            item["Lowering"],
            "FixedForwardKernelSpec.Lowering",
            frozenset({"WeightStaging", "LdsAddressHoist", "FixedAddressHoist"}),
        )
        return cls(
            work_group=_integer_tuple(geometry, "WorkGroup", 3),
            matrix_instruction=_integer_tuple(geometry, "MatrixInstruction", 9),
            macro_tile_tokens=_integer(geometry, "MacroTileTokens"),
            macro_tile_features=_integer(geometry, "MacroTileFeatures"),
            depth_u=_integer(geometry, "DepthU"),
            weight_staging=_enum(
                lowering,
                "WeightStaging",
                ForwardWeightStaging,
            ),
            lds_address_hoist=_string(lowering, "LdsAddressHoist"),
            fixed_address_hoist=_string(lowering, "FixedAddressHoist"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "Geometry": {
                "WorkGroup": list(self.work_group),
                "MatrixInstruction": list(self.matrix_instruction),
                "MacroTileTokens": self.macro_tile_tokens,
                "MacroTileFeatures": self.macro_tile_features,
                "DepthU": self.depth_u,
            },
            "Lowering": {
                "WeightStaging": self.weight_staging.value,
                "LdsAddressHoist": self.lds_address_hoist,
                "FixedAddressHoist": self.fixed_address_hoist,
            },
        }

    def forward_kernel_spec(
        self, contract: FixedForwardProblemContract
    ) -> ForwardKernelSpec:
        mechanism = forward_mechanism_contract(self.weight_staging)
        assert mechanism.physical_plan == "SignedInt8SmallMTiledLds"
        matrix_instruction = cast(
            tuple[int, int, int, int], self.matrix_instruction[:4]
        )
        threads = self.work_group[0] * self.work_group[1] * self.work_group[2]
        assert threads > 0 and threads % 32 == 0
        waves = threads // 32
        mi_wave_group = (1, waves) if mechanism.ownership == "WaveN" else (waves, 1)
        tile_m, tile_n, _, _ = matrix_instruction
        divisors = (tile_m * mi_wave_group[0], tile_n * mi_wave_group[1])
        assert not (
            self.macro_tile_tokens % divisors[0]
            or self.macro_tile_features % divisors[1]
        )
        return ForwardKernelSpec(
            geometry=GeometrySpec(self.work_group, matrix_instruction, self.depth_u),
            ownership=OwnershipSpec(
                mi_wave_group,
                (
                    self.macro_tile_tokens // divisors[0],
                    self.macro_tile_features // divisors[1],
                ),
            ),
            global_memory=GlobalMemorySpec(
                self.weight_staging,
                contract.activation_staging,
                None,
            ),
            lds=LdsSpec(self.lds_address_hoist),
            decode=DecodeSpec(
                mechanism.dataflow.metadata_conversion,
                FixedForwardDecodePolicy(),
            ),
            epilogue=EpilogueSpec(None),
            instruction_policy=InstructionPolicy(None, None, None),
            semantic_schedule=SemanticSchedulePolicy(),
            pipeline=ForwardPipelinePolicy.canonical(),
            data_movement=ForwardDataMovementPolicy(
                16,
                2,
                16,
                4,
                None,
            ),
        )


@dataclass(frozen=True)
class DerivedFixedForwardState:
    """All fixed-group values consumed by validation, lowering, and runtime."""

    problem: FixedForwardProblem
    contract: FixedForwardProblemContract
    spec: FixedForwardKernelSpec
    ordinary: DerivedForwardState
    physical: FixedQ8ForwardPhysicalPlan
    grid: tuple[int, int, int]

    @classmethod
    def from_problem_spec(
        cls,
        problem: FixedForwardProblem,
        fixed_spec: FixedForwardKernelSpec,
    ) -> "DerivedFixedForwardState":
        contract = FixedForwardProblemContract.for_problem(problem)
        kernel_spec = fixed_spec.forward_kernel_spec(contract)
        ordinary_state = DerivedForwardState.from_contract_spec(
            ProblemSize(
                problem.tokens,
                problem.output_features,
                problem.input_features,
            ),
            contract.ordinary(),
            kernel_spec,
            activation_rows=problem.total_activation_rows,
        )
        fixed_plan = fixed_q8_forward_physical_plan(
            kernel_spec, fixed_spec.fixed_address_hoist
        )
        for name, value, divisor in (
            ("tokens", problem.tokens, fixed_spec.macro_tile_tokens),
            (
                "output features",
                problem.output_features,
                fixed_spec.macro_tile_features,
            ),
            ("input features", problem.input_features, 4 * 32),
        ):
            assert not (value <= 0 or divisor <= 0 or value % divisor)
        return cls(
            problem=problem,
            contract=contract,
            spec=fixed_spec,
            ordinary=ordinary_state,
            physical=fixed_plan,
            grid=(
                problem.output_features // fixed_spec.macro_tile_features,
                problem.tokens // fixed_spec.macro_tile_tokens,
                problem.groups,
            ),
        )

    @property
    def expected_packed_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.problem.groups,
            self.problem.output_features,
            self.problem.packed_row_bytes,
        )

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.ordinary.activation_blocks_per_row,
            self.problem.total_activation_rows,
            self.contract.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int, int]:
        return (self.problem.tokens, self.problem.groups, self.problem.output_features)
