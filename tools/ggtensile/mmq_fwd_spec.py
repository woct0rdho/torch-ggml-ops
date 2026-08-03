"""Typed problem contracts, kernel specifications, and derived MMQ forward state."""

from dataclasses import asdict, dataclass
from typing import Literal, cast

from .model import ForwardSolution, ProblemSize, SolutionKey
from .quant_formats import (
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)


@dataclass(frozen=True)
class ForwardFormatTraits:
    """Packed-format and activation-workspace facts fixed by quant type."""

    quant_type: str
    block_values: int
    packed_weight_block_bytes: int
    activation_layout: str
    activation_block_bytes: int

    @classmethod
    def for_quant_type(cls, quant_type: str) -> "ForwardFormatTraits":
        if quant_type not in {"Q4_K", "Q5_K", "Q6_K"}:
            raise ValueError(f"unsupported MMQ forward quant type {quant_type!r}")
        quant_format = QUANT_FORMATS[quant_type]
        if quant_type == "Q6_K":
            activation_layout = "F32_D4"
            activation_block_bytes = Q8_1_F32_D4_BLOCK_BYTES
        else:
            activation_layout = "F16_D4S4"
            activation_block_bytes = Q8_1_F16_D4S4_BLOCK_BYTES
        return cls(
            quant_type=quant_type,
            block_values=quant_format.block_values,
            packed_weight_block_bytes=quant_format.block_bytes,
            activation_layout=activation_layout,
            activation_block_bytes=activation_block_bytes,
        )


@dataclass(frozen=True)
class PayloadPlaneSpec:
    """One semantic field in a packed 256-value weight block."""

    name: str
    byte_offset: int
    byte_count: int
    encoding: str


@dataclass(frozen=True)
class PackedFieldPart:
    """One source slice contributing bits to an unpacked metadata field."""

    metadata_word: int
    bit_offset: int
    bit_count: int
    destination_shift: int = 0


@dataclass(frozen=True)
class PackedScaleMinimumFields:
    """Packed metadata slices for one 32-value scale/minimum group."""

    scale: tuple[PackedFieldPart, ...]
    minimum: tuple[PackedFieldPart, ...]


@dataclass(frozen=True)
class HighBitReconstructionSpec:
    """Optional Q5 high-bit plane reconstruction semantics."""

    plane_name: str
    address_lane_mask: int
    address_lane_stride: int
    lane_shift_mask: int
    nibble_shift: int
    nibble_mask: int
    low_mask: int
    low_destination_shift: int
    high_mask: int
    high_destination_shift: int


@dataclass(frozen=True)
class Q6SignedDecodeSpec:
    """Bit-exact Q6 low/high merge and packed signed-byte normalization."""

    low_nibble_mask: int
    high_bits_mask: int
    high_bits_shift: int
    signed_add: int
    signed_xor: int

    def __post_init__(self) -> None:
        if (
            self.low_nibble_mask != 0x0F0F0F0F
            or self.high_bits_mask != 0x30303030
            or self.high_bits_shift != 4
            or self.signed_add != 0x60606060
            or self.signed_xor != 0x80808080
        ):
            raise ValueError("unsupported Q6 signed decode formula")


@dataclass(frozen=True)
class QuantForwardSemantics:
    """Project-owned packed payload and post-WMMA arithmetic semantics."""

    quant_type: str
    weight_bits: int
    payload_planes: tuple[PayloadPlaneSpec, ...]
    activation_components: tuple[str, ...]
    post_wmma_correction: str

    @classmethod
    def for_quant_type(cls, quant_type: str) -> "QuantForwardSemantics":
        if quant_type == "Q4_K":
            return cls(
                quant_type=quant_type,
                weight_bits=4,
                payload_planes=(
                    PayloadPlaneSpec("dm", 0, 4, "Float16Pair"),
                    PayloadPlaneSpec("scales", 4, 12, "PackedScaleMinimum6"),
                    PayloadPlaneSpec("ql", 16, 128, "UnsignedNibble"),
                ),
                activation_components=("q", "d", "s"),
                post_wmma_correction="ScaleAndMinimum",
            )
        if quant_type == "Q5_K":
            return cls(
                quant_type=quant_type,
                weight_bits=5,
                payload_planes=(
                    PayloadPlaneSpec("dm", 0, 4, "Float16Pair"),
                    PayloadPlaneSpec("scales", 4, 12, "PackedScaleMinimum6"),
                    PayloadPlaneSpec("qh", 16, 32, "HighBitMask"),
                    PayloadPlaneSpec("ql", 48, 128, "UnsignedNibble"),
                ),
                activation_components=("q", "d", "s"),
                post_wmma_correction="ScaleAndMinimum",
            )
        if quant_type == "Q6_K":
            return cls(
                quant_type=quant_type,
                weight_bits=6,
                payload_planes=(
                    PayloadPlaneSpec("ql", 0, 128, "UnsignedNibble"),
                    PayloadPlaneSpec("qh", 128, 64, "UnsignedTwoBit"),
                    PayloadPlaneSpec("scales", 192, 16, "SignedInt8"),
                    PayloadPlaneSpec("d", 208, 2, "Float16"),
                ),
                activation_components=("q", "d"),
                post_wmma_correction="SignedScaleTimesBlockFactors",
            )
        raise ValueError(f"unsupported MMQ forward quant type {quant_type!r}")

    def low_payload_group_offsets(self, group: int) -> tuple[int, int]:
        """Return the two 128-bit QL vectors consumed by one direct group."""
        if group not in range(8):
            raise ValueError("forward packed payload group must be in range 0..7")
        low = self.payload_plane("ql")
        return (
            low.byte_offset + 32 * (group // 2),
            low.byte_offset + 32 * (group // 2) + 16,
        )

    def packed_scale_minimum_fields(self, group: int) -> PackedScaleMinimumFields:
        if group not in range(8):
            raise ValueError("packed scale/minimum group must be in range 0..7")
        if self.post_wmma_correction != "ScaleAndMinimum":
            raise ValueError(f"{self.quant_type} has no packed scale/minimum fields")
        if group < 4:
            bit = 8 * group
            return PackedScaleMinimumFields(
                scale=(PackedFieldPart(1, bit, 6),),
                minimum=(PackedFieldPart(2, bit, 6),),
            )
        bit = 8 * (group - 4)
        return PackedScaleMinimumFields(
            scale=(
                PackedFieldPart(3, bit, 4),
                PackedFieldPart(1, bit + 6, 2, 4),
            ),
            minimum=(
                PackedFieldPart(3, bit + 4, 4),
                PackedFieldPart(2, bit + 6, 2, 4),
            ),
        )

    def high_bit_reconstruction(self) -> HighBitReconstructionSpec | None:
        if self.weight_bits != 5:
            return None
        return HighBitReconstructionSpec(
            plane_name="qh",
            address_lane_mask=1,
            address_lane_stride=16,
            lane_shift_mask=6,
            nibble_shift=4,
            nibble_mask=0x0F0F0F0F,
            low_mask=0x01010101,
            low_destination_shift=4,
            high_mask=0x02020202,
            high_destination_shift=3,
        )

    def q6_packed_payload_offset(
        self,
        atom: int,
        plane: Literal["ql", "qh"],
    ) -> int:
        if self.quant_type != "Q6_K":
            raise ValueError(f"{self.quant_type} has no Q6 packed payload")
        if atom not in range(16):
            raise ValueError(f"unsupported Q6 payload atom: {atom}")
        if plane not in ("ql", "qh"):
            raise ValueError(f"unsupported Q6 payload plane: {plane}")
        lane = atom % 8
        low_offset = 512 * ((5 * lane + atom // 8) % 8) + 64 * lane
        return low_offset if plane == "ql" else (low_offset + 128) % 4096

    def q6_signed_decode(self) -> Q6SignedDecodeSpec:
        if self.quant_type != "Q6_K":
            raise ValueError(f"{self.quant_type} has no signed Q6 decode")
        return Q6SignedDecodeSpec(
            low_nibble_mask=0x0F0F0F0F,
            high_bits_mask=0x30303030,
            high_bits_shift=4,
            signed_add=0x60606060,
            signed_xor=0x80808080,
        )

    def payload_plane(self, name: str) -> PayloadPlaneSpec:
        for plane in self.payload_planes:
            if plane.name == name:
                return plane
        raise ValueError(f"{self.quant_type} has no payload plane {name!r}")


@dataclass(frozen=True)
class ForwardProblemContract:
    """Non-tunable format, arithmetic, ISA, destination, and ABI contract."""

    quant_type: str
    block_values: int
    packed_weight_block_bytes: int
    activation_layout: str
    activation_block_bytes: int
    kernel_language: str
    isa: tuple[int, int, int]
    wavefront_size: int
    signed_weight: bool
    signed_activation: bool
    wmma_clamp: bool
    weight_decode: str
    scale_arithmetic: str
    arithmetic_contract: str
    destination_type: str = "BFloat16"
    bf16_rounding: str = "RNEPreserveNaN"
    abi: str = "PackedWeightQ81OutputV1"

    @classmethod
    def rejection_reason(
        cls,
        quant_type: str,
        solution: ForwardSolution,
    ) -> str | None:
        traits = ForwardFormatTraits.for_quant_type(quant_type)
        expected_clamp = quant_type != "Q6_K"
        expected_weight_decode = {
            "Q4_K": "DirectNibble",
            "Q5_K": "DirectNibbleHighBit",
            "Q6_K": "DirectQ6Signed",
        }[quant_type]
        expected_scale_arithmetic = "Int32ScaleF32" if quant_type == "Q6_K" else "FP16"
        checks = (
            (
                solution.activation_layout == traits.activation_layout,
                "forward activation layout does not match quant-format contract",
            ),
            (
                solution.activation_block_bytes == traits.activation_block_bytes,
                "forward activation block bytes do not match quant-format contract",
            ),
            (
                solution.packed_weight_block_bytes == traits.packed_weight_block_bytes,
                "forward packed-weight bytes do not match quant-format contract",
            ),
            (
                solution.kernel_language == "Assembly",
                "forward kernel language must be Assembly",
            ),
            (solution.isa == (11, 5, 1), "forward ISA must be gfx1151"),
            (solution.wavefront_size == 32, "forward wavefront size must be 32"),
            (
                solution.signed_weight and solution.signed_activation,
                "forward dot operands must be signed",
            ),
            (
                solution.wmma_clamp == expected_clamp,
                "forward WMMA clamp does not match the quant arithmetic contract",
            ),
            (
                solution.weight_decode == expected_weight_decode,
                "forward weight decode does not match the quant-format contract",
            ),
            (
                solution.scale_arithmetic == expected_scale_arithmetic,
                "forward scale arithmetic does not match the quant-format contract",
            ),
        )
        return next((message for accepted, message in checks if not accepted), None)

    @classmethod
    def from_solution(
        cls,
        quant_type: str,
        solution: ForwardSolution,
    ) -> "ForwardProblemContract":
        traits = ForwardFormatTraits.for_quant_type(quant_type)
        rejection = cls.rejection_reason(quant_type, solution)
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=traits.quant_type,
            block_values=traits.block_values,
            packed_weight_block_bytes=traits.packed_weight_block_bytes,
            activation_layout=traits.activation_layout,
            activation_block_bytes=traits.activation_block_bytes,
            kernel_language=solution.kernel_language,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            signed_weight=solution.signed_weight,
            signed_activation=solution.signed_activation,
            wmma_clamp=solution.wmma_clamp,
            weight_decode=solution.weight_decode,
            scale_arithmetic=solution.scale_arithmetic,
            arithmetic_contract=(
                "SignedQ6Int8ScaleIntegerWmmaF32Correction"
                if quant_type == "Q6_K"
                else "SignedKQuantIntegerWmmaFP16ScaleMinimumCorrection"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GeometrySpec:
    """Explicit launch and matrix-instruction geometry parameters."""

    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, int, int, int]
    depth_u: int


@dataclass(frozen=True)
class OwnershipSpec:
    """Explicit wave-group and per-wave output ownership."""

    mi_wave_group: tuple[int, int]
    mi_wave_tile: tuple[int, int]


@dataclass(frozen=True)
class GlobalMemorySpec:
    """Global operand dataflow and address/cache policies."""

    operand_source: str
    activation_addressing: str
    global_read_cache_policy: str | None


@dataclass(frozen=True)
class LdsSpec:
    """LDS representation/addressing policy for the selected dataflow family."""

    address_hoist: str


@dataclass(frozen=True)
class DecodedLdsLayout:
    """Derived shared LDS planes for the retained Q4/Q5 decoded path."""

    activation_base: int
    weight_data_base: int
    weight_metadata_base: int
    weight_row_stride: int
    activation_lane_stride: int

    @classmethod
    def for_contract(cls, contract: ForwardProblemContract) -> "DecodedLdsLayout":
        if contract.activation_layout != "F16_D4S4":
            raise ValueError("decoded LDS layout requires the F16_D4S4 workspace")
        weight_row_stride = 2 * contract.activation_block_bytes + 16
        return cls(
            activation_base=512,
            weight_data_base=18_944,
            weight_metadata_base=19_200,
            weight_row_stride=weight_row_stride,
            activation_lane_stride=512,
        )

    @property
    def metadata_row_stride(self) -> int:
        return self.weight_row_stride

    @property
    def metadata_group_stride(self) -> int:
        return 4 * self.metadata_row_stride


@dataclass(frozen=True)
class DecodeSpec:
    """Candidate-selectable metadata and traversal lowering policies."""

    metadata_conversion: str
    metadata_schedule: str | None


@dataclass(frozen=True)
class EpiloguePipelineSpec:
    """Only the active epilogue scheduling parameters for one family."""

    tiles_ahead: int | None
    dependency_width: int
    priority: int | None
    scope: str | None


@dataclass(frozen=True)
class EpilogueSpec:
    """Destination encoding and optional pipelined-store policy."""

    output_store: str
    pipeline: EpiloguePipelineSpec | None


@dataclass(frozen=True)
class ResourceLimits:
    """Fixed gfx1151 admission limits used by static candidate validation."""

    max_vgprs: int = 256
    max_sgprs: int = 106
    max_lds_bytes: int = 64 * 1024
    require_zero_spills: bool = True


@dataclass(frozen=True)
class ForwardResourceUsage:
    """Formula-derived static resources; never a candidate tuning dimension."""

    vgprs: int
    sgprs: int
    lds_bytes: int
    private_segment_bytes: int = 0
    vgpr_spills: int = 0
    sgpr_spills: int = 0

    def admit(self, limits: ResourceLimits) -> None:
        if self.vgprs > limits.max_vgprs:
            raise ValueError("forward VGPR usage exceeds the candidate limit")
        if self.sgprs > limits.max_sgprs:
            raise ValueError("forward SGPR usage exceeds the candidate limit")
        if self.lds_bytes > limits.max_lds_bytes:
            raise ValueError("forward LDS usage exceeds the candidate limit")
        if limits.require_zero_spills and (
            self.private_segment_bytes or self.vgpr_spills or self.sgpr_spills
        ):
            raise ValueError("forward candidate requires private storage or spills")


def structured_q6_resource_usage(mi_wave_tile_m: int) -> ForwardResourceUsage:
    if mi_wave_tile_m <= 0:
        raise ValueError("Q6 MIWaveTileM must be positive")
    lds = Q6LdsLayout(mi_wave_tile_m)
    return ForwardResourceUsage(
        vgprs=106 + 52 * mi_wave_tile_m,
        sgprs=27,
        lds_bytes=lds.total_bytes,
    )


def derive_forward_resource_usage(spec: "ForwardKernelSpec") -> ForwardResourceUsage:
    operand_source = spec.global_memory.operand_source
    if operand_source == "Q6StructuredDecoded":
        return structured_q6_resource_usage(spec.ownership.mi_wave_tile[0])
    if operand_source == "DecodedWeightLdsBatch8":
        return ForwardResourceUsage(vgprs=239, sgprs=16, lds_bytes=38_400)
    if operand_source == "Global":
        return ForwardResourceUsage(vgprs=88, sgprs=16, lds_bytes=0)
    raise ValueError(f"unsupported forward operand source {operand_source!r}")


@dataclass(frozen=True)
class InstructionPolicy:
    """Only active instruction-ordering policies not implied by dependencies."""

    accumulator_initialization: str | None
    dependency_delay_mode: str | None


@dataclass(frozen=True)
class SemanticSchedulePolicy:
    """Second-level deterministic lowering mechanisms, never an issue table."""

    traversal: str | None
    clustering: str | None
    latency: str | None
    pressure: str | None
    wait: str | None
    pairing: str | None

    @classmethod
    def structured_q6(cls) -> "SemanticSchedulePolicy":
        return cls(
            traversal="OutputRoleGroupMajor",
            clustering="StageDependencyOrder",
            latency="SerializedDependencyDistance",
            pressure="ExplicitRoleLifetime",
            wait="ProducerFirstUse",
            pairing="DependencyCompatibleDualIssue",
        )

    @classmethod
    def inactive(cls) -> "SemanticSchedulePolicy":
        return cls(None, None, None, None, None, None)

    def require_structured_q6(self) -> None:
        if self != self.structured_q6():
            raise ValueError("unsupported structured-Q6 semantic schedule policy")


def forward_kernel_spec_rejection_reason(solution: ForwardSolution) -> str | None:
    structured_q6 = solution.operand_source == "Q6StructuredDecoded"
    decoded_staged = solution.operand_source == "DecodedWeightLdsBatch8"
    serialized_q6_schedule = SemanticSchedulePolicy(
        traversal=solution.q6_output_traversal,
        clustering=solution.q6_stage_clustering,
        latency=solution.q6_latency_policy,
        pressure=solution.q6_pressure_policy,
        wait=solution.q6_wait_policy,
        pairing=solution.q6_pairing_policy,
    )
    if structured_q6:
        if serialized_q6_schedule != SemanticSchedulePolicy.structured_q6():
            return "unsupported structured-Q6 semantic schedule policy"
    elif serialized_q6_schedule != SemanticSchedulePolicy.structured_q6():
        return "Q6 semantic schedule is inactive for this lowering"
    expected_legacy_suffix = (
        (1, 1, 4, 4, 1) if structured_q6 or decoded_staged else (1, 1, 1, 1, 1)
    )
    if (
        len(solution.matrix_instruction) != 9
        or solution.matrix_instruction[4:] != expected_legacy_suffix
    ):
        return (
            "inactive legacy matrix-instruction fields must retain their "
            "canonical sentinel values"
        )
    inactive_checks: list[tuple[bool, str]] = []
    if structured_q6:
        inactive_checks.extend(
            (
                (
                    solution.metadata_schedule == "Serialized",
                    "metadata schedule is inactive for structured Q6",
                ),
                (
                    solution.epilogue_tiles_ahead == 8,
                    "tiles-ahead is inactive for structured Q6",
                ),
                (
                    solution.epilogue_priority == 0,
                    "epilogue priority is inactive for structured Q6",
                ),
                (
                    solution.accumulator_initialization == "ScalarCopy",
                    "accumulator initialization is inactive for structured Q6",
                ),
            )
        )
    elif decoded_staged:
        inactive_checks.extend(
            (
                (
                    solution.q6_epilogue_pipeline_scope == "StoreBatch",
                    "Q6 epilogue scope is inactive for decoded-weight LDS",
                ),
                (
                    solution.q6_dependency_delay_mode == "None",
                    "Q6 delay mode is inactive for decoded-weight LDS",
                ),
                (
                    solution.q6_global_read_cache_policy == "Default",
                    "Q6 cache policy is inactive for decoded-weight LDS",
                ),
            )
        )
    else:
        inactive_checks.extend(
            (
                (
                    solution.metadata_schedule == "Serialized",
                    "metadata schedule is inactive for direct-global lowering",
                ),
                (
                    solution.epilogue_tiles_ahead == 8
                    and solution.epilogue_dependency_width == 1
                    and solution.epilogue_priority == 0,
                    "epilogue pipeline is inactive for direct-global lowering",
                ),
                (
                    solution.accumulator_initialization == "ScalarCopy",
                    "accumulator initialization is inactive for direct-global lowering",
                ),
                (
                    solution.q6_epilogue_pipeline_scope == "StoreBatch"
                    and solution.q6_dependency_delay_mode == "None"
                    and solution.q6_global_read_cache_policy == "Default",
                    "Q6 policies are inactive for direct-global lowering",
                ),
            )
        )
    rejection = next(
        (message for accepted, message in inactive_checks if not accepted), None
    )
    if rejection is not None:
        return rejection
    tile_m, tile_n, tile_k, blocks = solution.matrix_instruction[:4]
    if min(tile_m, tile_n, tile_k, blocks) <= 0:
        return "forward matrix-instruction dimensions must be positive"
    num_threads = solution.num_threads
    if (
        num_threads <= 0
        or solution.wavefront_size <= 0
        or num_threads % solution.wavefront_size
    ):
        return "forward workgroup must contain a positive whole number of waves"
    mi_wave_group = (num_threads // solution.wavefront_size, 1)
    ownership_divisors = (
        tile_m * mi_wave_group[0],
        tile_n * mi_wave_group[1],
    )
    if (
        solution.macro_tile0 % ownership_divisors[0]
        or solution.macro_tile1 % ownership_divisors[1]
        or solution.depth_u % tile_k
    ):
        return "forward geometry is not divisible by matrix-instruction ownership"
    return None


@dataclass(frozen=True)
class ForwardKernelSpec:
    """Complete parameter-only kernel candidate after fixed contracts are removed."""

    geometry: GeometrySpec
    ownership: OwnershipSpec
    global_memory: GlobalMemorySpec
    lds: LdsSpec
    decode: DecodeSpec
    epilogue: EpilogueSpec
    instruction_policy: InstructionPolicy
    semantic_schedule: SemanticSchedulePolicy
    resource_limits: ResourceLimits

    @classmethod
    def from_solution(cls, solution: ForwardSolution) -> "ForwardKernelSpec":
        structured_q6 = solution.operand_source == "Q6StructuredDecoded"
        decoded_staged = solution.operand_source == "DecodedWeightLdsBatch8"
        operand_source = solution.operand_source
        serialized_q6_schedule = SemanticSchedulePolicy(
            traversal=solution.q6_output_traversal,
            clustering=solution.q6_stage_clustering,
            latency=solution.q6_latency_policy,
            pressure=solution.q6_pressure_policy,
            wait=solution.q6_wait_policy,
            pairing=solution.q6_pairing_policy,
        )
        rejection = forward_kernel_spec_rejection_reason(solution)
        if rejection is not None:
            raise ValueError(rejection)
        tile_m, tile_n, tile_k, blocks = solution.matrix_instruction[:4]
        num_threads = solution.num_threads
        mi_wave_group = (num_threads // solution.wavefront_size, 1)
        ownership_divisors = (
            tile_m * mi_wave_group[0],
            tile_n * mi_wave_group[1],
        )
        mi_wave_tile = (
            solution.macro_tile0 // ownership_divisors[0],
            solution.macro_tile1 // ownership_divisors[1],
        )
        pipeline = None
        if structured_q6:
            pipeline = EpiloguePipelineSpec(
                tiles_ahead=None,
                dependency_width=solution.epilogue_dependency_width,
                priority=None,
                scope=solution.q6_epilogue_pipeline_scope,
            )
        elif decoded_staged:
            pipeline = EpiloguePipelineSpec(
                tiles_ahead=solution.epilogue_tiles_ahead,
                dependency_width=solution.epilogue_dependency_width,
                priority=solution.epilogue_priority,
                scope=None,
            )
        return cls(
            geometry=GeometrySpec(
                work_group=solution.work_group,
                matrix_instruction=(tile_m, tile_n, tile_k, blocks),
                depth_u=solution.depth_u,
            ),
            ownership=OwnershipSpec(
                mi_wave_group=mi_wave_group,
                mi_wave_tile=mi_wave_tile,
            ),
            global_memory=GlobalMemorySpec(
                operand_source=operand_source,
                activation_addressing=solution.activation_addressing,
                global_read_cache_policy=(
                    solution.q6_global_read_cache_policy if structured_q6 else None
                ),
            ),
            lds=LdsSpec(address_hoist=solution.lds_address_hoist),
            decode=DecodeSpec(
                metadata_conversion=solution.metadata_conversion,
                metadata_schedule=(
                    solution.metadata_schedule if decoded_staged else None
                ),
            ),
            epilogue=EpilogueSpec(
                output_store=solution.output_store,
                pipeline=pipeline,
            ),
            instruction_policy=InstructionPolicy(
                accumulator_initialization=(
                    solution.accumulator_initialization if decoded_staged else None
                ),
                dependency_delay_mode=(
                    solution.q6_dependency_delay_mode if structured_q6 else None
                ),
            ),
            semantic_schedule=(
                serialized_q6_schedule
                if structured_q6
                else SemanticSchedulePolicy.inactive()
            ),
            resource_limits=ResourceLimits(),
        )

    @property
    def macro_tile(self) -> tuple[int, int]:
        tile_m, tile_n, _, _ = self.geometry.matrix_instruction
        return (
            tile_m * self.ownership.mi_wave_group[0] * self.ownership.mi_wave_tile[0],
            tile_n * self.ownership.mi_wave_group[1] * self.ownership.mi_wave_tile[1],
        )

    def to_solution(self, contract: ForwardProblemContract) -> ForwardSolution:
        """Reconstruct the normal serialized build input from a canonical spec."""
        if self.resource_limits != ResourceLimits():
            raise ValueError("forward resource limits are a fixed gfx1151 contract")
        source = self.global_memory.operand_source
        decoded_staged = source == "DecodedWeightLdsBatch8"
        structured_q6 = source == "Q6StructuredDecoded"
        if source not in {"Global", "DecodedWeightLdsBatch8", "Q6StructuredDecoded"}:
            raise ValueError(f"unsupported forward operand source {source!r}")
        expected_semantic_schedule = (
            SemanticSchedulePolicy.structured_q6()
            if structured_q6
            else SemanticSchedulePolicy.inactive()
        )
        if self.semantic_schedule != expected_semantic_schedule:
            raise ValueError(
                "forward semantic schedule does not match the lowering family"
            )
        expected_quant_types = (
            {"Q6_K"}
            if structured_q6
            else {"Q4_K", "Q5_K"}
            if decoded_staged
            else {"Q4_K"}
        )
        if contract.quant_type not in expected_quant_types:
            raise ValueError("forward contract does not match the lowering family")
        pipeline = self.epilogue.pipeline
        if (structured_q6 or decoded_staged) != (pipeline is not None):
            raise ValueError(
                "forward epilogue pipeline does not match the lowering family"
            )
        legacy_suffix = (
            (1, 1, 4, 4, 1) if structured_q6 or decoded_staged else (1, 1, 1, 1, 1)
        )
        metadata_schedule = "Serialized"
        epilogue_tiles_ahead = 8
        epilogue_dependency_width = 1
        epilogue_priority = 0
        accumulator_initialization = "ScalarCopy"
        q6_epilogue_pipeline_scope = "StoreBatch"
        q6_dependency_delay_mode = "None"
        q6_global_read_cache_policy = "Default"
        q6_output_traversal = "OutputRoleGroupMajor"
        q6_stage_clustering = "StageDependencyOrder"
        q6_latency_policy = "SerializedDependencyDistance"
        q6_pressure_policy = "ExplicitRoleLifetime"
        q6_wait_policy = "ProducerFirstUse"
        q6_pairing_policy = "DependencyCompatibleDualIssue"
        if decoded_staged:
            assert pipeline is not None
            if pipeline.tiles_ahead is None or pipeline.priority is None:
                raise ValueError("decoded-weight LDS requires complete epilogue policy")
            if self.decode.metadata_schedule is None:
                raise ValueError("decoded-weight LDS requires a metadata schedule")
            if self.instruction_policy.accumulator_initialization is None:
                raise ValueError(
                    "decoded-weight LDS requires accumulator initialization"
                )
            metadata_schedule = self.decode.metadata_schedule
            epilogue_tiles_ahead = pipeline.tiles_ahead
            epilogue_dependency_width = pipeline.dependency_width
            epilogue_priority = pipeline.priority
            accumulator_initialization = (
                self.instruction_policy.accumulator_initialization
            )
        elif structured_q6:
            assert pipeline is not None
            if pipeline.scope is None:
                raise ValueError("structured Q6 requires an epilogue scope")
            if self.instruction_policy.dependency_delay_mode is None:
                raise ValueError("structured Q6 requires a dependency-delay mode")
            if self.global_memory.global_read_cache_policy is None:
                raise ValueError("structured Q6 requires a global-read cache policy")
            epilogue_dependency_width = pipeline.dependency_width
            q6_epilogue_pipeline_scope = pipeline.scope
            q6_dependency_delay_mode = self.instruction_policy.dependency_delay_mode
            q6_global_read_cache_policy = self.global_memory.global_read_cache_policy
            q6_output_traversal = cast(str, self.semantic_schedule.traversal)
            q6_stage_clustering = cast(str, self.semantic_schedule.clustering)
            q6_latency_policy = cast(str, self.semantic_schedule.latency)
            q6_pressure_policy = cast(str, self.semantic_schedule.pressure)
            q6_wait_policy = cast(str, self.semantic_schedule.wait)
            q6_pairing_policy = cast(str, self.semantic_schedule.pairing)
        solution = ForwardSolution(
            kernel_language=contract.kernel_language,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            work_group=self.geometry.work_group,
            matrix_instruction=(*self.geometry.matrix_instruction, *legacy_suffix),
            macro_tile0=self.macro_tile[0],
            macro_tile1=self.macro_tile[1],
            depth_u=self.geometry.depth_u,
            activation_layout=contract.activation_layout,
            activation_block_bytes=contract.activation_block_bytes,
            packed_weight_block_bytes=contract.packed_weight_block_bytes,
            operand_source=source,
            weight_decode=contract.weight_decode,
            lds_address_hoist=self.lds.address_hoist,
            activation_addressing=self.global_memory.activation_addressing,
            metadata_conversion=self.decode.metadata_conversion,
            scale_arithmetic=contract.scale_arithmetic,
            output_store=self.epilogue.output_store,
            signed_weight=contract.signed_weight,
            signed_activation=contract.signed_activation,
            wmma_clamp=contract.wmma_clamp,
            metadata_schedule=metadata_schedule,
            epilogue_tiles_ahead=epilogue_tiles_ahead,
            epilogue_dependency_width=epilogue_dependency_width,
            epilogue_priority=epilogue_priority,
            accumulator_initialization=accumulator_initialization,
            q6_epilogue_pipeline_scope=q6_epilogue_pipeline_scope,
            q6_dependency_delay_mode=q6_dependency_delay_mode,
            q6_global_read_cache_policy=q6_global_read_cache_policy,
            q6_output_traversal=q6_output_traversal,
            q6_stage_clustering=q6_stage_clustering,
            q6_latency_policy=q6_latency_policy,
            q6_pressure_policy=q6_pressure_policy,
            q6_wait_policy=q6_wait_policy,
            q6_pairing_policy=q6_pairing_policy,
        )
        if (
            ForwardProblemContract.from_solution(contract.quant_type, solution)
            != contract
        ):
            raise ValueError("forward fixed contract cannot be represented by the ABI")
        if ForwardKernelSpec.from_solution(solution) != self:
            raise ValueError("forward kernel spec cannot be represented by the ABI")
        return solution

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Q6LdsPairRole:
    """One formula-derived LDS pair and its semantic stage lifetime."""

    name: str
    pair: int
    offset0: int
    offset1: int
    first_stage: int
    last_stage: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.pair < 0
            or min(self.offset0, self.offset1, self.first_stage) < 0
            or self.last_stage < self.first_stage
        ):
            raise ValueError("invalid Q6 LDS pair role")


@dataclass(frozen=True)
class Q6LdsLayout:
    """Formula-derived Q6 LDS planes and stage offsets."""

    output_rows_per_wave: int

    def __post_init__(self) -> None:
        if self.output_rows_per_wave <= 0:
            raise ValueError("Q6 LDS output rows per wave must be positive")

    @property
    def stage_stride_bytes(self) -> int:
        return 9_472 * self.output_rows_per_wave

    @property
    def total_bytes(self) -> int:
        return 19_456 + self.stage_stride_bytes

    @property
    def decoded_plane_base(self) -> int:
        return self.stage_stride_bytes + 256

    @property
    def scale_read_base(self) -> int:
        return self.stage_stride_bytes + 260

    @property
    def first_address_offset(self) -> int:
        return 2_320 + 256 * self.output_rows_per_wave

    @property
    def direct_read_offset(self) -> int:
        return 32 * self.output_rows_per_wave + 2

    @property
    def first_stage_write_offset(self) -> int:
        return 37 * self.output_rows_per_wave - 4

    @property
    def stage_read_offsets(self) -> tuple[int, ...]:
        return (
            64 * (self.output_rows_per_wave - 1),
            48,
            160 - 64 * self.output_rows_per_wave,
            64 * self.output_rows_per_wave - 48,
        )

    @property
    def read_pair_plane_stride(self) -> int:
        return 152

    @property
    def stage_read_roles(self) -> tuple[Q6LdsPairRole, ...]:
        return tuple(
            Q6LdsPairRole(
                name=f"stage_read.{index}",
                pair=index,
                offset0=offset,
                offset1=offset + self.read_pair_plane_stride,
                first_stage=4,
                last_stage=5,
            )
            for index, offset in enumerate(self.stage_read_offsets)
        )

    def cooperative_write_role(self, pair: int) -> Q6LdsPairRole:
        if pair < 0:
            raise ValueError("Q6 cooperative-write pair must be nonnegative")
        first = self.output_rows_per_wave + 4 * pair
        return Q6LdsPairRole(
            name=f"cooperative_write.{pair}",
            pair=pair,
            offset0=first,
            offset1=first + 2,
            first_stage=6,
            last_stage=7,
        )

    def factor_role(self, pair: int) -> Q6LdsPairRole:
        if pair < 0:
            raise ValueError("Q6 factor pair must be nonnegative")
        return Q6LdsPairRole(
            name=f"factor.{pair}",
            pair=pair,
            offset0=self.output_rows_per_wave + 18 * pair,
            offset1=self.output_rows_per_wave + 9 + 18 * pair,
            first_stage=5,
            last_stage=7,
        )


Q6SemanticStageKind = Literal[
    "Setup",
    "GlobalRead",
    "Decode",
    "LocalWrite",
    "BarrierLocalRead",
    "Dot",
    "Refill",
    "Epilogue",
]


@dataclass(frozen=True)
class Q6SemanticStage:
    """One named stage and its explicit dependencies in the Q6 pipeline."""

    kind: Q6SemanticStageKind
    dependencies: tuple[int, ...]
    phase: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "Dot":
            if self.phase is None or self.phase < 0:
                raise ValueError("Q6 dot stage requires a nonnegative phase")
        elif self.phase is not None:
            raise ValueError("only Q6 dot stages may carry a phase")


@dataclass(frozen=True)
class Q6SemanticPlan:
    """Dependency-checked first-level stage plan for one Q6 lowering."""

    stages: tuple[Q6SemanticStage, ...]

    @classmethod
    def from_schedule(cls, schedule: "Q6ForwardSchedule") -> "Q6SemanticPlan":
        if schedule.semantic_policy.clustering != "StageDependencyOrder":
            raise ValueError("unsupported Q6 semantic-stage clustering policy")
        if len(schedule.dot_register_shifts) != 2:
            raise ValueError("structured Q6 currently requires exactly two dot phases")
        stages = (
            Q6SemanticStage("Setup", ()),
            Q6SemanticStage("GlobalRead", (0,)),
            Q6SemanticStage("Decode", (1,)),
            Q6SemanticStage("LocalWrite", (2,)),
            Q6SemanticStage("BarrierLocalRead", (3,)),
            Q6SemanticStage("Dot", (4,), phase=0),
            Q6SemanticStage("Refill", (5,)),
            Q6SemanticStage("Dot", (6,), phase=1),
            Q6SemanticStage("Epilogue", (7,)),
        )
        plan = cls(stages)
        plan.validate()
        return plan

    def validate(self) -> None:
        for index, stage in enumerate(self.stages):
            if any(
                dependency < 0 or dependency >= index
                for dependency in stage.dependencies
            ):
                raise ValueError(
                    "Q6 semantic stage dependency must precede its consumer"
                )


@dataclass(frozen=True)
class Q6DotPhase:
    """One derived K phase in a structured Q6 lowering."""

    schedule: "Q6ForwardSchedule"
    phase: int

    @property
    def register_shift(self) -> int:
        return self.schedule.dot_register_shifts[self.phase]


@dataclass(frozen=True)
class Q6ForwardSchedule:
    """Complete parameter-only state for one structured Q6 lowering."""

    matrix_instruction: tuple[int, int, int, int]
    mi_wave_group: tuple[int, int]
    mi_wave_tile: tuple[int, int]
    semantic_policy: SemanticSchedulePolicy
    epilogue_dependency_width: int
    epilogue_pipeline_scope: Literal["StoreBatch", "FullTile"]
    dot_register_shifts: tuple[int, ...]
    dependency_delay_mode: Literal["None", "Explicit"]
    global_read_cache_policy: Literal["Default", "InvalidateL0"]

    def __post_init__(self) -> None:
        self.semantic_policy.require_structured_q6()
        _, _, _, blocks = self.matrix_instruction
        if blocks != 1:
            raise ValueError("Q6 matrix instruction implements one input block")
        if not self.dot_register_shifts:
            raise ValueError("Q6 requires at least one dot phase")
        if (
            self.epilogue_dependency_width not in (1, 2, 4, 8)
            or self.store_vector_width % self.epilogue_dependency_width
        ):
            raise ValueError(
                "unsupported Q6 epilogue dependency width: "
                f"{self.epilogue_dependency_width}"
            )
        if self.epilogue_pipeline_scope not in {"StoreBatch", "FullTile"}:
            raise ValueError(
                "unsupported Q6 epilogue pipeline scope: "
                f"{self.epilogue_pipeline_scope}"
            )
        if self.dependency_delay_mode not in ("None", "Explicit"):
            raise ValueError(
                f"unsupported Q6 dependency-delay mode: {self.dependency_delay_mode}"
            )
        if self.global_read_cache_policy not in ("Default", "InvalidateL0"):
            raise ValueError(
                "unsupported Q6 global-read cache policy: "
                f"{self.global_read_cache_policy}"
            )

    @property
    def macro_tile(self) -> tuple[int, int]:
        tile_m, tile_n, _, _ = self.matrix_instruction
        return (
            tile_m * self.mi_wave_group[0] * self.mi_wave_tile[0],
            tile_n * self.mi_wave_group[1] * self.mi_wave_tile[1],
        )

    @property
    def macro_tile0(self) -> int:
        return self.macro_tile[0]

    @property
    def work_group(self) -> tuple[int, int, int]:
        return (32, self.mi_wave_group[0] * self.mi_wave_group[1], 1)

    @property
    def depth_u(self) -> int:
        return self.matrix_instruction[2] * len(self.dot_register_shifts)

    @property
    def local_read_vector_width(self) -> int:
        return 2

    @property
    def store_vector_width(self) -> int:
        return 8

    @property
    def wmma_opcode(self) -> str:
        tile_m, tile_n, tile_k, _ = self.matrix_instruction
        return f"v_wmma_i32_{tile_m}x{tile_n}x{tile_k}_iu8"

    @property
    def resource_usage(self) -> ForwardResourceUsage:
        return structured_q6_resource_usage(self.mi_wave_tile[0])

    def phase(self, phase: int) -> Q6DotPhase:
        if phase not in range(len(self.dot_register_shifts)):
            raise ValueError(f"unsupported Q6 dot phase: {phase}")
        return Q6DotPhase(self, phase)


def q6_schedule_from_kernel_spec(spec: ForwardKernelSpec) -> Q6ForwardSchedule:
    """Derive structured Q6 lowering state from one canonical kernel spec."""
    if spec.global_memory.operand_source != "Q6StructuredDecoded":
        raise ValueError("Q6 schedule requires structured decoded operands")
    if spec.geometry.work_group[0] != 32 or spec.geometry.work_group[2] != 1:
        raise ValueError("structured Q6 requires WorkGroup=(32,waves,1)")
    if spec.ownership.mi_wave_group[0] * spec.ownership.mi_wave_group[1] <= 0:
        raise ValueError("structured Q6 requires at least one wave")
    pipeline = spec.epilogue.pipeline
    if pipeline is None or pipeline.scope is None:
        raise ValueError("structured Q6 requires an epilogue pipeline")
    delay_mode = spec.instruction_policy.dependency_delay_mode
    cache_policy = spec.global_memory.global_read_cache_policy
    if delay_mode is None or cache_policy is None:
        raise ValueError("structured Q6 requires delay and cache policies")
    _, _, tile_k, _ = spec.geometry.matrix_instruction
    return Q6ForwardSchedule(
        matrix_instruction=spec.geometry.matrix_instruction,
        mi_wave_group=spec.ownership.mi_wave_group,
        mi_wave_tile=spec.ownership.mi_wave_tile,
        semantic_policy=spec.semantic_schedule,
        epilogue_dependency_width=pipeline.dependency_width,
        epilogue_pipeline_scope=cast(Literal["StoreBatch", "FullTile"], pipeline.scope),
        dot_register_shifts=tuple(range(spec.geometry.depth_u // tile_k)),
        dependency_delay_mode=cast(Literal["None", "Explicit"], delay_mode),
        global_read_cache_policy=cast(Literal["Default", "InvalidateL0"], cache_policy),
    )


def q6_schedule_from_solution(solution: ForwardSolution) -> Q6ForwardSchedule:
    """Convert the legacy serialized solution and derive its Q6 schedule."""
    if solution.operand_source != "Q6StructuredDecoded":
        raise ValueError("Q6 schedule requires structured decoded operands")
    if solution.work_group[0] != 32 or solution.work_group[2] != 1:
        raise ValueError("structured Q6 requires WorkGroup=(32,waves,1)")
    wave_count = solution.num_threads // solution.wavefront_size
    if wave_count <= 0:
        raise ValueError("structured Q6 requires at least one wave")
    kernel_spec = ForwardKernelSpec.from_solution(solution)
    return q6_schedule_from_kernel_spec(kernel_spec)


@dataclass(frozen=True)
class ForwardKernelCandidate:
    """Canonical complete candidate independent of any exact problem shape."""

    problem_contract: ForwardProblemContract
    kernel_spec: ForwardKernelSpec

    @classmethod
    def from_solution(
        cls,
        quant_type: str,
        solution: ForwardSolution,
    ) -> "ForwardKernelCandidate":
        return cls(
            problem_contract=ForwardProblemContract.from_solution(
                quant_type,
                solution,
            ),
            kernel_spec=ForwardKernelSpec.from_solution(solution),
        )

    def to_solution(self) -> ForwardSolution:
        return self.kernel_spec.to_solution(self.problem_contract)

    def to_mapping(self) -> dict[str, object]:
        return {
            "SchemaVersion": 1,
            "ProblemContract": self.problem_contract.to_mapping(),
            "KernelSpec": self.kernel_spec.to_mapping(),
        }


@dataclass(frozen=True)
class DerivedForwardState:
    """All formula-derived values consumed by forward lowering and runtime."""

    problem_size: ProblemSize
    contract: ForwardProblemContract
    semantics: QuantForwardSemantics
    kernel_spec: ForwardKernelSpec
    decoded_lds: DecodedLdsLayout | None
    num_threads: int
    waves_per_workgroup: int
    mi_wave_group: tuple[int, int]
    mi_wave_tile: tuple[int, int]
    accumulator_count: int
    k_phases_per_iteration: int
    resources: ForwardResourceUsage
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    activation_plane_stride_bytes: int
    activation_weight_block_stride_bytes: int
    grid: tuple[int, int, int]

    @classmethod
    def from_solution_key(cls, key: SolutionKey) -> "DerivedForwardState":
        if not isinstance(key.solution, ForwardSolution):
            raise TypeError("MMQ forward derived state requires ForwardSolution")
        solution = key.solution
        size = key.problem_size
        contract = ForwardProblemContract.from_solution(
            key.problem_type.quant_data_type,
            solution,
        )
        semantics = QuantForwardSemantics.for_quant_type(contract.quant_type)
        payload_bytes = max(
            plane.byte_offset + plane.byte_count for plane in semantics.payload_planes
        )
        if payload_bytes != contract.packed_weight_block_bytes:
            raise ValueError("forward payload planes do not cover the packed block")
        kernel_spec = ForwardKernelSpec.from_solution(solution)
        decoded_lds = (
            DecodedLdsLayout.for_contract(contract)
            if kernel_spec.global_memory.operand_source == "DecodedWeightLdsBatch8"
            else None
        )
        resources = derive_forward_resource_usage(kernel_spec)
        resources.admit(kernel_spec.resource_limits)
        geometry = kernel_spec.geometry
        macro_tile_m, macro_tile_n = kernel_spec.macro_tile
        num_threads = solution.num_threads
        waves_per_workgroup = num_threads // contract.wavefront_size
        _, _, tile_k, _ = geometry.matrix_instruction
        mi_wave_group = kernel_spec.ownership.mi_wave_group
        mi_wave_tile = kernel_spec.ownership.mi_wave_tile
        for name, value, divisor in (
            ("M", size.m, macro_tile_m),
            ("N", size.n, macro_tile_n),
            ("K", size.k, contract.block_values),
        ):
            if value <= 0 or divisor <= 0 or value % divisor:
                raise ValueError(
                    f"forward {name} must be a positive multiple of {divisor}"
                )
        blocks_per_weight_row = size.k // contract.block_values
        activation_blocks_per_row = size.k // 128
        activation_plane_stride = size.m * contract.activation_block_bytes
        return cls(
            problem_size=size,
            contract=contract,
            semantics=semantics,
            kernel_spec=kernel_spec,
            decoded_lds=decoded_lds,
            num_threads=num_threads,
            waves_per_workgroup=waves_per_workgroup,
            mi_wave_group=mi_wave_group,
            mi_wave_tile=mi_wave_tile,
            accumulator_count=8 * mi_wave_tile[0] * mi_wave_tile[1],
            k_phases_per_iteration=geometry.depth_u // tile_k,
            resources=resources,
            blocks_per_weight_row=blocks_per_weight_row,
            activation_blocks_per_row=activation_blocks_per_row,
            packed_weight_row_bytes=(
                blocks_per_weight_row * contract.packed_weight_block_bytes
            ),
            activation_plane_stride_bytes=activation_plane_stride,
            activation_weight_block_stride_bytes=2 * activation_plane_stride,
            grid=(size.n // macro_tile_n, size.m // macro_tile_m, 1),
        )

    @property
    def lds_bytes(self) -> int:
        return self.resources.lds_bytes

    @property
    def expected_packed_weight_bytes(self) -> int:
        return self.problem_size.n * self.packed_weight_row_bytes

    @property
    def expected_activation_shape(self) -> tuple[int, int, int]:
        return (
            self.activation_blocks_per_row,
            self.problem_size.m,
            self.contract.activation_block_bytes,
        )

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        return (self.problem_size.m, self.problem_size.n)
