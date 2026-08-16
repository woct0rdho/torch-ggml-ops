"""Typed problem contracts, kernel specifications, and derived MMQ forward state."""

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from .model import ForwardSolution, ProblemSize, SolutionKey
from .quant_formats import (
    Q8_1_D4_BLOCK_VALUES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)

if TYPE_CHECKING:
    from .mmq_fwd_physical import ForwardPhysicalPlan


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
class Q3PackedFieldPart:
    """One byte-local field contributing to a signed Q3_K group scale."""

    source_byte: int
    bit_offset: int
    bit_count: int
    destination_shift: int = 0


@dataclass(frozen=True)
class Q3PayloadGroupSpec:
    """Packed payload and activation offsets for one 16-value Q3_K group."""

    group: int
    half: int
    low_payload_offset: int
    low_shift: int
    high_payload_offset: int
    high_shift: int
    activation_payload_offset: int
    activation_scale_offset: int


@dataclass(frozen=True)
class Q3SignedDecodeSpec:
    """Bit-exact packed Q3_K byte reconstruction and signed normalization."""

    low_mask: int
    high_mask: int
    high_destination_shift: int
    signed_add: int
    signed_xor: int

    def __post_init__(self) -> None:
        if (
            self.low_mask != 0x03030303
            or self.high_mask != 0x01010101
            or self.high_destination_shift != 2
            or self.signed_add != 0x7C7C7C7C
            or self.signed_xor != 0x80808080
        ):
            raise ValueError("unsupported Q3 signed decode formula")


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
        if quant_type == "IQ2_XXS":
            return cls(
                quant_type=quant_type,
                weight_bits=2,
                payload_planes=(
                    PayloadPlaneSpec("d", 0, 2, "Float16"),
                    PayloadPlaneSpec(
                        "grid_indices_and_signs",
                        2,
                        64,
                        "FourGridBytesThenParitySignsAndScale",
                    ),
                ),
                activation_components=("q", "d"),
                post_wmma_correction="IQ2XXSGroupScaleTimesActivationScale",
            )
        if quant_type == "IQ2_S":
            return cls(
                quant_type=quant_type,
                weight_bits=2,
                payload_planes=(
                    PayloadPlaneSpec("d", 0, 2, "Float16"),
                    PayloadPlaneSpec("grid_indices", 2, 32, "UnsignedInt8"),
                    PayloadPlaneSpec("signs", 34, 32, "SignBitMask"),
                    PayloadPlaneSpec("qh", 66, 8, "PackedTwoBitHigh"),
                    PayloadPlaneSpec("scales", 74, 8, "PackedUnsigned4"),
                ),
                activation_components=("q", "d"),
                post_wmma_correction="IQ2SGroupScaleTimesBlockFactors",
            )
        if quant_type == "Q2_K":
            return cls(
                quant_type=quant_type,
                weight_bits=2,
                payload_planes=(
                    PayloadPlaneSpec("scales", 0, 16, "PackedScaleMinimum4"),
                    PayloadPlaneSpec("qs", 16, 64, "UnsignedTwoBit"),
                    PayloadPlaneSpec("dm", 80, 4, "Float16Pair"),
                ),
                activation_components=("q", "d", "s"),
                post_wmma_correction="Q2ScaleAndMinimum",
            )
        if quant_type == "Q3_K":
            return cls(
                quant_type=quant_type,
                weight_bits=3,
                payload_planes=(
                    PayloadPlaneSpec("hmask", 0, 32, "HighBitMask"),
                    PayloadPlaneSpec("qs", 32, 64, "UnsignedTwoBit"),
                    PayloadPlaneSpec("scales", 96, 12, "PackedSigned6"),
                    PayloadPlaneSpec("d", 108, 2, "Float16"),
                ),
                activation_components=("q", "d"),
                post_wmma_correction="SignedGroupScaleTimesBlockFactors",
            )
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
        if quant_type == "Q8_0":
            return cls(
                quant_type=quant_type,
                weight_bits=8,
                payload_planes=(
                    PayloadPlaneSpec("d", 0, 2, "Float16"),
                    PayloadPlaneSpec("qs", 2, 32, "SignedInt8"),
                ),
                activation_components=("q", "d"),
                post_wmma_correction="SignedScaleTimesActivationScale",
            )
        raise ValueError(f"unsupported MMQ forward quant type {quant_type!r}")

    def q3_payload_group(self, group: int) -> Q3PayloadGroupSpec:
        if self.quant_type != "Q3_K":
            raise ValueError(f"{self.quant_type} has no Q3 payload groups")
        if group not in range(16):
            raise ValueError("Q3 payload group must be in range 0..15")
        half = group // 8
        local_group = group % 8
        return Q3PayloadGroupSpec(
            group=group,
            half=half,
            low_payload_offset=(
                self.payload_plane("qs").byte_offset
                + 32 * half
                + 16 * (local_group % 2)
            ),
            low_shift=2 * (local_group // 2),
            high_payload_offset=(
                self.payload_plane("hmask").byte_offset + 16 * (local_group % 2)
            ),
            high_shift=group // 2,
            activation_payload_offset=16 + 16 * local_group,
            activation_scale_offset=4 * (local_group // 2),
        )

    def q3_scale_fields(
        self,
        group: int,
    ) -> tuple[Q3PackedFieldPart, Q3PackedFieldPart]:
        if self.quant_type != "Q3_K":
            raise ValueError(f"{self.quant_type} has no Q3 scale fields")
        if group not in range(16):
            raise ValueError("Q3 scale group must be in range 0..15")
        return (
            Q3PackedFieldPart(group % 8, 4 * (group // 8), 4),
            Q3PackedFieldPart(8 + group % 4, 2 * (group // 4), 2, 4),
        )

    def q3_signed_decode(self) -> Q3SignedDecodeSpec:
        if self.quant_type != "Q3_K":
            raise ValueError(f"{self.quant_type} has no signed Q3 decode")
        return Q3SignedDecodeSpec(
            low_mask=0x03030303,
            high_mask=0x01010101,
            high_destination_shift=2,
            signed_add=0x7C7C7C7C,
            signed_xor=0x80808080,
        )

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
        try:
            traits = QUANT_FORMATS[quant_type]
        except KeyError:
            raise ValueError(f"unsupported MMQ forward quant type {quant_type!r}")
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
                solution.packed_weight_block_bytes == traits.block_bytes,
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
                solution.wmma_clamp == traits.wmma_clamp,
                "forward WMMA clamp does not match the quant arithmetic contract",
            ),
            (
                solution.weight_decode == traits.weight_decode,
                "forward weight decode does not match the quant-format contract",
            ),
            (
                solution.scale_arithmetic == traits.scale_arithmetic,
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
        try:
            traits = QUANT_FORMATS[quant_type]
        except KeyError:
            raise ValueError(f"unsupported MMQ forward quant type {quant_type!r}")
        rejection = cls.rejection_reason(quant_type, solution)
        if rejection is not None:
            raise ValueError(rejection)
        return cls(
            quant_type=quant_type,
            block_values=traits.block_values,
            packed_weight_block_bytes=traits.block_bytes,
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
            arithmetic_contract=traits.arithmetic_contract,
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
class F16D2S6ActivationMetadata:
    """Physical metadata for Q2_K's two-scale, six-sum workspace."""

    block_bytes: int

    GROUP_COUNT: ClassVar[int] = 8
    PAYLOAD_BASE: ClassVar[int] = 16
    GROUP_PAYLOAD_BYTES: ClassVar[int] = 16
    PAYLOAD_VECTOR_BYTES: ClassVar[int] = 16

    def __post_init__(self) -> None:
        if self.block_bytes <= 0:
            raise ValueError("F16_D2S6 activation block bytes must be positive")

    def payload_offset(self, group: int) -> int:
        if group not in range(self.GROUP_COUNT):
            raise ValueError("F16_D2S6 activation group must be in range 0..7")
        return self.PAYLOAD_BASE + self.GROUP_PAYLOAD_BYTES * group

    def scale_offset(self, group: int) -> int:
        if group not in range(self.GROUP_COUNT):
            raise ValueError("F16_D2S6 activation group must be in range 0..7")
        return 2 * (group // 4)

    def sum_offset(self, group: int) -> int | None:
        if group not in range(self.GROUP_COUNT):
            raise ValueError("F16_D2S6 activation group must be in range 0..7")
        return 4 + 2 * group if group < 6 else None


@dataclass(frozen=True)
class F16D4S4ActivationGroupRole:
    """One 32-value MMA group in the two-plane F16_D4S4 workspace."""

    index: int
    plane: int
    payload_offset: int
    payload_high_offset: int
    scale_sum_offset: int


@dataclass(frozen=True)
class F16D4S4ActivationMetadata:
    """Physical metadata for packed scale/minimum activation operands."""

    block_bytes: int

    GROUP_COUNT: ClassVar[int] = 8
    GROUPS_PER_PLANE: ClassVar[int] = 4
    PAYLOAD_BASE: ClassVar[int] = 16
    PAYLOAD_VECTOR_BYTES: ClassVar[int] = 16
    GROUP_PAYLOAD_BYTES: ClassVar[int] = 32
    GROUP_SCALE_SUM_BYTES: ClassVar[int] = 4

    def __post_init__(self) -> None:
        if self.block_bytes <= 0:
            raise ValueError("F16_D4S4 activation block bytes must be positive")

    @classmethod
    def for_contract(
        cls,
        contract: ForwardProblemContract,
    ) -> "F16D4S4ActivationMetadata":
        if contract.activation_layout != "F16_D4S4":
            raise ValueError("activation metadata requires the F16_D4S4 workspace")
        return cls(contract.activation_block_bytes)

    def group(self, index: int) -> F16D4S4ActivationGroupRole:
        if index not in range(self.GROUP_COUNT):
            raise ValueError("F16_D4S4 activation group must be in range 0..7")
        local_group = index % self.GROUPS_PER_PLANE
        payload_offset = self.PAYLOAD_BASE + self.GROUP_PAYLOAD_BYTES * local_group
        return F16D4S4ActivationGroupRole(
            index=index,
            plane=index // self.GROUPS_PER_PLANE,
            payload_offset=payload_offset,
            payload_high_offset=payload_offset + self.PAYLOAD_VECTOR_BYTES,
            scale_sum_offset=self.GROUP_SCALE_SUM_BYTES * local_group,
        )


@dataclass(frozen=True)
class DecodedLdsLayout:
    """Derived shared LDS planes for the retained Q4/Q5 decoded path."""

    activation_metadata: F16D2S6ActivationMetadata | F16D4S4ActivationMetadata
    activation_base: int
    weight_data_base: int
    weight_metadata_base: int
    weight_row_stride: int
    activation_lane_stride: int

    @classmethod
    def for_activation_block_bytes(
        cls,
        activation_block_bytes: int,
        activation_rows: int = 128,
    ) -> "DecodedLdsLayout":
        if activation_block_bytes <= 0:
            raise ValueError("decoded LDS layout requires a positive activation block")
        if activation_rows not in (64, 128):
            raise ValueError("decoded LDS layout requires 64 or 128 activation rows")
        activation_metadata = F16D4S4ActivationMetadata(activation_block_bytes)
        weight_row_stride = 2 * activation_block_bytes + 16
        activation_base = 512
        weight_data_base = activation_base + activation_rows * activation_block_bytes
        return cls(
            activation_metadata=activation_metadata,
            activation_base=activation_base,
            weight_data_base=weight_data_base,
            weight_metadata_base=weight_data_base + 256,
            weight_row_stride=weight_row_stride,
            activation_lane_stride=512,
        )

    @classmethod
    def for_q2_activation_block_bytes(
        cls,
        activation_block_bytes: int,
        activation_rows: int = 32,
    ) -> "DecodedLdsLayout":
        if activation_block_bytes <= 0:
            raise ValueError(
                "Q2 decoded LDS layout requires a positive activation block"
            )
        if activation_rows not in (32, 64, 128):
            raise ValueError(
                "Q2 decoded LDS layout requires 32, 64, or 128 activation rows"
            )
        activation_metadata = F16D2S6ActivationMetadata(activation_block_bytes)
        weight_row_stride = 320
        activation_base = 512
        weight_data_base = activation_base + activation_rows * activation_block_bytes
        return cls(
            activation_metadata=activation_metadata,
            activation_base=activation_base,
            weight_data_base=weight_data_base,
            weight_metadata_base=weight_data_base + 256,
            weight_row_stride=weight_row_stride,
            activation_lane_stride=512,
        )

    @classmethod
    def for_contract(cls, contract: ForwardProblemContract) -> "DecodedLdsLayout":
        if contract.activation_layout != "F16_D4S4":
            raise ValueError("decoded LDS layout requires the F16_D4S4 workspace")
        return cls.for_activation_block_bytes(contract.activation_block_bytes)

    @property
    def metadata_row_stride(self) -> int:
        return self.weight_row_stride

    @property
    def metadata_group_stride(self) -> int:
        return 4 * self.metadata_row_stride

    @property
    def total_bytes(self) -> int:
        return self.weight_data_base + 64 * self.weight_row_stride


@dataclass(frozen=True)
class Packed3BitTiledLdsLayout:
    """Formula-derived LDS planes for one 128-value Q3_K half block."""

    activation_rows: int = 128
    weight_rows: int = 64
    activation_row_stride: int = Q8_1_F32_D4_BLOCK_BYTES
    decoded_weight_values: int = 128
    scale_count: int = 8

    def __post_init__(self) -> None:
        if (
            min(
                self.activation_rows,
                self.weight_rows,
                self.activation_row_stride,
                self.decoded_weight_values,
                self.scale_count,
            )
            <= 0
        ):
            raise ValueError("Q3 half-tile LDS dimensions must be positive")

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_payload_bytes(self) -> int:
        return self.decoded_weight_values

    @property
    def weight_scale_bytes(self) -> int:
        return 4 * self.scale_count

    @property
    def weight_row_stride(self) -> int:
        return self.weight_payload_bytes + self.weight_scale_bytes

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


@dataclass(frozen=True)
class Q3FullWeightTiledLdsLayout:
    """Formula-derived LDS planes for the full 256-value Q3_K tile."""

    activation_rows: int = 128
    weight_rows: int = 64
    activation_row_stride: int = Q8_1_F32_D4_BLOCK_BYTES
    half_payload_bytes: int = 128
    half_payload_stride: int = 160
    weight_scale_offset: int = 128
    weight_scale_bytes: int = 32
    weight_row_stride: int = 336

    def __post_init__(self) -> None:
        if (
            self.activation_rows,
            self.weight_rows,
            self.activation_row_stride,
            self.half_payload_bytes,
            self.half_payload_stride,
            self.weight_scale_offset,
            self.weight_scale_bytes,
            self.weight_row_stride,
        ) != (128, 64, Q8_1_F32_D4_BLOCK_BYTES, 128, 160, 128, 32, 336):
            raise ValueError("Q3 full-weight LDS layout has fixed dimensions")
        if (
            self.weight_scale_offset + self.weight_scale_bytes
            > self.half_payload_stride
        ):
            raise ValueError("Q3 full-weight scale plane overlaps half payload stride")
        if 2 * self.half_payload_stride > self.weight_row_stride:
            raise ValueError("Q3 full-weight row does not fit its padded stride")

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_payload_bytes(self) -> int:
        return 2 * self.half_payload_bytes

    @property
    def weight_scale_total_bytes(self) -> int:
        return 2 * self.weight_scale_bytes

    @property
    def weight_padding_bytes(self) -> int:
        return self.weight_row_stride - 2 * self.half_payload_stride

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


@dataclass(frozen=True)
class SignedInt8SmallMTiledLdsLayout:
    """Formula-derived LDS planes for an exact M32 or M64 signed-int8 tile."""

    activation_rows: int
    weight_rows: int = 64
    activation_row_stride: int = Q8_1_F32_D4_BLOCK_BYTES
    weight_row_stride: int = 304
    weight_scale_offset: int = 256

    def __post_init__(self) -> None:
        if self.activation_rows not in (32, 64):
            raise ValueError("Q8 small-M LDS layout requires 32 or 64 rows")
        if (
            min(
                self.weight_rows,
                self.activation_row_stride,
                self.weight_row_stride,
                self.weight_scale_offset,
            )
            <= 0
        ):
            raise ValueError("Q8 small-M LDS dimensions must be positive")

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def weight_scale_element_stride(self) -> int:
        return 2 * self.weight_row_stride

    @property
    def weight_scale_pair_base_delta(self) -> int | None:
        return None

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


@dataclass(frozen=True)
class SignedInt8CompactDepth32TiledLdsLayout:
    """Formula-derived compact signed-int8 LDS planes for depth-32 tiles."""

    activation_rows: int
    weight_rows: int = 64
    activation_row_stride: int = Q8_1_F32_D4_BLOCK_BYTES
    weight_row_stride: int = 144
    weight_scale_offset: int = 128

    def __post_init__(self) -> None:
        if self.activation_rows not in (32, 64, 128):
            raise ValueError("Q8 compact depth-32 layout requires 32, 64, or 128 rows")
        if (
            self.weight_rows,
            self.activation_row_stride,
            self.weight_row_stride,
            self.weight_scale_offset,
        ) != (64, Q8_1_F32_D4_BLOCK_BYTES, 144, 128):
            raise ValueError("Q8 compact depth-32 layout has fixed row dimensions")

    @property
    def activation_bytes(self) -> int:
        return self.activation_rows * self.activation_row_stride

    @property
    def weight_base(self) -> int:
        return self.activation_bytes

    @property
    def weight_bytes(self) -> int:
        return self.weight_rows * self.weight_row_stride

    @property
    def weight_scale_element_stride(self) -> int:
        return 2 * self.weight_row_stride

    @property
    def weight_scale_pair_base_delta(self) -> int:
        return 4 * self.weight_scale_element_stride

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.weight_bytes


@dataclass(frozen=True)
class DecodeSpec:
    """Candidate-selectable metadata and traversal lowering policies."""

    metadata_conversion: str
    metadata_schedule: str | None

    @property
    def independent_metadata_extraction(self) -> bool:
        return self.metadata_schedule in (
            "IndependentExtraction",
            "IndependentExtractionMetadataAfterLowWmma",
        )

    @property
    def defer_metadata_reads(self) -> bool:
        return self.metadata_schedule in (
            "MetadataAfterLowWmma",
            "IndependentExtractionMetadataAfterLowWmma",
        )


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


def derive_forward_resource_usage(spec: "ForwardKernelSpec") -> ForwardResourceUsage:
    from .mmq_fwd_physical import derive_forward_physical_plan

    return derive_forward_physical_plan(spec).resources


@dataclass(frozen=True)
class InstructionPolicy:
    """Only active instruction-ordering policies not implied by dependencies."""

    accumulator_initialization: str | None
    dependency_delay_mode: str | None
    q6_physical_plan: str | None


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
    def structured_q6_wavefront(cls) -> "SemanticSchedulePolicy":
        """Return the typed row/role decode-wavefront policy."""
        return cls(
            traversal="OutputRoleWavefront",
            clustering="RowBatchedDecodeOrder",
            latency="WavefrontDependencyDistance",
            pressure="ExplicitRoleLifetime",
            wait="ProducerFirstUse",
            pairing="DependencyCompatibleDualIssue",
        )

    @classmethod
    def supported_structured_q6(cls) -> tuple["SemanticSchedulePolicy", ...]:
        return (cls.structured_q6(), cls.structured_q6_wavefront())

    @classmethod
    def inactive(cls) -> "SemanticSchedulePolicy":
        return cls(None, None, None, None, None, None)

    def require_structured_q6(self) -> None:
        if self not in self.supported_structured_q6():
            raise ValueError("unsupported structured-Q6 semantic schedule policy")


@dataclass(frozen=True)
class ForwardMechanismContract:
    """Data-contract capabilities implemented by one lowering mechanism."""

    lowering: Literal[
        "PackedScaleMinimumDirect",
        "DecodedWeightLds",
        "StructuredQ6",
        "Packed3BitTiledLds",
        "Packed3BitFullWeightTiledLds",
        "SignedInt8",
    ]
    activation_layout: str
    activation_block_bytes: int
    weight_block_values: int
    reduction_values: int
    wmma_clamp: bool
    weight_decodes: tuple[str, ...]
    scale_arithmetic: str
    arithmetic_contracts: tuple[str, ...]
    physical_plan: Literal[
        "PackedScaleMinimumDirect",
        "DecodedWeightLds",
        "StructuredQ6",
        "Packed3BitTiledLds",
        "Packed3BitFullWeightTiledLds",
        "SignedInt8Direct",
        "SignedInt8RegisterTiled",
        "SignedInt8WaveNTiledLds",
        "SignedInt8SmallMTiledLds",
    ]
    ownership: Literal["WaveM", "WaveN"] = "WaveM"
    uses_workitem_id: bool = False
    extended_legacy_matrix: bool = False

    def rejection_reason(self, contract: ForwardProblemContract) -> str | None:
        checks = (
            contract.activation_layout == self.activation_layout,
            contract.activation_block_bytes == self.activation_block_bytes,
            contract.block_values == self.weight_block_values,
            contract.wmma_clamp == self.wmma_clamp,
            contract.weight_decode in self.weight_decodes,
            contract.scale_arithmetic == self.scale_arithmetic,
            contract.arithmetic_contract in self.arithmetic_contracts,
        )
        if not all(checks):
            return "forward data contract does not match the lowering mechanism"
        return None


_PACKED_SCALE_MINIMUM_CONTRACT = ForwardMechanismContract(
    lowering="DecodedWeightLds",
    activation_layout="F16_D4S4",
    activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
    weight_block_values=256,
    reduction_values=2 * Q8_1_D4_BLOCK_VALUES,
    wmma_clamp=True,
    weight_decodes=("DirectNibble", "DirectNibbleHighBit"),
    scale_arithmetic="FP16",
    arithmetic_contracts=("SignedKQuantIntegerWmmaFP16ScaleMinimumCorrection",),
    physical_plan="DecodedWeightLds",
    extended_legacy_matrix=True,
)
_SIGNED_INT8_CONTRACT = ForwardMechanismContract(
    lowering="SignedInt8",
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    weight_block_values=32,
    reduction_values=Q8_1_D4_BLOCK_VALUES,
    wmma_clamp=False,
    weight_decodes=("DirectSignedInt8",),
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contracts=("SignedQ8Int8ScaleIntegerWmmaF32Correction",),
    physical_plan="SignedInt8Direct",
)
_FORWARD_MECHANISM_CONTRACTS = MappingProxyType(
    {
        "Global": replace(
            _PACKED_SCALE_MINIMUM_CONTRACT,
            lowering="PackedScaleMinimumDirect",
            weight_decodes=("DirectNibble",),
            physical_plan="PackedScaleMinimumDirect",
            extended_legacy_matrix=False,
        ),
        "DecodedWeightLdsBatch8": _PACKED_SCALE_MINIMUM_CONTRACT,
        "Q6StructuredDecoded": ForwardMechanismContract(
            lowering="StructuredQ6",
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            weight_block_values=256,
            reduction_values=2 * Q8_1_D4_BLOCK_VALUES,
            wmma_clamp=False,
            weight_decodes=("DirectQ6Signed",),
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contracts=("SignedQ6Int8ScaleIntegerWmmaF32Correction",),
            physical_plan="StructuredQ6",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
        "Q3HipTiledLds": ForwardMechanismContract(
            lowering="Packed3BitTiledLds",
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            weight_block_values=256,
            reduction_values=2 * Q8_1_D4_BLOCK_VALUES,
            wmma_clamp=False,
            weight_decodes=("DirectQ3Signed",),
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contracts=("SignedQ3Int8ScaleIntegerWmmaF32Correction",),
            physical_plan="Packed3BitTiledLds",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
        "Q3FullWeightTiledLds": ForwardMechanismContract(
            lowering="Packed3BitFullWeightTiledLds",
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            weight_block_values=256,
            reduction_values=2 * Q8_1_D4_BLOCK_VALUES,
            wmma_clamp=False,
            weight_decodes=("DirectQ3Signed",),
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contracts=("SignedQ3Int8ScaleIntegerWmmaF32Correction",),
            physical_plan="Packed3BitFullWeightTiledLds",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
        "Q8DirectGlobal": _SIGNED_INT8_CONTRACT,
        "Q8RegisterTiled": replace(
            _SIGNED_INT8_CONTRACT,
            physical_plan="SignedInt8RegisterTiled",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
        "Q8HipTiledLds": replace(
            _SIGNED_INT8_CONTRACT,
            physical_plan="SignedInt8WaveNTiledLds",
            ownership="WaveN",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
        "Q8SmallMTiledLds": replace(
            _SIGNED_INT8_CONTRACT,
            physical_plan="SignedInt8SmallMTiledLds",
            ownership="WaveN",
            uses_workitem_id=True,
            extended_legacy_matrix=True,
        ),
    }
)


def forward_mechanism_contract(operand_source: str) -> ForwardMechanismContract:
    """Return the explicit data-contract domain implemented by a mechanism."""
    try:
        return _FORWARD_MECHANISM_CONTRACTS[operand_source]
    except KeyError:
        raise ValueError(
            f"unsupported forward operand source {operand_source!r}"
        ) from None


def forward_kernel_spec_rejection_reason(solution: ForwardSolution) -> str | None:
    try:
        mechanism = forward_mechanism_contract(solution.operand_source)
    except ValueError as error:
        return str(error)
    structured_q6 = mechanism.lowering == "StructuredQ6"
    decoded_staged = mechanism.lowering == "DecodedWeightLds"
    full_weight_q3 = mechanism.lowering == "Packed3BitFullWeightTiledLds"
    signed_int8_wave_n_tiled_lds = mechanism.physical_plan == "SignedInt8WaveNTiledLds"
    signed_int8_small_m_tiled_lds = (
        mechanism.physical_plan == "SignedInt8SmallMTiledLds"
    )
    serialized_q6_schedule = SemanticSchedulePolicy(
        traversal=solution.q6_output_traversal,
        clustering=solution.q6_stage_clustering,
        latency=solution.q6_latency_policy,
        pressure=solution.q6_pressure_policy,
        wait=solution.q6_wait_policy,
        pairing=solution.q6_pairing_policy,
    )
    if structured_q6:
        if (
            serialized_q6_schedule
            not in SemanticSchedulePolicy.supported_structured_q6()
        ):
            return "unsupported structured-Q6 semantic schedule policy"
        if solution.q6_physical_plan not in {
            "CanonicalRegisterRoles",
            "WideScalarCarryFrontier",
        }:
            return "unsupported structured-Q6 physical plan"
        if solution.q6_physical_plan == "WideScalarCarryFrontier" and (
            solution.macro_tile0 != 64
            or serialized_q6_schedule
            != SemanticSchedulePolicy.structured_q6_wavefront()
        ):
            return "wide scalar-carry frontier requires wavefront MT64"
    elif serialized_q6_schedule != SemanticSchedulePolicy.structured_q6():
        return "Q6 semantic schedule is inactive for this lowering"
    elif solution.q6_physical_plan != "CanonicalRegisterRoles":
        return "Q6 physical plan is inactive for this lowering"
    expected_legacy_suffix = (
        (1, 1, 4, 4, 1) if mechanism.extended_legacy_matrix else (1, 1, 1, 1, 1)
    )
    if (
        len(solution.matrix_instruction) != 9
        or solution.matrix_instruction[4:] != expected_legacy_suffix
    ):
        return (
            "inactive legacy matrix-instruction fields must retain their "
            "canonical sentinel values"
        )
    if signed_int8_wave_n_tiled_lds and solution.depth_u not in (32, 64):
        return "Q8 HIP-shaped LDS controls require DepthU 32 or 64"
    if signed_int8_wave_n_tiled_lds and solution.lds_address_hoist not in {
        "HipTile",
        "CompactDepth32WeightRows",
    }:
        return "Q8 HIP-shaped LDS control has an unsupported layout"
    if signed_int8_small_m_tiled_lds and solution.lds_address_hoist not in {
        "SmallMTile",
        "CompactDepth32WeightRows",
    }:
        return "Q8 small-M LDS control has an unsupported layout"
    if solution.lds_address_hoist == "CompactDepth32WeightRows" and (
        solution.depth_u != 32
        or solution.work_group != (32, 4, 1)
        or solution.macro_tile0 not in (32, 64, 128)
        or solution.macro_tile1 != 64
    ):
        return "Q8 compact depth-32 control requires WG32x4, MT32/MT64/MT128 x N64"
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
    elif full_weight_q3:
        inactive_checks.extend(
            (
                (
                    solution.metadata_schedule == "Q3FullTileSharedDecode",
                    "metadata schedule is not the typed Q3 full-tile schedule",
                ),
                (
                    solution.epilogue_tiles_ahead == 8
                    and solution.epilogue_dependency_width == 1
                    and solution.epilogue_priority == 0,
                    "epilogue pipeline is inactive for full-weight Q3",
                ),
                (
                    solution.accumulator_initialization == "ScalarCopy",
                    "accumulator initialization is inactive for full-weight Q3",
                ),
                (
                    solution.q6_epilogue_pipeline_scope == "StoreBatch"
                    and solution.q6_dependency_delay_mode == "None"
                    and solution.q6_global_read_cache_policy == "Default",
                    "Q6 policies are inactive for full-weight Q3",
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
    wave_count = num_threads // solution.wavefront_size
    mi_wave_group = (
        (1, wave_count) if mechanism.ownership == "WaveN" else (wave_count, 1)
    )
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
        operand_source = solution.operand_source
        mechanism = forward_mechanism_contract(operand_source)
        structured_q6 = mechanism.lowering == "StructuredQ6"
        decoded_staged = mechanism.lowering == "DecodedWeightLds"
        full_weight_q3 = mechanism.lowering == "Packed3BitFullWeightTiledLds"
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
        wave_count = num_threads // solution.wavefront_size
        mi_wave_group = (
            (1, wave_count) if mechanism.ownership == "WaveN" else (wave_count, 1)
        )
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
                    solution.metadata_schedule
                    if decoded_staged or full_weight_q3
                    else None
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
                q6_physical_plan=(solution.q6_physical_plan if structured_q6 else None),
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
        mechanism = forward_mechanism_contract(source)
        decoded_staged = mechanism.lowering == "DecodedWeightLds"
        structured_q6 = mechanism.lowering == "StructuredQ6"
        full_weight_q3 = mechanism.lowering == "Packed3BitFullWeightTiledLds"
        if structured_q6:
            supported_schedules = SemanticSchedulePolicy.supported_structured_q6()
        else:
            supported_schedules = (SemanticSchedulePolicy.inactive(),)
        if self.semantic_schedule not in supported_schedules:
            raise ValueError(
                "forward semantic schedule does not match the lowering family"
            )
        contract_rejection = forward_mechanism_contract(source).rejection_reason(
            contract
        )
        if contract_rejection is not None:
            raise ValueError(contract_rejection)
        pipeline = self.epilogue.pipeline
        if (structured_q6 or decoded_staged) != (pipeline is not None):
            raise ValueError(
                "forward epilogue pipeline does not match the lowering family"
            )
        legacy_suffix = (
            (1, 1, 4, 4, 1) if mechanism.extended_legacy_matrix else (1, 1, 1, 1, 1)
        )
        metadata_schedule = "Q3FullTileSharedDecode" if full_weight_q3 else "Serialized"
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
        q6_physical_plan = "CanonicalRegisterRoles"
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
        elif full_weight_q3:
            if self.decode.metadata_schedule != "Q3FullTileSharedDecode":
                raise ValueError("full-weight Q3 requires its typed decode schedule")
            metadata_schedule = self.decode.metadata_schedule
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
            if self.instruction_policy.q6_physical_plan is None:
                raise ValueError("structured Q6 requires a physical plan")
            q6_physical_plan = self.instruction_policy.q6_physical_plan
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
            q6_physical_plan=q6_physical_plan,
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
    m_wave_groups: int = 1

    def __post_init__(self) -> None:
        if self.output_rows_per_wave <= 0 or self.m_wave_groups <= 0:
            raise ValueError(
                "Q6 LDS output rows per wave and M-wave groups must be positive"
            )

    @property
    def stage_stride_bytes(self) -> int:
        return 9_472 * self.output_rows_per_wave

    @property
    def total_bytes(self) -> int:
        return 19_456 + self.stage_stride_bytes * self.m_wave_groups

    @property
    def activation_wave_group_stride_bytes(self) -> int:
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
    physical_plan: Literal["CanonicalRegisterRoles", "WideScalarCarryFrontier"]
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
        if self.physical_plan not in {
            "CanonicalRegisterRoles",
            "WideScalarCarryFrontier",
        }:
            raise ValueError(f"unsupported Q6 physical plan: {self.physical_plan}")
        if self.physical_plan == "WideScalarCarryFrontier" and (
            self.macro_tile0 != 64
            or self.semantic_policy != SemanticSchedulePolicy.structured_q6_wavefront()
        ):
            raise ValueError("wide scalar-carry frontier requires wavefront MT64")
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
        from .mmq_fwd_physical import q6_structured_physical_plan

        return q6_structured_physical_plan(
            self.mi_wave_tile[0], self.mi_wave_group[0], self.physical_plan
        ).resources

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
    physical_plan = spec.instruction_policy.q6_physical_plan
    cache_policy = spec.global_memory.global_read_cache_policy
    if delay_mode is None or cache_policy is None or physical_plan is None:
        raise ValueError("structured Q6 requires delay, cache, and physical policies")
    _, _, tile_k, _ = spec.geometry.matrix_instruction
    return Q6ForwardSchedule(
        matrix_instruction=spec.geometry.matrix_instruction,
        mi_wave_group=spec.ownership.mi_wave_group,
        mi_wave_tile=spec.ownership.mi_wave_tile,
        semantic_policy=spec.semantic_schedule,
        physical_plan=cast(
            Literal["CanonicalRegisterRoles", "WideScalarCarryFrontier"],
            physical_plan,
        ),
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
    physical_plan: "ForwardPhysicalPlan"
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
        mechanism = forward_mechanism_contract(kernel_spec.global_memory.operand_source)
        contract_rejection = mechanism.rejection_reason(contract)
        if contract_rejection is not None:
            raise ValueError(contract_rejection)
        from .mmq_fwd_physical import derive_forward_physical_plan

        physical_plan = derive_forward_physical_plan(kernel_spec)
        resources = physical_plan.resources
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
            ("K", size.k, mechanism.reduction_values),
        ):
            if value <= 0 or divisor <= 0 or value % divisor:
                raise ValueError(
                    f"forward {name} must be a positive multiple of {divisor}"
                )
        blocks_per_weight_row = size.k // contract.block_values
        activation_blocks_per_row = size.k // Q8_1_D4_BLOCK_VALUES
        activation_plane_stride = size.m * contract.activation_block_bytes
        return cls(
            problem_size=size,
            contract=contract,
            semantics=semantics,
            kernel_spec=kernel_spec,
            physical_plan=physical_plan,
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
