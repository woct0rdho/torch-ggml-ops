"""Typed problem contracts, kernel specifications, and derived MMQ forward state."""

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from .identity import (
    GFX1151_TARGET,
    KernelFamily,
    KernelTarget,
    problem_type_mapping,
    quant_type_from_problem_type,
)
from .model import ProblemSize
from .physical_resources import (
    GFX1151_RESOURCE_CAPACITY,
    PhysicalResourceUsage,
)
from .quant_formats import (
    Q8_1_D4_BLOCK_VALUES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _strict_mapping
from .schema import strict_mapping_optional as _strict_mapping_optional
from .schema import string as _string

if TYPE_CHECKING:
    from .mmq_fwd_physical import ForwardPhysicalPlan


@dataclass(frozen=True)
class FixedForwardDecodePolicy:
    pass


@dataclass(frozen=True)
class DecodedLdsForwardDecodePolicy:
    independent_metadata_extraction: bool
    defer_metadata_reads: bool


@dataclass(frozen=True)
class Q3FullForwardDecodePolicy:
    decode_ready_frontier: bool


ForwardDecodePolicy = (
    FixedForwardDecodePolicy | DecodedLdsForwardDecodePolicy | Q3FullForwardDecodePolicy
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
        assert not (
            self.low_nibble_mask != 252645135
            or self.high_bits_mask != 808464432
            or self.high_bits_shift != 4
            or (self.signed_add != 1616928864)
            or (self.signed_xor != 2155905152)
        )


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
        assert not (
            self.low_mask != 50529027
            or self.high_mask != 16843009
            or self.high_destination_shift != 2
            or (self.signed_add != 2088533116)
            or (self.signed_xor != 2155905152)
        )


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
        raise AssertionError

    def q3_payload_group(self, group: int) -> Q3PayloadGroupSpec:
        assert self.quant_type == "Q3_K"
        assert group in range(16)
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
        assert self.quant_type == "Q3_K"
        assert group in range(16)
        return (
            Q3PackedFieldPart(group % 8, 4 * (group // 8), 4),
            Q3PackedFieldPart(8 + group % 4, 2 * (group // 4), 2, 4),
        )

    def q3_signed_decode(self) -> Q3SignedDecodeSpec:
        assert self.quant_type == "Q3_K"
        return Q3SignedDecodeSpec(
            low_mask=0x03030303,
            high_mask=0x01010101,
            high_destination_shift=2,
            signed_add=0x7C7C7C7C,
            signed_xor=0x80808080,
        )

    def low_payload_group_offsets(self, group: int) -> tuple[int, int]:
        """Return the two 128-bit QL vectors consumed by one direct group."""
        assert group in range(8)
        low = self.payload_plane("ql")
        return (
            low.byte_offset + 32 * (group // 2),
            low.byte_offset + 32 * (group // 2) + 16,
        )

    def packed_scale_minimum_fields(self, group: int) -> PackedScaleMinimumFields:
        assert group in range(8)
        assert self.post_wmma_correction == "ScaleAndMinimum"
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
        assert self.quant_type == "Q6_K"
        assert atom in range(16)
        assert plane in ("ql", "qh")
        lane = atom % 8
        low_offset = 512 * ((5 * lane + atom // 8) % 8) + 64 * lane
        return low_offset if plane == "ql" else (low_offset + 128) % 4096

    def q6_signed_decode(self) -> Q6SignedDecodeSpec:
        assert self.quant_type == "Q6_K"
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
        raise AssertionError


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
    def for_quant_type(cls, quant_type: str) -> "ForwardProblemContract":
        traits = QUANT_FORMATS.get(quant_type)
        assert traits is not None
        return cls(
            quant_type=quant_type,
            block_values=traits.block_values,
            packed_weight_block_bytes=traits.block_bytes,
            activation_layout=traits.activation_layout,
            activation_block_bytes=traits.activation_block_bytes,
            kernel_language="Assembly",
            isa=(11, 5, 1),
            wavefront_size=32,
            signed_weight=True,
            signed_activation=True,
            wmma_clamp=traits.wmma_clamp,
            weight_decode=traits.weight_decode,
            scale_arithmetic=traits.scale_arithmetic,
            arithmetic_contract=traits.arithmetic_contract,
        )


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
        assert self.block_bytes > 0

    def payload_offset(self, group: int) -> int:
        assert group in range(self.GROUP_COUNT)
        return self.PAYLOAD_BASE + self.GROUP_PAYLOAD_BYTES * group

    def scale_offset(self, group: int) -> int:
        assert group in range(self.GROUP_COUNT)
        return 2 * (group // 4)

    def sum_offset(self, group: int) -> int | None:
        assert group in range(self.GROUP_COUNT)
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
        assert self.block_bytes > 0

    @classmethod
    def for_contract(
        cls,
        contract: ForwardProblemContract,
    ) -> "F16D4S4ActivationMetadata":
        assert contract.activation_layout == "F16_D4S4"
        return cls(contract.activation_block_bytes)

    def group(self, index: int) -> F16D4S4ActivationGroupRole:
        assert index in range(self.GROUP_COUNT)
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
        assert activation_block_bytes > 0
        assert activation_rows in (64, 128)
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
        assert activation_block_bytes > 0
        assert activation_rows in (32, 64, 128)
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
        assert contract.activation_layout == "F16_D4S4"
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
        assert (
            min(
                self.activation_rows,
                self.weight_rows,
                self.activation_row_stride,
                self.decoded_weight_values,
                self.scale_count,
            )
            > 0
        )

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

    @property
    def activation_rows(self) -> int:
        return 128

    @property
    def weight_rows(self) -> int:
        return 64

    @property
    def activation_row_stride(self) -> int:
        return Q8_1_F32_D4_BLOCK_BYTES

    @property
    def half_payload_bytes(self) -> int:
        return 128

    @property
    def half_payload_stride(self) -> int:
        return 160

    @property
    def weight_scale_offset(self) -> int:
        return 128

    @property
    def weight_scale_bytes(self) -> int:
        return 32

    @property
    def weight_row_stride(self) -> int:
        return 336

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

    def __post_init__(self) -> None:
        assert self.activation_rows in (32, 64)

    @property
    def weight_rows(self) -> int:
        return 64

    @property
    def activation_row_stride(self) -> int:
        return Q8_1_F32_D4_BLOCK_BYTES

    @property
    def weight_row_stride(self) -> int:
        return 304

    @property
    def weight_scale_offset(self) -> int:
        return 256

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

    def __post_init__(self) -> None:
        assert self.activation_rows in (32, 64, 128)

    @property
    def weight_rows(self) -> int:
        return 64

    @property
    def activation_row_stride(self) -> int:
        return Q8_1_F32_D4_BLOCK_BYTES

    @property
    def weight_row_stride(self) -> int:
        return 144

    @property
    def weight_scale_offset(self) -> int:
        return 128

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
    policy: ForwardDecodePolicy

    @property
    def independent_metadata_extraction(self) -> bool:
        return (
            self.policy.independent_metadata_extraction
            if isinstance(self.policy, DecodedLdsForwardDecodePolicy)
            else False
        )

    @property
    def defer_metadata_reads(self) -> bool:
        return (
            self.policy.defer_metadata_reads
            if isinstance(self.policy, DecodedLdsForwardDecodePolicy)
            else False
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
    """Active pipelined-store policy for mechanisms that expose one."""

    pipeline: EpiloguePipelineSpec | None


def derive_forward_resource_usage(spec: "ForwardKernelSpec") -> PhysicalResourceUsage:
    from .mmq_fwd_physical import derive_forward_physical_plan

    return derive_forward_physical_plan(spec).resources


@dataclass(frozen=True)
class InstructionPolicy:
    """Only active instruction-ordering policies not implied by dependencies."""

    accumulator_initialization: str | None
    dependency_delay_mode: str | None
    physical_plan: str | None


class StructuredQ6ScheduleVariant(str, Enum):
    GroupMajor = "GroupMajor"
    Wavefront = "Wavefront"


@dataclass(frozen=True)
class SemanticSchedulePolicy:
    """Typed semantic schedule with a stable six-field serialized form."""

    variant: StructuredQ6ScheduleVariant | None = None

    def __post_init__(self) -> None:
        assert self.variant is None or isinstance(
            self.variant, StructuredQ6ScheduleVariant
        )

    @classmethod
    def from_serialized(
        cls,
        *,
        traversal: str | None,
        clustering: str | None,
        latency: str | None,
        pressure: str | None,
        wait: str | None,
        pairing: str | None,
    ) -> "SemanticSchedulePolicy":
        fields = (traversal, clustering, latency, pressure, wait, pairing)
        match fields:
            case (None, None, None, None, None, None):
                return cls()
            case (
                "OutputRoleGroupMajor",
                "StageDependencyOrder",
                "SerializedDependencyDistance",
                "ExplicitRoleLifetime",
                "ProducerFirstUse",
                "DependencyCompatibleDualIssue",
            ):
                return cls.structured_q6()
            case (
                "OutputRoleWavefront",
                "RowBatchedDecodeOrder",
                "WavefrontDependencyDistance",
                "ExplicitRoleLifetime",
                "ProducerFirstUse",
                "DependencyCompatibleDualIssue",
            ):
                return cls.structured_q6_wavefront()
            case _:
                raise AssertionError

    @classmethod
    def structured_q6(cls) -> "SemanticSchedulePolicy":
        return cls(StructuredQ6ScheduleVariant.GroupMajor)

    @classmethod
    def structured_q6_wavefront(cls) -> "SemanticSchedulePolicy":
        return cls(StructuredQ6ScheduleVariant.Wavefront)

    @classmethod
    def structured_q6_variants(cls) -> tuple["SemanticSchedulePolicy", ...]:
        return tuple(cls(variant) for variant in StructuredQ6ScheduleVariant)

    def require_structured_q6(self) -> StructuredQ6ScheduleVariant:
        assert self.variant is not None
        return self.variant

    def _serialized(self) -> tuple[str | None, ...]:
        match self.variant:
            case None:
                return (None, None, None, None, None, None)
            case StructuredQ6ScheduleVariant.GroupMajor:
                return (
                    "OutputRoleGroupMajor",
                    "StageDependencyOrder",
                    "SerializedDependencyDistance",
                    "ExplicitRoleLifetime",
                    "ProducerFirstUse",
                    "DependencyCompatibleDualIssue",
                )
            case StructuredQ6ScheduleVariant.Wavefront:
                return (
                    "OutputRoleWavefront",
                    "RowBatchedDecodeOrder",
                    "WavefrontDependencyDistance",
                    "ExplicitRoleLifetime",
                    "ProducerFirstUse",
                    "DependencyCompatibleDualIssue",
                )

    @property
    def traversal(self) -> str | None:
        return self._serialized()[0]

    @property
    def clustering(self) -> str | None:
        return self._serialized()[1]

    @property
    def latency(self) -> str | None:
        return self._serialized()[2]

    @property
    def pressure(self) -> str | None:
        return self._serialized()[3]

    @property
    def wait(self) -> str | None:
        return self._serialized()[4]

    @property
    def pairing(self) -> str | None:
        return self._serialized()[5]


@dataclass(frozen=True)
class ForwardDataflowContract:
    """Addressing and conversion choices owned by one mechanism."""

    activation_addressing: str
    lds_address_hoists: tuple[str, ...]
    metadata_conversion: str

    def validate(
        self,
        *,
        activation_addressing: str,
        lds_address_hoist: str,
        metadata_conversion: str,
    ) -> None:
        assert activation_addressing == self.activation_addressing
        self.validate_physical(lds_address_hoist)
        assert metadata_conversion == self.metadata_conversion

    def validate_physical(self, lds_address_hoist: str) -> None:
        assert lds_address_hoist in self.lds_address_hoists


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
    dataflow: ForwardDataflowContract
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
    extended_matrix_instruction: bool = False

    def validate(self, contract: ForwardProblemContract) -> None:
        assert contract.activation_layout == self.activation_layout
        assert contract.activation_block_bytes == self.activation_block_bytes
        assert contract.block_values == self.weight_block_values
        assert contract.wmma_clamp == self.wmma_clamp
        assert contract.weight_decode in self.weight_decodes
        assert contract.scale_arithmetic == self.scale_arithmetic
        assert contract.arithmetic_contract in self.arithmetic_contracts


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
    dataflow=ForwardDataflowContract(
        activation_addressing="MadU24",
        lds_address_hoists=("WeightMetadata",),
        metadata_conversion="DirectFloat16Unsigned16",
    ),
    physical_plan="DecodedWeightLds",
    extended_matrix_instruction=True,
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
    dataflow=ForwardDataflowContract(
        activation_addressing="MultiplyAdd",
        lds_address_hoists=("None",),
        metadata_conversion="Float16DToFloat32",
    ),
    physical_plan="SignedInt8Direct",
)
_FORWARD_MECHANISM_CONTRACTS = MappingProxyType(
    {
        "Global": replace(
            _PACKED_SCALE_MINIMUM_CONTRACT,
            lowering="PackedScaleMinimumDirect",
            weight_decodes=("DirectNibble",),
            dataflow=ForwardDataflowContract(
                activation_addressing="MultiplyAdd",
                lds_address_hoists=("None",),
                metadata_conversion="Float32ThenFloat16",
            ),
            physical_plan="PackedScaleMinimumDirect",
            extended_matrix_instruction=False,
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
            dataflow=ForwardDataflowContract(
                activation_addressing="MadU24",
                lds_address_hoists=("StructuredDecodeDot",),
                metadata_conversion="Float16DToFloat32Signed8Scale",
            ),
            physical_plan="StructuredQ6",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
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
            dataflow=ForwardDataflowContract(
                activation_addressing="MadU24",
                lds_address_hoists=("Q3HalfTile",),
                metadata_conversion="Float16DToFloat32Signed6Scale",
            ),
            physical_plan="Packed3BitTiledLds",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
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
            dataflow=ForwardDataflowContract(
                activation_addressing="ScalarPlaneBase",
                lds_address_hoists=("Q3FullTile336",),
                metadata_conversion="Float16DToFloat32Signed6ScaleShared",
            ),
            physical_plan="Packed3BitFullWeightTiledLds",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
        ),
        "Q8DirectGlobal": _SIGNED_INT8_CONTRACT,
        "Q8RegisterTiled": replace(
            _SIGNED_INT8_CONTRACT,
            physical_plan="SignedInt8RegisterTiled",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
        ),
        "Q8HipTiledLds": replace(
            _SIGNED_INT8_CONTRACT,
            dataflow=ForwardDataflowContract(
                activation_addressing="MadU24",
                lds_address_hoists=("HipTile", "CompactDepth32WeightRows"),
                metadata_conversion="Float16DToFloat32",
            ),
            physical_plan="SignedInt8WaveNTiledLds",
            ownership="WaveN",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
        ),
        "Q8SmallMTiledLds": replace(
            _SIGNED_INT8_CONTRACT,
            dataflow=ForwardDataflowContract(
                activation_addressing="MadU24",
                lds_address_hoists=("SmallMTile", "CompactDepth32WeightRows"),
                metadata_conversion="Float16DToFloat32",
            ),
            physical_plan="SignedInt8SmallMTiledLds",
            ownership="WaveN",
            uses_workitem_id=True,
            extended_matrix_instruction=True,
        ),
    }
)


def forward_mechanism_contract(operand_source: str) -> ForwardMechanismContract:
    """Return the explicit data-contract domain implemented by a mechanism."""
    mechanism = _FORWARD_MECHANISM_CONTRACTS.get(operand_source)
    assert mechanism is not None
    return mechanism


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

    @classmethod
    def from_mapping(cls, value: object) -> "ForwardKernelSpec":
        item = _strict_mapping_optional(
            value,
            name="ForwardKernelSpec",
            required=frozenset(
                {
                    "Geometry",
                    "Ownership",
                    "GlobalMemory",
                    "Lds",
                    "Decode",
                }
            ),
            optional=frozenset({"Epilogue", "InstructionPolicy", "SemanticSchedule"}),
        )
        geometry = _strict_mapping(
            item["Geometry"],
            name="ForwardKernelSpec.Geometry",
            keys=frozenset({"WorkGroup", "MatrixInstruction", "DepthU"}),
        )
        ownership = _strict_mapping(
            item["Ownership"],
            name="ForwardKernelSpec.Ownership",
            keys=frozenset({"MIWaveGroup", "MIWaveTile"}),
        )
        global_memory = _strict_mapping_optional(
            item["GlobalMemory"],
            name="ForwardKernelSpec.GlobalMemory",
            required=frozenset({"OperandSource", "ActivationAddressing"}),
            optional=frozenset({"GlobalReadCachePolicy"}),
        )
        lds = _strict_mapping(
            item["Lds"],
            name="ForwardKernelSpec.Lds",
            keys=frozenset({"AddressHoist"}),
        )
        source = _string(global_memory, "OperandSource")
        mechanism = forward_mechanism_contract(source)
        if mechanism.lowering == "DecodedWeightLds":
            decode_keys = frozenset(
                {
                    "MetadataConversion",
                    "IndependentMetadataExtraction",
                    "DeferMetadataReads",
                }
            )
        elif mechanism.lowering == "Packed3BitFullWeightTiledLds":
            decode_keys = frozenset({"MetadataConversion", "DecodeReadyFrontier"})
        else:
            decode_keys = frozenset({"MetadataConversion"})
        decode = _strict_mapping(
            item["Decode"],
            name="ForwardKernelSpec.Decode",
            keys=decode_keys,
        )
        if mechanism.lowering == "DecodedWeightLds":
            decode_policy: ForwardDecodePolicy = DecodedLdsForwardDecodePolicy(
                independent_metadata_extraction=_boolean(
                    decode, "IndependentMetadataExtraction"
                ),
                defer_metadata_reads=_boolean(decode, "DeferMetadataReads"),
            )
        elif mechanism.lowering == "Packed3BitFullWeightTiledLds":
            decode_policy = Q3FullForwardDecodePolicy(
                decode_ready_frontier=_boolean(decode, "DecodeReadyFrontier")
            )
        else:
            decode_policy = FixedForwardDecodePolicy()
        has_epilogue_policy = mechanism.lowering in {
            "DecodedWeightLds",
            "StructuredQ6",
        }
        if has_epilogue_policy != ("Epilogue" in item):
            raise SchemaError(
                "ForwardKernelSpec.Epilogue presence does not match the lowering family"
            )
        epilogue = (
            _strict_mapping(
                item["Epilogue"],
                name="ForwardKernelSpec.Epilogue",
                keys=frozenset({"Pipeline"}),
            )
            if has_epilogue_policy
            else None
        )
        pipeline = None
        if epilogue is not None:
            pipeline_item = _strict_mapping_optional(
                epilogue["Pipeline"],
                name="ForwardKernelSpec.Epilogue.Pipeline",
                required=frozenset({"DependencyWidth"}),
                optional=frozenset({"TilesAhead", "Priority", "Scope"}),
            )
            pipeline = EpiloguePipelineSpec(
                tiles_ahead=(
                    _integer(pipeline_item, "TilesAhead")
                    if "TilesAhead" in pipeline_item
                    else None
                ),
                dependency_width=_integer(pipeline_item, "DependencyWidth"),
                priority=(
                    _integer(pipeline_item, "Priority")
                    if "Priority" in pipeline_item
                    else None
                ),
                scope=(
                    _string(pipeline_item, "Scope")
                    if "Scope" in pipeline_item
                    else None
                ),
            )
        instruction = _strict_mapping_optional(
            item.get("InstructionPolicy", {}),
            name="ForwardKernelSpec.InstructionPolicy",
            required=frozenset(),
            optional=frozenset(
                {
                    "AccumulatorInitialization",
                    "DependencyDelayMode",
                    "PhysicalPlan",
                }
            ),
        )
        semantic = (
            _strict_mapping(
                item["SemanticSchedule"],
                name="ForwardKernelSpec.SemanticSchedule",
                keys=frozenset(
                    {
                        "Traversal",
                        "Clustering",
                        "Latency",
                        "Pressure",
                        "Wait",
                        "Pairing",
                    }
                ),
            )
            if "SemanticSchedule" in item
            else None
        )
        semantic_schedule = (
            SemanticSchedulePolicy.from_serialized(
                traversal=_string(semantic, "Traversal"),
                clustering=_string(semantic, "Clustering"),
                latency=_string(semantic, "Latency"),
                pressure=_string(semantic, "Pressure"),
                wait=_string(semantic, "Wait"),
                pairing=_string(semantic, "Pairing"),
            )
            if semantic is not None
            else SemanticSchedulePolicy()
        )
        return cls(
            geometry=GeometrySpec(
                work_group=_integer_tuple(geometry, "WorkGroup", 3),
                matrix_instruction=cast(
                    tuple[int, int, int, int],
                    _integer_tuple(geometry, "MatrixInstruction", 4),
                ),
                depth_u=_integer(geometry, "DepthU"),
            ),
            ownership=OwnershipSpec(
                mi_wave_group=cast(
                    tuple[int, int],
                    _integer_tuple(ownership, "MIWaveGroup", 2),
                ),
                mi_wave_tile=cast(
                    tuple[int, int],
                    _integer_tuple(ownership, "MIWaveTile", 2),
                ),
            ),
            global_memory=GlobalMemorySpec(
                operand_source=source,
                activation_addressing=_string(global_memory, "ActivationAddressing"),
                global_read_cache_policy=(
                    _string(
                        global_memory,
                        "GlobalReadCachePolicy",
                    )
                    if "GlobalReadCachePolicy" in global_memory
                    else None
                ),
            ),
            lds=LdsSpec(address_hoist=_string(lds, "AddressHoist")),
            decode=DecodeSpec(
                metadata_conversion=_string(decode, "MetadataConversion"),
                policy=decode_policy,
            ),
            epilogue=EpilogueSpec(pipeline=pipeline),
            instruction_policy=InstructionPolicy(
                accumulator_initialization=(
                    _string(
                        instruction,
                        "AccumulatorInitialization",
                    )
                    if "AccumulatorInitialization" in instruction
                    else None
                ),
                dependency_delay_mode=(
                    _string(
                        instruction,
                        "DependencyDelayMode",
                    )
                    if "DependencyDelayMode" in instruction
                    else None
                ),
                physical_plan=(
                    _string(instruction, "PhysicalPlan")
                    if "PhysicalPlan" in instruction
                    else None
                ),
            ),
            semantic_schedule=semantic_schedule,
        )

    def validate(self, contract: ForwardProblemContract) -> None:
        mechanism = forward_mechanism_contract(self.global_memory.operand_source)
        mechanism.validate(contract)
        mechanism.dataflow.validate(
            activation_addressing=self.global_memory.activation_addressing,
            lds_address_hoist=self.lds.address_hoist,
            metadata_conversion=self.decode.metadata_conversion,
        )
        work_group = self.geometry.work_group
        assert len(work_group) == 3
        assert min(work_group) > 0
        assert work_group[2] == 1
        tile_m, tile_n, tile_k, blocks = self.geometry.matrix_instruction
        assert min(tile_m, tile_n, tile_k, blocks) > 0
        threads = work_group[0] * work_group[1] * work_group[2]
        assert threads % contract.wavefront_size == 0
        wave_count = threads // contract.wavefront_size
        expected_group = (
            (1, wave_count) if mechanism.ownership == "WaveN" else (wave_count, 1)
        )
        assert self.ownership.mi_wave_group == expected_group
        assert min(self.ownership.mi_wave_tile) > 0
        assert self.macro_tile[0] % (tile_m * expected_group[0]) == 0
        assert self.macro_tile[1] % (tile_n * expected_group[1]) == 0
        assert self.geometry.depth_u % tile_k == 0
        structured_q6 = mechanism.lowering == "StructuredQ6"
        decoded_staged = mechanism.lowering == "DecodedWeightLds"
        full_weight_q3 = mechanism.lowering == "Packed3BitFullWeightTiledLds"
        pipeline = self.epilogue.pipeline
        if structured_q6:
            assert isinstance(self.decode.policy, FixedForwardDecodePolicy)
            assert pipeline is not None
            assert pipeline.tiles_ahead is None
            assert pipeline.priority is None
            assert pipeline.scope in {"StoreBatch", "FullTile"}
            assert self.instruction_policy.accumulator_initialization is None
            assert self.instruction_policy.dependency_delay_mode in {"None", "Explicit"}
            assert self.instruction_policy.physical_plan in {
                "CanonicalRegisterRoles",
                "WideScalarCarryFrontier",
            }
            assert self.global_memory.global_read_cache_policy in {
                "Default",
                "InvalidateL0",
            }
            assert self.semantic_schedule.variant is not None
            if self.instruction_policy.physical_plan == "WideScalarCarryFrontier":
                assert self.macro_tile[0] == 64
                assert (
                    self.semantic_schedule.variant
                    is StructuredQ6ScheduleVariant.Wavefront
                )
        elif decoded_staged:
            assert isinstance(self.decode.policy, DecodedLdsForwardDecodePolicy)
            assert pipeline is not None
            assert pipeline.tiles_ahead is not None
            assert pipeline.priority is not None
            assert pipeline.scope is None
            assert self.instruction_policy.accumulator_initialization in {
                "ScalarCopy",
                "VopdPair",
            }
            assert self.instruction_policy.dependency_delay_mode is None
            assert self.instruction_policy.physical_plan is None
            assert self.global_memory.global_read_cache_policy is None
            assert self.semantic_schedule.variant is None
        elif full_weight_q3:
            assert isinstance(self.decode.policy, Q3FullForwardDecodePolicy)
            assert pipeline is None
            assert self.instruction_policy == InstructionPolicy(None, None, None)
            assert self.global_memory.global_read_cache_policy is None
            assert self.semantic_schedule.variant is None
        else:
            assert isinstance(self.decode.policy, FixedForwardDecodePolicy)
            assert pipeline is None
            assert self.instruction_policy == InstructionPolicy(None, None, None)
            assert self.global_memory.global_read_cache_policy is None
            assert self.semantic_schedule.variant is None
        if mechanism.physical_plan == "SignedInt8WaveNTiledLds":
            assert self.geometry.depth_u in (32, 64)
        if self.lds.address_hoist == "CompactDepth32WeightRows":
            assert self.geometry.depth_u == 32
            assert work_group == (32, 4, 1)
            assert self.macro_tile[0] in (32, 64, 128)
            assert self.macro_tile[1] == 64

    @property
    def macro_tile(self) -> tuple[int, int]:
        tile_m, tile_n, _, _ = self.geometry.matrix_instruction
        return (
            tile_m * self.ownership.mi_wave_group[0] * self.ownership.mi_wave_tile[0],
            tile_n * self.ownership.mi_wave_group[1] * self.ownership.mi_wave_tile[1],
        )

    def to_mapping(self) -> dict[str, object]:
        global_memory: dict[str, object] = {
            "OperandSource": self.global_memory.operand_source,
            "ActivationAddressing": self.global_memory.activation_addressing,
        }
        if self.global_memory.global_read_cache_policy is not None:
            global_memory["GlobalReadCachePolicy"] = (
                self.global_memory.global_read_cache_policy
            )
        decode: dict[str, object] = {
            "MetadataConversion": self.decode.metadata_conversion
        }
        if isinstance(self.decode.policy, DecodedLdsForwardDecodePolicy):
            decode["IndependentMetadataExtraction"] = (
                self.decode.policy.independent_metadata_extraction
            )
            decode["DeferMetadataReads"] = self.decode.policy.defer_metadata_reads
        elif isinstance(self.decode.policy, Q3FullForwardDecodePolicy):
            decode["DecodeReadyFrontier"] = self.decode.policy.decode_ready_frontier
        else:
            assert isinstance(self.decode.policy, FixedForwardDecodePolicy)
        epilogue: dict[str, object] | None = None
        if self.epilogue.pipeline is not None:
            policy = self.epilogue.pipeline
            pipeline: dict[str, object] = {"DependencyWidth": policy.dependency_width}
            if policy.tiles_ahead is not None:
                pipeline["TilesAhead"] = policy.tiles_ahead
            if policy.priority is not None:
                pipeline["Priority"] = policy.priority
            if policy.scope is not None:
                pipeline["Scope"] = policy.scope
            epilogue = {"Pipeline": pipeline}
        mapping: dict[str, object] = {
            "Geometry": {
                "WorkGroup": list(self.geometry.work_group),
                "MatrixInstruction": list(self.geometry.matrix_instruction),
                "DepthU": self.geometry.depth_u,
            },
            "Ownership": {
                "MIWaveGroup": list(self.ownership.mi_wave_group),
                "MIWaveTile": list(self.ownership.mi_wave_tile),
            },
            "GlobalMemory": global_memory,
            "Lds": {"AddressHoist": self.lds.address_hoist},
            "Decode": decode,
        }
        if epilogue is not None:
            mapping["Epilogue"] = epilogue
        instruction: dict[str, object] = {}
        if self.instruction_policy.accumulator_initialization is not None:
            instruction["AccumulatorInitialization"] = (
                self.instruction_policy.accumulator_initialization
            )
        if self.instruction_policy.dependency_delay_mode is not None:
            instruction["DependencyDelayMode"] = (
                self.instruction_policy.dependency_delay_mode
            )
        if self.instruction_policy.physical_plan is not None:
            instruction["PhysicalPlan"] = self.instruction_policy.physical_plan
        if instruction:
            mapping["InstructionPolicy"] = instruction
        if self.semantic_schedule.variant is not None:
            mapping["SemanticSchedule"] = {
                "Traversal": self.semantic_schedule.traversal,
                "Clustering": self.semantic_schedule.clustering,
                "Latency": self.semantic_schedule.latency,
                "Pressure": self.semantic_schedule.pressure,
                "Wait": self.semantic_schedule.wait,
                "Pairing": self.semantic_schedule.pairing,
            }
        return mapping


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
        assert not (
            not self.name
            or self.pair < 0
            or min(self.offset0, self.offset1, self.first_stage) < 0
            or (self.last_stage < self.first_stage)
        )


@dataclass(frozen=True)
class Q6LdsLayout:
    """Formula-derived Q6 LDS planes and stage offsets."""

    output_rows_per_wave: int
    m_wave_groups: int = 1

    def __post_init__(self) -> None:
        assert not (self.output_rows_per_wave <= 0 or self.m_wave_groups <= 0)

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
        assert pair >= 0
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
        assert pair >= 0
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
        assert blocks == 1
        assert self.dot_register_shifts
        assert self.physical_plan in {
            "CanonicalRegisterRoles",
            "WideScalarCarryFrontier",
        }
        assert not (
            self.physical_plan == "WideScalarCarryFrontier"
            and (
                self.macro_tile0 != 64
                or self.semantic_policy
                != SemanticSchedulePolicy.structured_q6_wavefront()
            )
        )
        assert not (
            self.epilogue_dependency_width not in (1, 2, 4, 8)
            or self.store_vector_width % self.epilogue_dependency_width
        )
        assert self.epilogue_pipeline_scope in {"StoreBatch", "FullTile"}
        assert self.dependency_delay_mode in ("None", "Explicit")
        assert self.global_read_cache_policy in ("Default", "InvalidateL0")

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
    def resource_usage(self) -> PhysicalResourceUsage:
        from .mmq_fwd_physical import q6_structured_physical_plan

        return q6_structured_physical_plan(
            self.mi_wave_tile[0], self.mi_wave_group[0], self.physical_plan
        ).resources

    def phase(self, phase: int) -> Q6DotPhase:
        assert phase in range(len(self.dot_register_shifts))
        return Q6DotPhase(self, phase)


def validate_forward_kernel_spec_record(
    problem_size: ProblemSize,
    contract: ForwardProblemContract,
    spec: ForwardKernelSpec,
) -> None:
    spec.validate(contract)
    DerivedForwardState.from_contract_spec(problem_size, contract, spec)


def q6_schedule_from_kernel_spec(spec: ForwardKernelSpec) -> Q6ForwardSchedule:
    """Derive structured Q6 lowering state from one canonical kernel spec."""
    assert spec.global_memory.operand_source == "Q6StructuredDecoded"
    assert spec.geometry.work_group[0] == 32
    assert spec.geometry.work_group[2] == 1
    assert spec.ownership.mi_wave_group[0] * spec.ownership.mi_wave_group[1] > 0
    pipeline = spec.epilogue.pipeline
    assert pipeline is not None
    assert pipeline.scope is not None
    delay_mode = spec.instruction_policy.dependency_delay_mode
    physical_plan = spec.instruction_policy.physical_plan
    cache_policy = spec.global_memory.global_read_cache_policy
    assert delay_mode is not None
    assert cache_policy is not None
    assert physical_plan is not None
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


@dataclass(frozen=True)
class ForwardKernelCandidate:
    """Canonical complete candidate independent of any exact problem shape."""

    problem_contract: ForwardProblemContract
    kernel_spec: ForwardKernelSpec

    @classmethod
    def from_mapping(cls, value: object) -> "ForwardKernelCandidate":
        item = _strict_mapping(
            value,
            name="ForwardKernelCandidate",
            keys=frozenset({"KernelFamily", "Target", "ProblemType", "KernelSpec"}),
        )
        family = KernelFamily.OrdinaryForward
        if item["KernelFamily"] != family.value:
            raise SchemaError("forward candidate has the wrong KernelFamily")
        KernelTarget.from_mapping(item["Target"])
        quant_type = quant_type_from_problem_type(item["ProblemType"], family)
        contract = ForwardProblemContract.for_quant_type(quant_type)
        spec = ForwardKernelSpec.from_mapping(item["KernelSpec"])
        validate_forward_kernel_spec_record(
            ProblemSize(
                spec.macro_tile[0],
                spec.macro_tile[1],
                forward_mechanism_contract(
                    spec.global_memory.operand_source
                ).reduction_values,
            ),
            contract,
            spec,
        )
        return cls(problem_contract=contract, kernel_spec=spec)

    def to_mapping(self) -> dict[str, object]:
        family = KernelFamily.OrdinaryForward
        return {
            "KernelFamily": family.value,
            "Target": GFX1151_TARGET.to_mapping(),
            "ProblemType": problem_type_mapping(
                family, self.problem_contract.quant_type
            ),
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
    resources: PhysicalResourceUsage
    blocks_per_weight_row: int
    activation_blocks_per_row: int
    packed_weight_row_bytes: int
    activation_plane_stride_bytes: int
    activation_weight_block_stride_bytes: int
    grid: tuple[int, int, int]

    @classmethod
    def from_problem_spec(
        cls,
        size: ProblemSize,
        quant_type: str,
        spec: ForwardKernelSpec,
    ) -> "DerivedForwardState":
        contract = ForwardProblemContract.for_quant_type(quant_type)
        return cls.from_contract_spec(size, contract, spec)

    @classmethod
    def from_contract_spec(
        cls,
        size: ProblemSize,
        contract: ForwardProblemContract,
        kernel_spec: ForwardKernelSpec,
        *,
        activation_rows: int | None = None,
    ) -> "DerivedForwardState":
        semantics = QuantForwardSemantics.for_quant_type(contract.quant_type)
        payload_bytes = max(
            plane.byte_offset + plane.byte_count for plane in semantics.payload_planes
        )
        assert payload_bytes == contract.packed_weight_block_bytes
        mechanism = forward_mechanism_contract(kernel_spec.global_memory.operand_source)
        mechanism.validate(contract)
        from .mmq_fwd_physical import derive_forward_physical_plan

        physical_plan = derive_forward_physical_plan(kernel_spec)
        resources = physical_plan.resources
        resources.admit(GFX1151_RESOURCE_CAPACITY)
        geometry = kernel_spec.geometry
        macro_tile_m, macro_tile_n = kernel_spec.macro_tile
        num_threads = (
            geometry.work_group[0] * geometry.work_group[1] * geometry.work_group[2]
        )
        waves_per_workgroup = num_threads // contract.wavefront_size
        _, _, tile_k, _ = geometry.matrix_instruction
        mi_wave_group = kernel_spec.ownership.mi_wave_group
        mi_wave_tile = kernel_spec.ownership.mi_wave_tile
        for value, divisor in (
            (size.m, macro_tile_m),
            (size.n, macro_tile_n),
            (size.k, mechanism.reduction_values),
        ):
            assert value > 0
            assert divisor > 0
            assert value % divisor == 0
        blocks_per_weight_row = size.k // contract.block_values
        activation_blocks_per_row = size.k // Q8_1_D4_BLOCK_VALUES
        activation_plane_stride = (
            size.m if activation_rows is None else activation_rows
        ) * contract.activation_block_bytes
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
