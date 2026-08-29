"""Typed problem and solution state for MMQ backward assembly lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, cast

from .model import ProblemSize
from .quant_formats import BACKWARD_QUANT_FORMATS, QuantFormat
from .schema import SchemaError
from .schema import boolean as _boolean
from .schema import integer as _integer
from .schema import integer_tuple as _integer_tuple
from .schema import strict_mapping as _strict_mapping
from .schema import strict_mapping_optional as _strict_mapping_optional
from .schema import string as _string
from .work_group_mapping import work_group_mapping_shift

EnumT = TypeVar("EnumT", bound=Enum)


def _serialized_enum(enum_type: type[EnumT], value: object, name: str) -> EnumT:
    serialized = _string(value, name)
    if serialized not in enum_type._value2member_map_:
        raise SchemaError(f"invalid {name}: {serialized!r}")
    return enum_type(serialized)


class BackwardPairing(str, Enum):
    Partial = "Partial"
    Full = "Full"


class BackwardExtraction(str, Enum):
    Packed = "Packed"
    PackedVopd = "PackedVopd"
    Scalar = "Scalar"


@dataclass(frozen=True)
class BackwardIterationPolicy:
    prefetch_activation: bool
    interleave_wmma_waits: bool

    @property
    def uses_prefetch_activation_waits(self) -> bool:
        return self.prefetch_activation and not self.interleave_wmma_waits

    def supports_global_read_prefetch(self, count: int) -> bool:
        return count == 1 or (count == 2 and self.prefetch_activation)

    def supports_local_read_prefetch(self, count: int) -> bool:
        return count == 1 or (count == 2 and self.interleave_wmma_waits)


class BackwardBitfieldShiftPlacement(str, Enum):
    Inline = "Inline"
    Hoisted = "Hoisted"


class BackwardMetadataLoad(str, Enum):
    Scalar = "Scalar"
    Vector = "Vector"


class LdsBuffering(str, Enum):
    Single = "Single"
    Double = "Double"


class PackedLoadGrouping(str, Enum):
    PerLane = "PerLane"
    LanePair = "LanePair"

    @property
    def lane_share(self) -> int:
        return 1 if self is PackedLoadGrouping.PerLane else 2


class BackwardStorePriority(str, Enum):
    Normal = "Normal"
    Raised = "Raised"


class BackwardPackedRowAddress(str, Enum):
    Block32 = "Block32"
    SuperBlock256 = "SuperBlock256"


@dataclass(frozen=True)
class BackwardDecoderCapability:
    payload_register_count: int
    packed_loads_per_row: int
    scalar_packed_loads_per_row: int | None
    vector_metadata_packed_loads_per_row: int | None
    packed_pairing_temporary_count: int | None
    packed_vopd_temporary_base: int | None
    row_address: BackwardPackedRowAddress

    def packed_load_count(self, decode, rows: int) -> int:
        per_row = self.packed_loads_per_row
        if (
            self.row_address is BackwardPackedRowAddress.Block32
            and self.scalar_packed_loads_per_row is not None
            and decode.extraction is BackwardExtraction.Scalar
        ):
            per_row = self.scalar_packed_loads_per_row
        elif (
            self.vector_metadata_packed_loads_per_row is not None
            and decode.uses_vector_metadata_load
        ):
            per_row = self.vector_metadata_packed_loads_per_row
        return per_row * rows

    def temporary_register_count(self, decode, rows: int) -> int:
        candidates = [7, 3 + 2 * rows]
        if (
            self.packed_pairing_temporary_count is not None
            and decode.extraction is BackwardExtraction.Packed
        ):
            candidates.append(self.packed_pairing_temporary_count)
        if (
            self.packed_vopd_temporary_base is not None
            and decode.extraction is BackwardExtraction.PackedVopd
        ):
            candidates.append(self.packed_vopd_temporary_base + 2 * rows)
        return max(candidates)


@dataclass(frozen=True)
class BackwardAddressCapability:
    extended_quant_address_register_count: int

    def uses_extended_a(self, iteration: BackwardIterationPolicy, m_tiles: int) -> bool:
        return iteration.prefetch_activation and m_tiles > 2

    def register_count(self, iteration: BackwardIterationPolicy, m_tiles: int) -> int:
        if not self.uses_extended_a(iteration, m_tiles):
            return 8
        return 6 + m_tiles + self.extended_quant_address_register_count


@dataclass(frozen=True)
class BackwardQuantRegisterShape:
    dm_registers_per_row: int
    scale_registers_per_row: int
    scale_alias_offset: int | None
    vector_metadata_shape: BackwardQuantRegisterShape | None = None

    def for_decode(self, decode):
        if decode.uses_vector_metadata_load and self.vector_metadata_shape is not None:
            return self.vector_metadata_shape
        return self


@dataclass(frozen=True)
class BackwardMechanismContract:
    packed_load_grouping_values: frozenset[PackedLoadGrouping]
    padded_depth_values: frozenset[int]
    pipeline_n_values: frozenset[int]
    pipeline_depth_values: frozenset[int]
    next_packed_depth_values: frozenset[int]
    decoder: BackwardDecoderCapability
    address: BackwardAddressCapability
    quant_register_shape: BackwardQuantRegisterShape

    def supports_padded_depth(self, depth: int) -> bool:
        return depth in self.padded_depth_values

    def supports_pipeline_n(self, macro_tile1: int) -> bool:
        return macro_tile1 in self.pipeline_n_values

    def supports_pipeline_depth(self, depth: int) -> bool:
        return depth in self.pipeline_depth_values

    def supports_next_packed_depth(self, depth: int) -> bool:
        return depth in self.next_packed_depth_values

    @property
    def extended_quant_address_state(self) -> bool:
        return self.address.extended_quant_address_register_count > 0


def backward_mechanism_contract(quant_type: str) -> BackwardMechanismContract:
    assert quant_type in BACKWARD_QUANT_FORMATS
    return BackwardMechanismContract(
        packed_load_grouping_values=(
            frozenset({PackedLoadGrouping.PerLane, PackedLoadGrouping.LanePair})
            if quant_type in ("Q4_K", "Q5_K", "Q6_K")
            else frozenset({PackedLoadGrouping.PerLane})
        ),
        padded_depth_values=(
            frozenset({32, 64})
            if quant_type in ("Q3_K", "Q6_K", "Q8_0")
            else frozenset({32})
        ),
        pipeline_n_values=(
            frozenset({64, 128})
            if quant_type in ("Q2_K", "Q4_K", "Q6_K")
            else frozenset({128})
        ),
        pipeline_depth_values=(
            frozenset({32, 64}) if quant_type == "Q5_K" else frozenset({32})
        ),
        next_packed_depth_values=(
            frozenset({32, 64}) if quant_type == "Q6_K" else frozenset({32})
        ),
        decoder=BackwardDecoderCapability(
            payload_register_count=(
                4 if quant_type in ("Q2_K", "Q4_K", "Q8_0", "IQ2_S", "IQ2_XXS") else 8
            ),
            packed_loads_per_row=(
                3
                if quant_type == "Q2_K"
                else 5
                if quant_type in ("Q3_K", "Q4_K")
                else 4
                if quant_type == "Q6_K"
                else 2
                if quant_type == "Q8_0"
                else 5
                if quant_type == "IQ2_S"
                else 2
                if quant_type == "IQ2_XXS"
                else 6
            ),
            scalar_packed_loads_per_row=5 if quant_type == "Q8_0" else None,
            vector_metadata_packed_loads_per_row=3 if quant_type == "Q5_K" else None,
            packed_pairing_temporary_count=11 if quant_type == "Q3_K" else None,
            packed_vopd_temporary_base=(
                7 if quant_type == "Q6_K" else 5 if quant_type == "Q8_0" else None
            ),
            row_address=(
                BackwardPackedRowAddress.Block32
                if quant_type == "Q8_0"
                else BackwardPackedRowAddress.SuperBlock256
            ),
        ),
        address=BackwardAddressCapability(
            extended_quant_address_register_count=(
                2 if quant_type in ("Q3_K", "Q6_K") else 0
            )
        ),
        quant_register_shape=(
            BackwardQuantRegisterShape(1, 2, None)
            if quant_type == "Q2_K"
            else BackwardQuantRegisterShape(1, 2, None)
            if quant_type == "Q3_K"
            else BackwardQuantRegisterShape(1, 3, None)
            if quant_type == "Q4_K"
            else BackwardQuantRegisterShape(
                1,
                3,
                None,
                vector_metadata_shape=BackwardQuantRegisterShape(4, 0, 1),
            )
            if quant_type == "Q5_K"
            else BackwardQuantRegisterShape(1, 1, None)
            if quant_type == "Q6_K"
            else BackwardQuantRegisterShape(1, 4, None)
            if quant_type == "IQ2_S"
            else BackwardQuantRegisterShape(1, 2, None)
            if quant_type == "IQ2_XXS"
            else BackwardQuantRegisterShape(1, 0, 0)
        ),
    )


@dataclass(frozen=True)
class BackwardProblemContract:
    problem_size: ProblemSize
    quant_type: str
    quant_format: QuantFormat
    mechanism: BackwardMechanismContract
    kernel_language: str = "Assembly"
    isa: tuple[int, int, int] = (11, 5, 1)
    wavefront_size: int = 32
    code_object_version: int = 5

    @classmethod
    def for_problem_spec(
        cls, problem_size: ProblemSize, quant_type: str
    ) -> BackwardProblemContract:
        return cls(
            problem_size=problem_size,
            quant_type=quant_type,
            quant_format=BACKWARD_QUANT_FORMATS[quant_type],
            mechanism=backward_mechanism_contract(quant_type),
        )


@dataclass(frozen=True)
class BackwardGeometrySpec:
    isa: tuple[int, int, int]
    wavefront_size: int
    work_group: tuple[int, int, int]
    matrix_instruction: tuple[int, ...]
    macro_tile0: int
    macro_tile1: int
    depth_u: int
    work_group_mapping: int

    @property
    def num_threads(self) -> int:
        return self.work_group[0] * self.work_group[1] * self.work_group[2]

    def validate(self, contract: BackwardProblemContract) -> None:
        assert self.isa == contract.isa
        assert self.wavefront_size == contract.wavefront_size
        assert len(self.work_group) == 3
        assert min(self.work_group) > 0
        assert self.work_group[0] == self.wavefront_size
        assert self.work_group[2] == 1
        assert len(self.matrix_instruction) == 9
        assert self.matrix_instruction[:5] == (16, 16, 16, 1, 1)
        m_tiles, n_tiles, wave_count, k_groups = self.matrix_instruction[5:]
        assert m_tiles > 0
        assert n_tiles >= 2 and not (n_tiles & (n_tiles - 1))
        assert n_tiles % 2 == 0
        assert min(wave_count, k_groups) > 0
        assert self.num_threads == self.wavefront_size * wave_count
        assert self.work_group[1] == wave_count
        assert k_groups == 1
        assert self.macro_tile0 == 16 * m_tiles * wave_count
        assert self.macro_tile1 == 16 * n_tiles
        assert self.depth_u > 0 and self.depth_u % 32 == 0
        work_group_mapping_shift(self.work_group_mapping)


@dataclass(frozen=True)
class BackwardMemorySpec:
    lds_pad_b: int
    lds_swizzle_chunk_b: int

    @property
    def global_read_vector_width_a(self) -> int:
        return 16

    @property
    def global_read_vector_width_b(self) -> int:
        return 16

    @property
    def local_read_vector_width(self) -> int:
        return 16

    @property
    def transpose_lds(self) -> int:
        return 0

    @property
    def lds_block_size_per_pad_b(self) -> int:
        return 0

    def validate(self) -> None:
        assert self.lds_pad_b >= 0 and self.lds_pad_b % 8 == 0
        assert self.lds_swizzle_chunk_b in (0, 4, 8, 16)
        assert not (self.lds_pad_b and self.lds_swizzle_chunk_b)


@dataclass(frozen=True)
class BackwardPipelineSpec:
    global_read_prefetch: int
    local_read_prefetch: int
    lds_buffering: LdsBuffering
    iteration: BackwardIterationPolicy
    prefetch_packed_weight: bool
    prefetch_next_packed_weight: bool
    packed_load_grouping: PackedLoadGrouping

    @property
    def decoded_b_pipeline(self) -> bool:
        return self.lds_buffering is LdsBuffering.Double

    @property
    def prefetches_next_packed_tile(self) -> bool:
        return self.prefetch_next_packed_weight

    @property
    def uses_prefetched_local_read(self) -> bool:
        return self.local_read_prefetch == 2

    @property
    def uses_two_global_reads(self) -> bool:
        return self.global_read_prefetch == 2

    def validate(
        self,
        mechanism: BackwardMechanismContract,
        geometry: BackwardGeometrySpec,
    ) -> None:
        assert self.global_read_prefetch in (1, 2)
        assert self.local_read_prefetch in (1, 2)
        assert self.iteration.supports_global_read_prefetch(self.global_read_prefetch)
        assert self.iteration.supports_local_read_prefetch(self.local_read_prefetch)
        assert self.packed_load_grouping in mechanism.packed_load_grouping_values
        assert self.prefetch_packed_weight
        if self.lds_buffering is LdsBuffering.Double:
            assert self.iteration.prefetch_activation
            assert self.global_read_prefetch == 2
            assert self.local_read_prefetch == 1
            assert mechanism.supports_pipeline_n(geometry.macro_tile1)
            assert mechanism.supports_pipeline_depth(geometry.depth_u)
        if self.prefetch_next_packed_weight:
            assert mechanism.supports_next_packed_depth(geometry.depth_u)


@dataclass(frozen=True)
class BackwardDecodeSpec:
    decoder_width: int
    dependency_width: int = 1
    extraction: BackwardExtraction | None = None
    pairing: BackwardPairing | None = None
    bitfield_shift_placement: BackwardBitfieldShiftPlacement | None = None
    metadata_load: BackwardMetadataLoad | None = None

    @property
    def hoists_bitfield_shift(self) -> bool:
        return self.bitfield_shift_placement is BackwardBitfieldShiftPlacement.Hoisted

    @property
    def uses_vector_metadata_load(self) -> bool:
        return self.metadata_load is BackwardMetadataLoad.Vector

    def validate(
        self,
        quant_type: str,
        geometry: BackwardGeometrySpec,
    ) -> None:
        assert self.decoder_width == 16
        assert self.dependency_width > 0
        inactive = (
            self.extraction,
            self.pairing,
            self.bitfield_shift_placement,
            self.metadata_load,
        )
        if quant_type in {"Q2_K", "Q4_K"}:
            assert self.dependency_width in (1, 4)
            assert not any(value is not None for value in inactive[0:])
            return
        assert self.dependency_width == 1
        if quant_type == "Q3_K":
            assert self.extraction in {
                BackwardExtraction.Packed,
                BackwardExtraction.Scalar,
            }
            if self.extraction is BackwardExtraction.Packed:
                assert self.pairing in {
                    BackwardPairing.Partial,
                    BackwardPairing.Full,
                }
            else:
                assert self.pairing is None
            assert self.bitfield_shift_placement is None
            assert self.metadata_load is None
            if self.pairing is BackwardPairing.Full:
                rows = (
                    geometry.depth_u
                    * geometry.macro_tile1
                    // (min(geometry.num_threads, 128) * self.decoder_width)
                )
                assert rows <= 2
            return
        if quant_type == "Q5_K":
            assert self.extraction in {
                BackwardExtraction.Packed,
                BackwardExtraction.Scalar,
            }
            assert self.pairing is None
            assert self.bitfield_shift_placement is not None
            assert self.metadata_load is not None
            return
        if quant_type in {"Q6_K", "Q8_0"}:
            assert self.extraction is not None
            assert self.pairing is None
            assert self.bitfield_shift_placement is None
            assert self.metadata_load is None
            return
        assert not any(value is not None for value in inactive)


@dataclass(frozen=True)
class BackwardStoreSpec:
    priority: BackwardStorePriority

    @property
    def raises_priority(self) -> bool:
        return self.priority is BackwardStorePriority.Raised


@dataclass(frozen=True)
class BackwardKernelSpec:
    geometry: BackwardGeometrySpec
    memory: BackwardMemorySpec
    pipeline: BackwardPipelineSpec
    decode: BackwardDecodeSpec
    store: BackwardStoreSpec

    @property
    def mi_wave_tile(self) -> tuple[int, int]:
        return (
            self.geometry.matrix_instruction[5],
            self.geometry.matrix_instruction[6],
        )

    def to_mapping(self, quant_type: str) -> dict[str, object]:
        memory = self.memory
        if memory.lds_pad_b > 0 and memory.lds_swizzle_chunk_b == 0:
            lds_layout: dict[str, object] = {
                "Kind": "Padded",
                "LdsPadB": memory.lds_pad_b,
            }
        elif memory.lds_pad_b == 0 and memory.lds_swizzle_chunk_b > 0:
            lds_layout = {
                "Kind": "Swizzled",
                "LdsSwizzleChunkB": memory.lds_swizzle_chunk_b,
            }
        elif memory.lds_pad_b == 0 and memory.lds_swizzle_chunk_b == 0:
            lds_layout = {"Kind": "Linear"}
        else:
            raise AssertionError
        mapping: dict[str, object] = {
            "Geometry": {
                "WorkGroup": list(self.geometry.work_group),
                "MIWaveTile": list(self.mi_wave_tile),
                "DepthU": self.geometry.depth_u,
                "WorkGroupMapping": self.geometry.work_group_mapping,
            },
            "Memory": {"LdsLayout": lds_layout},
            "Pipeline": {
                "PrefetchGlobalRead": self.pipeline.global_read_prefetch,
                "PrefetchLocalRead": self.pipeline.local_read_prefetch,
                "LdsBuffering": self.pipeline.lds_buffering.value,
                "PrefetchActivation": self.pipeline.iteration.prefetch_activation,
                "InterleaveWmmaWaits": (self.pipeline.iteration.interleave_wmma_waits),
                "PrefetchPackedWeight": self.pipeline.prefetch_packed_weight,
                "PrefetchNextPackedWeight": (self.pipeline.prefetch_next_packed_weight),
                "PackedLoadGrouping": self.pipeline.packed_load_grouping.value,
            },
            "Store": {"Priority": self.store.priority.value},
        }
        if quant_type in {"Q2_K", "Q4_K"}:
            if self.decode.dependency_width != 1:
                mapping["Decode"] = {
                    "DecodeDependencyWidth": self.decode.dependency_width
                }
        elif quant_type == "Q3_K":
            extraction = self.decode.extraction
            pairing = self.decode.pairing
            if extraction is BackwardExtraction.Packed:
                assert pairing is not None
                mapping["Decode"] = {
                    "PayloadExtraction": extraction.value,
                    "Pairing": pairing.value,
                }
            elif extraction is BackwardExtraction.Scalar:
                assert pairing is None
                mapping["Decode"] = {"PayloadExtraction": extraction.value}
            else:
                raise AssertionError
        elif quant_type == "Q5_K":
            assert self.decode.extraction is not None
            assert self.decode.bitfield_shift_placement is not None
            assert self.decode.metadata_load is not None
            mapping["Decode"] = {
                "PayloadExtraction": self.decode.extraction.value,
                "BitfieldShiftPlacement": self.decode.bitfield_shift_placement.value,
                "MetadataLoad": self.decode.metadata_load.value,
            }
        elif quant_type in {"Q6_K", "Q8_0"}:
            assert self.decode.extraction is not None
            mapping["Decode"] = {"PayloadExtraction": self.decode.extraction.value}
        elif quant_type in {"IQ2_S", "IQ2_XXS"}:
            pass
        else:
            raise AssertionError
        return mapping

    @classmethod
    def from_mapping(cls, value: object, quant_type: str):
        decode_required = quant_type not in (
            "Q2_K",
            "Q4_K",
            "IQ2_S",
            "IQ2_XXS",
        )
        decode_optional = quant_type in ("Q2_K", "Q4_K")
        item = _strict_mapping_optional(
            value,
            name="BackwardKernelSpec",
            required=frozenset(
                {"Geometry", "Memory", "Pipeline", "Store"}
                | ({"Decode"} if decode_required else set())
            ),
            optional=frozenset({"Decode"}) if decode_optional else frozenset(),
        )
        geometry = _strict_mapping(
            item["Geometry"],
            name="BackwardKernelSpec.Geometry",
            keys=frozenset({"WorkGroup", "MIWaveTile", "DepthU", "WorkGroupMapping"}),
        )
        work_group = _integer_tuple(geometry, "WorkGroup", 3)
        mi_wave_tile = cast(
            tuple[int, int],
            _integer_tuple(geometry, "MIWaveTile", 2),
        )
        if min(*work_group, *mi_wave_tile) <= 0:
            raise SchemaError("backward workgroup and wave tile must be positive")
        thread_count = work_group[0] * work_group[1] * work_group[2]
        if thread_count % 32:
            raise SchemaError("backward workgroup must contain whole wave32 waves")
        wave_count = thread_count // 32
        matrix_instruction = (
            16,
            16,
            16,
            1,
            1,
            mi_wave_tile[0],
            mi_wave_tile[1],
            wave_count,
            1,
        )
        memory_item = _strict_mapping(
            item["Memory"],
            name="BackwardKernelSpec.Memory",
            keys=frozenset({"LdsLayout"}),
        )
        lds_layout_value = memory_item["LdsLayout"]
        if not isinstance(lds_layout_value, dict):
            raise SchemaError("BackwardKernelSpec.Memory.LdsLayout must be a mapping")
        kind = _string(lds_layout_value.get("Kind"), "LdsLayout.Kind")
        if kind == "Padded":
            layout_item = _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.Memory.LdsLayout",
                keys=frozenset({"Kind", "LdsPadB"}),
            )
            lds_pad_b = _integer(layout_item, "LdsPadB")
            lds_swizzle_chunk_b = 0
        elif kind == "Swizzled":
            layout_item = _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.Memory.LdsLayout",
                keys=frozenset({"Kind", "LdsSwizzleChunkB"}),
            )
            lds_pad_b = 0
            lds_swizzle_chunk_b = _integer(layout_item, "LdsSwizzleChunkB")
        elif kind == "Linear":
            _strict_mapping(
                lds_layout_value,
                name="BackwardKernelSpec.Memory.LdsLayout",
                keys=frozenset({"Kind"}),
            )
            lds_pad_b = 0
            lds_swizzle_chunk_b = 0
        else:
            raise SchemaError(f"invalid backward LDS layout kind {kind!r}")
        pipeline_item = _strict_mapping(
            item["Pipeline"],
            name="BackwardKernelSpec.Pipeline",
            keys=frozenset(
                {
                    "PrefetchGlobalRead",
                    "PrefetchLocalRead",
                    "LdsBuffering",
                    "PrefetchActivation",
                    "InterleaveWmmaWaits",
                    "PrefetchPackedWeight",
                    "PrefetchNextPackedWeight",
                    "PackedLoadGrouping",
                }
            ),
        )
        pipeline = BackwardPipelineSpec(
            global_read_prefetch=_integer(pipeline_item, "PrefetchGlobalRead"),
            local_read_prefetch=_integer(pipeline_item, "PrefetchLocalRead"),
            lds_buffering=_serialized_enum(
                LdsBuffering,
                pipeline_item["LdsBuffering"],
                "Pipeline.LdsBuffering",
            ),
            iteration=BackwardIterationPolicy(
                prefetch_activation=_boolean(pipeline_item, "PrefetchActivation"),
                interleave_wmma_waits=_boolean(pipeline_item, "InterleaveWmmaWaits"),
            ),
            prefetch_packed_weight=_boolean(pipeline_item, "PrefetchPackedWeight"),
            prefetch_next_packed_weight=_boolean(
                pipeline_item, "PrefetchNextPackedWeight"
            ),
            packed_load_grouping=_serialized_enum(
                PackedLoadGrouping,
                pipeline_item["PackedLoadGrouping"],
                "Pipeline.PackedLoadGrouping",
            ),
        )
        decode = BackwardDecodeSpec(decoder_width=16)
        if quant_type in {"Q2_K", "Q4_K"} and "Decode" in item:
            decode_item = _strict_mapping(
                item["Decode"],
                name="BackwardKernelSpec.Decode",
                keys=frozenset({"DecodeDependencyWidth"}),
            )
            dependency_width = _integer(decode_item, "DecodeDependencyWidth")
            if dependency_width != 4:
                raise SchemaError("decode dependency width must be 4 when specified")
            decode = BackwardDecodeSpec(16, dependency_width=dependency_width)
        elif decode_required:
            if quant_type == "Q3_K":
                decode_item = _strict_mapping_optional(
                    item["Decode"],
                    name="BackwardKernelSpec.Decode",
                    required=frozenset({"PayloadExtraction"}),
                    optional=frozenset({"Pairing"}),
                )
                extraction = _serialized_enum(
                    BackwardExtraction,
                    decode_item["PayloadExtraction"],
                    "Decode.PayloadExtraction",
                )
                if extraction is BackwardExtraction.Packed:
                    if "Pairing" not in decode_item:
                        raise SchemaError("packed Q3 extraction requires Pairing")
                    pairing = _serialized_enum(
                        BackwardPairing,
                        decode_item["Pairing"],
                        "Decode.Pairing",
                    )
                elif extraction is BackwardExtraction.Scalar:
                    if "Pairing" in decode_item:
                        raise SchemaError(
                            "scalar Q3 extraction does not accept Pairing"
                        )
                    pairing = None
                else:
                    raise SchemaError("Q3 decode requires Packed or Scalar extraction")
                decode = BackwardDecodeSpec(
                    16,
                    extraction=extraction,
                    pairing=pairing,
                )
            elif quant_type == "Q5_K":
                decode_item = _strict_mapping(
                    item["Decode"],
                    name="BackwardKernelSpec.Decode",
                    keys=frozenset(
                        {
                            "PayloadExtraction",
                            "BitfieldShiftPlacement",
                            "MetadataLoad",
                        }
                    ),
                )
                decode = BackwardDecodeSpec(
                    16,
                    extraction=_serialized_enum(
                        BackwardExtraction,
                        decode_item["PayloadExtraction"],
                        "Decode.PayloadExtraction",
                    ),
                    bitfield_shift_placement=_serialized_enum(
                        BackwardBitfieldShiftPlacement,
                        decode_item["BitfieldShiftPlacement"],
                        "Decode.BitfieldShiftPlacement",
                    ),
                    metadata_load=_serialized_enum(
                        BackwardMetadataLoad,
                        decode_item["MetadataLoad"],
                        "Decode.MetadataLoad",
                    ),
                )
            elif quant_type in {"Q6_K", "Q8_0"}:
                decode_item = _strict_mapping(
                    item["Decode"],
                    name="BackwardKernelSpec.Decode",
                    keys=frozenset({"PayloadExtraction"}),
                )
                decode = BackwardDecodeSpec(
                    16,
                    extraction=_serialized_enum(
                        BackwardExtraction,
                        decode_item["PayloadExtraction"],
                        "Decode.PayloadExtraction",
                    ),
                )
            else:
                raise SchemaError(f"unsupported backward quant type {quant_type!r}")
        store_item = _strict_mapping(
            item["Store"],
            name="BackwardKernelSpec.Store",
            keys=frozenset({"Priority"}),
        )
        return cls(
            geometry=BackwardGeometrySpec(
                isa=(11, 5, 1),
                wavefront_size=32,
                work_group=work_group,
                matrix_instruction=matrix_instruction,
                macro_tile0=16 * wave_count * mi_wave_tile[0],
                macro_tile1=16 * mi_wave_tile[1],
                depth_u=_integer(geometry, "DepthU"),
                work_group_mapping=_integer(geometry, "WorkGroupMapping"),
            ),
            memory=BackwardMemorySpec(lds_pad_b, lds_swizzle_chunk_b),
            pipeline=pipeline,
            decode=decode,
            store=BackwardStoreSpec(
                priority=_serialized_enum(
                    BackwardStorePriority, store_item["Priority"], "Store.Priority"
                )
            ),
        )

    def validate(self, contract: BackwardProblemContract) -> None:
        self.geometry.validate(contract)
        self.memory.validate()
        self.pipeline.validate(contract.mechanism, self.geometry)
        self.decode.validate(contract.quant_type, self.geometry)


@dataclass(frozen=True)
class DerivedBackwardState:
    contract: BackwardProblemContract
    spec: BackwardKernelSpec

    @classmethod
    def from_problem_spec(
        cls,
        problem_size: ProblemSize,
        quant_type: str,
        spec: BackwardKernelSpec,
    ) -> DerivedBackwardState:
        return cls.from_contract_spec(
            BackwardProblemContract.for_problem_spec(problem_size, quant_type), spec
        )

    @classmethod
    def from_contract_spec(
        cls, contract: BackwardProblemContract, spec: BackwardKernelSpec
    ):
        return cls(contract=contract, spec=spec)
