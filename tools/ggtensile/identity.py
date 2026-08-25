"""Canonical target, family, problem-type, and JSON naming identities."""

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .quant_formats import BACKWARD_QUANT_FORMATS, GROUPED_QUANT_FORMATS, QUANT_FORMATS
from .schema import SchemaError, integer_tuple, strict_mapping, string


class KernelFamily(str, Enum):
    OrdinaryForward = "OrdinaryForward"
    OrdinaryBackward = "OrdinaryBackward"
    GroupedForward = "GroupedForward"
    GroupedBackward = "GroupedBackward"
    GroupedForwardPair = "GroupedForwardPair"
    GroupedBackwardPair = "GroupedBackwardPair"
    FixedGroupedForward = "FixedGroupedForward"
    FixedGroupedBackward = "FixedGroupedBackward"

    @property
    def is_forward(self) -> bool:
        return self in {
            KernelFamily.OrdinaryForward,
            KernelFamily.GroupedForward,
            KernelFamily.GroupedForwardPair,
            KernelFamily.FixedGroupedForward,
        }


class MatrixInstructionSet(str, Enum):
    WmmaV1 = "WmmaV1"


@dataclass(frozen=True)
class KernelTarget:
    isa: tuple[int, int, int]
    wavefront_size: int
    matrix_instruction_set: MatrixInstructionSet
    code_object_version: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "ISA": list(self.isa),
            "WavefrontSize": self.wavefront_size,
            "MatrixInstructionSet": self.matrix_instruction_set.value,
            "CodeObjectVersion": self.code_object_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "KernelTarget":
        item = strict_mapping(
            value,
            "Target",
            frozenset(
                {
                    "ISA",
                    "WavefrontSize",
                    "MatrixInstructionSet",
                    "CodeObjectVersion",
                }
            ),
        )
        actual = {
            "ISA": list(integer_tuple(item, "ISA", 3)),
            "WavefrontSize": item["WavefrontSize"],
            "MatrixInstructionSet": string(item, "MatrixInstructionSet"),
            "CodeObjectVersion": item["CodeObjectVersion"],
        }
        if actual != GFX1151_TARGET.to_mapping():
            raise SchemaError("Target must be gfx1151 wave32 WMMA V1 code object v5")
        return GFX1151_TARGET


GFX1151_TARGET = KernelTarget((11, 5, 1), 32, MatrixInstructionSet.WmmaV1, 5)


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def exact_key_hash(value: object, *, prefix: str = "ggsol") -> str:
    return f"{prefix}_{canonical_sha256(value)[:16]}"


_FAMILY_QUANT_TYPES: dict[KernelFamily, frozenset[str]] = {
    KernelFamily.OrdinaryForward: frozenset({"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}),
    KernelFamily.OrdinaryBackward: frozenset({"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}),
    KernelFamily.GroupedForward: frozenset({"Q2_K", "Q4_K", "Q5_K", "IQ2_S"}),
    KernelFamily.GroupedBackward: frozenset({"Q2_K", "Q4_K", "Q5_K", "IQ2_S"}),
    KernelFamily.GroupedForwardPair: frozenset({"Q3_K", "IQ2_S", "IQ2_XXS"}),
    KernelFamily.GroupedBackwardPair: frozenset({"Q3_K", "IQ2_S", "IQ2_XXS"}),
    KernelFamily.FixedGroupedForward: frozenset({"Q8_0"}),
    KernelFamily.FixedGroupedBackward: frozenset({"Q8_0"}),
}


def problem_type_mapping(
    family: KernelFamily, quant_data_type: str
) -> dict[str, object]:
    if quant_data_type not in _FAMILY_QUANT_TYPES[family]:
        raise ValueError(f"{family.value} does not implement {quant_data_type!r}")
    formats = (
        GROUPED_QUANT_FORMATS
        if family
        in {
            KernelFamily.GroupedForward,
            KernelFamily.GroupedForwardPair,
        }
        else (QUANT_FORMATS if family.is_forward else BACKWARD_QUANT_FORMATS)
    )
    quant = formats[quant_data_type]
    mapping: dict[str, object] = {
        "QuantDataType": quant_data_type,
        "DataTypeA": "Q8_1" if family.is_forward else "BFloat16",
        "DataTypeB": quant_data_type,
        "DestinationDataType": "BFloat16",
        "ComputeDataType": "Float32",
        "TransposeA": False,
        "TransposeB": family.is_forward,
    }
    if family.is_forward:
        mapping["ActivationLayout"] = quant.activation_layout
    return mapping


def quant_type_from_problem_type(value: object, family: KernelFamily) -> str:
    required = {
        "QuantDataType",
        "DataTypeA",
        "DataTypeB",
        "DestinationDataType",
        "ComputeDataType",
        "TransposeA",
        "TransposeB",
    }
    if family.is_forward:
        required.add("ActivationLayout")
    item = strict_mapping(value, "ProblemType", frozenset(required))
    quant_type = string(item, "QuantDataType")
    if dict(item) != problem_type_mapping(family, quant_type):
        raise SchemaError("ProblemType is not canonical for KernelFamily")
    return quant_type
