from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Self

from .schema import SchemaError
from .schema import integer as _integer
from .schema import strict_mapping as _strict_mapping

if TYPE_CHECKING:
    from .kernel_instance import KernelInstance


@dataclass(frozen=True)
class ProblemType:
    operation_type: str
    quant_data_type: str
    data_type_a: str
    data_type_b: str
    dest_data_type: str
    compute_data_type: str
    transpose_a: bool
    transpose_b: bool

    @classmethod
    def mmq_backward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {
            "Q2_K",
            "Q3_K",
            "Q4_K",
            "Q5_K",
            "Q6_K",
            "Q8_0",
            "IQ2_S",
            "IQ2_XXS",
        }:
            raise ValueError(f"unsupported MMQ backward quant type {quant_data_type!r}")
        return cls(
            operation_type="MMQBackward",
            quant_data_type=quant_data_type,
            data_type_a="BFloat16",
            data_type_b=quant_data_type,
            dest_data_type="BFloat16",
            compute_data_type="Float",
            transpose_a=False,
            transpose_b=False,
        )

    @classmethod
    def grouped_mmq_backward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {"Q2_K", "Q4_K", "Q5_K", "IQ2_S"}:
            raise ValueError(
                f"unsupported grouped MMQ backward quant type {quant_data_type!r}"
            )
        ordinary = cls.mmq_backward(quant_data_type)
        return replace(ordinary, operation_type="GroupedMMQBackward")

    @classmethod
    def mmq_forward(cls, quant_data_type: str) -> Self:
        if quant_data_type not in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}:
            raise ValueError(f"unsupported MMQ forward quant type {quant_data_type!r}")
        return cls(
            operation_type="MMQForward",
            quant_data_type=quant_data_type,
            data_type_a="Q8_1",
            data_type_b=quant_data_type,
            dest_data_type="BFloat16",
            compute_data_type="Float",
            transpose_a=False,
            transpose_b=True,
        )


@dataclass(frozen=True)
class ProblemSize:
    """Exact GEMM coordinates interpreted by the selected ProblemType."""

    m: int
    n: int
    k: int

    def to_mapping(self) -> dict[str, int]:
        return {"M": self.m, "N": self.n, "K": self.k}

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _strict_mapping(
            value,
            name="Problem",
            keys=frozenset({"M", "N", "K"}),
        )
        problem_size = cls(
            m=_integer(item, "M"),
            n=_integer(item, "N"),
            k=_integer(item, "K"),
        )
        if min(problem_size.m, problem_size.n, problem_size.k) <= 0:
            raise SchemaError("Problem dimensions must be positive")
        return problem_size


@dataclass(frozen=True)
class KernelArtifact:
    instance: KernelInstance
    kernel_name: str
    assembly_path: Path
    object_path: Path | None
    code_object_path: Path | None
    assembly_sha256: str
    vgpr_count: int
    sgpr_count: int
    lds_num_bytes: int

    def to_mapping(self) -> dict[str, object]:
        from .family_registry import instance_hash, instance_name, mapping_for_instance

        return {
            "KernelSpecKey": mapping_for_instance(self.instance),
            "KernelSpecHash": instance_hash(self.instance),
            "KernelName": instance_name(self.instance),
            "AssemblyPath": str(self.assembly_path),
            "ObjectPath": str(self.object_path) if self.object_path else None,
            "CodeObjectPath": (
                str(self.code_object_path) if self.code_object_path else None
            ),
            "AssemblySHA256": self.assembly_sha256,
            "NumVgpr": self.vgpr_count,
            "NumSgpr": self.sgpr_count,
            "LdsNumBytes": self.lds_num_bytes,
        }
