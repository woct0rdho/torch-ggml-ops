"""Strict identities for the fixed-group Q8_0 backward experiment."""

import hashlib
import json
from dataclasses import dataclass, replace
from typing import ClassVar

from typing_extensions import Self

from .model import BackwardSolution
from .quant_formats import BACKWARD_QUANT_FORMATS
from .schema import SchemaError
from .schema import integer as _integer
from .schema import strict_mapping as _mapping


@dataclass(frozen=True)
class FixedBackwardProblem:
    quant_data_type: str
    tokens: int
    output_features: int
    input_features: int
    groups: int

    @classmethod
    def deepseek_q8_0(cls, tokens: int) -> Self:
        return cls("Q8_0", tokens, 1024, 4096, 8)

    @property
    def packed_row_bytes(self) -> int:
        quant = BACKWARD_QUANT_FORMATS[self.quant_data_type]
        if self.input_features % quant.block_values:
            raise ValueError("fixed backward input features are not block divisible")
        return self.input_features // quant.block_values * quant.block_bytes

    @property
    def bytes_per_group(self) -> int:
        return self.output_features * self.packed_row_bytes


@dataclass(frozen=True)
class FixedBackwardSolution:
    compute: BackwardSolution
    work_group_order: str
    store_schedule: str
    group_axis: str
    row_layout: str
    packed_weight_layout: str

    @classmethod
    def q8_0_m256_n64_k32(cls) -> Self:
        compute = replace(
            BackwardSolution.pilot(),
            matrix_instruction=(16, 16, 16, 1, 1, 4, 4, 4, 1),
            macro_tile0=256,
            macro_tile1=64,
            prefetch_global_read=2,
            schedule_iter_alg=5,
            lds_pad_b=8,
        )
        return cls(
            compute=compute,
            work_group_order="NMajor",
            store_schedule="ElementSerial",
            group_axis="WorkGroupZ",
            row_layout="TokenMajorGroupInterleaved",
            packed_weight_layout="GroupMajorRows",
        )

    @classmethod
    def q8_0_m256_n64_k32_m_major(cls) -> Self:
        return replace(cls.q8_0_m256_n64_k32(), work_group_order="MMajor")

    @classmethod
    def q8_0_m128_n128_k32(cls) -> Self:
        selected = cls.q8_0_m256_n64_k32()
        return replace(
            selected,
            compute=replace(
                selected.compute,
                matrix_instruction=(16, 16, 16, 1, 1, 2, 8, 4, 1),
                macro_tile0=128,
                macro_tile1=128,
            ),
        )

    @classmethod
    def q8_0_m128_n128_k32_packed_vopd(cls) -> Self:
        selected = cls.q8_0_m128_n128_k32()
        return replace(
            selected,
            compute=replace(selected.compute, q8_0_extraction="packed_vopd"),
        )

    @classmethod
    def q8_0_m64_n128_k32(cls) -> Self:
        selected = cls.q8_0_m128_n128_k32()
        return replace(
            selected,
            compute=replace(
                selected.compute,
                matrix_instruction=(16, 16, 16, 1, 1, 1, 8, 4, 1),
                macro_tile0=64,
            ),
        )

    @classmethod
    def q8_0_m128_n128_k32_clause_store(cls) -> Self:
        return replace(cls.q8_0_m128_n128_k32(), store_schedule="ClausePairs")

    @classmethod
    def q8_0_m128_n128_k64(cls) -> Self:
        selected = cls.q8_0_m128_n128_k32()
        return replace(
            selected,
            compute=replace(selected.compute, depth_u=64, schedule_iter_alg=4),
        )

    @classmethod
    def q8_0_m128_n128_k64_packed_vopd(cls) -> Self:
        selected = cls.q8_0_m128_n128_k64()
        return replace(
            selected,
            compute=replace(selected.compute, q8_0_extraction="packed_vopd"),
        )

    @classmethod
    def q8_0_m128_n128_k64_next_prefetch(cls) -> Self:
        selected = cls.q8_0_m128_n128_k64()
        return replace(
            selected,
            compute=replace(selected.compute, prefetch_packed_weight_next=True),
        )

    @classmethod
    def selected_q8_0(cls) -> Self:
        return cls.q8_0_m128_n128_k64()


@dataclass(frozen=True)
class FixedBackwardSolutionKey:
    problem: FixedBackwardProblem
    solution: FixedBackwardSolution

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"ProblemContract", "Problem", "KernelSpec"}
    )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        item = _mapping(value, "FixedBackwardSolutionKey", cls._KEYS)
        from .fixed_grouped_mmq_bwd_spec import (
            FixedBackwardKernelSpec,
            FixedBackwardProblemContract,
        )

        contract = FixedBackwardProblemContract.from_mapping(item["ProblemContract"])
        problem_item = _mapping(
            item["Problem"], "FixedBackwardProblem", frozenset({"tokens"})
        )
        problem = contract.problem(_integer(problem_item["tokens"], "tokens"))
        spec = FixedBackwardKernelSpec.from_mapping(item["KernelSpec"])
        solution = spec.to_solution(contract, problem.tokens)
        if FixedBackwardProblemContract.from_problem(problem, solution) != contract:
            raise SchemaError("fixed backward contract does not round-trip canonically")
        key = cls(problem, solution)
        from .fixed_grouped_mmq_bwd_validation import (
            validate_fixed_backward_solution_key,
        )

        validate_fixed_backward_solution_key(key)
        return key

    def to_mapping(self) -> dict[str, object]:
        from .fixed_grouped_mmq_bwd_spec import (
            FixedBackwardKernelSpec,
            FixedBackwardProblemContract,
        )

        contract = FixedBackwardProblemContract.from_problem(
            self.problem, self.solution
        )
        return {
            "ProblemContract": contract.to_mapping(),
            "Problem": {"tokens": self.problem.tokens},
            "KernelSpec": FixedBackwardKernelSpec.from_solution(
                self.solution
            ).to_mapping(),
        }

    @property
    def hash(self) -> str:
        identity = {
            "ArtifactKind": "ExactKernel",
            "KernelFamily": "FixedGroupedBackward",
            **self.to_mapping(),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return f"ggsol_{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    @property
    def kernel_name(self) -> str:
        problem = self.problem
        return (
            "torch_ggml_ops_ggtensile_gfx1151_v1_fixed_grouped_mmq_bwd_q8_0_"
            f"t{problem.tokens}_n{problem.input_features}_k{problem.output_features}_"
            f"{self.hash[6:]}"
        )
