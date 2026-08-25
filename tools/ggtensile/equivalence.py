"""Reusable structural evidence for exact GGTensile kernels."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .family_registry import (
    abi_for_instance,
    family_for_instance,
    instance_hash,
    instance_name,
    launch_for_instance,
    mapping_for_instance,
    writer_for_instance,
)
from .kernel_instance import KernelInstance
from .toolchain import Toolchain


@dataclass(frozen=True)
class StaticInstructionCounts:
    instructions: int
    wmma: int
    barriers: int
    valu_issues: int
    valu_operations: int
    vopd: int
    vmem: int
    lds: int
    waits: int
    clauses: int


@dataclass(frozen=True)
class StructuralEvidence:
    family: str
    quant_data_type: str
    problem: dict[str, object]
    exact_hash: str
    symbol: str
    source_sha256: str
    normalized_source_sha256: str
    abi_name: str
    kernarg_segment_size: int
    abi_arguments: tuple[tuple[str, int, int, str, str], ...]
    work_group: tuple[int, int, int]
    grid: tuple[int, int, int]
    static_group_segment_bytes: int
    ownership: str
    row_task_rows: int | None
    row_task_capacity: int | None
    route_split_factor: int
    total_vgprs: int
    total_sgprs: int
    instruction_counts: StaticInstructionCounts

    def to_mapping(self) -> dict[str, object]:
        counts = self.instruction_counts
        return {
            "KernelFamily": self.family,
            "QuantDataType": self.quant_data_type,
            "Problem": self.problem,
            "ExactHash": self.exact_hash,
            "Symbol": self.symbol,
            "SourceSHA256": self.source_sha256,
            "NormalizedSourceSHA256": self.normalized_source_sha256,
            "ABIName": self.abi_name,
            "KernargSegmentSize": self.kernarg_segment_size,
            "ABIArguments": self.abi_arguments,
            "WorkGroup": self.work_group,
            "Grid": self.grid,
            "StaticGroupSegmentBytes": self.static_group_segment_bytes,
            "Ownership": self.ownership,
            "RowTaskRows": self.row_task_rows,
            "RowTaskCapacity": self.row_task_capacity,
            "RouteSplitFactor": self.route_split_factor,
            "TotalVGPRs": self.total_vgprs,
            "TotalSGPRs": self.total_sgprs,
            "InstructionCounts": {
                "Instructions": counts.instructions,
                "Wmma": counts.wmma,
                "Barriers": counts.barriers,
                "ValuIssues": counts.valu_issues,
                "ValuOperations": counts.valu_operations,
                "Vopd": counts.vopd,
                "Vmem": counts.vmem,
                "Lds": counts.lds,
                "Waits": counts.waits,
                "Clauses": counts.clauses,
            },
        }


def normalized_source(source: str, symbol: str) -> str:
    return source.replace(symbol, "<KERNEL_SYMBOL>")


def static_instruction_counts(source: str) -> StaticInstructionCounts:
    instructions: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith((".", "//", "---", "..."))
            or stripped.endswith(":")
        ):
            continue
        instructions.append(stripped.split("//", 1)[0].rstrip())
    mnemonics = tuple(line.split(None, 1)[0] for line in instructions)
    vopd = sum(" :: " in line for line in instructions)
    valu_issues = sum(
        mnemonic.startswith("v_") and not mnemonic.startswith("v_wmma_")
        for mnemonic in mnemonics
    )
    return StaticInstructionCounts(
        instructions=len(instructions),
        wmma=sum(mnemonic.startswith("v_wmma_") for mnemonic in mnemonics),
        barriers=mnemonics.count("s_barrier"),
        valu_issues=valu_issues,
        valu_operations=valu_issues + vopd,
        vopd=vopd,
        vmem=sum(
            mnemonic.startswith(("global_", "flat_", "buffer_", "scratch_"))
            and mnemonic != "buffer_gl0_inv"
            for mnemonic in mnemonics
        ),
        lds=sum(mnemonic.startswith("ds_") for mnemonic in mnemonics),
        waits=sum(mnemonic.startswith("s_waitcnt") for mnemonic in mnemonics),
        clauses=mnemonics.count("s_clause"),
    )


def capture_structural_evidence(
    instance: KernelInstance, toolchain: Toolchain
) -> StructuralEvidence:
    writer = writer_for_instance(instance, toolchain)
    plan = writer.emission_plan()
    source = plan.render(writer.assembler)
    mapping = mapping_for_instance(instance)
    problem = mapping["Problem"]
    if not isinstance(problem, dict):
        raise TypeError("kernel instance problem is not a canonical dictionary")
    abi_name, abi = abi_for_instance(instance)
    launch = launch_for_instance(instance)
    kernel_name = instance_name(instance)
    normalized = normalized_source(source, kernel_name)
    return StructuralEvidence(
        family=family_for_instance(instance).value,
        quant_data_type=instance.problem_type.quant_data_type,
        problem=problem,
        exact_hash=instance_hash(instance),
        symbol=kernel_name,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        normalized_source_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        abi_name=abi_name.value,
        kernarg_segment_size=abi.segment_size,
        abi_arguments=abi.metadata_arguments,
        work_group=launch.work_group,
        grid=launch.grid,
        static_group_segment_bytes=launch.static_group_segment_bytes,
        ownership=launch.ownership,
        row_task_rows=launch.row_task_rows,
        row_task_capacity=launch.row_task_capacity,
        route_split_factor=launch.route_split_factor,
        total_vgprs=plan.total_vgprs,
        total_sgprs=plan.total_sgprs,
        instruction_counts=static_instruction_counts(source),
    )


def normalized_executable_text(disassembly: str, symbol: str) -> str:
    """Remove addresses and symbol identity from an llvm-objdump function body."""
    text = disassembly.replace(symbol, "<KERNEL_SYMBOL>")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+", "", line)
        if line and not line.startswith(("Disassembly of section", "file format")):
            lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def executable_text_sha256(
    code_object: Path, instance: KernelInstance, toolchain: Toolchain
) -> str:
    kernel_name = instance_name(instance)
    text = normalized_executable_text(
        toolchain.disassembly_output(code_object), kernel_name
    )
    return hashlib.sha256(text.encode()).hexdigest()
