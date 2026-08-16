import ast
import dis
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType
from typing import Literal, Protocol

import pytest

from tools.ggtensile.campaign import DeploymentCatalog, load_catalog
from tools.ggtensile.inspection import ArtifactInspection, inspect_artifact
from tools.ggtensile.model import (
    ProblemType,
    SolutionKey,
)
from tools.ggtensile.toolchain import Toolchain

_CONFIG_DIR = Path("tools/ggtensile/configs")
_REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_WRITER_SOURCE_PATH = _REPO_ROOT / "tools/ggtensile/kernel_writer_assembly.py"
FWD_WRITER_SOURCE_PATH = (
    _REPO_ROOT / "tools/ggtensile/kernel_writer_assembly_mmq_fwd.py"
)
FWD_PHYSICAL_SOURCE_PATH = _REPO_ROOT / "tools/ggtensile/mmq_fwd_physical.py"
FWD_LOWERING_SOURCE_PATHS = tuple(
    sorted((_REPO_ROOT / "tools/ggtensile").glob("mmq_fwd_lowering*.py"))
)
BWD_WRITER_SOURCE_PATH = (
    _REPO_ROOT / "tools/ggtensile/kernel_writer_assembly_mmq_bwd.py"
)
BWD_PHYSICAL_SOURCE_PATHS = tuple(
    sorted((_REPO_ROOT / "tools/ggtensile").glob("mmq_bwd_physical*.py"))
)
BWD_LOWERING_SOURCE_PATHS = tuple(
    sorted((_REPO_ROOT / "tools/ggtensile").glob("mmq_bwd_lowering*.py"))
)
BWD_IMPLEMENTATION_SOURCE_PATHS = tuple(
    path
    for path in sorted((_REPO_ROOT / "tools/ggtensile").glob("mmq_bwd_*.py"))
    if path.name != "mmq_bwd_search.py"
)
WRITER_SOURCE_PATHS = (
    SHARED_WRITER_SOURCE_PATH,
    FWD_WRITER_SOURCE_PATH,
    FWD_PHYSICAL_SOURCE_PATH,
    *FWD_LOWERING_SOURCE_PATHS,
    BWD_WRITER_SOURCE_PATH,
    *BWD_IMPLEMENTATION_SOURCE_PATHS,
)
WRITER_EXECUTED_LINES = {path: set() for path in WRITER_SOURCE_PATHS}
_WRITER_LINES_BY_FILENAME = {
    str(path): WRITER_EXECUTED_LINES[path] for path in WRITER_SOURCE_PATHS
}


def record_writer_line(
    frame: FrameType,
    event: str,
    argument: object,
) -> object:
    del argument
    lines = _WRITER_LINES_BY_FILENAME.get(frame.f_code.co_filename)
    if lines is None:
        return None
    if event == "line":
        lines.add(frame.f_lineno)
    return record_writer_line


def _code_lines(code: CodeType) -> set[int]:
    lines = {line for _, line in dis.findlinestarts(code) if line is not None}
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            lines.update(_code_lines(constant))
    return lines


def _writer_body_lines(path: Path) -> set[int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    executable = _code_lines(compile(source, str(path), "exec"))
    body_lines: set[int] = set()
    protocol_classes = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases
        )
    }
    tracked_classes = {
        SHARED_WRITER_SOURCE_PATH: {"Assembly"},
        FWD_WRITER_SOURCE_PATH: {"ForwardKernelWriterAssembly"},
        FWD_PHYSICAL_SOURCE_PATH: {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name not in protocol_classes
        },
        BWD_WRITER_SOURCE_PATH: {"_Assembly", "BackwardKernelWriterAssembly"},
    }.get(path, set())
    if path in FWD_LOWERING_SOURCE_PATHS:
        tracked_classes = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name != "ForwardBodyLowering"
            and node.name not in protocol_classes
        }
    if path in BWD_IMPLEMENTATION_SOURCE_PATHS:
        tracked_classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if path not in (
                SHARED_WRITER_SOURCE_PATH,
                FWD_WRITER_SOURCE_PATH,
                FWD_PHYSICAL_SOURCE_PATH,
                *FWD_LOWERING_SOURCE_PATHS,
                *BWD_IMPLEMENTATION_SOURCE_PATHS,
            ):
                continue
            first_body_line = node.body[0].lineno
            assert node.end_lineno is not None
            body_lines.update(range(first_body_line, node.end_lineno + 1))
            continue
        if not isinstance(node, ast.ClassDef) or node.name not in tracked_classes:
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first_body_line = member.body[0].lineno
            assert member.end_lineno is not None
            body_lines.update(range(first_body_line, member.end_lineno + 1))
    return executable & body_lines


def assert_writer_methods_have_complete_line_coverage(path: Path) -> None:
    missing = sorted(_writer_body_lines(path) - WRITER_EXECUTED_LINES[path])
    if not missing:
        return
    source_lines = path.read_text(encoding="utf-8").splitlines()
    details = "\n".join(
        f"  {line}: {source_lines[line - 1].strip()}" for line in missing
    )
    pytest.fail(
        f"writer method lines not covered in {path.relative_to(_REPO_ROOT)}:\n{details}"
    )


@dataclass(frozen=True)
class GGTensileInventoryCase:
    direction: Literal["fwd", "bwd"]
    quant_type: str
    entry_count: int
    m_values: frozenset[int]
    solution_count: int

    @property
    def id(self) -> str:
        return f"{self.direction}-{self.quant_type.lower()}"

    @property
    def problem_type(self) -> ProblemType:
        factory = (
            ProblemType.mmq_forward
            if self.direction == "fwd"
            else ProblemType.mmq_backward
        )
        return factory(self.quant_type)

    @property
    def catalog_path(self) -> Path:
        return (
            _CONFIG_DIR / f"mmq_{self.direction}_{self.quant_type.lower()}_catalog.json"
        )


MMQ_FWD_INVENTORY_CASES = (
    GGTensileInventoryCase(
        "fwd",
        "Q3_K",
        12,
        frozenset({2048, 8192, 32768}),
        1,
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q4_K",
        12,
        frozenset({2048, 8192, 32768}),
        5,
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q5_K",
        6,
        frozenset({2048, 8192, 32768}),
        5,
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q6_K",
        3,
        frozenset({64, 128, 256}),
        2,
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q8_0",
        23,
        frozenset({32, 64, 128, 256, 512, 2048, 8192, 32768}),
        4,
    ),
)
MMQ_FWD_INVENTORY_CASE_IDS = tuple(case.id for case in MMQ_FWD_INVENTORY_CASES)

MMQ_BWD_INVENTORY_CASES = (
    GGTensileInventoryCase(
        "bwd",
        "Q3_K",
        6,
        frozenset({2048, 8192, 32768}),
        5,
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q4_K",
        12,
        frozenset({2048, 8192, 32768}),
        4,
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q5_K",
        6,
        frozenset({2048, 8192, 32768}),
        4,
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q6_K",
        3,
        frozenset({64, 128, 256}),
        3,
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q8_0",
        23,
        frozenset({32, 64, 128, 256, 512, 2048, 8192, 32768}),
        9,
    ),
)
MMQ_BWD_INVENTORY_CASE_IDS = tuple(case.id for case in MMQ_BWD_INVENTORY_CASES)


def load_inventory_case(case: GGTensileInventoryCase) -> DeploymentCatalog:
    return load_catalog(case.catalog_path)


def selected_solution_keys(
    catalog: DeploymentCatalog,
) -> tuple[SolutionKey, ...]:
    return catalog.solution_keys


class AssemblyWriter(Protocol):
    def write(self, output: Path) -> str: ...


@dataclass(frozen=True)
class BuiltArtifact:
    source_hash: str
    source: str
    inspection: ArtifactInspection


def build_and_inspect(
    key: SolutionKey,
    writer: AssemblyWriter,
    toolchain: Toolchain,
    output_dir: Path,
    *,
    stem: str = "kernel",
) -> BuiltArtifact:
    assembly = output_dir / f"{stem}.s"
    object_path = output_dir / f"{stem}.o"
    code_object = output_dir / f"{stem}.hsaco"
    source_hash = writer.write(assembly)
    toolchain.assemble(assembly, object_path)
    toolchain.link(object_path, code_object)
    return BuiltArtifact(
        source_hash,
        assembly.read_text(encoding="utf-8"),
        inspect_artifact(key, code_object, toolchain),
    )


def assert_resource_clean(
    inspection: ArtifactInspection,
    *,
    sgpr_count: int = 16,
) -> None:
    assert inspection.sgpr_count == sgpr_count
    assert inspection.private_segment_bytes == 0
    assert inspection.vgpr_spill_count == 0
    assert inspection.sgpr_spill_count == 0
