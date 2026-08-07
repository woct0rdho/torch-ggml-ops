import ast
import dis
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType
from typing import Literal, Protocol

import pytest

from tools.ggtensile.campaign import (
    CampaignInventory,
    load_inventory,
    load_solution_catalog,
)
from tools.ggtensile.inspection import ArtifactInspection, inspect_artifact
from tools.ggtensile.model import (
    BackwardSolution,
    ForwardSolution,
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
WRITER_SOURCE_PATHS = (
    SHARED_WRITER_SOURCE_PATH,
    FWD_WRITER_SOURCE_PATH,
    FWD_PHYSICAL_SOURCE_PATH,
    *FWD_LOWERING_SOURCE_PATHS,
    BWD_WRITER_SOURCE_PATH,
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
    tracked_classes = {
        SHARED_WRITER_SOURCE_PATH: {"Assembly"},
        FWD_WRITER_SOURCE_PATH: {"ForwardKernelWriterAssembly"},
        FWD_PHYSICAL_SOURCE_PATH: {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        },
        BWD_WRITER_SOURCE_PATH: {"_Assembly", "BackwardKernelWriterAssembly"},
    }.get(path, set())
    if path in FWD_LOWERING_SOURCE_PATHS:
        tracked_classes = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name != "ForwardBodyLowering"
        }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if path not in (
                SHARED_WRITER_SOURCE_PATH,
                FWD_WRITER_SOURCE_PATH,
                FWD_PHYSICAL_SOURCE_PATH,
                *FWD_LOWERING_SOURCE_PATHS,
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
    families: frozenset[str]
    m_values: frozenset[int]
    status_counts: tuple[tuple[str, int], ...]
    solution_names: frozenset[str]

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
    def inventory_path(self) -> Path:
        return (
            _CONFIG_DIR
            / f"mmq_{self.direction}_{self.quant_type.lower()}_inventory.json"
        )

    @property
    def catalog_path(self) -> Path:
        return (
            _CONFIG_DIR
            / f"mmq_{self.direction}_{self.quant_type.lower()}_solutions.json"
        )


MMQ_FWD_INVENTORY_CASES = (
    GGTensileInventoryCase(
        "fwd",
        "Q4_K",
        12,
        frozenset({"narrow", "shared_down", "attention_output", "query"}),
        frozenset({2048, 8192, 32768}),
        (("open", 10), ("selected", 2)),
        frozenset(
            {
                "pilot_direct_global",
                "retained_metadata_after_low_wmma",
                "retained_independent_extraction_metadata_after_low_wmma",
                "retained_shared_down_m8192_extract_a1d4_p2",
                "retained_shared_down_m32768_extract_a1d2_p2",
                "composed_shared_down_m2048_extract_a1d2_p2",
                "composed_shared_down_m8192_extract_a1d4_p2",
                "composed_shared_down_m32768_extract_a1d2_p2",
                "composed_narrow_m32768_extract_a4d4_p2",
                "composed_query_m2048_extract_a1d2_p2",
                "composed_query_m8192_extract_a4d4_p2",
                "composed_query_m32768_extract_a2d2_p2",
            }
        ),
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q5_K",
        6,
        frozenset({"narrow", "shared_down"}),
        frozenset({2048, 8192, 32768}),
        (("selected", 6),),
        frozenset(
            {
                "retained_decoded_staged",
                "retained_metadata_after_low_wmma",
                "retained_independent_extraction_metadata_after_low_wmma",
                "selected_narrow_m2048_a1d8_p0",
                "selected_narrow_m8192_a8d1_p0",
                "selected_narrow_m32768_a7d3_p3_vopd_init",
                "selected_shared_down_m2048_a1d2_p2",
                "selected_shared_down_m8192_a1d4_p2_vopd_init",
                "selected_shared_down_m32768_a1d2_p2",
            }
        ),
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q6_K",
        3,
        frozenset({"lm_head"}),
        frozenset({64, 128, 256}),
        (("selected", 3),),
        frozenset(
            {
                "structured_decoded_mt64_selected",
                "structured_decoded_mt128_selected",
            }
        ),
    ),
    GGTensileInventoryCase(
        "fwd",
        "Q8_0",
        23,
        frozenset(
            {
                "attention_kv",
                "attention_output_b",
                "attention_q_a",
                "attention_q_b",
                "lm_head",
                "shared_down",
                "shared_gate_up",
            }
        ),
        frozenset({32, 64, 128, 256, 512, 2048, 8192, 32768}),
        (("selected", 23),),
        frozenset(
            {
                "direct_global_control",
                "register_tiled_128x32",
                "hip_tiled_lds_selected",
                "small_m32_tiled_lds_selected",
                "small_m64_tiled_lds_selected",
                "kv_compact_m64_tiled_lds_selected",
            }
        ),
    ),
)
MMQ_FWD_INVENTORY_CASE_IDS = tuple(case.id for case in MMQ_FWD_INVENTORY_CASES)

MMQ_BWD_INVENTORY_CASES = (
    GGTensileInventoryCase(
        "bwd",
        "Q3_K",
        6,
        frozenset({"narrow", "query"}),
        frozenset({2048, 8192, 32768}),
        (("selected", 6),),
        frozenset(
            {
                "retained_128x64_pad8_sia5",
                "retained_256x64_pad8_sia5",
                "retained_256x64_pad8_sia5_wgm2",
                "retained_128x128_pad8_sia5",
            }
        ),
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q4_K",
        12,
        frozenset({"narrow", "shared_down", "attention_output", "query"}),
        frozenset({2048, 8192, 32768}),
        (("selected", 12),),
        frozenset(
            {
                "retained_128x128_pipeline",
                "retained_128x64_pad8_sia5",
                "retained_256x64_pad8_sia5",
                "retained_256x64_pad8_sia5_wgm2",
            }
        ),
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q5_K",
        6,
        frozenset({"narrow", "shared_down"}),
        frozenset({2048, 8192, 32768}),
        (("selected", 6),),
        frozenset(
            {
                "retained_128x128_pipeline",
                "retained_128x128_pipeline_no_hoist",
                "retained_256x64_pad8_sia5",
                "retained_256x64_pad8_sia5_wgm2",
            }
        ),
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q6_K",
        3,
        frozenset({"lm_head"}),
        frozenset({64, 128, 256}),
        (("selected", 3),),
        frozenset(
            {
                "retained_m64x32x64_pad8_vopd",
                "retained_m128x32x64_pad8_vopd",
                "retained_m256x64x32_next_pad8_vopd",
            }
        ),
    ),
    GGTensileInventoryCase(
        "bwd",
        "Q8_0",
        23,
        frozenset(
            {
                "attention_kv",
                "attention_output_b",
                "attention_q_a",
                "attention_q_b",
                "lm_head",
                "shared_down",
                "shared_gate_up",
            }
        ),
        frozenset({32, 64, 128, 256, 512, 2048, 8192, 32768}),
        (("selected", 23),),
        frozenset(
            {
                "retained_128x128_pad8_sia5_packed",
                "retained_128x128_pad8_sia4_next_scalar",
                "retained_128x128_depth64_xor8_sia4_packed",
                "retained_256x64_pad8_sia5_packed",
                "retained_128x64_pad8_sia5_packed",
                "retained_lm_m512_256x64_pad8_next",
                "retained_lm_m32_depth64_pad8_vopd",
                "retained_lm_m64_depth64_pad8_vopd",
                "retained_lm_m128_depth64_pad8_vopd",
            }
        ),
    ),
)
MMQ_BWD_INVENTORY_CASE_IDS = tuple(case.id for case in MMQ_BWD_INVENTORY_CASES)

Solution = BackwardSolution | ForwardSolution
SolutionCatalog = dict[str, Solution]


def load_inventory_case(
    case: GGTensileInventoryCase,
) -> tuple[CampaignInventory, SolutionCatalog]:
    inventory = load_inventory(case.inventory_path)
    catalog = load_solution_catalog(
        case.catalog_path,
        problem_type=inventory.problem_type,
    )
    return inventory, catalog


def selected_solution_keys(
    inventory: CampaignInventory,
    catalog: Mapping[str, Solution],
) -> tuple[SolutionKey, ...]:
    return tuple(
        SolutionKey(
            inventory.problem_type,
            entry.problem_size,
            catalog[entry.selected_solution],
        )
        for entry in inventory.entries
        if entry.selected_solution != "hip_fallback"
    )


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
