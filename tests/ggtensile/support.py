from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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
from tools.ggtensile.toolchain import Toolchain, ToolchainError

_CONFIG_DIR = Path("tools/ggtensile/configs")


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
        entry.solution_key(inventory.problem_type, catalog[entry.selected_solution])
        for entry in inventory.entries
    )


def ggtensile_toolchain() -> Toolchain:
    try:
        return Toolchain.discover()
    except ToolchainError as error:
        pytest.skip(str(error))


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
