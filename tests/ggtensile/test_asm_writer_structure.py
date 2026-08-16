"""Structural debt inventory for the assembly-writer refactor."""

import ast
from pathlib import Path

from tests.ggtensile.support import (
    BWD_IMPLEMENTATION_SOURCE_PATHS,
    BWD_LOWERING_SOURCE_PATHS,
    BWD_PHYSICAL_SOURCE_PATHS,
    BWD_WRITER_SOURCE_PATH,
    FWD_LOWERING_SOURCE_PATHS,
    FWD_PHYSICAL_SOURCE_PATH,
    FWD_WRITER_SOURCE_PATH,
    WRITER_SOURCE_PATHS,
)

_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION = _ROOT / "tools/ggtensile/validation.py"
_INSPECTION = _ROOT / "tools/ggtensile/inspection.py"
_FORWARD_AUTHORITY_CONSUMERS = (
    FWD_WRITER_SOURCE_PATH,
    FWD_PHYSICAL_SOURCE_PATH,
    _INSPECTION,
)
_GROUPED_ROUTE = _ROOT / "tools/ggtensile/grouped_mmq_fwd_route.py"
_GROUPED_WRITER = _ROOT / "tools/ggtensile/kernel_writer_assembly_grouped_mmq_fwd.py"
_GROUPED_IQ2S_LOWERING = _ROOT / "tools/ggtensile/grouped_mmq_fwd_lowering_iq2_s.py"
_GROUPED_Q2_LOWERING = _ROOT / "tools/ggtensile/grouped_mmq_fwd_lowering_q2_k.py"
_BACKWARD_WRITER = _ROOT / "tools/ggtensile/kernel_writer_assembly_mmq_bwd.py"
_ASSEMBLY = _ROOT / "tools/ggtensile/kernel_writer_assembly.py"
_MMQ_FWD_PHYSICAL = _ROOT / "tools/ggtensile/mmq_fwd_physical.py"
_GROUPED_LOWERINGS = tuple(
    sorted((_ROOT / "tools/ggtensile").glob("grouped_mmq_fwd_lowering*.py"))
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _class_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.parse(_source(path), filename=str(path)).body
        if isinstance(node, ast.ClassDef)
    }


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(_source(path), filename=str(path))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_writer_source_discovery_tracks_directional_modules() -> None:
    expected = {
        FWD_WRITER_SOURCE_PATH,
        FWD_PHYSICAL_SOURCE_PATH,
        *FWD_LOWERING_SOURCE_PATHS,
        BWD_WRITER_SOURCE_PATH,
        *BWD_IMPLEMENTATION_SOURCE_PATHS,
    }
    assert expected <= set(WRITER_SOURCE_PATHS)


def test_known_exact_shape_validation_debt_does_not_grow() -> None:
    source = _source(_VALIDATION)
    known_markers = {
        marker for marker in ("problem_size.n.production",) if marker in source
    }
    assert known_markers == set()


def test_backward_capability_has_no_exact_shape_instruction_gate() -> None:
    sources = (_source(_VALIDATION),) + tuple(
        _source(path) for path in BWD_IMPLEMENTATION_SOURCE_PATHS
    )
    forbidden = (
        "problem_size.n.production",
        "allowed_geometries",
        "allowed_n",
        "problem_size.m ==",
        "problem_size.n ==",
        "problem_size.k ==",
        "size.m ==",
        "size.n ==",
        "size.k ==",
    )
    assert not {
        marker for source in sources for marker in forbidden if marker in source
    }


def test_capability_validation_has_no_numeric_geometry_table() -> None:
    tree = ast.parse(_source(_VALIDATION), filename=str(_VALIDATION))
    numeric_geometry_tables = {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set))
        and node.elts
        and all(
            isinstance(item, (ast.Tuple, ast.List))
            and item.elts
            and all(
                isinstance(value, ast.Constant) and type(value.value) is int
                for value in item.elts
            )
            for item in node.elts
        )
    }
    assert numeric_geometry_tables == set()


def test_known_writer_text_rewrite_debt_does_not_grow() -> None:
    offenders = {
        path.name
        for path in WRITER_SOURCE_PATHS
        if "re.fullmatch(" in _source(path)
        or "re.match(" in _source(path)
        or '.replace("v_' in _source(path)
    }
    assert offenders == set()


def test_backward_emission_inherits_ordinary_instruction_formatting() -> None:
    emission = next(
        path
        for path in BWD_IMPLEMENTATION_SOURCE_PATHS
        if path.name == "mmq_bwd_emission.py"
    )
    tree = ast.parse(_source(emission), filename=str(emission))
    assembly = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_Assembly"
    )
    methods = {node.name for node in assembly.body if isinstance(node, ast.FunctionDef)}
    assert "inst" not in methods


def test_planners_and_lowerers_preserve_import_boundaries() -> None:
    forbidden_lowerer_modules = ("campaign", "inspection", "runtime", "toolchain")
    for path in (*FWD_LOWERING_SOURCE_PATHS, *BWD_LOWERING_SOURCE_PATHS):
        assert not {
            module
            for module in _imported_modules(path)
            if any(name in module for name in forbidden_lowerer_modules)
        }, path

    forbidden_planner_modules = (
        "rocisa",
        "inspection",
        "toolchain",
        "lowering",
        "emission",
    )
    for path in (FWD_PHYSICAL_SOURCE_PATH, *BWD_PHYSICAL_SOURCE_PATHS):
        source = _source(path)
        assert not {
            module
            for module in _imported_modules(path)
            if any(name in module for name in forbidden_planner_modules)
        }, path
        assert "Assembly(" not in source


def test_backward_lowerers_consume_derived_state_without_one_hop_aliases() -> None:
    for path in BWD_LOWERING_SOURCE_PATHS:
        source = _source(path)
        assert "self.solution" not in source, path
        assert "self.solution_key" not in source, path
        assert "QUANT_FORMATS" not in source, path


def test_phase_six_removes_confirmed_one_hop_aliases_only() -> None:
    backward_writer = _source(_BACKWARD_WRITER)
    q2_lowering = _source(_GROUPED_Q2_LOWERING)
    iq2_lowering = _source(_GROUPED_IQ2S_LOWERING)
    assert "self.solution =" not in backward_writer
    assert "self.registers = self.physical.registers" in backward_writer
    assert "def _q2_physical" not in q2_lowering
    assert "_q2_physical(" not in q2_lowering
    assert "def _registers" not in iq2_lowering
    assert "self._registers" not in iq2_lowering
    assert "DeterministicRegisterPool" in _source(_ASSEMBLY)
    assert "DeterministicRegisterPool" in _source(_MMQ_FWD_PHYSICAL)


def test_forward_lowering_has_no_copy_projection_or_unused_protocol() -> None:
    lowering_classes = set().union(
        *(_class_names(path) for path in FWD_LOWERING_SOURCE_PATHS)
    )
    assert "ForwardBodyLowering" not in lowering_classes

    tree = ast.parse(_source(FWD_PHYSICAL_SOURCE_PATH))
    protocols = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases
        )
    }
    assert {
        "SignedInt8TiledLdsRegisters",
        "SignedInt8TiledLdsScaleLayout",
    } <= protocols
    projection_methods = {
        member.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in protocols
        for member in node.body
        if isinstance(member, ast.FunctionDef)
    }
    assert not projection_methods & {"from_plan", "from_layout"}


def test_inspection_consumes_backward_physical_resources_without_replay() -> None:
    source = _source(_INSPECTION)
    assert "derive_backward_physical_plan" in source
    assert "def allocate(cursor: int, count: int, alignment: int = 1)" not in source


def test_forward_authority_consumers_do_not_repeat_operand_source_sets() -> None:
    operand_sources = {
        "Q6StructuredDecoded",
        "Q3HipTiledLds",
        "Q3FullWeightTiledLds",
        "Q8DirectGlobal",
        "Q8RegisterTiled",
        "Q8HipTiledLds",
        "Q8SmallMTiledLds",
        "DecodedWeightLdsBatch8",
    }
    for path in _FORWARD_AUTHORITY_CONSUMERS:
        values = {
            node.value
            for node in ast.walk(ast.parse(_source(path), filename=str(path)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not operand_sources & values, path


def test_grouped_route_emitter_owns_the_complete_route_prologue() -> None:
    private_route_methods = (
        "_emit_kernarg_loads",
        "_emit_exact_shape_guard",
        "_emit_route_load_and_guard",
        "_emit_expert_pointer_rebase",
    )
    route_source = _source(_GROUPED_ROUTE)
    assert all(f"def {method}" in route_source for method in private_route_methods)
    for path in _GROUPED_LOWERINGS:
        source = _source(path)
        assert not {
            method for method in private_route_methods if f".{method}(" in source
        }, path


def test_iq2_s_lowering_owns_codebook_rodata_emission() -> None:
    facade = _source(_GROUPED_WRITER)
    lowering = _source(_GROUPED_IQ2S_LOWERING)
    assert "iq2_s_grid" not in facade
    assert "GroupedForwardLoweringResult" in lowering
    assert "iq2_s_grid_rodata" in lowering
