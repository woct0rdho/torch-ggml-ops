import ast
import re
from pathlib import Path

from tools.ggtensile.benchmark_routes import (
    distribution_summary,
    route_distributions,
    truncate_distribution,
)

_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS = tuple(sorted((_ROOT / "tools").glob("benchmark_ggtensile_*.py")))
_LOWER_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")
_QUANT_KEYS = frozenset({"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ2_S"})


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ggtensile_benchmarks_share_required_argument_names() -> None:
    expected = {
        "benchmark_ggtensile_mmq_fwd.py",
        "benchmark_ggtensile_mmq_bwd.py",
        "benchmark_ggtensile_grouped_mmq_bwd.py",
        "benchmark_ggtensile_grouped_mmq_fwd_pair.py",
        "benchmark_ggtensile_fixed_grouped_mmq_fwd.py",
    }
    assert expected <= {path.name for path in _BENCHMARKS}
    for path in _BENCHMARKS:
        source = _source(path)
        for option in (
            "--solution-key",
            "--code-object",
            "--warmup",
            "--repeats",
            "--seed",
        ):
            assert f'"{option}"' in source, (path, option)


def test_ggtensile_benchmark_report_fields_are_lower_snake_case() -> None:
    for path in _BENCHMARKS:
        tree = ast.parse(_source(path), filename=str(path))
        literal_keys = {
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        invalid = {
            key
            for key in literal_keys
            if key not in _QUANT_KEYS and _LOWER_SNAKE_CASE.fullmatch(key) is None
        }
        assert invalid == set(), (path, invalid)


def test_ggtensile_benchmarks_publish_common_identity_and_protocol_fields() -> None:
    for path in _BENCHMARKS:
        source = _source(path)
        for field in ("solution_key", "kernel_name", "protocol", "warmup", "repeats"):
            assert f'"{field}"' in source, (path, field)


def test_grouped_benchmark_routes_keep_deterministic_control_contract() -> None:
    first = route_distributions(16_384, 1)
    second = route_distributions(16_384, 1)
    assert first == second
    assert set(first) == {"uniform", "skewed", "sparse", "boundary"}
    assert all(distribution.rows == 16_384 for distribution in first.values())
    assert len(first["uniform"].expert_indices_cpu) == 256
    assert len(first["sparse"].expert_indices_cpu) == 192
    assert first["boundary"].group_sizes_cpu[:10] == (
        1,
        15,
        16,
        17,
        63,
        64,
        65,
        127,
        128,
        129,
    )
    summary = distribution_summary(first["boundary"])
    assert summary["active_experts"] == 256
    truncated = truncate_distribution(first["boundary"], 625)
    assert truncated.name == "boundary_correctness"
    assert truncated.rows == 625
