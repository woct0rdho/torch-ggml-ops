import ast
import re
from pathlib import Path

from tools.ggtensile.benchmark_routes import (
    distribution_summary,
    fitted_prior_distribution_for_rows,
    truncate_distribution,
)

_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS = tuple(
    sorted((_ROOT / "bench" / "ggtensile").glob("benchmark_ggtensile_*.py"))
)
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


def test_grouped_benchmark_routes_use_one_deterministic_fitted_profile() -> None:
    expected = {
        "qwen-learned": (16_384, 233, 1_297),
        "deepseek-learned": (12_288, 244, 982),
        "deepseek-hash": (12_288, 256, 264),
    }
    for prior, (rows, active, maximum) in expected.items():
        first = fitted_prior_distribution_for_rows(prior, rows)
        second = fitted_prior_distribution_for_rows(prior, rows)
        assert first == second
        assert first.rows == rows
        assert first.profile is not None
        assert first.profile.prior.value == prior
        assert first.rows_per_expert == first.profile.rows_per_expert
        summary = distribution_summary(first)
        assert summary["active_experts"] == active
        assert summary["max_rows"] == maximum

    truncated = truncate_distribution(
        fitted_prior_distribution_for_rows("qwen-learned", 16_384), 625
    )
    assert truncated.name.endswith("_correctness")
    assert truncated.rows == 625
