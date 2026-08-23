import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.ggtensile.support import WRITER_EXECUTED_LINES, record_writer_line

_GGTENSILE_TEST_ROOT = Path(__file__).resolve().parent
_WRITER_COVERAGE_MODULES = frozenset(
    {
        "test_fixed_grouped_mmq_bwd.py",
        "test_fixed_grouped_mmq_fwd.py",
        "test_grouped_mmq_bwd.py",
        "test_grouped_mmq_bwd_pair.py",
        "test_grouped_mmq_fwd.py",
        "test_grouped_mmq_fwd_pair.py",
        "test_kernel_writer_assembly_coverage.py",
        "test_mmq_bwd.py",
        "test_mmq_bwd_writer_coverage.py",
        "test_mmq_fwd.py",
        "test_mmq_fwd_search.py",
        "test_mmq_fwd_spec.py",
        "test_mmq_fwd_writer_coverage.py",
    }
)
_WRITER_COVERAGE_XDIST_GROUP = "ggtensile-writer-coverage"


@pytest.fixture(scope="session", autouse=True)
def trace_writer_lines() -> Iterator[None]:
    for lines in WRITER_EXECUTED_LINES.values():
        lines.clear()
    previous = sys.gettrace()
    sys.settrace(record_writer_line)
    yield
    sys.settrace(previous)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # The assertions consume line events accumulated by the direction suites.
    coverage_items = [
        item
        for item in items
        if item.name == "test_writer_methods_have_complete_line_coverage"
    ]
    if not coverage_items:
        return

    # Keep every test that executes tracked writer code with the assertions.
    for item in items:
        if (
            _GGTENSILE_TEST_ROOT in item.path.parents
            and item.path.name in _WRITER_COVERAGE_MODULES
        ):
            item.add_marker(pytest.mark.xdist_group(_WRITER_COVERAGE_XDIST_GROUP))

    items[:] = [item for item in items if item not in coverage_items] + coverage_items
