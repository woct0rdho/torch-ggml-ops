import sys
from collections.abc import Iterator

import pytest

from tests.ggtensile.support import WRITER_EXECUTED_LINES, record_writer_line


@pytest.fixture(scope="session", autouse=True)
def trace_writer_lines() -> Iterator[None]:
    for lines in WRITER_EXECUTED_LINES.values():
        lines.clear()
    previous = sys.gettrace()
    sys.settrace(record_writer_line)
    try:
        yield
    finally:
        sys.settrace(previous)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # The assertions consume line events accumulated by the direction suites.
    coverage_items = [
        item
        for item in items
        if item.name == "test_writer_methods_have_complete_line_coverage"
    ]
    if coverage_items:
        items[:] = [
            item for item in items if item not in coverage_items
        ] + coverage_items
