import pytest

# execnet, which carries every pytest-xdist report, encodes integers with a
# signed 32-bit format. Device-memory accounting plugins record byte counts that
# are larger than that, and such a value aborts the reporting worker instead of
# failing a test, which takes the whole parallel session down.
INT32_LIMIT = 2**31


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Keep oversized user properties serializable for pytest-xdist.

    The value is kept as text so it stays visible in verbose reports while the
    report payload stays inside the range execnet can send.
    """

    properties = getattr(report, "user_properties", None)
    if not properties:
        return
    report.user_properties = [
        (name, str(value))
        if isinstance(value, int) and not -INT32_LIMIT <= value < INT32_LIMIT
        else (name, value)
        for name, value in properties
    ]


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    if hasattr(config, "workerinput"):
        config.option.loadgroup = True
        return
    if config.getoption("numprocesses", default=None):
        config.option.dist = "loadgroup"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "xdist_group(name): group tests on one xdist worker for shared state",
    )
    if hasattr(config, "workerinput"):
        config.option.loadgroup = True
    elif config.getoption("numprocesses", default=None):
        config.option.dist = "loadgroup"
