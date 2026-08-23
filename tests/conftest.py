import pytest


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
