from tools.ggtensile.dense_mmq_bwd_runtime import (
    InstalledDenseBackwardModule,
    _select_control,
)
from tools.ggtensile.model import SolutionKey
from tools.mmq_deployment_spec import kernels
from tools.mmq_hip_control_spec import hip_control_specs

_DENSE_KEYS: tuple[SolutionKey, ...] = tuple(
    item.key
    for item in kernels()
    if item.operation == "OrdinaryBackward" and isinstance(item.key, SolutionKey)
)


def test_historical_dense_controls_cover_every_public_backward_route() -> None:
    available = {spec.symbol for spec in hip_control_specs()}
    symbols = set()
    for key in _DENSE_KEYS:
        control = _select_control(key)
        symbols.add(control.symbol)
        assert control.symbol in available
    assert len(symbols) == 14


def test_historical_dense_control_launch_geometry_is_stable() -> None:
    for key in _DENSE_KEYS:
        module = InstalledDenseBackwardModule.__new__(InstalledDenseBackwardModule)
        module.solution_key = key
        module.control = _select_control(key)
        grid, block, shared = module._launch_configuration()
        assert all(value > 0 for value in grid)
        assert block == (128, 1, 1)
        assert shared == 0
