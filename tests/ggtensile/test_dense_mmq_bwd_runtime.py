from tools.ggtensile.dense_mmq_bwd_runtime import (
    InstalledDenseBackwardModule,
    _select_control,
)
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.model import ProblemSize
from tools.mmq_deployment_spec import kernels
from tools.mmq_hip_control_spec import hip_control_specs

_DENSE_INSTANCES: tuple[KernelInstance, ...] = tuple(
    item.instance
    for item in kernels()
    if item.operation == "OrdinaryBackward" and item.instance is not None
)


def test_historical_dense_controls_cover_every_public_backward_route() -> None:
    available = {spec.symbol for spec in hip_control_specs()}
    symbols = set()
    for instance in _DENSE_INSTANCES:
        assert isinstance(instance.problem, ProblemSize)
        control = _select_control(
            instance.problem, instance.problem_type.quant_data_type
        )
        symbols.add(control.symbol)
        assert control.symbol in available
    assert len(symbols) == 14


def test_historical_dense_control_launch_geometry_is_stable() -> None:
    for instance in _DENSE_INSTANCES:
        assert isinstance(instance.problem, ProblemSize)
        module = InstalledDenseBackwardModule.__new__(InstalledDenseBackwardModule)
        module.problem_size = instance.problem
        module.control = _select_control(
            instance.problem, instance.problem_type.quant_data_type
        )
        grid, block, shared = module._launch_configuration()
        assert all(value > 0 for value in grid)
        assert block == (128, 1, 1)
        assert shared == 0
