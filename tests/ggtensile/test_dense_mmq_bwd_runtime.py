"""The deployed dense-backward module launches the catalogued control.

`tools/configs/hip_deployment.json` fixes the symbol per exact backward key and
derives its launch geometry. This module must not carry a second table.
"""

from tools.ggtensile.dense_mmq_bwd_runtime import InstalledDenseBackwardModule
from tools.ggtensile.hip_deployment import select_hip_control
from tools.ggtensile.kernel_instance import KernelInstance
from tools.ggtensile.model import ProblemSize
from tools.mmq_deployment_spec import kernels

_DENSE_INSTANCES: tuple[KernelInstance, ...] = tuple(
    item.instance
    for item in kernels()
    if item.operation == "OrdinaryBackward" and item.instance is not None
)


def test_deployed_dense_backward_launch_geometry_follows_the_catalog() -> None:
    assert _DENSE_INSTANCES
    for instance in _DENSE_INSTANCES:
        problem = instance.problem
        assert isinstance(problem, ProblemSize)
        control = select_hip_control(
            "OrdinaryBackward",
            instance.problem_type.quant_data_type,
            problem.m,
            problem.n,
            problem.k,
        )
        module = InstalledDenseBackwardModule.__new__(InstalledDenseBackwardModule)
        module.problem_size = problem
        module.hip_control = control
        assert module._launch_configuration() == control.launch_configuration(
            problem.m, problem.n, problem.k
        )
