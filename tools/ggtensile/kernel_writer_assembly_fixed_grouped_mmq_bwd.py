"""Assembly writer facade for fixed-group Q8_0 backward."""

from .fixed_grouped_mmq_bwd_lowering import FixedGroupedQ8BackwardLowering
from .fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from .fixed_grouped_mmq_bwd_spec import (
    DerivedFixedBackwardState,
    FixedBackwardKernelSpec,
)
from .fixed_grouped_mmq_bwd_validation import (
    validate_fixed_backward_solution,
)
from .kernel_abi import FIXED_GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .toolchain import Toolchain


class FixedGroupedBackwardKernelWriterAssembly(AssemblyKernelWriter):
    """Emit one strict six-argument fixed-group Q8_0 backward kernel."""

    def __init__(
        self,
        problem: FixedBackwardProblem,
        kernel_spec: FixedBackwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_fixed_backward_solution(problem, kernel_spec)
        self.kernel_name = kernel_name
        self.state = DerivedFixedBackwardState.from_problem_spec(problem, kernel_spec)
        self.assembler = toolchain.assembler

    def emission_plan(self) -> KernelEmissionPlan:
        state = self.state
        resources = state.physical.resources
        geometry = state.spec.compute.geometry
        emission = FixedGroupedQ8BackwardLowering(self.kernel_name, state).emission()
        return KernelEmissionPlan(
            module_name="GGTensileFixedGroupedBackwardKernel",
            kernel_name=self.kernel_name,
            isa=geometry.isa,
            wavefront_size=geometry.wavefront_size,
            temporary_prefix="ggtensile-fixed-backward-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 1),
            vgpr_work_item=1,
            flat_workgroup_size=geometry.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=FIXED_GROUPED_BACKWARD_ABI,
            description="GGTensile fixed-group Q8_0 backward",
            lowering=emission,
        )
