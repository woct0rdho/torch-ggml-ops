"""Assembly writer facade for the fixed-group Q8_0 forward ABI."""

from .fixed_grouped_mmq_fwd_lowering import (
    FixedForwardLoweringContext,
    FixedGroupedQ8ForwardLowering,
)
from .fixed_grouped_mmq_fwd_model import FixedForwardProblem
from .fixed_grouped_mmq_fwd_spec import (
    DerivedFixedForwardState,
    FixedForwardKernelSpec,
)
from .fixed_grouped_mmq_fwd_validation import validate_fixed_forward_solution
from .kernel_abi import FIXED_GROUPED_FORWARD_ABI
from .kernel_writer_assembly import (
    AssemblyKernelWriter,
    KernelEmissionPlan,
    LoweringResult,
)
from .toolchain import Toolchain


class FixedGroupedForwardKernelWriterAssembly(AssemblyKernelWriter):
    """Emit one strict six-argument fixed-group Q8_0 kernel."""

    def __init__(
        self,
        problem: FixedForwardProblem,
        kernel_spec: FixedForwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_fixed_forward_solution(problem, kernel_spec)
        state = DerivedFixedForwardState.from_problem_spec(problem, kernel_spec)
        self.kernel_name = kernel_name
        self.state = state
        self.context = FixedForwardLoweringContext(kernel_name, state)
        self.assembler = toolchain.assembler

    def emission_plan(self) -> KernelEmissionPlan:
        resources = self.state.physical.resources
        return KernelEmissionPlan(
            module_name="GGTensileFixedGroupedForwardKernel",
            kernel_name=self.kernel_name,
            isa=self.state.contract.isa,
            wavefront_size=self.state.contract.wavefront_size,
            temporary_prefix="ggtensile-fixed-forward-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 1),
            vgpr_work_item=1,
            flat_workgroup_size=self.state.ordinary.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=FIXED_GROUPED_FORWARD_ABI,
            description=("GGTensile fixed-group Q8_0 forward, Q8_1 F32_D4 activations"),
            lowering=LoweringResult(FixedGroupedQ8ForwardLowering(self.context).body()),
        )
