"""Assembly writer facade for isolated grouped MMQ backward kernels."""

from .grouped_mmq_bwd_lowering import GroupedBackwardKernelLowering
from .grouped_mmq_bwd_physical import derive_grouped_backward_physical_plan
from .grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
)
from .kernel_abi import GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .model import ProblemSize
from .toolchain import Toolchain
from .validation import validate_grouped_backward_solution


class GroupedBackwardKernelWriterAssembly(AssemblyKernelWriter):
    """Public facade for one exact routed gfx1151 backward solution."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: GroupedBackwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_grouped_backward_solution(problem_size, quant_type, kernel_spec)
        self.kernel_name = kernel_name
        self.assembler = toolchain.assembler
        self.state = DerivedGroupedBackwardState.from_problem_spec(
            problem_size, quant_type, kernel_spec
        )
        self.physical = derive_grouped_backward_physical_plan(self.state)
        self.lowering = GroupedBackwardKernelLowering(
            kernel_name,
            self.state,
            self.physical,
        )

    def emission_plan(self) -> KernelEmissionPlan:
        compute = self.state.spec.compute
        resources = self.physical.primary.resources
        emission = self.lowering.emission()
        return KernelEmissionPlan(
            module_name="GGTensileKernel",
            kernel_name=self.kernel_name,
            isa=compute.geometry.isa,
            wavefront_size=compute.geometry.wavefront_size,
            temporary_prefix="ggtensile-rocisa-grouped-bwd-",
            code_object_version=self.state.contract.code_object_version,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(
                1,
                1,
                int(self.state.spec.ownership.split_factor > 1),
            ),
            vgpr_work_item=1,
            flat_workgroup_size=compute.geometry.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=GROUPED_BACKWARD_ABI,
            description=(
                f"GGTensile {self.state.contract.quant_type} grouped MMQ backward"
            ),
            lowering=emission,
        )
