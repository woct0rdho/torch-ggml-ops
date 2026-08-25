"""Assembly writer facade for paired grouped MMQ backward kernels."""

from .grouped_mmq_bwd_pair_lowering import GroupedBackwardPairKernelLowering
from .grouped_mmq_bwd_pair_model import GroupedBackwardPairProblem
from .grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from .grouped_mmq_bwd_pair_spec import (
    DerivedGroupedBackwardPairState,
    GroupedBackwardPairKernelSpec,
)
from .grouped_mmq_bwd_pair_validation import (
    validate_grouped_backward_pair_solution,
)
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .toolchain import Toolchain


class GroupedBackwardPairKernelWriterAssembly(AssemblyKernelWriter):
    """Emit one strict fused routed backward-pair artifact."""

    def __init__(
        self,
        problem: GroupedBackwardPairProblem,
        kernel_spec: GroupedBackwardPairKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_grouped_backward_pair_solution(problem, kernel_spec)
        self.kernel_name = kernel_name
        self.assembler = toolchain.assembler
        self.state = DerivedGroupedBackwardPairState.from_problem_spec(
            problem, kernel_spec
        )
        self.physical = derive_grouped_backward_pair_physical_plan(self.state)
        self.lowering = GroupedBackwardPairKernelLowering(
            kernel_name,
            self.state,
            self.physical,
        )

    def emission_plan(self) -> KernelEmissionPlan:
        compute = self.state.kernel_spec.compute
        resources = self.physical.ordinary.resources
        emission = self.lowering.emission()
        return KernelEmissionPlan(
            module_name="GGTensileGroupedBackwardPairKernel",
            kernel_name=self.kernel_name,
            isa=compute.geometry.isa,
            wavefront_size=compute.geometry.wavefront_size,
            temporary_prefix="ggtensile-grouped-backward-pair-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=1,
            flat_workgroup_size=compute.geometry.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=GROUPED_BACKWARD_PAIR_ABI,
            description=(
                f"GGTensile fused grouped {self.state.contract.quant_type} "
                "MMQ backward pair, "
                f"M{compute.geometry.macro_tile0}/N{compute.geometry.macro_tile1}/K32"
            ),
            lowering=emission,
        )
