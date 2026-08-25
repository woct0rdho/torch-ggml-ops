"""Assembly writer facade for isolated grouped MMQ forward kernels."""

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
    GroupedPackedScaleMinimumDirectLowering,
)
from .grouped_mmq_fwd_lowering_decoded_lds import grouped_decoded_lowering
from .grouped_mmq_fwd_lowering_iq2_s import GroupedIQ2SFullWeightLdsLowering
from .grouped_mmq_fwd_lowering_q2_k import grouped_q2_decoded_lowering
from .grouped_mmq_fwd_model import (
    GroupedForwardProblem,
    GroupedOperandSource,
    GroupedQ2DecodePolicy,
)
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState, GroupedForwardKernelSpec
from .grouped_mmq_fwd_validation import validate_grouped_forward_solution
from .kernel_abi import GROUPED_FORWARD_ABI
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .toolchain import Toolchain


class GroupedForwardKernelWriterAssembly(AssemblyKernelWriter):
    """Emit strict routed packed-weight forward controls."""

    def __init__(
        self,
        problem: GroupedForwardProblem,
        kernel_spec: GroupedForwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_grouped_forward_solution(problem, kernel_spec)
        self.kernel_name = kernel_name
        self.state = DerivedGroupedForwardState.from_problem_spec(problem, kernel_spec)
        self.context = GroupedForwardLoweringContext(kernel_name, problem, self.state)
        self.assembler = toolchain.assembler

    def emission_plan(self) -> KernelEmissionPlan:
        spec = self.state.kernel_spec
        contract = self.state.contract
        resources = self.state.physical_plan.resources
        quant_type = self.state.problem.quant_data_type
        if spec.operand_source is GroupedOperandSource.GroupedDirectGlobal:
            emission = GroupedPackedScaleMinimumDirectLowering(self.context).emission()
        elif spec.operand_source is GroupedOperandSource.GroupedDecodedWeightLds:
            if isinstance(self.state.kernel_spec.decode, GroupedQ2DecodePolicy):
                emission = grouped_q2_decoded_lowering(self.context).emission()
            else:
                emission = grouped_decoded_lowering(self.context).emission()
        elif spec.operand_source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
            emission = GroupedIQ2SFullWeightLdsLowering(self.context).emission()
        else:
            raise TypeError(
                f"unsupported grouped operand source {spec.operand_source!r}"
            )
        return KernelEmissionPlan(
            module_name="GGTensileGroupedForwardKernel",
            kernel_name=self.kernel_name,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            temporary_prefix="ggtensile-grouped-forward-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=1,
            flat_workgroup_size=(
                spec.geometry.work_group[0]
                * spec.geometry.work_group[1]
                * spec.geometry.work_group[2]
            ),
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=GROUPED_FORWARD_ABI,
            description=(
                f"GGTensile grouped {quant_type} MMQ forward, serial GEMM "
                f"ownership, fixed Q8_1 {contract.activation_layout} producer"
            ),
            lowering=emission,
        )
