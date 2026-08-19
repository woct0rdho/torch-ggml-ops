"""Assembly writer facade for isolated grouped MMQ forward kernels."""

from pathlib import Path

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
    GroupedPackedScaleMinimumDirectLowering,
)
from .grouped_mmq_fwd_lowering_decoded_lds import grouped_decoded_lowering
from .grouped_mmq_fwd_lowering_iq2_s import GroupedIQ2SFullWeightLdsLowering
from .grouped_mmq_fwd_lowering_q2_k import grouped_q2_decoded_lowering
from .grouped_mmq_fwd_model import GroupedForwardSolutionKey, GroupedOperandSource
from .grouped_mmq_fwd_spec import (
    DerivedGroupedForwardState,
    GroupedQ2SchedulePolicy,
)
from .grouped_mmq_fwd_validation import validate_grouped_forward_solution
from .kernel_abi import GROUPED_FORWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class GroupedForwardKernelWriterAssembly:
    """Emit strict routed packed-weight forward controls."""

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        toolchain: Toolchain,
    ) -> None:
        reasons = validate_grouped_forward_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"grouped solution rejected: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardState.from_solution_key(solution_key)
        self.context = GroupedForwardLoweringContext(solution_key, self.state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        solution = self.solution_key.solution
        resources = self.state.physical_plan.resources
        quant_type = self.solution_key.problem.quant_data_type
        envelope = KernelEnvelope(
            module_name="GGTensileGroupedForwardKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-grouped-forward-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=1,
            flat_workgroup_size=solution.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=GROUPED_FORWARD_ABI,
            description=(
                f"GGTensile grouped {quant_type} MMQ forward, serial GEMM "
                f"ownership, fixed Q8_1 {solution.activation_layout} producer"
            ),
        )
        envelope.initialize()
        if solution.operand_source is GroupedOperandSource.GroupedDirectGlobal:
            emission = GroupedPackedScaleMinimumDirectLowering(self.context).emission()
        elif solution.operand_source is GroupedOperandSource.GroupedDecodedWeightLds:
            if isinstance(self.state.kernel_spec.decode, GroupedQ2SchedulePolicy):
                emission = grouped_q2_decoded_lowering(self.context).emission()
            else:
                emission = grouped_decoded_lowering(self.context).emission()
        elif solution.operand_source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
            emission = GroupedIQ2SFullWeightLdsLowering(self.context).emission()
        else:
            raise TypeError(
                f"unsupported grouped operand source {solution.operand_source!r}"
            )
        return envelope.render(
            emission.body,
            trailing_sections=emission.trailing_sections,
        )
