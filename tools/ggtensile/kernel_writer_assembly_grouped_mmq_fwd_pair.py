"""Assembly writer facade for paired grouped forward research kernels."""

from pathlib import Path

from .grouped_mmq_fwd_pair_lowering_common import GroupedForwardPairLoweringContext
from .grouped_mmq_fwd_pair_lowering_iq2_s import GroupedIQ2SPairedK128Lowering
from .grouped_mmq_fwd_pair_lowering_iq2_xxs import (
    GroupedIQ2XXSPairedK128Lowering,
)
from .grouped_mmq_fwd_pair_lowering_q3_k import GroupedQ3KPairedK128Lowering
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from .grouped_mmq_fwd_pair_validation import validate_grouped_forward_pair_solution
from .kernel_abi import (
    GROUPED_FORWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
)
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class GroupedForwardPairKernelWriterAssembly:
    """Emit one strict two-projection routed research artifact."""

    def __init__(
        self,
        solution_key: GroupedForwardPairSolutionKey,
        toolchain: Toolchain,
    ) -> None:
        reasons = validate_grouped_forward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"paired solution rejected: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardPairState.from_solution_key(solution_key)
        self.context = GroupedForwardPairLoweringContext(solution_key, self.state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        solution = self.solution_key.solution
        resources = self.state.physical_plan.resources
        row_tasks = (
            self.state.kernel_spec.route_ownership
            is GroupedPairRouteOwnership.DeviceRowTasks64
        )
        quant_type = self.solution_key.problem.quant_data_type
        ownership = (
            "device 64-row task ownership" if row_tasks else "serial GEMM ownership"
        )
        envelope = KernelEnvelope(
            module_name="GGTensileGroupedForwardPairKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=solution.isa,
            wavefront_size=solution.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-grouped-forward-pair-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=1,
            flat_workgroup_size=solution.num_threads,
            total_vgprs=resources.vgprs,
            total_sgprs=resources.sgprs,
            abi=(
                GROUPED_FORWARD_PAIR_ROW_TASK_ABI
                if row_tasks
                else GROUPED_FORWARD_PAIR_ABI
            ),
            description=(
                f"GGTensile paired grouped {quant_type} MMQ forward, K128 "
                f"interleaved {ownership}, one fixed Q8_1 F32_D4 workspace"
            ),
        )
        envelope.initialize()
        if quant_type == "IQ2_S":
            emission = GroupedIQ2SPairedK128Lowering(self.context).emission()
        elif quant_type == "IQ2_XXS":
            emission = GroupedIQ2XXSPairedK128Lowering(self.context).emission()
        elif quant_type == "Q3_K":
            emission = GroupedQ3KPairedK128Lowering(self.context).emission()
        else:
            raise ForwardKernelWriterError(
                f"paired lowering is unavailable for {quant_type!r}"
            )
        return envelope.render(
            emission.body,
            trailing_sections=emission.trailing_sections,
        )
