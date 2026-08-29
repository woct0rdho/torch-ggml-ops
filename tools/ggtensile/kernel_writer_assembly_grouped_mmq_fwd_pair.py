"""Assembly writer facade for paired grouped forward research kernels."""

from .grouped_mmq_fwd_pair_lowering_common import GroupedForwardPairLoweringContext
from .grouped_mmq_fwd_pair_lowering_iq2_s import GroupedIQ2SPairedK128Lowering
from .grouped_mmq_fwd_pair_lowering_iq2_xxs import (
    GroupedIQ2XXSPairedK128Lowering,
)
from .grouped_mmq_fwd_pair_lowering_q3_k import GroupedQ3KPairedK128Lowering
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairProblem,
    GroupedPairRouteOwnership,
    GroupedPairWeightStaging,
)
from .grouped_mmq_fwd_pair_spec import (
    DerivedGroupedForwardPairState,
    GroupedForwardPairKernelSpec,
)
from .grouped_mmq_fwd_pair_validation import validate_grouped_forward_pair_solution
from .kernel_abi import (
    GROUPED_FORWARD_PAIR_ABI,
    GROUPED_FORWARD_PAIR_ROW_TASK_ABI,
)
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class GroupedForwardPairKernelWriterAssembly(AssemblyKernelWriter):
    """Emit one strict two-projection routed research artifact."""

    def __init__(
        self,
        problem: GroupedForwardPairProblem,
        kernel_spec: GroupedForwardPairKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_grouped_forward_pair_solution(problem, kernel_spec)
        self.kernel_name = kernel_name
        self.state = DerivedGroupedForwardPairState.from_problem_spec(
            problem, kernel_spec
        )
        self.context = GroupedForwardPairLoweringContext(
            kernel_name, problem, self.state
        )
        self.assembler = toolchain.assembler

    def emission_plan(self) -> KernelEmissionPlan:
        spec = self.state.kernel_spec
        contract = self.state.contract
        resources = self.state.physical_plan.resources
        row_tasks = spec.route_ownership is GroupedPairRouteOwnership.DeviceRowTasks
        quant_type = self.state.problem.quant_data_type
        ownership = (
            f"device {self.state.kernel_spec.row_task_rows}-row task ownership"
            if row_tasks
            else "serial GEMM ownership"
        )
        weight_staging = self.state.kernel_spec.weight_staging
        if weight_staging is GroupedPairWeightStaging.GridHalfWeightLds:
            emission = GroupedIQ2SPairedK128Lowering(self.context).emission()
        elif weight_staging is GroupedPairWeightStaging.ParityGridHalfWeightLds:
            emission = GroupedIQ2XXSPairedK128Lowering(self.context).emission()
        elif weight_staging is GroupedPairWeightStaging.SignedThreeBitHalfWeightLds:
            emission = GroupedQ3KPairedK128Lowering(self.context).emission()
        else:
            raise ForwardKernelWriterError(
                f"paired lowering is unavailable for {weight_staging!r}"
            )
        return KernelEmissionPlan(
            module_name="GGTensileGroupedForwardPairKernel",
            kernel_name=self.kernel_name,
            isa=contract.isa,
            wavefront_size=contract.wavefront_size,
            temporary_prefix="ggtensile-grouped-forward-pair-rocisa-",
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
            abi=(
                GROUPED_FORWARD_PAIR_ROW_TASK_ABI
                if row_tasks
                else GROUPED_FORWARD_PAIR_ABI
            ),
            description=(
                f"GGTensile paired grouped {quant_type} MMQ forward, K128 "
                f"interleaved {ownership}, one fixed Q8_1 F32_D4 workspace"
            ),
            lowering=emission,
        )
