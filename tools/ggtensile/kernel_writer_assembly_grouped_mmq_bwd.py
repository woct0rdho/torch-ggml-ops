"""Assembly writer facade for isolated grouped MMQ backward kernels."""

from pathlib import Path

from .grouped_mmq_bwd_lowering import GroupedBackwardKernelLowering
from .kernel_abi import GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_bwd_emission import BackwardKernelWriterError
from .model import GroupedBackwardSolution, SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class GroupedBackwardKernelWriterAssembly:
    """Public facade for one exact routed gfx1151 backward solution."""

    def __init__(self, solution_key: SolutionKey, toolchain: Toolchain) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise BackwardKernelWriterError(f"solution rejected: {details}")
        if not isinstance(solution_key.solution, GroupedBackwardSolution):
            raise BackwardKernelWriterError(
                "grouped backward writer requires GroupedBackwardSolution"
            )
        self.solution_key = solution_key
        self.toolchain = toolchain
        self.lowering = GroupedBackwardKernelLowering(solution_key)
        self.state = self.lowering.grouped_state
        self.physical = self.lowering.grouped_physical

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        compute = self.state.spec.compute
        resources = self.physical.resources
        envelope = KernelEnvelope(
            module_name="GGTensileKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=compute.geometry.isa,
            wavefront_size=compute.geometry.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-rocisa-grouped-bwd-",
            code_object_version=self.state.contract.code_object_version,
            group_segment_size=resources.lds_num_bytes,
            sgpr_work_group=(
                1,
                1,
                int(self.state.spec.ownership.split_factor > 1),
            ),
            vgpr_work_item=1,
            flat_workgroup_size=compute.geometry.num_threads,
            total_vgprs=resources.total_vgprs,
            total_sgprs=resources.total_sgprs,
            abi=GROUPED_BACKWARD_ABI,
            description=(
                f"GGTensile {self.state.contract.quant_type} grouped MMQ backward"
            ),
        )
        envelope.initialize()
        return envelope.render(self.lowering.body())
