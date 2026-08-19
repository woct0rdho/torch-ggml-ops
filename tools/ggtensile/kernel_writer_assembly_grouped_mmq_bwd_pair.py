"""Assembly writer facade for paired grouped MMQ backward kernels."""

from pathlib import Path

from .grouped_mmq_bwd_pair_lowering import GroupedBackwardPairKernelLowering
from .grouped_mmq_bwd_pair_model import GroupedBackwardPairSolutionKey
from .grouped_mmq_bwd_pair_physical import (
    derive_grouped_backward_pair_physical_plan,
)
from .grouped_mmq_bwd_pair_spec import DerivedGroupedBackwardPairState
from .grouped_mmq_bwd_pair_validation import (
    validate_grouped_backward_pair_solution,
)
from .kernel_abi import GROUPED_BACKWARD_PAIR_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_bwd_emission import BackwardKernelWriterError
from .toolchain import Toolchain


class GroupedBackwardPairKernelWriterAssembly:
    """Emit one strict fused routed backward-pair artifact."""

    def __init__(
        self,
        solution_key: GroupedBackwardPairSolutionKey,
        toolchain: Toolchain,
    ) -> None:
        reasons = validate_grouped_backward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise BackwardKernelWriterError(f"paired solution rejected: {details}")
        self.solution_key = solution_key
        self.toolchain = toolchain
        self.state = DerivedGroupedBackwardPairState.from_solution_key(solution_key)
        self.physical = derive_grouped_backward_pair_physical_plan(self.state)
        self.lowering = GroupedBackwardPairKernelLowering(
            solution_key.kernel_name,
            self.state,
            self.physical,
        )

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        compute = self.solution_key.solution.compute
        resources = self.physical.ordinary.resources
        envelope = KernelEnvelope(
            module_name="GGTensileGroupedBackwardPairKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=compute.isa,
            wavefront_size=compute.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-grouped-backward-pair-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_num_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=1,
            flat_workgroup_size=compute.num_threads,
            total_vgprs=resources.total_vgprs,
            total_sgprs=resources.total_sgprs,
            abi=GROUPED_BACKWARD_PAIR_ABI,
            description=(
                "GGTensile fused grouped IQ2_S MMQ backward pair, "
                f"M{compute.macro_tile0}/N{compute.macro_tile1}/K32"
            ),
        )
        envelope.initialize()
        emission = self.lowering.emission()
        return envelope.render(
            emission.body,
            trailing_sections=emission.trailing_sections,
        )
