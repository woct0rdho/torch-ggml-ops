"""Assembly writer facade for fixed-group Q8_0 backward."""

from pathlib import Path

from .fixed_grouped_mmq_bwd_lowering import FixedGroupedQ8BackwardLowering
from .fixed_grouped_mmq_bwd_model import FixedBackwardSolutionKey
from .fixed_grouped_mmq_bwd_spec import DerivedFixedBackwardState
from .fixed_grouped_mmq_bwd_validation import (
    validate_fixed_backward_solution_key,
)
from .kernel_abi import FIXED_GROUPED_BACKWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .toolchain import Toolchain


class FixedGroupedBackwardKernelWriterAssembly:
    """Emit one strict six-argument fixed-group Q8_0 backward kernel."""

    def __init__(
        self, solution_key: FixedBackwardSolutionKey, toolchain: Toolchain
    ) -> None:
        validate_fixed_backward_solution_key(solution_key)
        self.solution_key = solution_key
        self.state = DerivedFixedBackwardState.from_solution_key(solution_key)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        state = self.state
        resources = state.physical.resources
        geometry = state.spec.compute.geometry
        envelope = KernelEnvelope(
            module_name="GGTensileFixedGroupedBackwardKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=geometry.isa,
            wavefront_size=geometry.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-fixed-backward-rocisa-",
            code_object_version=5,
            group_segment_size=resources.lds_num_bytes,
            sgpr_work_group=(1, 1, 1),
            vgpr_work_item=1,
            flat_workgroup_size=geometry.num_threads,
            total_vgprs=resources.total_vgprs,
            total_sgprs=resources.total_sgprs,
            abi=FIXED_GROUPED_BACKWARD_ABI,
            description="GGTensile fixed-group Q8_0 backward",
        )
        envelope.initialize()
        emission = FixedGroupedQ8BackwardLowering(
            self.solution_key.kernel_name, state
        ).emission()
        return envelope.render(
            emission.body, trailing_sections=emission.trailing_sections
        )
