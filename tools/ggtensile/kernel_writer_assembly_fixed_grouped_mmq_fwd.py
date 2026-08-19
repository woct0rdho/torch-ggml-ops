"""Assembly writer facade for the fixed-group Q8_0 forward ABI."""

from pathlib import Path

from .fixed_grouped_mmq_fwd_lowering import (
    FixedForwardLoweringContext,
    FixedGroupedQ8ForwardLowering,
)
from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from .fixed_grouped_mmq_fwd_validation import validate_fixed_forward_solution_key
from .kernel_abi import FIXED_GROUPED_FORWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .toolchain import Toolchain


class FixedGroupedForwardKernelWriterAssembly:
    """Emit one strict six-argument fixed-group Q8_0 kernel."""

    def __init__(self, solution_key: FixedForwardSolutionKey, toolchain: Toolchain):
        validate_fixed_forward_solution_key(solution_key)
        state = DerivedFixedForwardState.from_solution_key(solution_key)
        self.solution_key = solution_key
        self.state = state
        self.context = FixedForwardLoweringContext(solution_key.kernel_name, state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        resources = self.state.physical.resources
        envelope = KernelEnvelope(
            module_name="GGTensileFixedGroupedForwardKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=self.state.contract.isa,
            wavefront_size=self.state.contract.wavefront_size,
            assembler=self.toolchain.assembler,
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
        )
        envelope.initialize()
        return envelope.render(FixedGroupedQ8ForwardLowering(self.context).body())
