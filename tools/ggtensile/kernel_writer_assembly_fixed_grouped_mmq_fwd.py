"""Assembly writer facade for the fixed-group Q8_0 forward ABI."""

from pathlib import Path

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .fixed_grouped_mmq_fwd_lowering import (
    FixedForwardLoweringContext,
    FixedGroupedQ8ForwardLowering,
)
from .fixed_grouped_mmq_fwd_model import FixedForwardSolutionKey
from .fixed_grouped_mmq_fwd_spec import DerivedFixedForwardState
from .fixed_grouped_mmq_fwd_validation import validate_fixed_forward_solution_key
from .kernel_writer_assembly import initialize_rocisa, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class FixedGroupedForwardKernelWriterAssembly:
    """Emit one strict six-argument fixed-group Q8_0 kernel."""

    def __init__(self, solution_key: FixedForwardSolutionKey, toolchain: Toolchain):
        try:
            validate_fixed_forward_solution_key(solution_key)
            state = DerivedFixedForwardState.from_solution_key(solution_key)
        except (TypeError, ValueError) as error:
            raise ForwardKernelWriterError(f"solution rejected: {error}") from error
        self.solution_key = solution_key
        self.state = state
        self.context = FixedForwardLoweringContext(solution_key, state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        initialize_rocisa(
            self.solution_key.solution.isa,
            self.solution_key.solution.wavefront_size,
            self.toolchain.assembler,
            temporary_prefix="ggtensile-fixed-forward-rocisa-",
        )
        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=self.state.resources.lds_bytes,
            sgprWorkGroup=(1, 1, 1),
            vgprWorkItem=1,
            flatWorkGroupSize=self.state.num_threads,
            totalVgprs=self.state.resources.vgprs,
            totalAgprs=0,
            totalSgprs=self.state.resources.sgprs,
        )
        signature.addDescriptionTopic(
            "GGTensile fixed-group Q8_0 forward, Q8_1 F32_D4 activations"
        )
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("tokens", SVK.SIG_VALUE, "u32")
        signature.addArg("out_features", SVK.SIG_VALUE, "u32")
        signature.addArg("bytes_per_group", SVK.SIG_VALUE, "u64")

        module = code.Module("GGTensileFixedGroupedForwardKernel")
        module.add(signature)
        module.add(code.TextBlock(FixedGroupedQ8ForwardLowering(self.context).body()))
        return str(module)
