from pathlib import Path

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .kernel_writer_assembly import (
    initialize_rocisa,
    write_assembly_source,
)
from .mmq_fwd_lowering import ForwardKernelWriterError, ForwardLoweringContext
from .mmq_fwd_lowering_decoded_lds import DecodedWeightLdsLowering
from .mmq_fwd_lowering_packed_3bit import Packed3BitTiledLdsLowering
from .mmq_fwd_lowering_packed_direct import PackedScaleMinimumDirectLowering
from .mmq_fwd_lowering_q6 import Q6StructuredLowering
from .mmq_fwd_lowering_signed_i8 import SignedInt8ForwardLowering
from .mmq_fwd_spec import (
    DerivedForwardState,
)
from .model import ForwardSolution, SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class ForwardKernelWriterAssembly:
    """Emit strict packed K-quant/Q8_1 forward controls."""

    def __init__(self, solution_key: SolutionKey, toolchain: Toolchain) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"solution rejected: {details}")
        if not isinstance(solution_key.solution, ForwardSolution):
            raise ForwardKernelWriterError("forward writer requires ForwardSolution")
        self.solution_key = solution_key
        self.state = DerivedForwardState.from_solution_key(solution_key)
        self.context = ForwardLoweringContext(solution_key, self.state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        initialize_rocisa(
            self.state.contract.isa,
            self.state.contract.wavefront_size,
            self.toolchain.assembler,
            temporary_prefix="ggtensile-forward-rocisa-",
        )

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=self.state.resources.lds_bytes,
            sgprWorkGroup=(1, 1, 0),
            vgprWorkItem=(
                1
                if self.state.kernel_spec.global_memory.operand_source
                in {
                    "Q6StructuredDecoded",
                    "Q3HipTiledLds",
                    "Q8RegisterTiled",
                    "Q8HipTiledLds",
                    "Q8SmallMTiledLds",
                }
                else 0
            ),
            flatWorkGroupSize=self.state.num_threads,
            totalVgprs=self.state.resources.vgprs,
            totalAgprs=0,
            totalSgprs=self.state.resources.sgprs,
        )
        signature.addDescriptionTopic(
            f"GGTensile {self.state.contract.quant_type} MMQ forward, fixed Q8_1 "
            f"{self.state.contract.activation_layout} producer"
        )
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("nrows_weight", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation_padded", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileForwardKernel")
        module.add(signature)
        module.add(code.TextBlock(self._body()))
        return str(module)

    def _body(self) -> str:
        operand_source = self.state.kernel_spec.global_memory.operand_source
        if operand_source == "Q6StructuredDecoded":
            return Q6StructuredLowering(self.context).body()
        if operand_source == "Q3HipTiledLds":
            return Packed3BitTiledLdsLowering(self.context).body()
        if operand_source == PackedScaleMinimumDirectLowering.OPERAND_SOURCE:
            return PackedScaleMinimumDirectLowering(self.context).body()
        if operand_source == DecodedWeightLdsLowering.OPERAND_SOURCE:
            return DecodedWeightLdsLowering(self.context).body()
        if operand_source in SignedInt8ForwardLowering.OPERAND_SOURCES:
            return SignedInt8ForwardLowering(self.context).body()
        raise TypeError(f"unsupported forward operand source {operand_source!r}")
