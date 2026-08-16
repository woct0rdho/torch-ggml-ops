from pathlib import Path

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .kernel_writer_assembly import initialize_rocisa, write_assembly_source
from .mmq_bwd_emission import BackwardDiagnosticMode, BackwardKernelWriterError
from .mmq_bwd_lowering import BackwardKernelLowering
from .model import BackwardSolution, SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class BackwardKernelWriterAssembly:
    """Public facade for one exact gfx1151 MMQ backward solution."""

    def __init__(
        self,
        solution_key: SolutionKey,
        toolchain: Toolchain,
        *,
        diagnostic_mode: BackwardDiagnosticMode | None = None,
    ) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise BackwardKernelWriterError(f"solution rejected: {details}")
        if not isinstance(solution_key.solution, BackwardSolution):
            raise BackwardKernelWriterError("backward writer requires BackwardSolution")
        self.solution_key = solution_key
        self.toolchain = toolchain
        self.diagnostic_mode = diagnostic_mode
        self.lowering = BackwardKernelLowering(
            solution_key,
            diagnostic_mode=diagnostic_mode,
        )
        self.state = self.lowering.state
        self.physical = self.lowering.physical
        self.registers = self.physical.registers

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        initialize_rocisa(
            self.state.spec.geometry.isa,
            self.state.spec.geometry.wavefront_size,
            self.toolchain.assembler,
            temporary_prefix="ggtensile-rocisa-",
        )

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion=str(self.state.contract.code_object_version),
            groupSegmentSize=self.physical.resources.lds_num_bytes,
            sgprWorkGroup=(1, 1, 1),
            vgprWorkItem=1,
            flatWorkGroupSize=self.state.spec.geometry.num_threads,
            totalVgprs=self.physical.resources.total_vgprs,
            totalAgprs=0,
            totalSgprs=self.physical.resources.total_sgprs,
        )
        description = f"GGTensile {self.state.contract.quant_type} MMQ backward"
        if self.diagnostic_mode is not None:
            description += f" {self.diagnostic_mode.value} diagnostic"
        signature.addDescriptionTopic(description)
        signature.addArg("grad_output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("grad_input", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("rows", SVK.SIG_VALUE, "u32")
        signature.addArg("out_features", SVK.SIG_VALUE, "u32")
        signature.addArg("in_features", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileKernel")
        module.add(signature)
        module.add(code.TextBlock(self.lowering.body()))
        return str(module)
