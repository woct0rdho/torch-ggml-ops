from pathlib import Path

from .iq2_s_grid import iq2_s_grid_rodata
from .kernel_abi import ORDINARY_BACKWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_bwd_emission import BackwardDiagnosticMode, BackwardKernelWriterError
from .mmq_bwd_lowering import BackwardKernelLowering
from .mmq_bwd_lowering_quant import IQ2_S_GRID_SYMBOL
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
        description = f"GGTensile {self.state.contract.quant_type} MMQ backward"
        if self.diagnostic_mode is not None:
            description += f" {self.diagnostic_mode.value} diagnostic"
        envelope = KernelEnvelope(
            module_name="GGTensileKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=self.state.spec.geometry.isa,
            wavefront_size=self.state.spec.geometry.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-rocisa-",
            code_object_version=self.state.contract.code_object_version,
            group_segment_size=self.physical.resources.lds_num_bytes,
            sgpr_work_group=(1, 1, 1),
            vgpr_work_item=1,
            flat_workgroup_size=self.state.spec.geometry.num_threads,
            total_vgprs=self.physical.resources.total_vgprs,
            total_sgprs=self.physical.resources.total_sgprs,
            abi=ORDINARY_BACKWARD_ABI,
            description=description,
        )
        envelope.initialize()
        body = self.lowering.body()
        if self.state.contract.quant_type == "IQ2_S":
            return envelope.render(
                body,
                trailing_sections=(iq2_s_grid_rodata(IQ2_S_GRID_SYMBOL),),
            )
        return envelope.render(body)
