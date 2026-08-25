from .kernel_abi import ORDINARY_BACKWARD_ABI
from .kernel_writer_assembly import AssemblyKernelWriter, KernelEmissionPlan
from .mmq_bwd_emission import BackwardDiagnosticMode
from .mmq_bwd_lowering import BackwardTileComputeEmitter
from .mmq_bwd_physical import derive_backward_physical_plan
from .mmq_bwd_spec import BackwardKernelSpec, DerivedBackwardState
from .model import ProblemSize
from .toolchain import Toolchain
from .validation import validate_backward_solution


class BackwardKernelWriterAssembly(AssemblyKernelWriter):
    """Public facade for one exact gfx1151 MMQ backward solution."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: BackwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
        *,
        diagnostic_mode: BackwardDiagnosticMode | None = None,
    ) -> None:
        validate_backward_solution(problem_size, quant_type, kernel_spec)
        self.kernel_name = kernel_name
        self.assembler = toolchain.assembler
        self.diagnostic_mode = diagnostic_mode
        self.state = DerivedBackwardState.from_problem_spec(
            problem_size, quant_type, kernel_spec
        )
        self.physical = derive_backward_physical_plan(self.state)
        self.lowering = BackwardTileComputeEmitter(
            self.state,
            self.physical,
            diagnostic_mode=diagnostic_mode,
        )
        self.registers = self.physical.registers

    def emission_plan(self) -> KernelEmissionPlan:
        description = f"GGTensile {self.state.contract.quant_type} MMQ backward"
        if self.diagnostic_mode is not None:
            description += f" {self.diagnostic_mode.value} diagnostic"
        emission = self.lowering.ordinary_emission(self.kernel_name)
        return KernelEmissionPlan(
            module_name="GGTensileKernel",
            kernel_name=self.kernel_name,
            isa=self.state.spec.geometry.isa,
            wavefront_size=self.state.spec.geometry.wavefront_size,
            temporary_prefix="ggtensile-rocisa-",
            code_object_version=self.state.contract.code_object_version,
            group_segment_size=self.physical.resources.lds_bytes,
            sgpr_work_group=(1, 1, 1),
            vgpr_work_item=1,
            flat_workgroup_size=self.state.spec.geometry.num_threads,
            total_vgprs=self.physical.resources.vgprs,
            total_sgprs=self.physical.resources.sgprs,
            abi=ORDINARY_BACKWARD_ABI,
            description=description,
            lowering=emission,
        )
