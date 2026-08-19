from pathlib import Path

from .kernel_abi import ORDINARY_FORWARD_ABI
from .kernel_writer_assembly import KernelEnvelope, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError, ForwardLoweringContext
from .mmq_fwd_lowering_decoded_lds import DecodedWeightLdsLowering
from .mmq_fwd_lowering_packed_3bit import Packed3BitTiledLdsLowering
from .mmq_fwd_lowering_packed_direct import PackedScaleMinimumDirectLowering
from .mmq_fwd_lowering_q3_full import FullWeightQ3TiledLdsLowering
from .mmq_fwd_lowering_q6 import Q6StructuredLowering
from .mmq_fwd_lowering_signed_i8 import SignedInt8ForwardLowering
from .mmq_fwd_spec import (
    DerivedForwardState,
    forward_mechanism_contract,
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
        self.context = ForwardLoweringContext(solution_key.kernel_name, self.state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        envelope = KernelEnvelope(
            module_name="GGTensileForwardKernel",
            kernel_name=self.solution_key.kernel_name,
            isa=self.state.contract.isa,
            wavefront_size=self.state.contract.wavefront_size,
            assembler=self.toolchain.assembler,
            temporary_prefix="ggtensile-forward-rocisa-",
            code_object_version=5,
            group_segment_size=self.state.resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=int(
                forward_mechanism_contract(
                    self.state.kernel_spec.global_memory.operand_source
                ).uses_workitem_id
            ),
            flat_workgroup_size=self.state.num_threads,
            total_vgprs=self.state.resources.vgprs,
            total_sgprs=self.state.resources.sgprs,
            abi=ORDINARY_FORWARD_ABI,
            description=(
                f"GGTensile {self.state.contract.quant_type} MMQ forward, fixed "
                f"Q8_1 {self.state.contract.activation_layout} producer"
            ),
        )
        envelope.initialize()
        return envelope.render(self._body())

    def _body(self) -> str:
        operand_source = self.state.kernel_spec.global_memory.operand_source
        lowering = forward_mechanism_contract(operand_source).lowering
        if lowering == "StructuredQ6":
            return Q6StructuredLowering(self.context).body()
        if lowering == "Packed3BitTiledLds":
            return Packed3BitTiledLdsLowering(self.context).body()
        if lowering == "Packed3BitFullWeightTiledLds":
            return FullWeightQ3TiledLdsLowering(self.context).body()
        if lowering == "PackedScaleMinimumDirect":
            return PackedScaleMinimumDirectLowering(self.context).body()
        if lowering == "DecodedWeightLds":
            return DecodedWeightLdsLowering(self.context).body()
        if lowering == "SignedInt8":
            return SignedInt8ForwardLowering(self.context).body()
        raise TypeError(f"unsupported forward operand source {operand_source!r}")
