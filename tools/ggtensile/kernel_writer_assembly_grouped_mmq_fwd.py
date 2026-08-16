"""Assembly writer facade for isolated grouped MMQ forward kernels."""

from pathlib import Path

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .grouped_mmq_fwd_lowering import (
    GroupedForwardLoweringContext,
    GroupedPackedScaleMinimumDirectLowering,
)
from .grouped_mmq_fwd_lowering_decoded_lds import grouped_decoded_lowering
from .grouped_mmq_fwd_lowering_iq2_s import GroupedIQ2SFullWeightLdsLowering
from .grouped_mmq_fwd_lowering_q2_k import grouped_q2_decoded_lowering
from .grouped_mmq_fwd_model import GroupedForwardSolutionKey, GroupedOperandSource
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
from .grouped_mmq_fwd_validation import validate_grouped_forward_solution
from .kernel_writer_assembly import initialize_rocisa, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class GroupedForwardKernelWriterAssembly:
    """Emit strict routed packed-weight forward controls."""

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        toolchain: Toolchain,
    ) -> None:
        reasons = validate_grouped_forward_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"grouped solution rejected: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardState.from_solution_key(solution_key)
        self.context = GroupedForwardLoweringContext(solution_key, self.state)
        self.toolchain = toolchain

    def write(self, output: Path) -> str:
        return write_assembly_source(output, self.source())

    def source(self) -> str:
        solution = self.solution_key.solution
        resources = self.state.physical_plan.resources
        initialize_rocisa(
            solution.isa,
            solution.wavefront_size,
            self.toolchain.assembler,
            temporary_prefix="ggtensile-grouped-forward-rocisa-",
        )

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=resources.lds_bytes,
            sgprWorkGroup=(1, 1, 0),
            vgprWorkItem=1,
            flatWorkGroupSize=solution.num_threads,
            totalVgprs=resources.vgprs,
            totalAgprs=0,
            totalSgprs=resources.sgprs,
        )
        quant_type = self.solution_key.problem.quant_data_type
        signature.addDescriptionTopic(
            f"GGTensile grouped {quant_type} MMQ forward, serial GEMM ownership, "
            f"fixed Q8_1 {solution.activation_layout} producer"
        )
        signature.addArg("weights", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("dst", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("expert_indices", SVK.SIG_GLOBALBUFFER, "i64", "generic")
        signature.addArg("expert_offsets", SVK.SIG_GLOBALBUFFER, "i32", "generic")
        signature.addArg("num_experts", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_weight", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")
        signature.addArg("bytes_per_expert", SVK.SIG_VALUE, "u64")

        module = code.Module("GGTensileGroupedForwardKernel")
        module.add(signature)
        if solution.operand_source is GroupedOperandSource.GroupedDirectGlobal:
            emission = GroupedPackedScaleMinimumDirectLowering(self.context).emission()
        elif solution.operand_source is GroupedOperandSource.GroupedDecodedWeightLds:
            if self.solution_key.problem.quant_data_type == "Q2_K":
                emission = grouped_q2_decoded_lowering(self.context).emission()
            else:
                emission = grouped_decoded_lowering(self.context).emission()
        elif solution.operand_source is GroupedOperandSource.GroupedIQ2SFullWeightLds:
            emission = GroupedIQ2SFullWeightLdsLowering(self.context).emission()
        else:
            raise TypeError(
                f"unsupported grouped operand source {solution.operand_source!r}"
            )
        module.add(code.TextBlock(emission.body))
        source = str(module)
        for section in emission.trailing_sections:
            source += "\n" + section
        return source
