"""Assembly writer facade for paired grouped forward research kernels."""

from pathlib import Path

from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .grouped_mmq_fwd_pair_lowering_iq2_s import (
    GroupedForwardPairLoweringContext,
    GroupedIQ2SPairedK128Lowering,
)
from .grouped_mmq_fwd_pair_lowering_iq2_xxs import (
    GroupedIQ2XXSPairedK128Lowering,
)
from .grouped_mmq_fwd_pair_lowering_q3_k import GroupedQ3KPairedK128Lowering
from .grouped_mmq_fwd_pair_model import (
    GroupedForwardPairSolutionKey,
    GroupedPairRouteOwnership,
)
from .grouped_mmq_fwd_pair_spec import DerivedGroupedForwardPairState
from .grouped_mmq_fwd_pair_validation import validate_grouped_forward_pair_solution
from .kernel_writer_assembly import initialize_rocisa, write_assembly_source
from .mmq_fwd_lowering import ForwardKernelWriterError
from .toolchain import Toolchain


class GroupedForwardPairKernelWriterAssembly:
    """Emit one strict two-projection routed research artifact."""

    def __init__(
        self,
        solution_key: GroupedForwardPairSolutionKey,
        toolchain: Toolchain,
    ) -> None:
        reasons = validate_grouped_forward_pair_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"paired solution rejected: {details}")
        self.solution_key = solution_key
        self.state = DerivedGroupedForwardPairState.from_solution_key(solution_key)
        self.context = GroupedForwardPairLoweringContext(solution_key, self.state)
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
            temporary_prefix="ggtensile-grouped-forward-pair-rocisa-",
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
        row_tasks = (
            self.state.contract.route_ownership
            is GroupedPairRouteOwnership.DeviceRowTasks64
        )
        quant_type = self.solution_key.problem.quant_data_type
        ownership = (
            "device 64-row task ownership" if row_tasks else "serial GEMM ownership"
        )
        signature.addDescriptionTopic(
            f"GGTensile paired grouped {quant_type} MMQ forward, K128 interleaved "
            f"{ownership}, one fixed Q8_1 F32_D4 workspace"
        )
        signature.addArg("weights_first", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("weights_second", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("dst_first", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("dst_second", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        if row_tasks:
            signature.addArg("task_count", SVK.SIG_GLOBALBUFFER, "i32", "generic")
            signature.addArg("task_experts", SVK.SIG_GLOBALBUFFER, "i32", "generic")
            signature.addArg("task_row_starts", SVK.SIG_GLOBALBUFFER, "i32", "generic")
            signature.addArg("task_row_ends", SVK.SIG_GLOBALBUFFER, "i32", "generic")
        else:
            signature.addArg("expert_indices", SVK.SIG_GLOBALBUFFER, "i64", "generic")
            signature.addArg("expert_offsets", SVK.SIG_GLOBALBUFFER, "i32", "generic")
        signature.addArg("num_experts", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_weight", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")
        signature.addArg("bytes_per_expert", SVK.SIG_VALUE, "u64")

        module = code.Module("GGTensileGroupedForwardPairKernel")
        module.add(signature)
        if quant_type == "IQ2_S":
            emission = GroupedIQ2SPairedK128Lowering(self.context).emission()
        elif quant_type == "IQ2_XXS":
            emission = GroupedIQ2XXSPairedK128Lowering(self.context).emission()
        elif quant_type == "Q3_K":
            emission = GroupedQ3KPairedK128Lowering(self.context).emission()
        else:
            raise ForwardKernelWriterError(
                f"paired lowering is unavailable for {quant_type!r}"
            )
        module.add(code.TextBlock(emission.body))
        source = str(module)
        for section in emission.trailing_sections:
            source += "\n" + section
        return source
