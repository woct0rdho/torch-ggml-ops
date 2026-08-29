from .kernel_abi import ORDINARY_FORWARD_ABI
from .kernel_writer_assembly import (
    AssemblyKernelWriter,
    KernelEmissionPlan,
    LoweringResult,
)
from .mmq_fwd_lowering import ForwardLoweringContext
from .mmq_fwd_lowering_decoded_lds import DecodedWeightLdsLowering
from .mmq_fwd_lowering_packed_3bit import Packed3BitTiledLdsLowering
from .mmq_fwd_lowering_packed_direct import PackedScaleMinimumDirectLowering
from .mmq_fwd_lowering_q3_full import FullWeightQ3TiledLdsLowering
from .mmq_fwd_lowering_q6 import Q6StructuredLowering
from .mmq_fwd_lowering_signed_i8 import SignedInt8ForwardLowering
from .mmq_fwd_spec import (
    DerivedForwardState,
    ForwardKernelSpec,
    forward_mechanism_contract,
)
from .model import ProblemSize
from .toolchain import Toolchain
from .validation import validate_forward_solution


class ForwardKernelWriterAssembly(AssemblyKernelWriter):
    """Emit strict packed K-quant/Q8_1 forward controls."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: ForwardKernelSpec,
        kernel_name: str,
        toolchain: Toolchain,
    ) -> None:
        validate_forward_solution(problem_size, quant_type, kernel_spec)
        self.kernel_name = kernel_name
        self.state = DerivedForwardState.from_problem_spec(
            problem_size, quant_type, kernel_spec
        )
        self.context = ForwardLoweringContext(kernel_name, self.state)
        self.assembler = toolchain.assembler

    def emission_plan(self) -> KernelEmissionPlan:
        return KernelEmissionPlan(
            module_name="GGTensileForwardKernel",
            kernel_name=self.kernel_name,
            isa=self.state.contract.isa,
            wavefront_size=self.state.contract.wavefront_size,
            temporary_prefix="ggtensile-forward-rocisa-",
            code_object_version=5,
            group_segment_size=self.state.resources.lds_bytes,
            sgpr_work_group=(1, 1, 0),
            vgpr_work_item=int(
                forward_mechanism_contract(
                    self.state.kernel_spec.global_memory.weight_staging
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
            lowering=LoweringResult(self._body()),
        )

    def _body(self) -> str:
        weight_staging = self.state.kernel_spec.global_memory.weight_staging
        lowering = forward_mechanism_contract(weight_staging).lowering
        lowering_types = {
            "StructuredQ6": Q6StructuredLowering,
            "Packed3BitTiledLds": Packed3BitTiledLdsLowering,
            "Packed3BitFullWeightTiledLds": FullWeightQ3TiledLdsLowering,
            "PackedScaleMinimumDirect": PackedScaleMinimumDirectLowering,
            "DecodedWeightLds": DecodedWeightLdsLowering,
            "SignedInt8": SignedInt8ForwardLowering,
        }
        assert lowering in lowering_types
        lowering_type = lowering_types[lowering]
        return lowering_type(self.context).body()
