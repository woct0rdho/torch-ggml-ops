"""Launch machinery for compiled MMQ kernels.

Covers the GGTensile bundle artifacts and the installed HIP controls alike: the
shared HIP module lifetime, the family launchers the benchmark and the
deployment runner use, and the installed-control wrappers that the HIP
deployment catalog selects. Artifact lookup goes through `tools.mmq_hip_paths`.
"""

import ctypes
import ctypes.util
import os
import sysconfig
from pathlib import Path
from types import TracebackType

import torch
from typing_extensions import Self

from tools.ggtensile.fixed_grouped_mmq_bwd_model import FixedBackwardProblem
from tools.ggtensile.fixed_grouped_mmq_bwd_spec import (
    DerivedFixedBackwardState,
    FixedBackwardKernelSpec,
)
from tools.ggtensile.fixed_grouped_mmq_bwd_validation import (
    validate_fixed_backward_solution,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_model import FixedForwardProblem
from tools.ggtensile.fixed_grouped_mmq_fwd_spec import (
    DerivedFixedForwardState,
    FixedForwardKernelSpec,
)
from tools.ggtensile.fixed_grouped_mmq_fwd_validation import (
    validate_fixed_forward_solution,
)
from tools.ggtensile.grouped_mmq_bwd_spec import (
    DerivedGroupedBackwardState,
    GroupedBackwardKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_model import GroupedForwardProblem
from tools.ggtensile.grouped_mmq_fwd_spec import (
    DerivedGroupedForwardState,
    GroupedForwardKernelSpec,
)
from tools.ggtensile.grouped_mmq_fwd_validation import validate_grouped_forward_solution
from tools.ggtensile.mmq_bwd_spec import BackwardKernelSpec
from tools.ggtensile.mmq_fwd_spec import DerivedForwardState, ForwardKernelSpec
from tools.ggtensile.model import ProblemSize
from tools.ggtensile.validation import (
    validate_backward_solution,
    validate_forward_solution,
    validate_grouped_backward_solution,
)
from tools.mmq_abi import (
    FIXED_GROUPED_BACKWARD_ABI,
    FIXED_GROUPED_FORWARD_ABI,
    GROUPED_BACKWARD_ABI,
    GROUPED_FORWARD_ABI,
    ORDINARY_BACKWARD_ABI,
    ORDINARY_FORWARD_ABI,
    Q8_1_QUANTIZER_ABI,
)
from tools.mmq_hip_deployment import GroupedForwardControl, select_hip_control
from tools.mmq_hip_paths import locate_control
from tools.mmq_quant_formats import (
    BACKWARD_QUANT_FORMATS,
    Q8_1_F16_D2S6_BLOCK_BYTES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)
from tools.mmq_work_group_mapping import mapped_grid_extent, mapped_m_tile_count


class HIPRuntimeError(RuntimeError):
    pass


def _resolve_code_object(code_object: Path, kernel_name: str) -> Path:
    located = locate_control(kernel_name, code_object)
    if located is not None:
        return located
    if code_object.is_dir():
        raise HIPRuntimeError(
            f"code object directory does not contain {kernel_name}.hsaco: {code_object}"
        )
    raise HIPRuntimeError(f"code object does not exist: {code_object}")


def find_control(symbol: str, code_object: Path | None = None) -> Path:
    """Return the artifact path of one installed HIP control, failing closed.

    An explicit `code_object` may be the artifact itself or a control
    directory. Otherwise the shared control roots are searched.
    """

    if code_object is not None:
        return _resolve_code_object(code_object, symbol)
    located = locate_control(symbol)
    if located is None:
        raise HIPRuntimeError(
            f"cannot find HIP control {symbol}. Pass --hip-code-object or set "
            "GGTENSILE_HIP_CONTROL_ROOT"
        )
    return located


# HIP defaults the dynamic LDS of a function to 48 KiB, and a tile above
# that has to ask for the space it uses (`hipFuncAttributeMaxDynamicSharedMemorySize`
# is attribute 8 in the HIP runtime API).
_DEFAULT_DYNAMIC_SHARED_BYTES = 48 * 1024
_HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE = 8


class _HIPModule:
    """Shared HIP module lifetime and runtime API configuration."""

    def __init__(
        self,
        code_object: Path,
        hip_library: Path | None,
        kernel_name: str,
    ) -> None:
        code_object = _resolve_code_object(code_object, kernel_name)
        self.code_object = code_object
        self._lib = ctypes.CDLL(str(hip_library or _find_hip_library()))
        self._configure_api()
        self._module = ctypes.c_void_p()
        self._function = ctypes.c_void_p()
        self._check(
            self._lib.hipModuleLoad(
                ctypes.byref(self._module), str(code_object).encode()
            ),
            "hipModuleLoad",
        )
        function_loaded = False
        try:
            self._check(
                self._lib.hipModuleGetFunction(
                    ctypes.byref(self._function),
                    self._module,
                    kernel_name.encode(),
                ),
                "hipModuleGetFunction",
            )
            function_loaded = True
        finally:
            if not function_loaded:
                self.close()

    def close(self) -> None:
        if self._module:
            self._check(self._lib.hipModuleUnload(self._module), "hipModuleUnload")
            self._module = ctypes.c_void_p()
            self._function = ctypes.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _check(self, status: int, operation: str) -> None:
        if status:
            message = self._lib.hipGetErrorString(status).decode()
            raise HIPRuntimeError(f"{operation} failed ({status}): {message}")

    def _set_dynamic_shared_bytes(self, shared_bytes: int) -> None:
        """Raise the per-function dynamic LDS limit above the 48 KiB default.

        A control whose tile needs more than the default must opt in before it
        can be launched, and a tile of that size also fixes the occupancy at one
        workgroup per compute unit.
        """

        if shared_bytes <= _DEFAULT_DYNAMIC_SHARED_BYTES:
            return
        self._check(
            self._lib.hipFuncSetAttribute(
                self._function,
                _HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                shared_bytes,
            ),
            "hipFuncSetAttribute",
        )

    def _configure_api(self) -> None:
        self._lib.hipGetErrorString.argtypes = [ctypes.c_int]
        self._lib.hipGetErrorString.restype = ctypes.c_char_p
        self._lib.hipModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        self._lib.hipModuleLoad.restype = ctypes.c_int
        self._lib.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib.hipModuleGetFunction.restype = ctypes.c_int
        self._lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._lib.hipModuleLaunchKernel.restype = ctypes.c_int
        self._lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        self._lib.hipModuleUnload.restype = ctypes.c_int
        self._lib.hipFuncSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.hipFuncSetAttribute.restype = ctypes.c_int


class _OrdinaryHIPModule(_HIPModule):
    """HIP module over explicit ordinary problem and specification values."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: ForwardKernelSpec | BackwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        self.problem_size = problem_size
        self.quant_type = quant_type
        self.kernel_spec = kernel_spec
        super().__init__(code_object, hip_library, kernel_name)


class BackwardModule(_OrdinaryHIPModule):
    """Direct HIP module launcher for the fixed MMQ backward kernarg ABI."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: BackwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_backward_solution(problem_size, quant_type, kernel_spec)
        super().__init__(
            problem_size,
            quant_type,
            kernel_spec,
            code_object,
            kernel_name,
            hip_library,
        )

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        size = self.problem_size
        tensors = (grad_output, packed_weight, grad_input)
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be contiguous")
        if grad_output.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_output must be BF16")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if grad_input.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_input must be BF16")
        if tuple(grad_output.shape) != (size.m, size.k):
            raise HIPRuntimeError("grad_output shape does not match ProblemSize")
        if tuple(grad_input.shape) != (size.m, size.n):
            raise HIPRuntimeError("grad_input shape does not match ProblemSize")
        quant_format = QUANT_FORMATS[self.quant_type]
        expected_weight_bytes = (
            size.k * (size.n // quant_format.block_values) * quant_format.block_bytes
        )
        if packed_weight.numel() != expected_weight_bytes:
            raise HIPRuntimeError(
                f"packed_weight size does not match {self.quant_type} ProblemSize"
            )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise HIPRuntimeError("all launch tensors must be on the same device")

        packed_arguments = ORDINARY_BACKWARD_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "rows": size.m,
                "out_features": size.k,
                "in_features": size.n,
                "blocks_per_weight_row": size.n
                // BACKWARD_QUANT_FORMATS[self.quant_type].block_values,
            }
        )
        grid, block, shared_memory = self._launch_configuration()
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        assert isinstance(self.kernel_spec, BackwardKernelSpec)
        geometry = self.kernel_spec.geometry
        group_m = geometry.work_group_mapping
        m_blocks = self.problem_size.m // geometry.macro_tile0
        mapped_m_tile_count(m_blocks, group_m)
        return (
            (
                mapped_grid_extent(1, group_m),
                self.problem_size.n // geometry.macro_tile1,
                m_blocks // group_m,
            ),
            geometry.work_group,
            0,
        )


class GroupedBackwardModule(_HIPModule):
    """Direct launcher for the exact routed grouped backward ABI."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: GroupedBackwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_grouped_backward_solution(problem_size, quant_type, kernel_spec)
        self.problem_size = problem_size
        self.quant_type = quant_type
        self.kernel_spec = kernel_spec
        _HIPModule.__init__(
            self,
            code_object,
            hip_library,
            kernel_name,
        )

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = DerivedGroupedBackwardState.from_problem_spec(
            self.problem_size, self.quant_type, self.kernel_spec
        )
        size = state.contract.problem_size
        tensors = (
            grad_output,
            packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be contiguous")
        if grad_output.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_output must be BF16")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if grad_input.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_input must be BF16")
        if expert_indices.dtype != torch.int64:
            raise HIPRuntimeError("expert_indices must be int64")
        if expert_offsets.dtype != torch.int32:
            raise HIPRuntimeError("expert_offsets must be int32")
        if tuple(grad_output.shape) != (size.m, size.k):
            raise HIPRuntimeError("grad_output shape does not match grouped key")
        if tuple(grad_input.shape) != (size.m, size.n):
            raise HIPRuntimeError("grad_input shape does not match grouped key")
        row_bytes = (
            size.n
            // state.contract.quant_format.block_values
            * state.contract.quant_format.block_bytes
        )
        expected_weight_shape = (
            state.contract.physical_experts,
            size.k,
            row_bytes,
        )
        if tuple(packed_weight.shape) != expected_weight_shape:
            raise HIPRuntimeError("packed_weight shape does not match grouped key")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("route metadata must be one-dimensional")
        if expert_indices.numel() != expert_offsets.numel():
            raise HIPRuntimeError("route metadata lengths differ")
        num_routes = expert_indices.numel()
        if not 0 < num_routes <= state.contract.max_route_entries:
            raise HIPRuntimeError("route count is outside the grouped key")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("all launch tensors must be on the same device")

        bytes_per_expert = size.k * row_bytes
        packed_arguments = GROUPED_BACKWARD_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": state.contract.physical_experts,
                "rows": size.m,
                "bytes_per_expert": bytes_per_expert,
            }
        )
        grid, block, shared_memory = self._launch_configuration_for_state(
            state, num_routes
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    @staticmethod
    def _launch_configuration_for_state(
        state: DerivedGroupedBackwardState,
        route_entries: int,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        geometry = state.spec.compute.geometry
        return (
            (
                mapped_grid_extent(
                    state.contract.problem_size.n // geometry.macro_tile1,
                    geometry.work_group_mapping,
                ),
                route_entries,
                state.spec.ownership.split_factor,
            ),
            geometry.work_group,
            0,
        )


class ForwardModule(_OrdinaryHIPModule):
    """Direct launcher for an exact packed K-quant/Q8_1 forward kernel."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: ForwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_forward_solution(problem_size, quant_type, kernel_spec)
        super().__init__(
            problem_size,
            quant_type,
            kernel_spec,
            code_object,
            kernel_name,
            hip_library,
        )

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        assert isinstance(self.kernel_spec, ForwardKernelSpec)
        state = DerivedForwardState.from_problem_spec(
            self.problem_size, self.quant_type, self.kernel_spec
        )
        size = state.problem_size
        tensors = (packed_weight, activations, output)
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be contiguous")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if activations.dtype != torch.uint8:
            raise HIPRuntimeError(
                "activations must be the uint8 Q8_1 "
                f"{state.contract.activation_layout} workspace"
            )
        if output.dtype != torch.bfloat16:
            raise HIPRuntimeError("output must be BF16")
        if packed_weight.numel() != state.expected_packed_weight_bytes:
            raise HIPRuntimeError("packed_weight size does not match ProblemSize")
        expected_activation_shape = state.expected_activation_shape
        if tuple(activations.shape) != expected_activation_shape:
            raise HIPRuntimeError(
                "activations shape does not match the exact Q8_1 "
                f"{state.contract.activation_layout} workspace contract"
            )
        if tuple(output.shape) != state.expected_output_shape:
            raise HIPRuntimeError("output shape does not match ProblemSize")
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise HIPRuntimeError("all launch tensors must be on the same device")

        packed_arguments = ORDINARY_FORWARD_ABI.pack(
            {
                "packed_weight": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "output": output.data_ptr(),
                "nrows_weight": size.n,
                "nrows_activation": size.m,
                "nrows_activation_padded": size.m,
                "blocks_per_weight_row": state.blocks_per_weight_row,
            }
        )
        grid, block, shared_memory = self._launch_configuration()
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        assert isinstance(self.kernel_spec, ForwardKernelSpec)
        state = DerivedForwardState.from_problem_spec(
            self.problem_size, self.quant_type, self.kernel_spec
        )
        return (state.grid, state.kernel_spec.geometry.work_group, 0)


class GroupedForwardModule(_HIPModule):
    """Research-only launcher for a routed grouped forward artifact."""

    def __init__(
        self,
        problem: GroupedForwardProblem,
        kernel_spec: GroupedForwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_grouped_forward_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedGroupedForwardState.from_problem_spec(problem, kernel_spec)
        super().__init__(code_object, hip_library, kernel_name)

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = self.state
        tensors = (
            packed_weight,
            activations,
            output,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all grouped launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all grouped launch tensors must be contiguous")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if activations.dtype != torch.uint8:
            raise HIPRuntimeError("activations must be uint8 Q8_1 F16_D4S4")
        if output.dtype != torch.bfloat16:
            raise HIPRuntimeError("output must be BF16")
        if expert_indices.dtype != torch.int64:
            raise HIPRuntimeError("expert_indices must be int64")
        if expert_offsets.dtype != torch.int32:
            raise HIPRuntimeError("expert_offsets must be int32")
        if tuple(packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError("packed_weight shape does not match grouped bank")
        if tuple(activations.shape) != state.expected_activation_shape:
            raise HIPRuntimeError(
                "activations shape does not match grouped Q8_1 workspace"
            )
        if tuple(output.shape) != state.expected_output_shape:
            raise HIPRuntimeError("output shape does not match grouped problem")
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError("route metadata must be one-dimensional")
        route_entries = expert_indices.numel()
        if route_entries <= 0 or route_entries > self.problem.max_route_entries:
            raise HIPRuntimeError("route entry count is outside the grouped contract")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("route metadata lengths must match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError(
                "all grouped launch tensors must be on the same device"
            )

        packed_arguments = GROUPED_FORWARD_ABI.pack(
            {
                "weights": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "dst": output.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.problem.physical_experts,
                "nrows_weight": self.problem.output_features,
                "nrows_activation": self.problem.aggregate_rows,
                "blocks_per_weight_row": state.blocks_per_weight_row,
                "bytes_per_expert": state.bytes_per_expert,
            }
        )
        grid, block, shared_memory = self._launch_configuration(route_entries)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        return (
            self.state_grid(route_entries),
            self.state.kernel_spec.geometry.work_group,
            0,
        )

    def state_grid(self, route_entries: int) -> tuple[int, int, int]:
        return self.state.grid(route_entries)


class InstalledGroupedForwardModule(GroupedForwardModule):
    """Direct launcher for one deployed HIP grouped-forward control.

    `tools.mmq_hip_deployment` picks the body for the route bank and derives
    the launch geometry from the control's build record, so retuning a family
    only changes the deployment catalog.
    """

    def __init__(
        self,
        problem: GroupedForwardProblem,
        kernel_spec: GroupedForwardKernelSpec,
        control: GroupedForwardControl,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.control = control
        super().__init__(
            problem,
            kernel_spec,
            code_object or _find_installed_kernel(control.symbol),
            control.symbol,
            hip_library,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        return self.control.launch_configuration(
            self.problem.output_features, route_entries
        )


class FixedQ81F16D2S6QuantizerModule(_HIPModule):
    """Direct launcher for the installed HIP Q8_1 F16_D2S6 producer."""

    SYMBOL = "quantize_bf16_q8_1_f16_d2s6"

    def __init__(
        self,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        selected = code_object or _find_installed_kernel(self.SYMBOL)
        super().__init__(selected, hip_library, self.SYMBOL)

    def allocate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 2 or input_tensor.shape[1] % 128:
            raise HIPRuntimeError(
                "quantizer input must be [rows, K] with K divisible by 128"
            )
        rows, k = input_tensor.shape
        return torch.empty(
            (k // 128, rows, Q8_1_F16_D2S6_BLOCK_BYTES),
            dtype=torch.uint8,
            device=input_tensor.device,
        )

    def launch(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        if not input_tensor.is_cuda or not output.is_cuda:
            raise HIPRuntimeError("quantizer tensors must be on a HIP device")
        if not input_tensor.is_contiguous() or not output.is_contiguous():
            raise HIPRuntimeError("quantizer tensors must be contiguous")
        if input_tensor.dtype != torch.bfloat16:
            raise HIPRuntimeError("quantizer input must be BF16")
        if output.dtype != torch.uint8:
            raise HIPRuntimeError("quantizer output must be uint8")
        if input_tensor.ndim != 2:
            raise HIPRuntimeError("quantizer input must be two-dimensional")
        rows, k = input_tensor.shape
        expected_shape = (k // 128, rows, Q8_1_F16_D2S6_BLOCK_BYTES)
        if k % 128 or tuple(output.shape) != expected_shape:
            raise HIPRuntimeError(
                "quantizer output does not match the Q8_1 F16_D2S6 contract"
            )
        if input_tensor.device != output.device:
            raise HIPRuntimeError("quantizer tensors must be on the same device")
        packed_arguments = Q8_1_QUANTIZER_ABI.pack(
            {
                "input": input_tensor.data_ptr(),
                "output": output.data_ptr(),
                "rows": rows,
                "rows_padded": rows,
                "k": k,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                rows,
                1,
                1,
                512,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class FixedHipForwardModule(ForwardModule):
    """Direct prequantized launcher for the deployed dense forward control."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: ForwardKernelSpec,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.hip_control = select_hip_control(
            "OrdinaryForward",
            quant_type,
            problem_size.m,
            problem_size.n,
            problem_size.k,
        )
        super().__init__(
            problem_size,
            quant_type,
            kernel_spec,
            code_object or _find_installed_kernel(self.hip_control.symbol),
            self.hip_control.symbol,
            hip_library,
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        size = self.problem_size
        return self.hip_control.launch_configuration(size.m, size.n, size.k)


class FixedGroupedQ8BackwardModule(_HIPModule):
    """Research-only launcher for the fixed-group six-argument backward ABI."""

    def __init__(
        self,
        problem: FixedBackwardProblem,
        kernel_spec: FixedBackwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_fixed_backward_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedFixedBackwardState.from_problem_spec(problem, kernel_spec)
        super().__init__(code_object, hip_library, kernel_name)

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = self.state
        tensors = (grad_output, packed_weight, grad_input)
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all fixed backward tensors must be on HIP")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all fixed backward tensors must be contiguous")
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError("fixed backward tensors need zero storage offsets")
        if grad_output.data_ptr() % 16 or packed_weight.data_ptr() % 16:
            raise HIPRuntimeError(
                "fixed backward input pointers must be 16-byte aligned"
            )
        if grad_output.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_output must be BF16")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if grad_input.dtype != torch.bfloat16:
            raise HIPRuntimeError("grad_input must be BF16")
        if tuple(grad_output.shape) != state.expected_grad_output_shape:
            raise HIPRuntimeError("grad_output shape does not match fixed key")
        if tuple(packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError("packed_weight shape does not match fixed key")
        if tuple(grad_input.shape) != state.expected_grad_input_shape:
            raise HIPRuntimeError("grad_input shape does not match fixed key")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("fixed backward tensors must share one device")

        problem = state.problem
        packed_arguments = FIXED_GROUPED_BACKWARD_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "tokens": problem.tokens,
                "out_features": problem.output_features,
                "bytes_per_group": problem.bytes_per_group,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *self._launch_configuration(),
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(self) -> tuple[int, int, int, int, int, int, int]:
        geometry = self.state.spec.compute.geometry
        return (*self.state.grid, *geometry.work_group, 0)


class InstalledFixedGroupedQ8BackwardModule(FixedGroupedQ8BackwardModule):
    """Direct launcher for the deployed fixed Q8_0 research control.

    The catalog selects the tuned M192/N64 body at every deployed token count.
    The M256/N64 body is built with the same arithmetic and bitwise-identical
    output but is slower. This control exists for comparison against GGTensile
    and is not part of the public API.
    """

    def __init__(
        self,
        problem: FixedBackwardProblem,
        kernel_spec: FixedBackwardKernelSpec,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.hip_control = select_hip_control(
            "FixedGroupedBackward",
            "Q8_0",
            problem.tokens,
            problem.output_features,
            problem.input_features,
        )
        self.m_tile = self.hip_control.fixed_m_tile()
        super().__init__(
            problem,
            kernel_spec,
            code_object or _find_installed_kernel(self.hip_control.symbol),
            self.hip_control.symbol,
            hip_library,
        )

    def _launch_configuration(self) -> tuple[int, int, int, int, int, int, int]:
        problem = self.state.problem
        return (
            problem.input_features // 64,
            (problem.tokens + self.m_tile - 1) // self.m_tile,
            problem.groups,
            128,
            1,
            1,
            0,
        )


class FixedGroupedQ8ForwardModule(_HIPModule):
    """Direct prequantized launcher for the fixed-group six-argument ABI."""

    def __init__(
        self,
        problem: FixedForwardProblem,
        kernel_spec: FixedForwardKernelSpec,
        code_object: Path,
        kernel_name: str,
        hip_library: Path | None = None,
    ) -> None:
        validate_fixed_forward_solution(problem, kernel_spec)
        self.problem = problem
        self.kernel_spec = kernel_spec
        self.state = DerivedFixedForwardState.from_problem_spec(problem, kernel_spec)
        super().__init__(code_object, hip_library, kernel_name)

    def launch(
        self,
        packed_weight: torch.Tensor,
        activations: torch.Tensor,
        output: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not self._module or not self._function:
            raise HIPRuntimeError("HIP module is closed")
        state = self.state
        tensors = (packed_weight, activations, output)
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all fixed launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all fixed launch tensors must be contiguous")
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError("fixed launch tensors must have zero storage offsets")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if activations.dtype != torch.uint8:
            raise HIPRuntimeError("activations must be uint8 Q8_1 F32_D4")
        if output.dtype != torch.bfloat16:
            raise HIPRuntimeError("output must be BF16")
        if tuple(packed_weight.shape) != state.expected_packed_weight_shape:
            raise HIPRuntimeError("packed_weight shape does not match fixed bank")
        if tuple(activations.shape) != state.expected_activation_shape:
            raise HIPRuntimeError(
                "activations shape does not match fixed Q8_1 F32_D4 workspace"
            )
        if tuple(output.shape) != state.expected_output_shape:
            raise HIPRuntimeError("output shape does not match fixed problem")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError("all fixed launch tensors must be on the same device")

        packed_arguments = FIXED_GROUPED_FORWARD_ABI.pack(
            {
                "packed_weight": packed_weight.data_ptr(),
                "activations": activations.data_ptr(),
                "output": output.data_ptr(),
                "tokens": state.problem.tokens,
                "out_features": state.problem.output_features,
                "bytes_per_group": state.problem.bytes_per_group,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *self._launch_configuration(),
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(self) -> tuple[int, int, int, int, int, int, int]:
        state = self.state
        return (*state.grid, *state.ordinary.kernel_spec.geometry.work_group, 0)


class InstalledFixedGroupedQ8ForwardModule(FixedGroupedQ8ForwardModule):
    """Direct launcher for the deployed fixed-group HIP multiply."""

    SYMBOL = "grouped_fwd_fixed_q8_0_g8_k4096_j64_full"

    def __init__(
        self,
        problem: FixedForwardProblem,
        kernel_spec: FixedForwardKernelSpec,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.hip_control = select_hip_control(
            "FixedGroupedForward",
            "Q8_0",
            problem.tokens,
            problem.output_features,
            problem.input_features,
        )
        super().__init__(
            problem,
            kernel_spec,
            code_object or _find_installed_kernel(self.hip_control.symbol),
            self.hip_control.symbol,
            hip_library,
        )

    def _launch_configuration(self) -> tuple[int, int, int, int, int, int, int]:
        state = self.state
        return (*state.grid, *state.ordinary.kernel_spec.geometry.work_group, 28_928)


class _Q81QuantizerModule(_HIPModule):
    """Installed HIP Q8_1 producer with the size dispatch between the two bodies.

    The grouped producer gives a thread half a 32-value group: the group needs one
    lane exchange instead of three LDS permutes, the loads and stores are 16-byte
    vectors and the index math happens once. Measured on this APU it is 1.4-1.6x
    faster than the row producer while the activation row set stays inside the
    cache and a few percent behind once the rows stream from memory, so the
    dispatch picks the row producer past the byte threshold.
    """

    GROUPED_SYMBOL = ""
    GROUPED_MAX_INPUT_BYTES = 20 * 1024 * 1024
    SYMBOL = ""
    LABEL = ""
    BLOCK_BYTES = 0

    def __init__(
        self,
        code_object: Path | None = None,
        hip_library: Path | None = None,
        symbol: str | None = None,
    ) -> None:
        self._grouped_module: _HIPModule | None = None
        self._hip_library = hip_library
        selected = code_object or _find_installed_kernel(symbol or self.SYMBOL)
        super().__init__(selected, hip_library, symbol or self.SYMBOL)

    def allocate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 2 or input_tensor.shape[1] % 128:
            raise HIPRuntimeError(
                "quantizer input must be [rows, K] with K divisible by 128"
            )
        rows, k = input_tensor.shape
        return torch.empty(
            (k // 128, rows, self.BLOCK_BYTES),
            dtype=torch.uint8,
            device=input_tensor.device,
        )

    def _uses_grouped_producer(self, rows: int, k: int) -> bool:
        return rows * k * 2 <= self.GROUPED_MAX_INPUT_BYTES

    def _grouped(self) -> _HIPModule:
        if self._grouped_module is None:
            self._grouped_module = _HIPModule(
                _find_installed_kernel(self.GROUPED_SYMBOL),
                self._hip_library,
                self.GROUPED_SYMBOL,
            )
        return self._grouped_module

    def close(self) -> None:
        if self._grouped_module is not None:
            self._grouped_module.close()
            self._grouped_module = None
        super().close()

    def launch(
        self,
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        if not input_tensor.is_cuda or not output.is_cuda:
            raise HIPRuntimeError("quantizer tensors must be on a HIP device")
        if not input_tensor.is_contiguous() or not output.is_contiguous():
            raise HIPRuntimeError("quantizer tensors must be contiguous")
        if input_tensor.dtype != torch.bfloat16:
            raise HIPRuntimeError("quantizer input must be BF16")
        if output.dtype != torch.uint8:
            raise HIPRuntimeError("quantizer output must be uint8")
        if input_tensor.ndim != 2:
            raise HIPRuntimeError("quantizer input must be two-dimensional")
        rows, k = input_tensor.shape
        if k % 128 or tuple(output.shape) != (k // 128, rows, self.BLOCK_BYTES):
            raise HIPRuntimeError(
                f"quantizer output does not match the Q8_1 {self.LABEL} contract"
            )
        if input_tensor.device != output.device:
            raise HIPRuntimeError("quantizer tensors must be on the same device")
        producer = self
        if self._uses_grouped_producer(rows, k):
            producer = self._grouped()
        if not producer._module or not producer._function:
            raise HIPRuntimeError("HIP module is closed")
        packed_arguments = Q8_1_QUANTIZER_ABI.pack(
            {
                "input": input_tensor.data_ptr(),
                "output": output.data_ptr(),
                "rows": rows,
                "rows_padded": rows,
                "k": k,
            }
        )
        producer._check(
            producer._lib.hipModuleLaunchKernel(
                producer._function,
                rows,
                1,
                1,
                512,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                packed_arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class FixedQ81F16D4S4QuantizerModule(_Q81QuantizerModule):
    """Direct launcher for the installed HIP Q8_1 F16_D4S4 producer."""

    SYMBOL = "quantize_bf16_q8_1_f16_d4s4"
    GROUPED_SYMBOL = "quantize_bf16_q8_1_f16_d4s4_grouped"
    LABEL = "F16_D4S4"
    BLOCK_BYTES = Q8_1_F16_D4S4_BLOCK_BYTES


class FixedQ81F32D4QuantizerModule(_Q81QuantizerModule):
    """Direct launcher for the installed HIP Q8_1 F32_D4 producer."""

    SYMBOL = "quantize_bf16_q8_1_f32_d4"
    GROUPED_SYMBOL = "quantize_bf16_q8_1_f32_d4_grouped"
    LABEL = "F32_D4"
    BLOCK_BYTES = Q8_1_F32_D4_BLOCK_BYTES


def _find_hip_library() -> Path:
    override = os.environ.get("GGTENSILE_HIP_LIBRARY")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise HIPRuntimeError(f"GGTENSILE_HIP_LIBRARY does not exist: {candidate}")
    found = ctypes.util.find_library("amdhip64")
    if found:
        return Path(found)
    purelib = Path(sysconfig.get_paths()["purelib"])
    for package in ("_rocm_sdk_core", "_rocm_sdk_devel"):
        candidate = purelib / package / "lib" / "libamdhip64.so"
        if candidate.is_file():
            return candidate
    raise HIPRuntimeError("cannot find libamdhip64.so. Set GGTENSILE_HIP_LIBRARY")


def _find_installed_kernel(symbol: str) -> Path:
    override_name = (
        "GGTENSILE_Q8_1_F16_D4S4_CODE_OBJECT"
        if symbol == FixedQ81F16D4S4QuantizerModule.SYMBOL
        else "GGTENSILE_Q8_1_F32_D4_CODE_OBJECT"
        if symbol == FixedQ81F32D4QuantizerModule.SYMBOL
        else "GGTENSILE_FIXED_GROUPED_Q8_CODE_OBJECT"
        if symbol == InstalledFixedGroupedQ8ForwardModule.SYMBOL
        else "GGTENSILE_HIP_FORWARD_CODE_OBJECT"
    )
    override = os.environ.get(override_name)
    if override:
        return _resolve_code_object(Path(override), symbol)
    located = locate_control(symbol)
    if located is None:
        raise HIPRuntimeError(
            f"cannot find installed kernel {symbol}. Set {override_name}"
        )
    return located
