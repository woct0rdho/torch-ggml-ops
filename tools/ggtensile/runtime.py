import ctypes
import ctypes.util
import os
import sysconfig
from pathlib import Path
from types import TracebackType

import torch
from typing_extensions import Self

from .grouped_mmq_fwd_model import GroupedForwardSolutionKey
from .grouped_mmq_fwd_spec import DerivedGroupedForwardState
from .grouped_mmq_fwd_validation import validate_grouped_forward_solution
from .mmq_fwd_spec import DerivedForwardState
from .model import BackwardSolution, ForwardSolution, SolutionKey
from .quant_formats import (
    Q8_1_F16_D2S6_BLOCK_BYTES,
    Q8_1_F16_D4S4_BLOCK_BYTES,
    Q8_1_F32_D4_BLOCK_BYTES,
    QUANT_FORMATS,
)
from .validation import validate_solution


class HIPRuntimeError(RuntimeError):
    pass


class _HIPModule:
    """Shared HIP module lifetime and runtime API configuration."""

    def __init__(
        self,
        code_object: Path,
        hip_library: Path | None,
        kernel_name: str,
    ) -> None:
        if not code_object.is_file():
            raise HIPRuntimeError(f"code object does not exist: {code_object}")
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
        try:
            self._check(
                self._lib.hipModuleGetFunction(
                    ctypes.byref(self._function),
                    self._module,
                    kernel_name.encode(),
                ),
                "hipModuleGetFunction",
            )
        except Exception:
            self.close()
            raise

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


class _SolutionHIPModule(_HIPModule):
    """HIP module whose artifact and symbol are described by a solution key."""

    def __init__(
        self,
        solution_key: SolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected solution: {details}")
        self.solution_key = solution_key
        super().__init__(
            code_object,
            hip_library,
            kernel_name or solution_key.kernel_name,
        )


class BackwardModule(_SolutionHIPModule):
    """Direct HIP module launcher for the fixed MMQ backward kernarg ABI."""

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
        size = self.solution_key.problem_size
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
        quant_format = QUANT_FORMATS[self.solution_key.problem_type.quant_data_type]
        expected_weight_bytes = (
            size.k * (size.n // quant_format.block_values) * quant_format.block_bytes
        )
        if packed_weight.numel() != expected_weight_bytes:
            raise HIPRuntimeError(
                "packed_weight size does not match "
                f"{self.solution_key.problem_type.quant_data_type} ProblemSize"
            )
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise HIPRuntimeError("all launch tensors must be on the same device")

        arguments = (
            ctypes.c_uint64(grad_output.data_ptr()),
            ctypes.c_uint64(packed_weight.data_ptr()),
            ctypes.c_uint64(grad_input.data_ptr()),
            ctypes.c_uint32(size.m),
            ctypes.c_uint32(size.k),
            ctypes.c_uint32(size.n),
            ctypes.c_uint32(size.n // 256),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        solution = self.solution_key.solution
        if not isinstance(solution, BackwardSolution):
            raise HIPRuntimeError("MMQ backward requires BackwardSolution")
        group_m = solution.work_group_mapping
        m_blocks = size.m // solution.macro_tile0
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                group_m,
                size.n // solution.macro_tile1,
                m_blocks // group_m,
                *solution.work_group,
                0,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class ForwardModule(_SolutionHIPModule):
    """Direct launcher for an exact packed K-quant/Q8_1 forward kernel."""

    def __init__(
        self,
        solution_key: SolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        if not isinstance(solution_key.solution, ForwardSolution):
            raise HIPRuntimeError("MMQ forward requires ForwardSolution")
        super().__init__(
            solution_key,
            code_object,
            hip_library,
            kernel_name=kernel_name,
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
        state = DerivedForwardState.from_solution_key(self.solution_key)
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

        arguments = (
            ctypes.c_uint64(packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_uint32(size.n),
            ctypes.c_uint32(size.m),
            ctypes.c_uint32(size.m),
            ctypes.c_uint32(state.blocks_per_weight_row),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        grid, block, shared_memory = self._launch_configuration()
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        state = DerivedForwardState.from_solution_key(self.solution_key)
        return (state.grid, state.kernel_spec.geometry.work_group, 0)


class GroupedForwardModule(_HIPModule):
    """Research-only launcher for a routed grouped forward artifact."""

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        reasons = validate_grouped_forward_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected grouped solution: {details}")
        self.solution_key = solution_key
        super().__init__(
            code_object, hip_library, kernel_name or solution_key.kernel_name
        )

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
        state = DerivedGroupedForwardState.from_solution_key(self.solution_key)
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
        if (
            route_entries <= 0
            or route_entries > self.solution_key.problem.max_route_entries
        ):
            raise HIPRuntimeError("route entry count is outside the grouped contract")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("route metadata lengths must match")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError(
                "all grouped launch tensors must be on the same device"
            )

        arguments = (
            ctypes.c_uint64(packed_weight.data_ptr()),
            ctypes.c_uint64(activations.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_uint64(expert_indices.data_ptr()),
            ctypes.c_uint64(expert_offsets.data_ptr()),
            ctypes.c_uint32(self.solution_key.problem.physical_experts),
            ctypes.c_uint32(self.solution_key.problem.output_features),
            ctypes.c_uint32(self.solution_key.problem.aggregate_rows),
            ctypes.c_uint32(state.blocks_per_weight_row),
            ctypes.c_uint64(state.bytes_per_expert),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
        )
        grid, block, shared_memory = self._launch_configuration(route_entries)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                *grid,
                *block,
                shared_memory,
                ctypes.c_void_p(stream),
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        return (
            self.state_grid(route_entries),
            self.solution_key.solution.work_group,
            0,
        )

    def state_grid(self, route_entries: int) -> tuple[int, int, int]:
        return DerivedGroupedForwardState.from_solution_key(self.solution_key).grid(
            route_entries
        )


class InstalledGroupedForwardModule(GroupedForwardModule):
    """Direct launcher for the installed HIP Q4_K grouped serial control."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q4_k_n2048_k512_j64"

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            kernel_name=self.SYMBOL,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        return ((32, route_entries, 1), (32, 4, 1), 28_928)


class InstalledGroupedForwardQ5Module(GroupedForwardModule):
    """Direct launcher for the installed HIP Q5_K J64 grouped control."""

    J64_SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q5_k_n2048_k512_j64"
    J32_SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q5_k_n2048_k512_j32"

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if solution_key.problem.quant_data_type != "Q5_K":
            raise HIPRuntimeError("installed Q5_K control requires a Q5_K problem")
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(self.J64_SYMBOL),
            hip_library,
            kernel_name=self.J64_SYMBOL,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        if self.solution_key.problem.aggregate_rows < 128 * route_entries:
            raise HIPRuntimeError(
                "the installed Q5_K J32 control requires its dedicated module"
            )
        return ((32, route_entries, 1), (32, 4, 1), 28_928)


class InstalledGroupedForwardQ5J32Module(GroupedForwardModule):
    """Direct launcher for the installed HIP Q5_K small-route J32 control."""

    SYMBOL = InstalledGroupedForwardQ5Module.J32_SYMBOL

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if solution_key.problem.quant_data_type != "Q5_K":
            raise HIPRuntimeError("installed Q5_K J32 control requires a Q5_K problem")
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            kernel_name=self.SYMBOL,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        if self.solution_key.problem.aggregate_rows >= 128 * route_entries:
            raise HIPRuntimeError("installed Q5_K dispatch selects J64 for this route")
        return ((32, route_entries, 1), (32, 4, 1), 24_192)


class InstalledGroupedForwardQ2J32Module(GroupedForwardModule):
    """Direct launcher for the installed pure Q2_K J32 control."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q2_k_n4096_k2048_j32"

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if solution_key.problem.quant_data_type != "Q2_K":
            raise HIPRuntimeError("installed Q2_K J32 control requires a Q2_K problem")
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            kernel_name=self.SYMBOL,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        rows = self.solution_key.problem.aggregate_rows
        if rows == 49_152 or rows < 64 * route_entries:
            raise HIPRuntimeError("installed Q2_K dispatch selects mixed J32/J16")
        return ((64, route_entries, 1), (32, 4, 1), 30_336)


class InstalledGroupedForwardQ2J32J16Module(GroupedForwardModule):
    """Direct launcher for the installed mixed Q2_K J32/J16 control."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_grouped_fwd_serial_q2_k_n4096_k2048_j32_j16"

    def __init__(
        self,
        solution_key: GroupedForwardSolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        if solution_key.problem.quant_data_type != "Q2_K":
            raise HIPRuntimeError(
                "installed Q2_K mixed control requires a Q2_K problem"
            )
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(self.SYMBOL),
            hip_library,
            kernel_name=self.SYMBOL,
        )

    def _launch_configuration(
        self, route_entries: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        rows = self.solution_key.problem.aggregate_rows
        if rows != 49_152 and rows >= 64 * route_entries:
            raise HIPRuntimeError("installed Q2_K dispatch selects pure J32")
        return ((64, route_entries, 1), (32, 4, 1), 30_336)


class FixedQ81F16D2S6QuantizerModule(_HIPModule):
    """Direct launcher for the installed HIP Q8_1 F16_D2S6 producer."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f16_d2s6"

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
        arguments = (
            ctypes.c_uint64(input_tensor.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_int64(rows),
            ctypes.c_int64(rows),
            ctypes.c_int64(k),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
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
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class FixedHipForwardModule(ForwardModule):
    """Direct prequantized launcher for an installed exact HIP multiply."""

    def __init__(
        self,
        solution_key: SolutionKey,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        quant_type = solution_key.problem_type.quant_data_type
        k = solution_key.problem_size.k
        allowed_k = {
            "Q3_K": (2048, 4096),
            "Q4_K": (512, 2048, 4096),
            "Q5_K": (512, 2048),
            "Q6_K": (2048,),
            "Q8_0": (1024, 2048, 4096, 8192),
        }.get(quant_type)
        if allowed_k is None or k not in allowed_k:
            raise HIPRuntimeError(
                f"installed {quant_type} control does not support K={k}"
            )
        symbol_quant = quant_type.lower()
        # The installed bundle ABI explicitly contrasts ordinary (`dense_fwd`) and
        # grouped entry points, so preserve its external symbol spelling here.
        m = solution_key.problem_size.m
        j = 64 if quant_type == "Q6_K" and m == 64 else 128
        suffix = f"k{k}_j{j}_full"
        if quant_type == "Q3_K" and k == 4096:
            suffix = "j128"
        elif quant_type == "Q8_0" and m in (32, 64):
            if k != 4096:
                raise HIPRuntimeError(
                    f"installed Q8_0 J64 control does not support K={k}"
                )
            j = 64
            suffix = f"k4096_j64_{'bounded' if m == 32 else 'full'}"
        symbol = f"torch_ggml_ops_mmq_gfx1151_v1_dense_fwd_{symbol_quant}_{suffix}"
        super().__init__(
            solution_key,
            code_object or _find_installed_kernel(symbol),
            hip_library,
            kernel_name=symbol,
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        size = self.solution_key.problem_size
        quant_type = self.solution_key.problem_type.quant_data_type
        j = 64 if quant_type in ("Q6_K", "Q8_0") and size.m in (32, 64) else 128
        if quant_type == "Q3_K":
            # Match mmq_bundle.cpp: Q3 uses a 36-dword activation tile and
            # the 84-dword packed Q3 row stride.
            lds_bytes = 40_448
        else:
            lds_bytes = 28_928 if j == 64 else 38_400
        return (
            (size.n // 64, (size.m + j - 1) // j, 1),
            (32, 4, 1),
            lds_bytes,
        )


class FixedQ81F16D4S4QuantizerModule(_HIPModule):
    """Direct launcher for the installed HIP Q8_1 F16_D4S4 producer."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f16_d4s4"

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
            (k // 128, rows, Q8_1_F16_D4S4_BLOCK_BYTES),
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
        expected_shape = (k // 128, rows, Q8_1_F16_D4S4_BLOCK_BYTES)
        if k % 128 or tuple(output.shape) != expected_shape:
            raise HIPRuntimeError(
                "quantizer output does not match the Q8_1 F16_D4S4 contract"
            )
        if input_tensor.device != output.device:
            raise HIPRuntimeError("quantizer tensors must be on the same device")

        arguments = (
            ctypes.c_uint64(input_tensor.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_int64(rows),
            ctypes.c_int64(rows),
            ctypes.c_int64(k),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
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
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


class FixedQ81F32D4QuantizerModule(_HIPModule):
    """Direct launcher for the installed HIP Q8_1 F32_D4 producer."""

    SYMBOL = "torch_ggml_ops_mmq_gfx1151_v1_quantize_bf16_q8_1_f32_d4"

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
            (k // 128, rows, Q8_1_F32_D4_BLOCK_BYTES),
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
        if input_tensor.dtype != torch.bfloat16 or output.dtype != torch.uint8:
            raise HIPRuntimeError("quantizer requires BF16 input and uint8 output")
        if input_tensor.ndim != 2:
            raise HIPRuntimeError("quantizer input must be two-dimensional")
        rows, k = input_tensor.shape
        expected_shape = (k // 128, rows, Q8_1_F32_D4_BLOCK_BYTES)
        if k % 128 or tuple(output.shape) != expected_shape:
            raise HIPRuntimeError(
                "quantizer output does not match the Q8_1 F32_D4 contract"
            )
        if input_tensor.device != output.device:
            raise HIPRuntimeError("quantizer tensors must be on the same device")
        arguments = (
            ctypes.c_uint64(input_tensor.data_ptr()),
            ctypes.c_uint64(output.data_ptr()),
            ctypes.c_int64(rows),
            ctypes.c_int64(rows),
            ctypes.c_int64(k),
        )
        parameters = (ctypes.c_void_p * len(arguments))(
            *(
                ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
                for argument in arguments
            )
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
                parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )


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
    raise HIPRuntimeError("cannot find libamdhip64.so; set GGTENSILE_HIP_LIBRARY")


def _find_installed_kernel(symbol: str) -> Path:
    override_name = (
        "GGTENSILE_Q8_1_F16_D4S4_CODE_OBJECT"
        if symbol == FixedQ81F16D4S4QuantizerModule.SYMBOL
        else "GGTENSILE_Q8_1_F32_D4_CODE_OBJECT"
        if symbol == FixedQ81F32D4QuantizerModule.SYMBOL
        else "GGTENSILE_HIP_FORWARD_CODE_OBJECT"
    )
    override = os.environ.get(override_name)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise HIPRuntimeError(f"{override_name} does not exist: {candidate}")
    package_roots = (
        Path(__file__).resolve().parents[2] / "torch_ggml_ops",
        Path(sysconfig.get_paths()["purelib"]) / "torch_ggml_ops",
    )
    for package_root in package_roots:
        candidate = package_root / "kernels" / "gfx1151" / f"{symbol}.hsaco"
        if candidate.is_file():
            return candidate
    raise HIPRuntimeError(f"cannot find installed kernel {symbol}; set {override_name}")
