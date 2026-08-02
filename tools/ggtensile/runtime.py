import ctypes
import ctypes.util
import os
import sysconfig
from pathlib import Path
from types import TracebackType

import torch
from typing_extensions import Self

from .model import DenseForwardSolution, Solution, SolutionKey
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


class DenseBackwardModule(_SolutionHIPModule):
    """Direct HIP module launcher for the fixed dense-backward kernarg ABI."""

    def launch(
        self,
        grad_output: torch.Tensor,
        packed_weight: torch.Tensor,
        grad_input: torch.Tensor,
        *,
        stream: int,
    ) -> None:
        import torch

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
        block_bytes = {
            "Q3_K": 110,
            "Q4_K": 144,
            "Q5_K": 176,
            "Q6_K": 210,
            "Q8_0": 34,
        }[self.solution_key.problem_type.quant_data_type]
        values_per_block = (
            32 if self.solution_key.problem_type.quant_data_type == "Q8_0" else 256
        )
        expected_weight_bytes = size.k * (size.n // values_per_block) * block_bytes
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
        if not isinstance(solution, Solution):
            raise HIPRuntimeError("dense backward requires Solution")
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


class DenseForwardModule(_SolutionHIPModule):
    """Direct launcher for an exact packed K-quant/Q8_1 forward kernel."""

    def __init__(
        self,
        solution_key: SolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
        *,
        kernel_name: str | None = None,
    ) -> None:
        if not isinstance(solution_key.solution, DenseForwardSolution):
            raise HIPRuntimeError("dense forward requires DenseForwardSolution")
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
        size = self.solution_key.problem_size
        tensors = (packed_weight, activations, output)
        if any(not tensor.is_cuda for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be on a HIP device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError("all launch tensors must be contiguous")
        if packed_weight.dtype != torch.uint8:
            raise HIPRuntimeError("packed_weight must be uint8")
        if activations.dtype != torch.uint8:
            raise HIPRuntimeError(
                "activations must be the uint8 Q8_1 F16_D4S4 workspace"
            )
        if output.dtype != torch.bfloat16:
            raise HIPRuntimeError("output must be BF16")
        block_bytes = {
            "Q4_K": 144,
            "Q5_K": 176,
        }.get(self.solution_key.problem_type.quant_data_type)
        if block_bytes is None:
            raise HIPRuntimeError("unsupported dense-forward quant type")
        expected_weight_bytes = size.n * (size.k // 256) * block_bytes
        if packed_weight.numel() != expected_weight_bytes:
            raise HIPRuntimeError("packed_weight size does not match ProblemSize")
        expected_activation_shape = (size.k // 128, size.m, 144)
        if tuple(activations.shape) != expected_activation_shape:
            raise HIPRuntimeError(
                "activations shape does not match the exact Q8_1 F16_D4S4 "
                "workspace contract"
            )
        if tuple(output.shape) != (size.m, size.n):
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
            ctypes.c_uint32(size.k // 256),
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
        size = self.solution_key.problem_size
        solution = self.solution_key.solution
        return (
            (size.n // solution.macro_tile1, size.m // solution.macro_tile0, 1),
            solution.work_group,
            0,
        )


class FixedHipDenseForwardModule(DenseForwardModule):
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
            "Q4_K": (512, 2048, 4096),
            "Q5_K": (512, 2048),
        }.get(quant_type)
        if allowed_k is None or k not in allowed_k:
            raise HIPRuntimeError(
                f"installed {quant_type} control does not support K={k}"
            )
        symbol_quant = quant_type.lower()
        symbol = (
            f"torch_ggml_ops_mmq_gfx1151_v1_dense_fwd_{symbol_quant}_k{k}_j128_full"
        )
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
        return ((size.n // 64, size.m // 128, 1), (32, 4, 1), 38_400)


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
        import torch

        if input_tensor.ndim != 2 or input_tensor.shape[1] % 128:
            raise HIPRuntimeError(
                "quantizer input must be [rows, K] with K divisible by 128"
            )
        rows, k = input_tensor.shape
        return torch.empty(
            (k // 128, rows, 144),
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
        import torch

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
        expected_shape = (k // 128, rows, 144)
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
