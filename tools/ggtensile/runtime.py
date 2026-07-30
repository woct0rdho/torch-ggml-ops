import ctypes
import ctypes.util
import os
import sysconfig
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from .model import SolutionKey
from .validation import validate_solution

if TYPE_CHECKING:
    import torch


class HIPRuntimeError(RuntimeError):
    pass


class DenseBackwardModule:
    """Direct HIP module launcher for the fixed dense-backward kernarg ABI."""

    def __init__(
        self,
        solution_key: SolutionKey,
        code_object: Path,
        hip_library: Path | None = None,
    ) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(reason.rule_id for reason in reasons)
            raise HIPRuntimeError(f"cannot launch rejected solution: {details}")
        if not code_object.is_file():
            raise HIPRuntimeError(f"code object does not exist: {code_object}")
        self.solution_key = solution_key
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
                    solution_key.kernel_name.encode(),
                ),
                "hipModuleGetFunction",
            )
        except Exception:
            self.close()
            raise

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
        block_bytes = (
            144 if self.solution_key.problem_type.quant_data_type == "Q4_K" else 176
        )
        expected_weight_bytes = size.k * (size.n // 256) * block_bytes
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
