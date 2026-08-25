"""Direct launchers for installed standalone grouped-backward controls."""

import ctypes
import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import torch

from .kernel_abi import GROUPED_BACKWARD_ABI
from .runtime import HIPRuntimeError, _HIPModule, _resolve_code_object


@dataclass(frozen=True)
class _ControlSpec:
    symbol: str
    out_features: int
    in_features: int
    packed_row_bytes: int

    @property
    def bytes_per_expert(self) -> int:
        return self.out_features * self.packed_row_bytes


_PREFIX = "grouped_bwd_single_"
_SPECS = {
    "Q2_K": {
        "m64": (_PREFIX + "q2_k_n4096_k2048_mt64_nt64", 4096, 2048, 672),
        "m128": (_PREFIX + "q2_k_n4096_k2048_mt128_nt64", 4096, 2048, 672),
        "m128_u2": (
            _PREFIX + "q2_k_n4096_k2048_mt128_nt64_u2",
            4096,
            2048,
            672,
        ),
    },
    "Q4_K": {
        "m64": (_PREFIX + "q4_k_n2048_k512_mt64_nt64", 2048, 512, 288),
        "m128": (_PREFIX + "q4_k_n2048_k512_mt128_nt64", 2048, 512, 288),
    },
    "Q5_K": {
        "m64": (_PREFIX + "q5_k_n2048_k512_mt64_nt64", 2048, 512, 352),
    },
    "IQ2_S": {
        "m64": (_PREFIX + "iq2_s_n2048_k512_mt64_nt64", 2048, 512, 164),
        "m128": (_PREFIX + "iq2_s_n2048_k512_mt128_nt64", 2048, 512, 164),
    },
}


def _control_path(symbol: str, supplied: Path | None) -> Path:
    if supplied is not None:
        return _resolve_code_object(supplied, symbol)
    roots = []
    configured = os.environ.get("GGTENSILE_HIP_CONTROL_ROOT")
    if configured:
        configured_path = Path(configured)
        roots.extend(
            (
                configured_path,
                configured_path / "gfx1151",
                configured_path / "hip_controls",
                configured_path / "gfx1151" / "hip_controls",
            )
        )
    repo_root = Path(__file__).resolve().parents[2]
    purelib = Path(sysconfig.get_paths()["purelib"])
    roots.extend(
        (
            repo_root / "build/mmq_hip_controls/gfx1151",
            repo_root / "torch_ggml_ops/kernels/gfx1151",
            repo_root / "torch_ggml_ops/kernels/gfx1151/hip_controls",
            purelib / "torch_ggml_ops/kernels/gfx1151",
            purelib / "torch_ggml_ops/kernels/gfx1151/hip_controls",
        )
    )
    for root in roots:
        path = root / f"{symbol}.hsaco"
        if path.is_file():
            return path
    raise HIPRuntimeError(
        f"cannot find HIP control {symbol}; pass --hip-code-object or set "
        "GGTENSILE_HIP_CONTROL_ROOT"
    )


class InstalledGroupedBackwardControl(_HIPModule):
    """Launch the old specialized grouped-backward HIP body directly."""

    PHYSICAL_EXPERTS = 256

    def __init__(
        self,
        quant_type: str,
        rows: int,
        route_entries: int,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.spec = self.select_spec(quant_type, rows, route_entries)
        super().__init__(
            _control_path(self.spec.symbol, code_object),
            hip_library,
            self.spec.symbol,
        )

    @classmethod
    def select_spec(
        cls, quant_type: str, rows: int, route_entries: int
    ) -> _ControlSpec:
        if rows <= 0 or route_entries <= 0:
            raise HIPRuntimeError(
                "grouped-backward control dimensions must be positive"
            )
        choices = _SPECS.get(quant_type)
        if choices is None:
            raise HIPRuntimeError(
                f"no standalone grouped-backward control for {quant_type}"
            )
        if quant_type == "Q2_K":
            name = (
                "m64"
                if rows < route_entries * 128
                else "m128_u2"
                if rows < route_entries * 512
                else "m128"
            )
        elif quant_type == "Q5_K":
            name = "m64"
        else:
            name = "m128" if rows >= route_entries * 80 else "m64"
        symbol, out_features, in_features, packed_row_bytes = choices[name]
        return _ControlSpec(symbol, out_features, in_features, packed_row_bytes)

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
        tensors = (
            grad_output,
            packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise HIPRuntimeError(
                "grouped-backward control requires contiguous HIP tensors"
            )
        if any(tensor.storage_offset() != 0 for tensor in tensors):
            raise HIPRuntimeError(
                "grouped-backward control requires zero storage offsets"
            )
        if (
            grad_output.dtype != torch.bfloat16
            or packed_weight.dtype != torch.uint8
            or grad_input.dtype != torch.bfloat16
            or expert_indices.dtype != torch.int64
            or expert_offsets.dtype != torch.int32
        ):
            raise HIPRuntimeError("grouped-backward control tensor dtypes are invalid")
        spec = self.spec
        rows = grad_output.shape[0]
        if tuple(grad_output.shape) != (rows, spec.out_features):
            raise HIPRuntimeError("grouped-backward control gradient shape is invalid")
        if tuple(packed_weight.shape) != (
            self.PHYSICAL_EXPERTS,
            spec.out_features,
            spec.packed_row_bytes,
        ):
            raise HIPRuntimeError("grouped-backward control weight shape is invalid")
        if tuple(grad_input.shape) != (rows, spec.in_features):
            raise HIPRuntimeError("grouped-backward control output shape is invalid")
        route_entries = expert_indices.numel()
        if expert_indices.ndim != 1 or expert_offsets.ndim != 1:
            raise HIPRuntimeError(
                "grouped-backward control routes must be one-dimensional"
            )
        if route_entries <= 0 or route_entries > self.PHYSICAL_EXPERTS:
            raise HIPRuntimeError("grouped-backward control route count is invalid")
        if expert_offsets.numel() != route_entries:
            raise HIPRuntimeError("grouped-backward control route lengths differ")
        if len({tensor.device for tensor in tensors}) != 1:
            raise HIPRuntimeError(
                "grouped-backward control tensors must share a device"
            )
        arguments = GROUPED_BACKWARD_ABI.pack(
            {
                "grad_output": grad_output.data_ptr(),
                "packed_weight": packed_weight.data_ptr(),
                "grad_input": grad_input.data_ptr(),
                "expert_indices": expert_indices.data_ptr(),
                "expert_offsets": expert_offsets.data_ptr(),
                "num_experts": self.PHYSICAL_EXPERTS,
                "rows": rows,
                "bytes_per_expert": spec.bytes_per_expert,
            }
        )
        self._check(
            self._lib.hipModuleLaunchKernel(
                self._function,
                spec.in_features // 64,
                route_entries,
                1,
                128,
                1,
                1,
                0,
                ctypes.c_void_p(stream),
                arguments.parameters,
                None,
            ),
            "hipModuleLaunchKernel",
        )
