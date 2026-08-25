"""Optional launchers for historical standalone dense-backward controls."""

import math
from dataclasses import dataclass
from pathlib import Path

from .mmq_bwd_spec import BackwardKernelSpec
from .model import ProblemSize
from .runtime import (
    BackwardModule,
    HIPRuntimeError,
    _find_installed_kernel,
    _resolve_code_object,
)


@dataclass(frozen=True)
class _DenseControl:
    symbol: str
    n_tiles: int
    k_iteration: int
    group_m: int = 0
    m_tiles_per_wave: int = 1
    active_waves: int = 4
    full_tiles: bool = False


_Q8_EXACT = {
    (1024, 4096): "n1024k4096",
    (32768, 1024): "n32768k1024",
    (512, 4096): "n512k4096",
    (4096, 8192): "n4096k8192",
    (2048, 4096): "n2048k4096",
    (4096, 2048): "n4096k2048",
}


def _select_control(problem_size: ProblemSize, quant_type: str) -> _DenseControl:
    size = problem_size
    if quant_type in {"Q3_K", "Q4_K", "Q5_K"}:
        return _DenseControl(
            f"dense_bwd_{quant_type.lower()}_nt64_ki16_g0",
            n_tiles=4,
            k_iteration=16,
        )
    if quant_type == "Q6_K":
        if size.m == 64:
            return _DenseControl(
                "dense_bwd_q6_k_m64_nt32_ki64_bounded",
                n_tiles=2,
                k_iteration=64,
            )
        if size.m in {128, 256}:
            return _DenseControl(
                f"dense_bwd_q6_k_m{size.m}_nt64_ki32_bounded",
                n_tiles=4,
                k_iteration=32,
                m_tiles_per_wave=2,
            )
        raise HIPRuntimeError(f"no historical Q6_K control for M={size.m}")
    if quant_type == "Q8_0":
        label = _Q8_EXACT.get((size.k, size.n))
        if label is not None:
            return _DenseControl(
                f"dense_bwd_q8_0_exact_{label}",
                n_tiles=4,
                k_iteration=16,
                full_tiles=True,
            )
        if (size.k, size.n) == (129280, 4096) and size.m > 0:
            return _DenseControl(
                "dense_bwd_q8_0_exact_lm_head_"
                f"{'full' if size.m % 64 == 0 else 'bounded'}",
                n_tiles=4,
                k_iteration=16,
                full_tiles=size.m % 64 == 0,
            )
        raise HIPRuntimeError(
            f"no historical Q8_0 control for M={size.m}, N={size.n}, K={size.k}"
        )
    raise HIPRuntimeError(f"no historical dense-backward control for {quant_type}")


class InstalledDenseBackwardModule(BackwardModule):
    """Launch a historical dense control while retaining the public ABI checks."""

    def __init__(
        self,
        problem_size: ProblemSize,
        quant_type: str,
        kernel_spec: BackwardKernelSpec,
        code_object: Path | None = None,
        hip_library: Path | None = None,
    ) -> None:
        self.control = _select_control(problem_size, quant_type)
        selected = (
            _resolve_code_object(code_object, self.control.symbol)
            if code_object is not None
            else _find_installed_kernel(self.control.symbol)
        )
        super().__init__(
            problem_size,
            quant_type,
            kernel_spec,
            selected,
            self.control.symbol,
            hip_library,
        )

    def _launch_configuration(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        size = self.problem_size
        m_per_block = 16 * self.control.m_tiles_per_wave * self.control.active_waves
        n_per_block = 16 * self.control.n_tiles
        m_blocks = math.ceil(size.m / m_per_block)
        n_blocks = math.ceil(size.n / n_per_block)
        if self.control.group_m:
            grid = (
                self.control.group_m,
                n_blocks,
                math.ceil(m_blocks / self.control.group_m),
            )
        else:
            grid = (m_blocks, n_blocks, 1)
        return grid, (128, 1, 1), 0


__all__ = ["InstalledDenseBackwardModule"]
