import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from mmq_bundle_wrapper_source import (
    DenseBackwardConfig,
    ForwardConfig,
    ForwardKind,
    GroupedBackwardConfig,
    GroupedBackwardKind,
    KernelConfig,
    QuantType,
    render_wrapper,
)

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
PACKAGE_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"
GENERATED_HEADER = CSRC / "generated" / "mmq_bundle_table.cuh"
GENERATED_SOURCE_DIR = ROOT / "build" / "mmq_bundle_sources" / "gfx1151"
BUILD_INPUT_STAMP = ".mmq-build-input"
ARCH = "gfx1151"
ABI_PREFIX = "torch_ggml_ops_mmq_gfx1151_v1_"

QUANT_TYPES = tuple((quant_type.name, quant_type) for quant_type in QuantType)
BACKWARD_QUANT_TYPES = tuple(
    item
    for item in QUANT_TYPES
    if item[0] in {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_S"}
)
ROW_TASK_TYPES = tuple(
    item for item in QUANT_TYPES if item[0] in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_S"}
)


@dataclass(frozen=True)
class KernelSpec:
    cpp_id: str
    suffix: str
    config: KernelConfig
    enforce_resource_gate: bool = False

    @property
    def symbol(self) -> str:
        return ABI_PREFIX + self.suffix

    @property
    def filename(self) -> str:
        return f"{self.symbol}.hsaco"

    @property
    def cuid(self) -> str:
        return hashlib.sha256(self.symbol.encode()).hexdigest()[:16]


def _forward_spec(
    cpp_id: str,
    suffix: str,
    kind: ForwardKind,
    *,
    quant_type: QuantType | None = None,
    j: int = 0,
    nrows_weight: int = 0,
    blocks_per_weight_row: int = 0,
    groups: int = 0,
    fallback: bool = False,
    rolled_q2: bool = False,
    mixed_j32_tails: bool = False,
    mixed_q2_k: bool = False,
    mixed_j32_rows: tuple[int, int] = (0, 0),
    full_i: bool = False,
    full_j: bool = False,
    enforce_resource_gate: bool = False,
) -> KernelSpec:
    config = ForwardConfig(
        kind=kind,
        quant_type=quant_type,
        j=j,
        nrows_weight=nrows_weight,
        blocks_per_weight_row=blocks_per_weight_row,
        groups=groups,
        fallback=fallback,
        rolled_q2=rolled_q2,
        mixed_j32_tails=mixed_j32_tails,
        mixed_q2_k=mixed_q2_k,
        mixed_j32_rows=mixed_j32_rows,
        full_i=full_i,
        full_j=full_j,
    )
    return KernelSpec(cpp_id, suffix, config, enforce_resource_gate)


def _grouped_forward_specs() -> list[KernelSpec]:
    specs = [
        _forward_spec(
            "GroupedFwdFixedQ80G8K4096J64Full",
            "grouped_fwd_fixed_q8_0_g8_k4096_j64_full",
            ForwardKind.FIXED_GROUPED,
            quant_type=QuantType.Q8_0,
            j=64,
            blocks_per_weight_row=16,
            groups=8,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "GroupedFwdFixedQ80G8K4096J64Bounded",
            "grouped_fwd_fixed_q8_0_g8_k4096_j64_bounded",
            ForwardKind.FIXED_GROUPED,
            quant_type=QuantType.Q8_0,
            j=64,
            blocks_per_weight_row=16,
            groups=8,
            fallback=True,
        ),
    ]
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = quant_name.lower()
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}GenericJ128",
                f"grouped_fwd_serial_{suffix}_generic_j128",
                ForwardKind.GROUPED_SERIAL,
                quant_type=quant_type,
                j=128,
            )
        )
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = quant_name.lower()
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}N512K2048J64",
                f"grouped_fwd_serial_{suffix}_n512_k2048_j64",
                ForwardKind.GROUPED_SERIAL,
                quant_type=quant_type,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
                enforce_resource_gate=quant_name in {"Q3_K", "IQ2_S"},
            )
        )
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = quant_name.lower()
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}N2048K512J64",
                f"grouped_fwd_serial_{suffix}_n2048_k512_j64",
                ForwardKind.GROUPED_SERIAL,
                quant_type=quant_type,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                mixed_j32_tails=quant_name == "Q4_K",
                mixed_j32_rows=(16384, 65536) if quant_name == "Q4_K" else (0, 0),
                enforce_resource_gate=quant_name in {"Q4_K", "Q5_K", "IQ2_S"},
            )
        )
    for quant_name, quant_type in ROW_TASK_TYPES:
        label = quant_name.replace("_", "")
        suffix = quant_name.lower()
        specs.append(
            _forward_spec(
                f"GroupedFwdRowTask{label}N512K2048J64",
                f"grouped_fwd_row_task_{suffix}_n512_k2048_j64",
                ForwardKind.GROUPED_ROW_TASK,
                quant_type=quant_type,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
                enforce_resource_gate=quant_name in {"Q3_K", "IQ2_S"},
            )
        )
    specs.extend(
        (
            _forward_spec(
                "GroupedFwdSerialIQ2SN2048K512J64J32",
                "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.IQ2_S,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                mixed_j32_tails=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialIQ2XXSN2048K4096J64",
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j64",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.IQ2_XXS,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialIQ2XXSN2048K4096J80",
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.IQ2_XXS,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ2KN4096K2048J32",
                "grouped_fwd_serial_q2_k_n4096_k2048_j32",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.Q2_K,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ2KN4096K2048J32J16",
                "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.Q2_K,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                mixed_q2_k=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ5KN2048K512J32",
                "grouped_fwd_serial_q5_k_n2048_k512_j32",
                ForwardKind.GROUPED_SERIAL,
                quant_type=QuantType.Q5_K,
                j=32,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                enforce_resource_gate=True,
            ),
        )
    )
    assert len(specs) == 37
    return specs


def _dense_backward_spec(
    cpp_id: str,
    suffix: str,
    quant_type: QuantType,
    n_tiles: int,
    k_iteration: int,
    *,
    group_m: int = 2,
    m_tiles_per_wave: int = 1,
    decoder_width: int = 0,
    prefetch_local: bool = False,
    full_tiles: bool = False,
    prefetch_packed: bool = False,
    lds_padding: int = 0,
    vector_local_load: bool = False,
    lds_swizzle_chunk: int = 0,
    pack_q5_quant_bytes: bool = False,
    pack_q6_quant_bytes: bool = False,
    exact_out_features: int = 0,
    exact_in_features: int = 0,
    active_waves: int = 4,
) -> KernelSpec:
    config = DenseBackwardConfig(
        quant_type=quant_type,
        n_tiles=n_tiles,
        k_iteration=k_iteration,
        group_m=group_m,
        m_tiles_per_wave=m_tiles_per_wave,
        decoder_width=decoder_width,
        prefetch_local=prefetch_local,
        full_tiles=full_tiles,
        prefetch_packed=prefetch_packed,
        lds_padding=lds_padding,
        vector_local_load=vector_local_load,
        lds_swizzle_chunk=lds_swizzle_chunk,
        pack_q5_quant_bytes=pack_q5_quant_bytes,
        pack_q6_quant_bytes=pack_q6_quant_bytes,
        exact_out_features=exact_out_features,
        exact_in_features=exact_in_features,
        active_waves=active_waves,
    )
    return KernelSpec(cpp_id, suffix, config, enforce_resource_gate=True)


def _dense_backward_specs() -> list[KernelSpec]:
    specs: list[KernelSpec] = [
        _dense_backward_spec(
            "DenseBwdQ80NT64KI16G0",
            "dense_bwd_q8_0_nt64_ki16_g0",
            QuantType.Q8_0,
            4,
            16,
            group_m=0,
            decoder_width=16,
        )
    ]
    for label, out_features, in_features in (
        ("N1024K4096", 1024, 4096),
        ("N32768K1024", 32768, 1024),
        ("N512K4096", 512, 4096),
        ("N4096K8192", 4096, 8192),
        ("N2048K4096", 2048, 4096),
        ("N4096K2048", 4096, 2048),
    ):
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80Exact{label}",
                f"dense_bwd_q8_0_exact_{label.lower()}",
                QuantType.Q8_0,
                4,
                16,
                group_m=0,
                decoder_width=16,
                full_tiles=True,
                exact_out_features=out_features,
                exact_in_features=in_features,
            )
        )
    for label, out_features, in_features in (
        ("N1024K4096", 1024, 4096),
        ("N32768K1024", 32768, 1024),
        ("N512K4096", 512, 4096),
        ("N4096K8192", 4096, 8192),
        ("N2048K4096", 2048, 4096),
        ("N4096K2048", 4096, 2048),
    ):
        for geometry, n_tiles, m_tiles_per_wave in (
            ("G1", 4, 2),
            ("G2", 8, 2),
            ("G3", 4, 4),
        ):
            specs.append(
                _dense_backward_spec(
                    f"DenseBwdQ80Exact{label}{geometry}",
                    f"dense_bwd_q8_0_exact_{label.lower()}_{geometry.lower()}",
                    QuantType.Q8_0,
                    n_tiles,
                    32,
                    group_m=0,
                    m_tiles_per_wave=m_tiles_per_wave,
                    decoder_width=16,
                    full_tiles=True,
                    exact_out_features=out_features,
                    exact_in_features=in_features,
                )
            )
    for label, out_features, in_features in (
        ("N1024K4096", 1024, 4096),
        ("N32768K1024", 32768, 1024),
        ("N512K4096", 512, 4096),
        ("N4096K8192", 4096, 8192),
        ("N2048K4096", 2048, 4096),
        ("N4096K2048", 4096, 2048),
    ):
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80Exact{label}G2Padding8",
                f"dense_bwd_q8_0_exact_{label.lower()}_g2_padding8",
                QuantType.Q8_0,
                8,
                32,
                group_m=0,
                m_tiles_per_wave=2,
                decoder_width=16,
                full_tiles=True,
                lds_padding=8,
                exact_out_features=out_features,
                exact_in_features=in_features,
            )
        )
    for label, out_features, in_features in (
        ("N32768K1024", 32768, 1024),
        ("N4096K8192", 4096, 8192),
    ):
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80Exact{label}G2GroupM2",
                f"dense_bwd_q8_0_exact_{label.lower()}_g2_group_m2",
                QuantType.Q8_0,
                8,
                32,
                group_m=2,
                m_tiles_per_wave=2,
                decoder_width=16,
                full_tiles=True,
                exact_out_features=out_features,
                exact_in_features=in_features,
            )
        )
    specs.append(
        _dense_backward_spec(
            "DenseBwdQ80ExactLMHeadM32Active2",
            "dense_bwd_q8_0_exact_lm_head_m32_active2",
            QuantType.Q8_0,
            4,
            16,
            group_m=0,
            decoder_width=16,
            exact_out_features=129280,
            exact_in_features=4096,
            active_waves=2,
        )
    )
    for geometry, n_tiles, m_tiles_per_wave in (
        ("G1", 4, 2),
        ("G2", 8, 2),
        ("G3", 4, 4),
    ):
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80ExactLMHead{geometry}",
                f"dense_bwd_q8_0_exact_lm_head_{geometry.lower()}",
                QuantType.Q8_0,
                n_tiles,
                32,
                group_m=0,
                m_tiles_per_wave=m_tiles_per_wave,
                decoder_width=16,
                full_tiles=True,
                exact_out_features=129280,
                exact_in_features=4096,
            )
        )
    for full in (False, True):
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80ExactLMHead{'Full' if full else 'Bounded'}",
                f"dense_bwd_q8_0_exact_lm_head_{'full' if full else 'bounded'}",
                QuantType.Q8_0,
                4,
                16,
                group_m=0,
                decoder_width=16,
                full_tiles=full,
                exact_out_features=129280,
                exact_in_features=4096,
            )
        )

    def generic(
        label: str,
        quant_type: QuantType,
        n_tiles: int,
        group_m: int,
    ) -> None:
        nt = n_tiles * 16
        cpp_label = label.replace("_", "")
        specs.append(
            _dense_backward_spec(
                f"DenseBwd{cpp_label}NT{nt}KI16G{group_m}",
                f"dense_bwd_{label.lower()}_nt{nt}_ki16_g{group_m}",
                quant_type,
                n_tiles,
                16,
                group_m=group_m,
            )
        )

    for label, quant_type, variants in (
        ("Q3_K", QuantType.Q3_K, ((1, 0), (4, 0), (4, 2), (8, 2), (12, 2), (16, 2))),
        ("Q4_K", QuantType.Q4_K, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("Q5_K", QuantType.Q5_K, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("IQ2_S", QuantType.IQ2_S, ((1, 0), (4, 0), (4, 2), (12, 2), (16, 2))),
    ):
        for n_tiles, group_m in variants:
            generic(label, quant_type, n_tiles, group_m)

    specs.extend(
        (
            _dense_backward_spec(
                "DenseBwdQ3KFullWide",
                "dense_bwd_q3_k_mt128_nt128_ki32_full_wide",
                QuantType.Q3_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
            _dense_backward_spec(
                "DenseBwdQ3KFullNarrow",
                "dense_bwd_q3_k_mt128_nt128_ki32_full_narrow",
                QuantType.Q3_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                lds_padding=8,
                vector_local_load=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK4096",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k4096",
                QuantType.Q4_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                lds_padding=8,
                vector_local_load=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK2048",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k2048",
                QuantType.Q4_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK512",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k512",
                QuantType.Q4_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=16,
            ),
            _dense_backward_spec(
                "DenseBwdQ5KFullK2048",
                "dense_bwd_q5_k_mt128_nt128_ki32_full_k2048",
                QuantType.Q5_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
                pack_q5_quant_bytes=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ5KFullK512",
                "dense_bwd_q5_k_mt128_nt128_ki32_full_k512",
                QuantType.Q5_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=4,
            ),
        )
    )

    for rows, n_tiles, k_iteration, m_tiles, swizzle, pack_q6 in (
        (64, 2, 64, 1, 16, False),
        (128, 4, 32, 2, 8, False),
        (256, 4, 32, 2, 8, True),
    ):
        for full in (False, True):
            specs.append(
                _dense_backward_spec(
                    f"DenseBwdQ6KM{rows}{'Full' if full else 'Bounded'}",
                    f"dense_bwd_q6_k_m{rows}_nt{n_tiles * 16}_ki{k_iteration}_"
                    f"{'full' if full else 'bounded'}",
                    QuantType.Q6_K,
                    n_tiles,
                    k_iteration,
                    group_m=0,
                    m_tiles_per_wave=m_tiles,
                    prefetch_local=True,
                    full_tiles=full,
                    vector_local_load=True,
                    lds_swizzle_chunk=swizzle,
                    pack_q6_quant_bytes=pack_q6,
                )
            )
    specs.extend(
        (
            _dense_backward_spec(
                "DenseBwdQ6KNT128KI16G2",
                "dense_bwd_q6_k_nt128_ki16_g2",
                QuantType.Q6_K,
                8,
                16,
            ),
            _dense_backward_spec(
                "DenseBwdQ6KNT256KI16G2",
                "dense_bwd_q6_k_nt256_ki16_g2",
                QuantType.Q6_K,
                16,
                16,
            ),
        )
    )
    specs.extend(
        (
            _dense_backward_spec(
                "DenseBwdQ5KFullK2048ScalarExtraction",
                "dense_bwd_q5_k_full_k2048_scalar_extraction",
                QuantType.Q5_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
        )
    )
    assert len(specs) == 76
    return specs


def _db8_dense_backward_specs() -> list[KernelSpec]:
    specs = []
    for label, out_features, in_features, padding, group_m in (
        ("N1024K4096", 1024, 4096, 0, 2),
        ("N1024K4096", 1024, 4096, 8, 2),
        ("N32768K1024", 32768, 1024, 8, 1),
        ("N512K4096", 512, 4096, 8, 2),
        ("N2048K4096", 2048, 4096, 0, 2),
        ("N2048K4096", 2048, 4096, 8, 2),
        ("N4096K2048", 4096, 2048, 0, 2),
    ):
        padding_label = "Padding8" if padding else ""
        padding_suffix = "_padding8" if padding else ""
        specs.append(
            _dense_backward_spec(
                f"DenseBwdQ80Exact{label}G2GroupM{group_m}{padding_label}",
                f"dense_bwd_q8_0_exact_{label.lower()}_g2_group_m{group_m}"
                f"{padding_suffix}",
                QuantType.Q8_0,
                8,
                32,
                group_m=group_m,
                m_tiles_per_wave=2,
                decoder_width=16,
                full_tiles=True,
                lds_padding=padding,
                exact_out_features=out_features,
                exact_in_features=in_features,
            )
        )
    assert len(specs) == 7
    return specs


def _grouped_backward_spec(
    cpp_id: str,
    suffix: str,
    kind: GroupedBackwardKind,
    quant_type: QuantType | None = None,
    enforce_resource_gate: bool = False,
) -> KernelSpec:
    config = GroupedBackwardConfig(kind=kind, quant_type=quant_type)
    return KernelSpec(cpp_id, suffix, config, enforce_resource_gate)


def _grouped_backward_specs() -> list[KernelSpec]:
    specs: list[KernelSpec] = []
    for quant_name, quant_type in BACKWARD_QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _grouped_backward_spec(
                f"GroupedBwdSingle{label}Generic",
                f"grouped_bwd_single_{quant_name.lower()}_generic",
                GroupedBackwardKind.GENERIC_SINGLE,
                quant_type,
            )
        )
    for quant_name, quant_type in BACKWARD_QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _grouped_backward_spec(
                f"GroupedBwdPair{label}Generic",
                f"grouped_bwd_pair_{quant_name.lower()}_generic",
                GroupedBackwardKind.GENERIC_PAIR,
                quant_type,
            )
        )
    specs.extend(
        (
            _grouped_backward_spec(
                "GroupedBwdFixedQ80G8K4096",
                "grouped_bwd_fixed_q8_0_g8_k4096",
                GroupedBackwardKind.FIXED_Q8_0_GENERIC,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M64N64",
                "grouped_bwd_single_q2_k_n4096_k2048_mt64_nt64",
                GroupedBackwardKind.Q2_K_SINGLE_M64_U1,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M128N64",
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64",
                GroupedBackwardKind.Q2_K_SINGLE_M128_U1,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2XXSN2048K4096M64N64",
                "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt64_nt64",
                GroupedBackwardKind.IQ2_XXS_PAIR_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M128N64U2",
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64_u2",
                GroupedBackwardKind.Q2_K_SINGLE_M128_U2,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdFixedQ80G8K4096M256N64",
                "grouped_bwd_fixed_q8_0_g8_k4096_mt256_nt64",
                GroupedBackwardKind.FIXED_Q8_0_M256,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ4KN2048K512M64N64",
                "grouped_bwd_single_q4_k_n2048_k512_mt64_nt64",
                GroupedBackwardKind.Q4_SINGLE_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ4KN2048K512M128N64",
                "grouped_bwd_single_q4_k_n2048_k512_mt128_nt64",
                GroupedBackwardKind.Q4_SINGLE_M128,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairQ3KN512K2048M64N64",
                "grouped_bwd_pair_q3_k_n512_k2048_mt64_nt64",
                GroupedBackwardKind.Q3_PAIR_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairQ3KN512K2048M128N64",
                "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64",
                GroupedBackwardKind.Q3_PAIR_M128,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ5KN2048K512M64N64",
                "grouped_bwd_single_q5_k_n2048_k512_mt64_nt64",
                GroupedBackwardKind.Q5_SINGLE_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleIQ2SN2048K512M64N64",
                "grouped_bwd_single_iq2_s_n2048_k512_mt64_nt64",
                GroupedBackwardKind.IQ2_S_SINGLE_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleIQ2SN2048K512M128N64",
                "grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64",
                GroupedBackwardKind.IQ2_S_SINGLE_M128,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2SN512K2048M64N64",
                "grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64",
                GroupedBackwardKind.IQ2_S_PAIR_M64,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2SN512K2048M128N64",
                "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64",
                GroupedBackwardKind.IQ2_S_PAIR_M128,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskQ4KN2048K512M128N128",
                "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128",
                GroupedBackwardKind.Q4_ROW_TASK,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskQ5KN2048K512M128N128",
                "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128",
                GroupedBackwardKind.Q5_ROW_TASK,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskIQ2SN2048K512M128N128",
                "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128",
                GroupedBackwardKind.IQ2_S_ROW_TASK,
                enforce_resource_gate=True,
            ),
        )
    )
    assert len(specs) == 32
    return specs


def kernel_specs() -> tuple[KernelSpec, ...]:
    specs: list[KernelSpec] = [
        _forward_spec(
            "QuantizeQ81F32D4",
            "quantize_bf16_q8_1_f32_d4",
            ForwardKind.QUANTIZE,
            quant_type=QuantType.Q8_0,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "QuantizeQ81F16D4S4",
            "quantize_bf16_q8_1_f16_d4s4",
            ForwardKind.QUANTIZE,
            quant_type=QuantType.Q4_K,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "QuantizeQ81F16D2S6",
            "quantize_bf16_q8_1_f16_d2s6",
            ForwardKind.QUANTIZE,
            quant_type=QuantType.Q2_K,
            enforce_resource_gate=True,
        ),
    ]
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _forward_spec(
                f"DenseFwd{label}J128",
                f"dense_fwd_{quant_name.lower()}_j128",
                ForwardKind.DENSE,
                quant_type=quant_type,
                j=128,
                enforce_resource_gate=quant_type == QuantType.Q8_0,
            )
        )
    specs.append(
        _forward_spec(
            "DenseFwdQ6KJ64",
            "dense_fwd_q6_k_j64",
            ForwardKind.DENSE,
            quant_type=QuantType.Q6_K,
            j=64,
            enforce_resource_gate=True,
        )
    )
    for k, j, full_j in (
        (1024, 128, True),
        (2048, 128, True),
        (4096, 64, False),
        (4096, 64, True),
        (4096, 128, True),
        (8192, 128, True),
    ):
        body = "Full" if full_j else "Bounded"
        specs.append(
            _forward_spec(
                f"DenseFwdQ80K{k}J{j}{body}",
                f"dense_fwd_q8_0_k{k}_j{j}_{body.lower()}",
                ForwardKind.DENSE,
                quant_type=QuantType.Q8_0,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=full_j,
                enforce_resource_gate=True,
            )
        )
    for quant_type, k, j in (
        (QuantType.Q3_K, 2048, 128),
        (QuantType.Q4_K, 512, 128),
        (QuantType.Q4_K, 2048, 128),
        (QuantType.Q4_K, 4096, 128),
        (QuantType.Q5_K, 512, 128),
        (QuantType.Q5_K, 2048, 128),
        (QuantType.Q6_K, 2048, 64),
        (QuantType.Q6_K, 2048, 128),
    ):
        label = quant_type.name.replace("_", "")
        specs.append(
            _forward_spec(
                f"DenseFwd{label}K{k}J{j}Full",
                f"dense_fwd_{quant_type.name.lower()}_k{k}_j{j}_full",
                ForwardKind.DENSE,
                quant_type=quant_type,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=True,
                enforce_resource_gate=True,
            )
        )
    specs.append(
        _forward_spec(
            "GroupedRowTaskSetup",
            "grouped_row_task_setup",
            ForwardKind.ROW_TASK_SETUP,
        )
    )
    specs.extend(_grouped_forward_specs())
    specs.extend(_dense_backward_specs())
    specs.extend(_grouped_backward_specs())
    specs.extend(_db8_dense_backward_specs())
    assert len(specs) == 179
    assert len({spec.cpp_id for spec in specs}) == len(specs)
    assert len({spec.symbol for spec in specs}) == len(specs)
    return tuple(specs)


def _find_tool(hipcc: Path, name: str) -> Path:
    candidates = (
        hipcc.with_name(name),
        hipcc.parent.parent / "lib" / "llvm" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"cannot locate {name} beside {hipcc}")


def _compiler_identity(hipcc: Path) -> str:
    return subprocess.run(
        [str(hipcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _common_args(hipcc: Path) -> list[str]:
    return [
        str(hipcc),
        "--genco",
        "--no-gpu-bundle-output",
        "-c",
        "-O3",
        "-std=c++17",
        f"--offload-arch={ARCH}",
        f"-I{CSRC}",
        f"-ffile-prefix-map={ROOT}=.",
    ]


def _ccache_namespace(hipcc: Path, compiler_identity: str) -> str:
    digest = hashlib.sha256()
    digest.update(compiler_identity.encode())
    digest.update("\0".join(_common_args(hipcc)[1:]).encode())
    return f"torch-ggml-ops-mmq-{digest.hexdigest()[:16]}"


def _build_input_digest(
    hipcc: Path, compiler_identity: str, specs: tuple[KernelSpec, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(compiler_identity.encode())
    digest.update("\0".join(_common_args(hipcc)[1:]).encode())
    digest.update(json.dumps([asdict(spec) for spec in specs], sort_keys=True).encode())
    device_headers = [
        path
        for path in sorted(CSRC.rglob("*.cuh"))
        if "generated" not in path.parts and not path.name.endswith("_hip.cuh")
    ]
    wrapper_renderer = Path(__file__).with_name("mmq_bundle_wrapper_source.py")
    for path in (Path(__file__), wrapper_renderer, *device_headers):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _header_text(specs: tuple[KernelSpec, ...]) -> str:
    enum_values = "\n".join(
        f"    {spec.cpp_id} = {index}," for index, spec in enumerate(specs)
    )
    symbols = "\n".join(f'    "{spec.symbol}",' for spec in specs)
    return f"""// Generated by tools/build_mmq_bundle.py. Do not edit directly.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace torch_ggml_ops::mmq_bundle {{

enum class MMQKernelId : std::uint16_t {{
{enum_values}
    Count = {len(specs)},
}};

inline constexpr std::array<const char *, {len(specs)}> kMMQKernelSymbols{{{{
{symbols}
}}}};

inline constexpr const char * mmq_kernel_symbol(MMQKernelId id) {{
    return kMMQKernelSymbols[static_cast<std::size_t>(id)];
}}

}} // namespace torch_ggml_ops::mmq_bundle
"""


def _verify_artifact(artifact: Path, spec: KernelSpec, readelf: Path) -> bytes:
    data = artifact.read_bytes()
    if not data.startswith(b"\x7fELF"):
        raise RuntimeError(f"{artifact} is not an ELF code object")
    if ARCH.encode() not in data:
        raise RuntimeError(f"{artifact} does not identify target {ARCH}")
    symbols = subprocess.run(
        [str(readelf), "--symbols", "--wide", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported_functions = {
        line.split()[-1]
        for line in symbols.splitlines()
        if " FUNC " in line and " GLOBAL " in line
    }
    if exported_functions != {spec.symbol}:
        raise RuntimeError(
            f"{artifact} exports {sorted(exported_functions)}, expected only {spec.symbol}"
        )
    notes = subprocess.run(
        [str(readelf), "--notes", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def metadata_value(name: str) -> str:
        match = re.search(rf"\.{name}:\s+([^\s]+)", notes)
        if match is None:
            raise RuntimeError(f"{artifact} has no {name} metadata")
        return match.group(1)

    resources: dict[str, int | bool] = {
        "private_segment_bytes": int(metadata_value("private_segment_fixed_size")),
        "sgpr_spills": int(metadata_value("sgpr_spill_count")),
        "uses_dynamic_stack": metadata_value("uses_dynamic_stack") == "true",
        "vgpr_spills": int(metadata_value("vgpr_spill_count")),
    }
    if spec.enforce_resource_gate and (
        resources["private_segment_bytes"] != 0
        or resources["sgpr_spills"] != 0
        or resources["vgpr_spills"] != 0
        or resources["uses_dynamic_stack"]
    ):
        raise RuntimeError(f"kernel {spec.cpp_id} fails the resource gate: {resources}")
    return data


def _compile_one(
    spec: KernelSpec,
    output_dir: Path,
    hipcc: Path,
    readelf: Path,
    env: dict[str, str],
    ccache: Path | None,
) -> tuple[str, bytes]:
    temporary = output_dir / f"{spec.cpp_id}.hsaco"
    source_text = render_wrapper(spec.symbol, spec.config)
    source_digest = hashlib.sha256(source_text.encode()).hexdigest()[:16]
    source = GENERATED_SOURCE_DIR / f"{spec.cpp_id}-{source_digest}.cu"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        source.write_text(source_text)
    elif source.read_text() != source_text:
        raise RuntimeError(f"generated source hash collision at {source}")
    command = [
        *([str(ccache)] if ccache is not None else []),
        *_common_args(hipcc),
        f"-cuid={spec.cuid}",
        str(source),
        "-o",
        str(temporary),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to compile {spec.cpp_id}\ncommand: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
            f"generated source:\n{source_text}"
        )
    data = _verify_artifact(temporary, spec, readelf)
    temporary.chmod(0o644)
    temporary.rename(output_dir / spec.filename)
    return spec.cpp_id, data


def _compile_all(
    specs: tuple[KernelSpec, ...],
    hipcc: Path,
    jobs: int,
    ccache: Path | None,
    ccache_namespace: str,
) -> tuple[Path, list[bytes]]:
    readelf = _find_tool(hipcc, "llvm-readelf")
    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mmq-gfx1151-", dir=PACKAGE_DIR.parent))
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "SOURCE_DATE_EPOCH": "0"})
    if ccache is not None:
        inherited_namespace = env.get("CCACHE_NAMESPACE")
        env["CCACHE_NAMESPACE"] = ":".join(
            part for part in (inherited_namespace, ccache_namespace) if part
        )
        env.setdefault("CCACHE_COMPILERCHECK", "content")
    try:
        images_by_id: dict[str, bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _compile_one, spec, staging, hipcc, readelf, env, ccache
                ): spec
                for spec in specs
            }
            for future in concurrent.futures.as_completed(futures):
                cpp_id, image = future.result()
                images_by_id[cpp_id] = image
                print(f"built {cpp_id}", flush=True)
        return staging, [images_by_id[spec.cpp_id] for spec in specs]
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _bundle_is_current(build_input: str, specs: tuple[KernelSpec, ...]) -> bool:
    build_input_stamp = PACKAGE_DIR / BUILD_INPUT_STAMP
    if (
        not PACKAGE_DIR.is_dir()
        or not GENERATED_HEADER.is_file()
        or not build_input_stamp.is_file()
    ):
        return False
    expected_files = {spec.filename for spec in specs}
    actual_files = {path.name for path in PACKAGE_DIR.glob("*.hsaco")}
    return (
        actual_files == expected_files
        and GENERATED_HEADER.read_text() == _header_text(specs)
        and build_input_stamp.read_text() == build_input + "\n"
    )


def _install_bundle(
    staging: Path,
    specs: tuple[KernelSpec, ...],
    build_input: str,
) -> None:
    (staging / BUILD_INPUT_STAMP).write_text(build_input + "\n")
    generated_text = _header_text(specs)
    generated_changed = (
        not GENERATED_HEADER.is_file() or GENERATED_HEADER.read_text() != generated_text
    )
    generated_staging = GENERATED_HEADER.with_suffix(".cuh.tmp")
    if generated_changed:
        generated_staging.parent.mkdir(parents=True, exist_ok=True)
        generated_staging.write_text(generated_text)

    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    old_dir = PACKAGE_DIR.with_name(PACKAGE_DIR.name + ".old")
    shutil.rmtree(old_dir, ignore_errors=True)
    if PACKAGE_DIR.exists():
        PACKAGE_DIR.rename(old_dir)
    staging.rename(PACKAGE_DIR)
    shutil.rmtree(old_dir, ignore_errors=True)
    if generated_changed:
        os.replace(generated_staging, GENERATED_HEADER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-reproducible", action="store_true")
    parser.add_argument("--no-ccache", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    discovered_hipcc = shutil.which("hipcc")
    hipcc_value = args.hipcc or (
        Path(discovered_hipcc) if discovered_hipcc is not None else None
    )
    if hipcc_value is None or not hipcc_value.is_file():
        raise FileNotFoundError("hipcc is required to build the MMQ bundle")
    hipcc = hipcc_value.resolve()
    specs = kernel_specs()
    compiler_identity = _compiler_identity(hipcc)
    build_input = _build_input_digest(hipcc, compiler_identity, specs)

    if args.check:
        if not _bundle_is_current(build_input, specs):
            raise SystemExit("MMQ gfx1151 bundle is stale")
        print(f"MMQ gfx1151 bundle is current ({len(specs)} kernels)")
        return
    if (
        not args.force
        and not args.verify_reproducible
        and _bundle_is_current(build_input, specs)
    ):
        print(f"MMQ gfx1151 bundle is current ({len(specs)} kernels)")
        return

    disable_ccache = args.no_ccache or os.environ.get(
        "TORCH_GGML_OPS_DISABLE_CCACHE", ""
    ).lower() in {"1", "true", "yes", "on"}
    ccache_value = None if disable_ccache else shutil.which("ccache")
    ccache = Path(ccache_value).resolve() if ccache_value else None
    if args.verify_reproducible:
        ccache = None
    ccache_namespace = _ccache_namespace(hipcc, compiler_identity)
    if ccache is not None:
        print(f"using ccache: {ccache}", flush=True)

    first_dir, first_images = _compile_all(
        specs, hipcc, args.jobs, ccache, ccache_namespace
    )
    try:
        if args.verify_reproducible:
            second_dir, second_images = _compile_all(
                specs, hipcc, args.jobs, None, ccache_namespace
            )
            try:
                if first_images != second_images:
                    mismatches = [
                        spec.cpp_id
                        for spec, first, second in zip(
                            specs, first_images, second_images, strict=True
                        )
                        if first != second
                    ]
                    raise RuntimeError(
                        "non-reproducible MMQ artifacts: " + ", ".join(mismatches)
                    )
            finally:
                shutil.rmtree(second_dir, ignore_errors=True)
        _install_bundle(first_dir, specs, build_input)
    except Exception:
        shutil.rmtree(first_dir, ignore_errors=True)
        raise
    print(f"installed {len(specs)} MMQ kernels in {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
