"""Historical HIP control specifications kept separate from public deployment."""

from dataclasses import dataclass
from typing import Any

from tools.mmq_bundle_wrapper_source import (
    DenseBackwardConfig,
    ForwardConfig,
    ForwardKind,
    GroupedBackwardConfig,
    GroupedBackwardKind,
    QuantType,
    render_wrapper,
)

ABI_PREFIX = "torch_ggml_ops_mmq_gfx1151_v1_"
QUANT_TYPES = tuple(QuantType)
BACKWARD_QUANT_TYPES = tuple(
    quant
    for quant in QUANT_TYPES
    if quant.name in {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_S"}
)
ROW_TASK_TYPES = tuple(
    quant
    for quant in QUANT_TYPES
    if quant.name in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_S"}
)


@dataclass(frozen=True)
class HIPControlSpec:
    symbol: str
    config: ForwardConfig | DenseBackwardConfig | GroupedBackwardConfig

    @property
    def filename(self) -> str:
        return f"{self.symbol}.hsaco"


def _forward(
    suffix: str,
    kind: ForwardKind,
    quant_type: QuantType | None = None,
    **kwargs: Any,
) -> HIPControlSpec:
    return HIPControlSpec(
        ABI_PREFIX + suffix,
        ForwardConfig(kind=kind, quant_type=quant_type, **kwargs),
    )


def _dense_backward(
    suffix: str,
    quant_type: QuantType,
    n_tiles: int,
    k_iteration: int,
    **kwargs: Any,
) -> HIPControlSpec:
    values: dict[str, Any] = {
        "group_m": 2,
        "m_tiles_per_wave": 1,
        "decoder_width": 0,
        "prefetch_local": False,
        "full_tiles": False,
        "prefetch_packed": False,
        "lds_padding": 0,
        "vector_local_load": False,
        "lds_swizzle_chunk": 0,
        "pack_q5_quant_bytes": False,
        "pack_q6_quant_bytes": False,
        **kwargs,
    }
    return HIPControlSpec(
        ABI_PREFIX + suffix,
        DenseBackwardConfig(
            quant_type=quant_type,
            n_tiles=n_tiles,
            k_iteration=k_iteration,
            **values,
        ),
    )


def _grouped_backward(
    suffix: str,
    kind: GroupedBackwardKind,
    quant_type: QuantType | None = None,
    **kwargs: Any,
) -> HIPControlSpec:
    return HIPControlSpec(
        ABI_PREFIX + suffix,
        GroupedBackwardConfig(kind=kind, quant_type=quant_type, **kwargs),
    )


def _forward_controls() -> list[HIPControlSpec]:
    specs = [
        _forward(
            "quantize_bf16_q8_1_f32_d4",
            ForwardKind.QUANTIZE,
            QuantType.Q8_0,
        ),
        _forward(
            "quantize_bf16_q8_1_f16_d4s4",
            ForwardKind.QUANTIZE,
            QuantType.Q4_K,
        ),
        _forward(
            "quantize_bf16_q8_1_f16_d2s6",
            ForwardKind.QUANTIZE,
            QuantType.Q2_K,
        ),
    ]
    for quant in QUANT_TYPES:
        specs.append(
            _forward(
                f"dense_fwd_{quant.name.lower()}_j128",
                ForwardKind.DENSE,
                quant,
                j=128,
            )
        )
    specs.append(
        _forward(
            "dense_fwd_q6_k_j64",
            ForwardKind.DENSE,
            QuantType.Q6_K,
            j=64,
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
        body = "full" if full_j else "bounded"
        specs.append(
            _forward(
                f"dense_fwd_q8_0_k{k}_j{j}_{body}",
                ForwardKind.DENSE,
                QuantType.Q8_0,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=full_j,
            )
        )
    for quant, k, j in (
        (QuantType.Q3_K, 2048, 128),
        (QuantType.Q4_K, 512, 128),
        (QuantType.Q4_K, 2048, 128),
        (QuantType.Q4_K, 4096, 128),
        (QuantType.Q5_K, 512, 128),
        (QuantType.Q5_K, 2048, 128),
        (QuantType.Q6_K, 2048, 64),
        (QuantType.Q6_K, 2048, 128),
    ):
        specs.append(
            _forward(
                f"dense_fwd_{quant.name.lower()}_k{k}_j{j}_full",
                ForwardKind.DENSE,
                quant,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=True,
            )
        )
    specs.extend(
        [
            _forward(
                "grouped_row_task_setup",
                ForwardKind.ROW_TASK_SETUP,
            ),
            _forward(
                "grouped_fwd_fixed_q8_0_g8_k4096_j64_full",
                ForwardKind.FIXED_GROUPED,
                QuantType.Q8_0,
                j=64,
                blocks_per_weight_row=16,
                groups=8,
            ),
            _forward(
                "grouped_fwd_fixed_q8_0_g8_k4096_j64_bounded",
                ForwardKind.FIXED_GROUPED,
                QuantType.Q8_0,
                j=64,
                blocks_per_weight_row=16,
                groups=8,
                fallback=True,
            ),
        ]
    )
    for quant in QUANT_TYPES:
        specs.append(
            _forward(
                f"grouped_fwd_serial_{quant.name.lower()}_generic_j128",
                ForwardKind.GROUPED_SERIAL,
                quant,
                j=128,
            )
        )
    for quant in QUANT_TYPES:
        specs.append(
            _forward(
                f"grouped_fwd_serial_{quant.name.lower()}_n512_k2048_j64",
                ForwardKind.GROUPED_SERIAL,
                quant,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
            )
        )
    for quant in QUANT_TYPES:
        specs.append(
            _forward(
                f"grouped_fwd_serial_{quant.name.lower()}_n2048_k512_j64",
                ForwardKind.GROUPED_SERIAL,
                quant,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                mixed_j32_tails=quant is QuantType.Q4_K,
                mixed_j32_rows=(16384, 65536) if quant is QuantType.Q4_K else (0, 0),
            )
        )
    for quant in ROW_TASK_TYPES:
        specs.append(
            _forward(
                f"grouped_fwd_row_task_{quant.name.lower()}_n512_k2048_j64",
                ForwardKind.GROUPED_ROW_TASK,
                quant,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
            )
        )
    specs.extend(
        [
            _forward(
                "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_S,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                mixed_j32_tails=True,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j64",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=16,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
            ),
            _forward(
                "grouped_fwd_serial_q2_k_n4096_k2048_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q2_K,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
            ),
            _forward(
                "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q2_K,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                mixed_q2_k=True,
            ),
            _forward(
                "grouped_fwd_serial_q5_k_n2048_k512_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q5_K,
                j=32,
                nrows_weight=2048,
                blocks_per_weight_row=2,
            ),
        ]
    )
    return specs


def _dense_backward_controls() -> list[HIPControlSpec]:
    specs = [
        _dense_backward(
            "dense_bwd_q8_0_nt64_ki16_g0", QuantType.Q8_0, 4, 16, decoder_width=16
        )
    ]
    exact_shapes = (
        ("n1024k4096", 1024, 4096),
        ("n32768k1024", 32768, 1024),
        ("n512k4096", 512, 4096),
        ("n4096k8192", 4096, 8192),
        ("n2048k4096", 2048, 4096),
        ("n4096k2048", 4096, 2048),
    )
    for label, out_features, in_features in exact_shapes:
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}",
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
        for geometry, n_tiles, m_tiles in (("g1", 4, 2), ("g2", 8, 2), ("g3", 4, 4)):
            specs.append(
                _dense_backward(
                    f"dense_bwd_q8_0_exact_{label}_{geometry}",
                    QuantType.Q8_0,
                    n_tiles,
                    32,
                    group_m=0,
                    m_tiles_per_wave=m_tiles,
                    decoder_width=16,
                    full_tiles=True,
                    exact_out_features=out_features,
                    exact_in_features=in_features,
                )
            )
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}_g2_padding8",
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
        ("n32768k1024", 32768, 1024),
        ("n4096k8192", 4096, 8192),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}_g2_group_m2",
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
    for suffix, full_tiles in (("bounded", False), ("full", True)):
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_lm_head_{suffix}",
                QuantType.Q8_0,
                4,
                16,
                group_m=0,
                decoder_width=16,
                full_tiles=full_tiles,
                exact_out_features=129280,
                exact_in_features=4096,
            )
        )
    specs.append(
        _dense_backward(
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
    for geometry, n_tiles, m_tiles in (("g1", 4, 2), ("g2", 8, 2), ("g3", 4, 4)):
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_lm_head_{geometry}",
                QuantType.Q8_0,
                n_tiles,
                32,
                group_m=0,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                full_tiles=True,
                exact_out_features=129280,
                exact_in_features=4096,
            )
        )
    for label, quant, variants in (
        ("q3_k", QuantType.Q3_K, ((1, 0), (4, 0), (4, 2), (8, 2), (12, 2), (16, 2))),
        ("q4_k", QuantType.Q4_K, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("q5_k", QuantType.Q5_K, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("iq2_s", QuantType.IQ2_S, ((1, 0), (4, 0), (4, 2), (12, 2), (16, 2))),
    ):
        for n_tiles, group_m in variants:
            specs.append(
                _dense_backward(
                    f"dense_bwd_{label}_nt{n_tiles * 16}_ki16_g{group_m}",
                    quant,
                    n_tiles,
                    16,
                    group_m=group_m,
                )
            )
    specs.extend(
        [
            _dense_backward(
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
            _dense_backward(
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
        ]
    )
    for quant, k, swizzle, pack_q5 in (
        (QuantType.Q4_K, 4096, 0, False),
        (QuantType.Q4_K, 2048, 8, False),
        (QuantType.Q4_K, 512, 16, False),
        (QuantType.Q5_K, 2048, 8, True),
        (QuantType.Q5_K, 512, 4, True),
    ):
        name = quant.name.lower()
        specs.append(
            _dense_backward(
                f"dense_bwd_{name}_mt128_nt128_ki32_full_k{k}",
                quant,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                lds_padding=8 if quant is QuantType.Q4_K and k == 4096 else 0,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pack_q5_quant_bytes=pack_q5,
            )
        )
    for rows, n_tiles, k_iteration, m_tiles, swizzle, pack_q6 in (
        (64, 2, 64, 1, 16, False),
        (128, 4, 32, 2, 8, False),
        (256, 4, 32, 2, 8, True),
    ):
        for suffix, full_tiles in (("bounded", False), ("full", True)):
            specs.append(
                _dense_backward(
                    f"dense_bwd_q6_k_m{rows}_nt{n_tiles * 16}_ki{k_iteration}_{suffix}",
                    QuantType.Q6_K,
                    n_tiles,
                    k_iteration,
                    group_m=0,
                    m_tiles_per_wave=m_tiles,
                    prefetch_local=True,
                    full_tiles=full_tiles,
                    vector_local_load=True,
                    lds_swizzle_chunk=swizzle,
                    pack_q6_quant_bytes=pack_q6,
                )
            )
    for n_tiles in (8, 16):
        specs.append(
            _dense_backward(
                f"dense_bwd_q6_k_nt{n_tiles * 16}_ki16_g2",
                QuantType.Q6_K,
                n_tiles,
                16,
            )
        )
    specs.append(
        _dense_backward(
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
        )
    )
    return specs


def _db8_dense_backward_controls() -> list[HIPControlSpec]:
    specs = []
    for label, out_features, in_features, padding, group_m in (
        ("n1024k4096", 1024, 4096, 0, 2),
        ("n1024k4096", 1024, 4096, 8, 2),
        ("n32768k1024", 32768, 1024, 8, 1),
        ("n512k4096", 512, 4096, 8, 2),
        ("n2048k4096", 2048, 4096, 0, 2),
        ("n2048k4096", 2048, 4096, 8, 2),
        ("n4096k2048", 4096, 2048, 0, 2),
    ):
        suffix = "_padding8" if padding else ""
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}_g2_group_m{group_m}{suffix}",
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
    return specs


def _grouped_backward_controls() -> list[HIPControlSpec]:
    specs: list[HIPControlSpec] = []
    for quant in BACKWARD_QUANT_TYPES:
        name = quant.name.lower()
        specs.append(
            _grouped_backward(
                f"grouped_bwd_single_{name}_generic",
                GroupedBackwardKind.GENERIC_SINGLE,
                quant,
            )
        )
    for quant in BACKWARD_QUANT_TYPES:
        name = quant.name.lower()
        specs.append(
            _grouped_backward(
                f"grouped_bwd_pair_{name}_generic",
                GroupedBackwardKind.GENERIC_PAIR,
                quant,
            )
        )
    specs.extend(
        [
            _grouped_backward(
                "grouped_bwd_fixed_q8_0_g8_k4096",
                GroupedBackwardKind.FIXED_Q8_0_GENERIC,
            ),
            _grouped_backward(
                "grouped_bwd_single_q2_k_n4096_k2048_mt64_nt64",
                GroupedBackwardKind.Q2_K_SINGLE_M64_U1,
                QuantType.Q2_K,
            ),
            _grouped_backward(
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64",
                GroupedBackwardKind.Q2_K_SINGLE_M128_U1,
                QuantType.Q2_K,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt64_nt64",
                GroupedBackwardKind.IQ2_XXS_PAIR_M64,
                QuantType.IQ2_XXS,
            ),
            _grouped_backward(
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64_u2",
                GroupedBackwardKind.Q2_K_SINGLE_M128_U2,
                QuantType.Q2_K,
            ),
            _grouped_backward(
                "grouped_bwd_fixed_q8_0_g8_k4096_mt256_nt64",
                GroupedBackwardKind.FIXED_Q8_0_M256,
            ),
            _grouped_backward(
                "grouped_bwd_single_q4_k_n2048_k512_mt64_nt64",
                GroupedBackwardKind.Q4_SINGLE_M64,
                QuantType.Q4_K,
            ),
            _grouped_backward(
                "grouped_bwd_single_q4_k_n2048_k512_mt128_nt64",
                GroupedBackwardKind.Q4_SINGLE_M128,
                QuantType.Q4_K,
            ),
            _grouped_backward(
                "grouped_bwd_pair_q3_k_n512_k2048_mt64_nt64",
                GroupedBackwardKind.Q3_PAIR_M64,
                QuantType.Q3_K,
            ),
            _grouped_backward(
                "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64",
                GroupedBackwardKind.Q3_PAIR_M128,
                QuantType.Q3_K,
            ),
            _grouped_backward(
                "grouped_bwd_single_q5_k_n2048_k512_mt64_nt64",
                GroupedBackwardKind.Q5_SINGLE_M64,
                QuantType.Q5_K,
            ),
            _grouped_backward(
                "grouped_bwd_single_iq2_s_n2048_k512_mt64_nt64",
                GroupedBackwardKind.IQ2_S_SINGLE_M64,
                QuantType.IQ2_S,
            ),
            _grouped_backward(
                "grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64",
                GroupedBackwardKind.IQ2_S_SINGLE_M128,
                QuantType.IQ2_S,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64",
                GroupedBackwardKind.IQ2_S_PAIR_M64,
                QuantType.IQ2_S,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64",
                GroupedBackwardKind.IQ2_S_PAIR_M128,
                QuantType.IQ2_S,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128",
                GroupedBackwardKind.Q4_ROW_TASK,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128",
                GroupedBackwardKind.Q5_ROW_TASK,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128",
                GroupedBackwardKind.IQ2_S_ROW_TASK,
            ),
        ]
    )
    specs.extend(
        [
            _grouped_backward(
                "grouped_bwd_tuned_pair_iq2_xxs_n2048_k4096_mt128_nt64",
                GroupedBackwardKind.TUNED_DEEPSEEK_PAIR,
                QuantType.IQ2_XXS,
                n_tiles=4,
                m_tiles_per_wave=2,
                reduction_unroll=1,
            ),
            _grouped_backward(
                "grouped_bwd_tuned_fixed_q8_0_g8_k4096_mt192_nt64",
                GroupedBackwardKind.TUNED_FIXED_Q8_0,
                n_tiles=4,
                m_tiles_per_wave=3,
                reduction_unroll=1,
            ),
        ]
    )
    return specs


def hip_control_specs() -> tuple[HIPControlSpec, ...]:
    specs = (
        _forward_controls()
        + _dense_backward_controls()
        + _db8_dense_backward_controls()
        + _grouped_backward_controls()
    )
    symbols = [spec.symbol for spec in specs]
    if len(specs) != 181:
        raise ValueError(f"historical HIP control inventory has {len(specs)} entries")
    if len(symbols) != len(set(symbols)):
        raise ValueError("HIP control symbols must be unique")
    return tuple(specs)


def render_control(spec: HIPControlSpec) -> str:
    return render_wrapper(spec.symbol, spec.config)


def kernel_specs() -> tuple[HIPControlSpec, ...]:
    """Compatibility name for callers that consume the historical inventory."""
    return hip_control_specs()
