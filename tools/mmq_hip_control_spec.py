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
    SplitKReduceConfig,
    render_wrapper,
)

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


def control_filename(symbol: str) -> str:
    return f"{symbol}.hsaco"


@dataclass(frozen=True)
class HIPControlSpec:
    symbol: str
    config: (
        ForwardConfig | DenseBackwardConfig | GroupedBackwardConfig | SplitKReduceConfig
    )

    @property
    def filename(self) -> str:
        return control_filename(self.symbol)


def _forward(
    suffix: str,
    kind: ForwardKind,
    quant_type: QuantType | None = None,
    **kwargs: Any,
) -> HIPControlSpec:
    return HIPControlSpec(
        suffix,
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
        suffix,
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
        suffix,
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
    for k, tail in ((2560, 0), (640, 128)):
        specs.append(
            _forward(
                f"dense_fwd_q2_0_k{k}_j128_full",
                ForwardKind.DENSE,
                QuantType.Q2_0,
                j=128,
                blocks_per_weight_row=(k + 255) // 256,
                full_i=True,
                full_j=True,
                tail_values=tail,
            )
        )
    specs.append(
        _forward(
            "dense_fwd_q5_0_k640_j128_full_table",
            ForwardKind.DENSE,
            QuantType.Q5_0,
            j=128,
            blocks_per_weight_row=3,
            full_i=True,
            full_j=True,
            tail_values=128,
            table_decode=True,
        )
    )
    specs.append(
        _forward(
            "dense_fwd_iq4_xs_k2560_j128_full",
            ForwardKind.DENSE,
            QuantType.IQ4_XS,
            j=128,
            blocks_per_weight_row=10,
            full_i=True,
            full_j=True,
        )
    )
    specs.append(
        _forward(
            "dense_fwd_iq4_nl_k640_j128_full",
            ForwardKind.DENSE,
            QuantType.IQ4_NL,
            j=128,
            blocks_per_weight_row=3,
            full_i=True,
            full_j=True,
            tail_values=128,
        )
    )
    specs.append(
        _forward(
            "dense_fwd_q5_0_k640_j128_full",
            ForwardKind.DENSE,
            QuantType.Q5_0,
            j=128,
            blocks_per_weight_row=3,
            full_i=True,
            full_j=True,
            tail_values=128,
        )
    )
    specs.append(
        _forward(
            "dense_fwd_q4_0_k640_j128_full",
            ForwardKind.DENSE,
            QuantType.Q4_0,
            j=128,
            blocks_per_weight_row=3,
            full_i=True,
            full_j=True,
            tail_values=128,
        )
    )
    specs.append(
        _forward(
            "dense_fwd_q2_0_k2560_j128_full_table",
            ForwardKind.DENSE,
            QuantType.Q2_0,
            j=128,
            blocks_per_weight_row=10,
            full_i=True,
            full_j=True,
            table_decode=True,
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
        (QuantType.Q3_K, 2560, 128),
        (QuantType.Q4_K, 2560, 128),
        (QuantType.Q4_K, 6144, 128),
        (QuantType.Q5_K, 2560, 128),
        (QuantType.Q5_K, 6144, 128),
    ):
        specs.append(
            _forward(
                f"dense_fwd_{quant.name.lower()}_k{k}_j{j}_wide",
                ForwardKind.DENSE,
                quant,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=True,
                wide_tile=True,
            )
        )
    specs.append(
        _forward(
            "dense_fwd_q8_0_k640_j128_full",
            ForwardKind.DENSE,
            QuantType.Q8_0,
            j=128,
            blocks_per_weight_row=3,
            full_i=True,
            full_j=True,
            tail_values=128,
        )
    )
    for quant, k, j in (
        (QuantType.Q3_K, 2048, 128),
        (QuantType.Q3_K, 2560, 128),
        (QuantType.Q4_K, 512, 128),
        (QuantType.Q4_K, 2048, 128),
        (QuantType.Q4_K, 2560, 128),
        (QuantType.Q4_K, 4096, 128),
        (QuantType.Q4_K, 6144, 128),
        (QuantType.Q5_K, 512, 128),
        (QuantType.Q5_K, 2048, 128),
        (QuantType.Q5_K, 2560, 64),
        (QuantType.Q5_K, 2560, 128),
        (QuantType.Q5_K, 6144, 128),
        (QuantType.Q6_K, 2048, 64),
        (QuantType.Q6_K, 2048, 128),
        (QuantType.Q6_K, 2560, 128),
        (QuantType.Q6_K, 6144, 128),
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
    for quant, k, j in (
        (QuantType.Q3_K, 2048, 128),
        (QuantType.Q3_K, 2560, 128),
        (QuantType.Q4_K, 512, 128),
        (QuantType.Q4_K, 2048, 128),
        (QuantType.Q4_K, 4096, 128),
        (QuantType.Q5_K, 512, 128),
        (QuantType.Q5_K, 2048, 128),
        (QuantType.Q6_K, 2048, 64),
        (QuantType.Q6_K, 2048, 128),
        (QuantType.Q6_K, 2560, 128),
        (QuantType.Q6_K, 6144, 128),
        (QuantType.Q8_0, 1024, 128),
        (QuantType.Q8_0, 2048, 128),
        (QuantType.Q8_0, 4096, 128),
        (QuantType.Q8_0, 8192, 128),
    ):
        specs.append(
            _forward(
                f"dense_fwd_{quant.name.lower()}_k{k}_j{j}_full_hoisted",
                ForwardKind.DENSE,
                quant,
                j=j,
                blocks_per_weight_row=k // 256,
                full_i=True,
                full_j=True,
                hoisted_epilogue=True,
            )
        )
    for j, full_j in ((64, True), (64, False)):
        body = "full" if full_j else "bounded"
        specs.append(
            _forward(
                f"dense_fwd_q8_0_k4096_j{j}_{body}_hoisted",
                ForwardKind.DENSE,
                QuantType.Q8_0,
                j=j,
                blocks_per_weight_row=16,
                full_i=True,
                full_j=full_j,
                hoisted_epilogue=True,
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
                "grouped_fwd_fixed_q8_0_g8_k4096_j64_full_hoisted",
                ForwardKind.FIXED_GROUPED,
                QuantType.Q8_0,
                j=64,
                blocks_per_weight_row=16,
                groups=8,
                hoisted_epilogue=True,
            ),
            _forward(
                "grouped_fwd_fixed_q8_0_g8_k4096_j64_bounded",
                ForwardKind.FIXED_GROUPED,
                QuantType.Q8_0,
                j=64,
                blocks_per_weight_row=16,
                groups=8,
                bounded_tokens=True,
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
                prefetch_activation=quant
                in {QuantType.Q4_K, QuantType.Q5_K, QuantType.IQ2_S},
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
                prefetch_activation=True,
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
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80_pipe1",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                pipeline_depth=1,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80_pipe2",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                pipeline_depth=2,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80_pipe4",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                pipeline_depth=4,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j88",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=88,
                nrows_weight=2048,
                blocks_per_weight_row=16,
            ),
            _forward(
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j128",
                ForwardKind.GROUPED_SERIAL,
                QuantType.IQ2_XXS,
                j=128,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                wide_tile=True,
            ),
            _forward(
                "grouped_fwd_serial_q2_k_n4096_k2048_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q2_K,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                compact_tile=True,
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
                compact_tile=True,
            ),
            _forward(
                "grouped_fwd_serial_q5_k_n2048_k512_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q5_K,
                j=32,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                prefetch_activation=True,
            ),
            _forward(
                "grouped_fwd_serial_q4_k_n2048_k512_j32",
                ForwardKind.GROUPED_SERIAL,
                QuantType.Q4_K,
                j=32,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                prefetch_activation=True,
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
        ("n640k2560", 2560, 640),
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
    specs.append(HIPControlSpec("dense_bwd_split_k_reduce", SplitKReduceConfig()))
    for suffix, n_tiles, k_iteration, m_tiles, slices, full_tiles in (
        ("m32_s2", 4, 16, 1, 2, False),
        ("m64_s4", 4, 16, 1, 4, True),
        ("m128_s16", 4, 32, 2, 16, True),
        ("m256_s16", 4, 32, 4, 16, True),
        ("m512_s32", 4, 32, 4, 32, True),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_lm_head_splitk_{suffix}",
                QuantType.Q8_0,
                n_tiles,
                k_iteration,
                group_m=0,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                full_tiles=full_tiles,
                exact_out_features=129280,
                exact_in_features=4096,
                split_k=slices,
            )
        )
    for suffix, n_tiles, k_iteration, m_tiles, swizzle, pack_q6, slices in (
        ("m64_s2", 2, 64, 1, 16, False, 2),
        ("m128_s40", 4, 32, 2, 8, False, 40),
        ("m256_s40", 4, 32, 2, 8, True, 40),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q6_k_exact_lm_head_splitk_{suffix}",
                QuantType.Q6_K,
                n_tiles,
                k_iteration,
                group_m=0,
                m_tiles_per_wave=m_tiles,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pack_q6_quant_bytes=pack_q6,
                exact_out_features=248320,
                exact_in_features=2048,
                split_k=slices,
            )
        )
    for suffix, slices in (("s2", 2), ("s4", 4)):
        specs.append(
            _dense_backward(
                f"dense_bwd_q5_k_exact_n8192k2048_splitk_{suffix}",
                QuantType.Q5_K,
                8,
                32,
                group_m=0,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
                pack_q5_quant_bytes=True,
                exact_out_features=8192,
                exact_in_features=2048,
                split_k=slices,
            )
        )
    for suffix, n_tiles, k_iteration, m_tiles, slices in (
        ("m64_s2", 4, 32, 1, 2),
        ("m64_s4", 4, 32, 1, 4),
        ("m128_s8", 4, 32, 2, 8),
        ("m128_s16", 4, 32, 2, 16),
        ("m256_s16", 4, 32, 4, 16),
        ("m256_s32", 4, 32, 4, 32),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q5_k_exact_lm_head_splitk_{suffix}",
                QuantType.Q5_K,
                n_tiles,
                k_iteration,
                group_m=0,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                full_tiles=True,
                exact_out_features=248320,
                exact_in_features=2560,
                split_k=slices,
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
            _dense_backward(
                "dense_bwd_q3_k_mt128_nt128_ki32_full_narrow_sw8",
                QuantType.Q3_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
            _dense_backward(
                "dense_bwd_q3_k_mt128_nt128_ki32_full_narrow_sw16",
                QuantType.Q3_K,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                lds_swizzle_chunk=16,
            ),
        ]
    )
    for quant, k, swizzle, pack_q5, pack_q6 in (
        (QuantType.Q4_K, 4096, 0, False, False),
        (QuantType.Q4_K, 2048, 8, False, False),
        (QuantType.Q4_K, 512, 16, False, False),
        (QuantType.Q5_K, 2048, 8, True, False),
        (QuantType.Q5_K, 512, 4, True, False),
        (QuantType.Q6_K, 2048, 8, False, True),
        (QuantType.Q4_K, 4096, 8, False, True),
        (QuantType.Q4_K, 4096, 16, False, True),
        (QuantType.Q6_K, 512, 4, False, True),
        (QuantType.Q6_K, 2048, 8, False, False),
    ):
        name = quant.name.lower()
        suffix = ""
        if quant is QuantType.Q6_K and not pack_q6:
            suffix = "_scalar_extraction"
        elif quant is QuantType.Q4_K and k == 4096 and swizzle:
            suffix = f"_sw{swizzle}"
        specs.append(
            _dense_backward(
                f"dense_bwd_{name}_mt128_nt128_ki32_full_k{k}{suffix}",
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
                pack_q6_quant_bytes=pack_q6,
            )
        )
    for suffix, n_tiles, k_iteration, m_tiles, swizzle, padding, decoder, prefetch in (
        ("k4096", 8, 32, 2, 0, 8, 16, True),
        ("k2048_pad8", 8, 32, 2, 8, 8, 16, True),
        ("k2048_dw0", 8, 32, 2, 8, 0, 0, True),
        ("k2048_noprefetch", 8, 32, 2, 8, 0, 16, False),
        ("k2048_ki64", 8, 64, 2, 16, 0, 16, True),
        ("k2048_nt16", 16, 32, 2, 8, 0, 16, True),
        ("k2048_nt4", 4, 32, 2, 8, 0, 16, True),
        ("k2048_nt2", 2, 32, 2, 8, 0, 16, True),
        ("k2048_nt4_ki64", 4, 64, 2, 16, 0, 16, True),
        ("k2048_nt4_ki64_mw4", 4, 64, 4, 16, 0, 16, True),
        ("k2048_nt8_ki64_mw4", 8, 64, 4, 16, 0, 16, True),
        ("k2048_nt4_mw4", 4, 32, 4, 8, 0, 16, True),
        ("k2048_nt4_ki64_mw8", 4, 64, 8, 16, 0, 16, True),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q6_k_mt128_nt128_ki32_full_{suffix}",
                QuantType.Q6_K,
                n_tiles,
                k_iteration,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=decoder,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=prefetch,
                lds_padding=padding,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pack_q6_quant_bytes=True,
            )
        )
    for suffix, n_tiles, k_iteration, m_tiles, swizzle, group_m, padding in (
        ("k2048", 8, 32, 2, 8, 1, 0),
        ("k2048_nt4", 4, 32, 2, 8, 1, 0),
        ("k2048_nt4_g0", 4, 32, 2, 8, 0, 0),
        ("k2048_nt4_ki64_mw4", 4, 64, 4, 16, 1, 0),
        ("k2048_nt4_ki64_mw2", 4, 64, 2, 16, 1, 0),
        ("k2048_nt8_mw4", 8, 32, 4, 8, 1, 0),
        ("nt4mw2_sw4", 4, 64, 2, 4, 1, 0),
        ("nt4mw2_sw16", 4, 64, 2, 16, 1, 0),
        ("nt4mw2_ki32", 4, 32, 2, 8, 1, 0),
        ("nt4mw1", 4, 64, 1, 16, 1, 0),
        ("nt2mw2", 2, 64, 2, 16, 1, 0),
        ("nt8mw2_ki64", 8, 64, 2, 16, 1, 0),
        ("nt4mw2_pad8", 4, 64, 2, 16, 1, 8),
        ("nt4mw2_pad4", 4, 64, 2, 16, 1, 4),
        ("nt4mw2_pad16", 4, 64, 2, 16, 1, 16),
        ("nt4_pad8", 4, 32, 2, 8, 1, 8),
        ("nt4mw2_ki128_pad8", 4, 128, 2, 16, 1, 8),
        ("nt4mw2_ki128", 4, 128, 2, 16, 1, 0),
        ("nt8_ki32_pad8", 8, 32, 2, 8, 1, 8),
        ("nt8_ki64_pad8", 8, 64, 2, 16, 1, 8),
        ("nt4mw4_ki64_pad8", 4, 64, 4, 16, 1, 8),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q2_0_mt128_nt128_ki32_full_{suffix}",
                QuantType.Q2_0,
                n_tiles,
                k_iteration,
                group_m=group_m,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                lds_padding=padding,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
            )
        )
    for suffix, n_tiles, k_iteration, m_tiles, swizzle, padding in (
        ("pipea_nt4_ki64_pad8", 4, 64, 2, 16, 8),
        ("pipea_nt4_ki64_sw0_pad8", 4, 64, 2, 0, 8),
        ("pipea_nt4_ki64_mw4_pad8", 4, 64, 4, 16, 8),
        ("pipea_nt4_ki64", 4, 64, 2, 16, 0),
        ("pipea_nt4_ki64_sw0", 4, 64, 2, 0, 0),
        ("pipea_nt2_ki64_pad8", 2, 64, 2, 8, 8),
        ("pipea_nt4_ki32", 4, 32, 2, 8, 0),
        ("pipea_nt4_ki64_mw4", 4, 64, 4, 16, 0),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q2_0_mt128_nt128_ki32_full_{suffix}",
                QuantType.Q2_0,
                n_tiles,
                k_iteration,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                lds_padding=padding,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=True,
                prefetch_a_fragments=suffix.startswith("pipea_"),
            )
        )
    for suffix, swizzle, decoder, waves, m_tiles in (
        ("nt4mw2_sw0_pad8", 0, 16, 4, 2),
        ("nt4mw2_sw8_pad8", 8, 16, 4, 2),
        ("nt4mw2_dw8_pad8", 16, 8, 4, 2),
        ("nt4mw2_dw32_pad8", 16, 32, 4, 2),
        ("nt4mw2_aw2_pad8", 16, 16, 2, 2),
        ("nt4mw3_pad8", 16, 16, 4, 3),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_q2_0_mt128_nt128_ki32_full_{suffix}",
                QuantType.Q2_0,
                4,
                64,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=decoder,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                lds_padding=8,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                active_waves=waves,
            )
        )
    for suffix, quant, pipeline, a_prefetch, n_tiles, k_iteration, m_tiles, swizzle in (
        ("q6_k_pipe_nt4_ki64_mw4_sw16", QuantType.Q6_K, True, True, 4, 64, 4, 16),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_{suffix}",
                quant,
                n_tiles,
                k_iteration,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=pipeline,
                prefetch_a_fragments=a_prefetch,
            )
        )
    # Pipelined twins of the deployed K-quant backward tiles: the same
    # geometries with the two-tile stage order, at the deployed swizzle and
    # with the padding that the narrow-result families measured faster.
    for (
        quant,
        name,
        suffix,
        n_tiles,
        k_iteration,
        m_tiles,
        swizzle,
        padding,
        a_prefetch,
    ) in (
        (QuantType.Q3_K, "q3_k", "pipea_nt4_ki64_mw4_sw16", 4, 64, 4, 16, 0, True),
        (QuantType.Q4_K, "q4_k", "pipea_nt4_ki64_mw4_sw16", 4, 64, 4, 16, 0, True),
        (QuantType.Q5_K, "q5_k", "pipea_nt4_ki64_mw4_sw16", 4, 64, 4, 16, 0, True),
        (QuantType.Q5_K, "q5_k", "pipea_nt4_ki64_mw2_sw16", 4, 64, 2, 16, 0, True),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_{name}_mt128_nt128_ki32_full_{suffix}",
                quant,
                n_tiles,
                k_iteration,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                lds_padding=padding,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=True,
                prefetch_a_fragments=a_prefetch,
            )
        )
    for label, out_features, in_features, group_m, swizzle, padding in (
        ("n1024k4096", 1024, 4096, 2, 8, 0),
        ("n1024k4096", 1024, 4096, 2, 0, 8),
        ("n32768k1024", 32768, 1024, 1, 8, 0),
        ("n512k4096", 512, 4096, 2, 8, 0),
        ("n512k4096", 512, 4096, 2, 0, 8),
        ("n2048k4096", 2048, 4096, 2, 8, 0),
        ("n4096k2048", 4096, 2048, 0, 8, 0),
        ("n4096k2048", 4096, 2048, 0, 0, 8),
        ("n4096k8192", 4096, 8192, 0, 8, 0),
        ("n640k2560", 2560, 640, 2, 8, 0),
    ):
        parts = f"_group_m{group_m}" if group_m else ""
        parts += "_padding8" if padding else "_sw8"
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}_g2{parts}_pipe",
                QuantType.Q8_0,
                8,
                32,
                group_m=group_m,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=False,
                full_tiles=True,
                lds_padding=padding,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=True,
                exact_out_features=out_features,
                exact_in_features=in_features,
            )
        )
    # Pipelined split-contraction twins of the deployed head bodies, and
    # prefetched-decode twins of the pipelined projection bodies.
    for prefix, quant, n_tiles, k_iteration, m_tiles, swizzle, slices in (
        ("q5_k_pipesplit_m64_s4", QuantType.Q5_K, 4, 64, 1, 16, 4),
        ("q5_k_pipesplit_m128_s8", QuantType.Q5_K, 4, 32, 2, 8, 8),
        ("q5_k_pipesplit_m256_s32", QuantType.Q5_K, 4, 32, 4, 8, 32),
        ("q8_0_pipesplit_m64_s4", QuantType.Q8_0, 4, 16, 1, 0, 4),
        ("q8_0_pipesplit_m128_s16", QuantType.Q8_0, 4, 32, 2, 0, 16),
        ("q8_0_pipesplit_m256_s16", QuantType.Q8_0, 4, 32, 4, 0, 16),
        ("q8_0_pipesplit_m512_s32", QuantType.Q8_0, 4, 32, 4, 0, 32),
        ("q6_k_pipesplit_m256_s40", QuantType.Q6_K, 4, 32, 2, 8, 40),
        ("q6_k_pipesplit_m64_s2", QuantType.Q6_K, 2, 64, 1, 16, 2),
        ("q6_k_pipesplit_m128_s40", QuantType.Q6_K, 4, 32, 2, 8, 40),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_{prefix}",
                quant,
                n_tiles,
                k_iteration,
                group_m=0,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=True,
                prefetch_packed=True,
                exact_out_features=(
                    248320 if quant in (QuantType.Q5_K, QuantType.Q6_K) else 129280
                ),
                exact_in_features=(
                    2560
                    if quant is QuantType.Q5_K
                    else (2048 if quant is QuantType.Q6_K else 4096)
                ),
                split_k=slices,
            )
        )
    for prefix, quant, n_tiles, m_tiles, swizzle, pack_q5, pack_q6 in (
        (
            "q3_k_pipea_nt4_ki64_mw4_sw16_prefetch",
            QuantType.Q3_K,
            4,
            4,
            16,
            False,
            False,
        ),
        (
            "q4_k_pipea_nt4_ki64_mw4_sw16_prefetch",
            QuantType.Q4_K,
            4,
            4,
            16,
            False,
            False,
        ),
        (
            "q5_k_pipea_nt4_ki64_mw2_sw16_prefetch",
            QuantType.Q5_K,
            4,
            2,
            16,
            True,
            False,
        ),
        (
            "q6_k_pipea_nt4_ki64_mw4_sw16_prefetch",
            QuantType.Q6_K,
            4,
            4,
            16,
            False,
            True,
        ),
    ):
        specs.append(
            _dense_backward(
                f"dense_bwd_{prefix}",
                quant,
                n_tiles,
                64,
                group_m=1,
                m_tiles_per_wave=m_tiles,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                vector_local_load=True,
                prefetch_packed=True,
                lds_swizzle_chunk=swizzle,
                pipeline_tiles=True,
                prefetch_a_fragments=True,
                pack_q5_quant_bytes=pack_q5,
                pack_q6_quant_bytes=pack_q6,
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
    # Exact-dimension twins of the three deployed bodies whose contraction is
    # 2048 or more and which still resolve both bounds at runtime: the language
    # model head at M256, and the two wide Q4_K chunks. Their geometry is
    # copied from the deployed bodies so the only difference is the bound.
    for (
        label,
        quant,
        n_tiles,
        k_iteration,
        group_m,
        m_tiles,
        swizzle,
        pack_q6,
        out_features,
        in_features,
    ) in (
        (
            "q6_k_m256_nt64_ki32_full",
            QuantType.Q6_K,
            4,
            32,
            0,
            2,
            8,
            True,
            248320,
            2048,
        ),
        (
            "q4_k_mt128_nt128_ki32_full_k2048",
            QuantType.Q4_K,
            8,
            32,
            1,
            2,
            8,
            False,
            8192,
            2048,
        ),
        (
            "q4_k_mt128_nt128_ki32_full_k512",
            QuantType.Q4_K,
            8,
            32,
            1,
            2,
            16,
            False,
            2048,
            512,
        ),
    ):
        for exacts in (True,):
            specs.append(
                _dense_backward(
                    f"dense_bwd_{label}_exact",
                    quant,
                    n_tiles,
                    k_iteration,
                    group_m=group_m,
                    m_tiles_per_wave=m_tiles,
                    decoder_width=16 if quant is QuantType.Q4_K else 0,
                    prefetch_local=True,
                    full_tiles=True,
                    prefetch_packed=quant is QuantType.Q4_K,
                    vector_local_load=True,
                    lds_swizzle_chunk=swizzle,
                    pack_q6_quant_bytes=pack_q6,
                    exact_out_features=out_features,
                    exact_in_features=in_features,
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
    for label, out_features, in_features, padding, group_m, swizzle in (
        ("n1024k4096", 1024, 4096, 0, 2, 0),
        ("n1024k4096", 1024, 4096, 8, 2, 0),
        ("n32768k1024", 32768, 1024, 8, 1, 0),
        ("n512k4096", 512, 4096, 8, 2, 0),
        ("n2048k4096", 2048, 4096, 0, 2, 0),
        ("n2048k4096", 2048, 4096, 8, 2, 0),
        ("n4096k2048", 4096, 2048, 0, 2, 0),
        ("n640k2560", 2560, 640, 0, 1, 0),
        ("n640k2560", 2560, 640, 0, 2, 0),
        ("n640k2560", 2560, 640, 8, 2, 0),
        # Swizzle-only twins of the padding-8 bodies: the same tile with the
        # conflict handled by the LDS swizzle instead of two KiB of padding.
        ("n4096k2048", 4096, 2048, 0, 2, 8),
        ("n640k2560", 2560, 640, 0, 2, 8),
    ):
        parts = f"_group_m{group_m}" if group_m else ""
        if padding:
            parts += "_padding8"
        if swizzle:
            parts += f"_sw{swizzle}"
        specs.append(
            _dense_backward(
                f"dense_bwd_q8_0_exact_{label}_g2{parts}",
                QuantType.Q8_0,
                8,
                32,
                group_m=group_m,
                m_tiles_per_wave=2,
                decoder_width=16,
                full_tiles=True,
                lds_padding=padding,
                lds_swizzle_chunk=swizzle,
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
            _grouped_backward(
                "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt64_s3",
                GroupedBackwardKind.Q4_ROW_TASK_N64_S3,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt64_s2",
                GroupedBackwardKind.Q5_ROW_TASK_N64_S2,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt64_s2",
                GroupedBackwardKind.IQ2_S_ROW_TASK_N64_S2,
            ),
            _grouped_backward(
                "grouped_bwd_row_task_q2_k_n4096_k2048_mt128_nt64_s3",
                GroupedBackwardKind.Q2_K_ROW_TASK_N64_S3,
            ),
            _grouped_backward(
                "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64_s2_skip",
                GroupedBackwardKind.Q3_PAIR_STAGED_M128,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64_s2",
                GroupedBackwardKind.IQ2_S_PAIR_STAGED_M128,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_s_n512_k2048_mt256_nt64_s3_skip",
                GroupedBackwardKind.IQ2_S_PAIR_STAGED_M256,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt128_nt64_s2",
                GroupedBackwardKind.IQ2_XXS_PAIR_STAGED_M128,
            ),
            _grouped_backward(
                "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip",
                GroupedBackwardKind.IQ2_XXS_PAIR_STAGED_M128_SKIP,
            ),
            _grouped_backward(
                "grouped_bwd_pair_task_q3_k_n512_k2048_mt128_nt64_s2_skip",
                GroupedBackwardKind.Q3_PAIR_TASK,
            ),
            _grouped_backward(
                "grouped_bwd_pair_task_iq2_s_n512_k2048_mt128_nt64_s2_skip",
                GroupedBackwardKind.IQ2_S_PAIR_TASK,
            ),
            _grouped_backward(
                "grouped_bwd_pair_task_iq2_xxs_n2048_k4096_mt128_nt64_s2_skip",
                GroupedBackwardKind.IQ2_XXS_PAIR_TASK,
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
    if len(specs) != 374:
        raise ValueError(f"historical HIP control inventory has {len(specs)} entries")
    if len(symbols) != len(set(symbols)):
        raise ValueError("HIP control symbols must be unique")
    return tuple(specs)


def render_control(spec: HIPControlSpec) -> str:
    return render_wrapper(spec.symbol, spec.config)
