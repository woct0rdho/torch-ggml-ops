from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

Q8_1_F16_D2S6_BLOCK_BYTES = 144
Q8_1_F16_D4S4_BLOCK_BYTES = 144
Q8_1_F32_D4_BLOCK_BYTES = 144
Q8_1_D4_BLOCK_VALUES = 128


@dataclass(frozen=True)
class QuantFormat:
    block_values: int
    block_bytes: int
    activation_layout: str
    activation_block_bytes: int
    wmma_clamp: bool
    weight_decode: str
    scale_arithmetic: str
    arithmetic_contract: str


Q2_K_FORMAT = QuantFormat(
    block_values=256,
    block_bytes=84,
    activation_layout="F16_D2S6",
    activation_block_bytes=Q8_1_F16_D2S6_BLOCK_BYTES,
    wmma_clamp=True,
    weight_decode="DirectTwoBitNibbleScaleMinimum",
    scale_arithmetic="FP16",
    arithmetic_contract="UnsignedQ2IntegerWmmaFP16ScaleMinimumCorrection",
)

IQ2_S_FORMAT = QuantFormat(
    block_values=256,
    block_bytes=82,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="DirectIQ2SGridSigned",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedIQ2SInt8ScaleIntegerWmmaF32Correction",
)

IQ2_XXS_FORMAT = QuantFormat(
    block_values=256,
    block_bytes=66,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="DirectIQ2XXSGridSigned",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedIQ2XXSInt8ScaleIntegerWmmaF32Correction",
)


QUANT_FORMATS: Mapping[str, QuantFormat] = MappingProxyType(
    {
        "Q3_K": QuantFormat(
            block_values=256,
            block_bytes=110,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            wmma_clamp=False,
            weight_decode="DirectQ3Signed",
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contract="SignedQ3Int8ScaleIntegerWmmaF32Correction",
        ),
        "Q4_K": QuantFormat(
            block_values=256,
            block_bytes=144,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            wmma_clamp=True,
            weight_decode="DirectNibble",
            scale_arithmetic="FP16",
            arithmetic_contract="SignedKQuantIntegerWmmaFP16ScaleMinimumCorrection",
        ),
        "Q5_K": QuantFormat(
            block_values=256,
            block_bytes=176,
            activation_layout="F16_D4S4",
            activation_block_bytes=Q8_1_F16_D4S4_BLOCK_BYTES,
            wmma_clamp=True,
            weight_decode="DirectNibbleHighBit",
            scale_arithmetic="FP16",
            arithmetic_contract="SignedKQuantIntegerWmmaFP16ScaleMinimumCorrection",
        ),
        "Q6_K": QuantFormat(
            block_values=256,
            block_bytes=210,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            wmma_clamp=False,
            weight_decode="DirectQ6Signed",
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contract="SignedQ6Int8ScaleIntegerWmmaF32Correction",
        ),
        "Q8_0": QuantFormat(
            block_values=32,
            block_bytes=34,
            activation_layout="F32_D4",
            activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
            wmma_clamp=False,
            weight_decode="DirectSignedInt8",
            scale_arithmetic="Int32ScaleF32",
            arithmetic_contract="SignedQ8Int8ScaleIntegerWmmaF32Correction",
        ),
    }
)


# The two simple dense-only types carry no codebook and only one metadata
# value per block, so their levels are the whole decode. They are listed in this
# superset rather than in QUANT_FORMATS because that table also drives the
# GGTensile forward type registry, which does not know them yet, while the
# correctness machinery reads packed shapes from here.
Q2_0_FORMAT = QuantFormat(
    block_values=64,
    block_bytes=18,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="DirectTwoBitConsecutiveLevelMinusOne",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedQ2Int8ScaleIntegerWmmaF32Correction",
)

Q4_0_FORMAT = QuantFormat(
    block_values=32,
    block_bytes=18,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="DirectNibblePlanePairLevelMinusEight",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedQ4Int8ScaleIntegerWmmaF32Correction",
)

Q5_0_FORMAT = QuantFormat(
    block_values=32,
    block_bytes=22,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="DirectNibblePlanePairFifthBitLevelMinusSixteen",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedQ5Int8ScaleIntegerWmmaF32Correction",
)

IQ4_NL_FORMAT = QuantFormat(
    block_values=32,
    block_bytes=18,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="NonlinearFourBitCodebook",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedCodebookInt8ScaleIntegerWmmaF32Correction",
)

IQ4_XS_FORMAT = QuantFormat(
    block_values=256,
    block_bytes=136,
    activation_layout="F32_D4",
    activation_block_bytes=Q8_1_F32_D4_BLOCK_BYTES,
    wmma_clamp=False,
    weight_decode="NonlinearFourBitCodebookSixBitSubblockScales",
    scale_arithmetic="Int32ScaleF32",
    arithmetic_contract="SignedCodebookInt8ScaleIntegerWmmaF32Correction",
)

BACKWARD_QUANT_FORMATS: Mapping[str, QuantFormat] = MappingProxyType(
    {
        **QUANT_FORMATS,
        "Q2_K": Q2_K_FORMAT,
        "IQ2_S": IQ2_S_FORMAT,
        "IQ2_XXS": IQ2_XXS_FORMAT,
        "Q2_0": Q2_0_FORMAT,
        "Q4_0": Q4_0_FORMAT,
        "Q5_0": Q5_0_FORMAT,
        "IQ4_NL": IQ4_NL_FORMAT,
        "IQ4_XS": IQ4_XS_FORMAT,
    }
)


GROUPED_QUANT_FORMATS: Mapping[str, QuantFormat] = MappingProxyType(
    {
        **BACKWARD_QUANT_FORMATS,
        "IQ2_S": IQ2_S_FORMAT,
        "IQ2_XXS": IQ2_XXS_FORMAT,
    }
)
