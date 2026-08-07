from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
