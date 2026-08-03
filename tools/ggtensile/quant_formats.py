from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class QuantFormat:
    block_values: int
    block_bytes: int


QUANT_FORMATS: Mapping[str, QuantFormat] = MappingProxyType(
    {
        "Q3_K": QuantFormat(block_values=256, block_bytes=110),
        "Q4_K": QuantFormat(block_values=256, block_bytes=144),
        "Q5_K": QuantFormat(block_values=256, block_bytes=176),
        "Q6_K": QuantFormat(block_values=256, block_bytes=210),
        "Q8_0": QuantFormat(block_values=32, block_bytes=34),
    }
)

Q8_1_F16_D4S4_BLOCK_BYTES = 144
Q8_1_F32_D4_BLOCK_BYTES = 144
