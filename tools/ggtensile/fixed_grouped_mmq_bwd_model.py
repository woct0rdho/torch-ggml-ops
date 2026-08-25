"""Strict identities for the fixed-group Q8_0 backward experiment."""

from __future__ import annotations

from dataclasses import dataclass

from .quant_formats import BACKWARD_QUANT_FORMATS


@dataclass(frozen=True)
class FixedBackwardProblem:
    quant_data_type: str
    tokens: int
    output_features: int
    input_features: int
    groups: int

    @property
    def packed_row_bytes(self) -> int:
        quant = BACKWARD_QUANT_FORMATS[self.quant_data_type]
        assert self.input_features % quant.block_values == 0
        return self.input_features // quant.block_values * quant.block_bytes

    @property
    def bytes_per_group(self) -> int:
        return self.output_features * self.packed_row_bytes
