"""Strict identities for the fixed-group Q8_0 forward experiment."""

from __future__ import annotations

from dataclasses import dataclass

from tools.mmq_quant_formats import QUANT_FORMATS


@dataclass(frozen=True)
class FixedForwardProblem:
    """One exact fixed-group problem, including its non-routed group axis."""

    quant_data_type: str
    tokens: int
    out_features: int
    in_features: int
    groups: int

    @property
    def packed_row_bytes(self) -> int:
        quant = QUANT_FORMATS[self.quant_data_type]
        assert self.in_features % quant.block_values == 0
        return self.in_features // quant.block_values * quant.block_bytes

    @property
    def bytes_per_group(self) -> int:
        return self.out_features * self.packed_row_bytes

    @property
    def total_activation_rows(self) -> int:
        return self.tokens * self.groups
