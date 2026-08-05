import math
import struct

import pytest

from tools.ggtensile.mmq_fwd_reference import (
    decode_q8_0_block,
    decode_q8_0_rows,
    q8_0_matmul_reference,
)


def _block(scale: float, values: tuple[int, ...]) -> bytes:
    assert len(values) == 32
    return struct.pack("<e", scale) + bytes(value & 0xFF for value in values)


def test_q8_0_reference_decodes_every_signed_payload_byte() -> None:
    values = tuple(range(-16, 16))
    decoded = decode_q8_0_block(_block(0.5, values))
    assert decoded == tuple(value * 0.5 for value in values)
    extrema = decode_q8_0_block(_block(2.0, (-128, 127) + (0,) * 30))
    assert extrema[:2] == (-256.0, 254.0)
    assert decode_q8_0_block(_block(0.0, values)) == (0.0,) * 32
    with pytest.raises(ValueError, match="exactly 34 bytes"):
        decode_q8_0_block(b"\0" * 33)


def test_q8_0_reference_preserves_block_and_row_boundaries() -> None:
    packed = b"".join(
        (
            _block(1.0, (1,) * 32),
            _block(2.0, (2,) * 32),
            _block(-1.0, (3,) * 32),
            _block(0.5, (-4,) * 32),
        )
    )
    rows = decode_q8_0_rows(packed, 2, 64)
    assert rows[0] == (1.0,) * 32 + (4.0,) * 32
    assert rows[1] == (-3.0,) * 32 + (-2.0,) * 32
    with pytest.raises(ValueError, match="positive K divisible by 32"):
        decode_q8_0_rows(packed, 2, 33)
    with pytest.raises(ValueError, match="exactly 136 bytes"):
        decode_q8_0_rows(packed[:-1], 2, 64)


def test_q8_0_reference_matmul_uses_logical_weight_transpose() -> None:
    packed = _block(0.25, tuple(range(32))) + _block(0.5, tuple(range(-16, 16)))
    activations = ((1.0,) * 32, tuple(float(index) for index in range(32)))
    output = q8_0_matmul_reference(activations, packed, 2)
    assert output[0] == (sum(range(32)) * 0.25, sum(range(-16, 16)) * 0.5)
    assert math.isclose(
        output[1][0],
        sum(index * index for index in range(32)) * 0.25,
    )
    with pytest.raises(ValueError, match="equal width"):
        q8_0_matmul_reference(((1.0,), (1.0, 2.0)), packed, 2)
