import math
import struct

import pytest

from tools.ggtensile.mmq_fwd_reference import (
    decode_q3_k_block,
    decode_q3_k_rows,
    decode_q8_0_block,
    decode_q8_0_rows,
    q3_k_matmul_reference,
    q8_0_matmul_reference,
)


def _block(scale: float, values: tuple[int, ...]) -> bytes:
    assert len(values) == 32
    return struct.pack("<e", scale) + bytes(value & 0xFF for value in values)


def _q3_block(
    block_scale: float,
    scales: tuple[int, ...],
    values: tuple[int, ...],
) -> bytes:
    assert len(scales) == 16
    assert len(values) == 256
    hmask = bytearray(32)
    payload = bytearray(64)
    packed_scales = bytearray(12)
    for index, value in enumerate(values):
        assert -4 <= value <= 3
        code = value + 4
        hmask[index % 32] |= (code >> 2) << (index // 32)
        low_remainder = index % 128
        low_byte = 32 * (index // 128) + index % 32
        payload[low_byte] |= (code & 0x03) << (2 * (low_remainder // 32))
    for group, scale in enumerate(scales):
        assert -32 <= scale <= 31
        code = scale + 32
        packed_scales[group % 8] |= (code & 0x0F) << (4 * (group // 8))
        packed_scales[8 + group % 4] |= (code >> 4) << (2 * (group // 4))
    return bytes(hmask + payload + packed_scales) + struct.pack("<e", block_scale)


def test_q3_k_reference_decodes_payload_and_signed_scale_planes() -> None:
    scales = tuple(range(-8, 8))
    values = tuple(index % 8 - 4 for index in range(256))
    decoded = decode_q3_k_block(_q3_block(0.5, scales, values))
    assert decoded == tuple(
        0.5 * scales[index // 16] * values[index] for index in range(256)
    )
    extrema = decode_q3_k_block(_q3_block(1.0, (-32, 31) + (1,) * 14, (-4, 3) * 128))
    assert extrema[0] == 128.0
    assert extrema[16] == -124.0
    with pytest.raises(ValueError, match="exactly 110 bytes"):
        decode_q3_k_block(b"\0" * 109)


def test_q3_k_reference_preserves_rows_and_weight_transpose() -> None:
    positive = _q3_block(1.0, (1,) * 16, (1,) * 256)
    negative = _q3_block(0.5, (-2,) * 16, (-1,) * 256)
    rows = decode_q3_k_rows(positive + negative, 2, 256)
    assert rows[0] == (1.0,) * 256
    assert rows[1] == (1.0,) * 256
    output = q3_k_matmul_reference(((1.0,) * 256,), positive + negative, 2)
    assert output == ((256.0, 256.0),)
    with pytest.raises(ValueError, match="positive K divisible by 256"):
        decode_q3_k_rows(positive, 1, 128)
    with pytest.raises(ValueError, match="exactly 220 bytes"):
        decode_q3_k_rows((positive + negative)[:-1], 2, 256)
    with pytest.raises(ValueError, match="equal width"):
        q3_k_matmul_reference(((1.0,), (1.0, 2.0)), positive + negative, 2)


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
