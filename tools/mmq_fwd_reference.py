"""Independent packed-weight references for MMQ forward campaign tests."""

import struct
from collections.abc import Sequence

Q3_K_BLOCK_BYTES = 110
Q3_K_BLOCK_VALUES = 256
Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_VALUES = 32


def decode_q3_k_block(block: bytes | bytearray | memoryview) -> tuple[float, ...]:
    """Decode one GGUF Q3_K block without using the production lowering."""
    view = memoryview(block)
    if len(view) != Q3_K_BLOCK_BYTES:
        raise ValueError("Q3_K block must contain exactly 110 bytes")
    hmask = view[0:32]
    payload = view[32:96]
    packed_scales = view[96:108]
    block_scale = struct.unpack_from("<e", view, 108)[0]
    scales = []
    for group in range(16):
        low = (packed_scales[group % 8] >> (4 * (group // 8))) & 0x0F
        high = (packed_scales[8 + group % 4] >> (2 * (group // 4))) & 0x03
        scales.append((low | high << 4) - 32)

    values = []
    for index in range(Q3_K_BLOCK_VALUES):
        low_remainder = index % 128
        low_byte = 32 * (index // 128) + index % 32
        low_shift = 2 * (low_remainder // 32)
        low = (payload[low_byte] >> low_shift) & 0x03
        high = (hmask[index % 32] >> (index // 32)) & 0x01
        quant = (low | high << 2) - 4
        values.append(block_scale * scales[index // 16] * quant)
    return tuple(values)


def decode_q3_k_rows(
    packed: bytes | bytearray | memoryview,
    rows: int,
    values_per_row: int,
) -> tuple[tuple[float, ...], ...]:
    """Decode row-major packed Q3_K weights into logical row tuples."""
    if rows <= 0 or values_per_row <= 0 or values_per_row % Q3_K_BLOCK_VALUES:
        raise ValueError("Q3_K rows require positive K divisible by 256")
    view = memoryview(packed)
    blocks_per_row = values_per_row // Q3_K_BLOCK_VALUES
    expected = rows * blocks_per_row * Q3_K_BLOCK_BYTES
    if len(view) != expected:
        raise ValueError(f"Q3_K payload must contain exactly {expected} bytes")
    decoded = []
    for row in range(rows):
        values: list[float] = []
        row_start = row * blocks_per_row * Q3_K_BLOCK_BYTES
        for block in range(blocks_per_row):
            start = row_start + block * Q3_K_BLOCK_BYTES
            values.extend(decode_q3_k_block(view[start : start + Q3_K_BLOCK_BYTES]))
        decoded.append(tuple(values))
    return tuple(decoded)


def q3_k_matmul_reference(
    activations: Sequence[Sequence[float]],
    packed: bytes | bytearray | memoryview,
    out_features: int,
) -> tuple[tuple[float, ...], ...]:
    """Compute logical output from independently decoded Q3_K rows."""
    if not activations or not activations[0]:
        raise ValueError("activations must be nonempty")
    k = len(activations[0])
    if any(len(row) != k for row in activations):
        raise ValueError("activation rows must have equal width")
    weights = decode_q3_k_rows(packed, out_features, k)
    return tuple(
        tuple(
            sum(value * weight for value, weight in zip(row, weights[column]))
            for column in range(out_features)
        )
        for row in activations
    )


def decode_q8_0_block(block: bytes | bytearray | memoryview) -> tuple[float, ...]:
    """Decode one GGUF Q8_0 block without using the production lowering."""
    view = memoryview(block)
    if len(view) != Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 block must contain exactly 34 bytes")
    scale = struct.unpack_from("<e", view, 0)[0]
    return tuple(
        scale * int.from_bytes(view[2 + index : 3 + index], "little", signed=True)
        for index in range(Q8_0_BLOCK_VALUES)
    )


def decode_q8_0_rows(
    packed: bytes | bytearray | memoryview,
    rows: int,
    values_per_row: int,
) -> tuple[tuple[float, ...], ...]:
    """Decode row-major packed Q8_0 weights into logical row tuples."""
    if rows <= 0 or values_per_row <= 0 or values_per_row % Q8_0_BLOCK_VALUES:
        raise ValueError("Q8_0 rows require positive K divisible by 32")
    view = memoryview(packed)
    blocks_per_row = values_per_row // Q8_0_BLOCK_VALUES
    expected = rows * blocks_per_row * Q8_0_BLOCK_BYTES
    if len(view) != expected:
        raise ValueError(f"Q8_0 payload must contain exactly {expected} bytes")
    decoded: list[tuple[float, ...]] = []
    for row in range(rows):
        values: list[float] = []
        row_start = row * blocks_per_row * Q8_0_BLOCK_BYTES
        for block in range(blocks_per_row):
            start = row_start + block * Q8_0_BLOCK_BYTES
            values.extend(decode_q8_0_block(view[start : start + Q8_0_BLOCK_BYTES]))
        decoded.append(tuple(values))
    return tuple(decoded)


def q8_0_matmul_reference(
    activations: Sequence[Sequence[float]],
    packed: bytes | bytearray | memoryview,
    out_features: int,
) -> tuple[tuple[float, ...], ...]:
    """Compute BF32-style logical output from decoded Q8_0 rows."""
    if not activations or not activations[0]:
        raise ValueError("activations must be nonempty")
    k = len(activations[0])
    if any(len(row) != k for row in activations):
        raise ValueError("activation rows must have equal width")
    weights = decode_q8_0_rows(packed, out_features, k)
    return tuple(
        tuple(
            sum(value * weight for value, weight in zip(row, weights[column]))
            for column in range(out_features)
        )
        for row in activations
    )
