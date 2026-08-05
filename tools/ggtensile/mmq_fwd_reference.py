"""Independent packed-weight references for MMQ forward campaign tests."""

import struct
from collections.abc import Sequence

Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_VALUES = 32


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
