"""Shared matrix-instruction emission for MMQ forward lowerings."""

from .kernel_writer_assembly import Assembly


def emit_signed_i8_wmma(
    asm: Assembly,
    *,
    destination: int,
    weight: int,
    activation: int,
    accumulator: int,
    clamp: bool,
) -> None:
    """Emit one signed int8 16x16x16 WMMA without changing ownership."""
    clamp_suffix = " clamp" if clamp else ""
    asm.inst(
        f"v_wmma_i32_16x16x16_iu8 v[{destination}:{destination + 7}], "
        f"v[{weight}:{weight + 3}], "
        f"v[{activation}:{activation + 3}], "
        f"v[{accumulator}:{accumulator + 7}] neg_lo:[1,1,0]{clamp_suffix}"
    )
