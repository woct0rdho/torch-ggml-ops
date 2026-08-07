"""Shared packed metadata reconstruction for forward lowerings."""

from .kernel_writer_assembly import Assembly
from .mmq_fwd_spec import QuantForwardSemantics


def emit_packed_scale_minimum(
    asm: Assembly,
    group: int,
    metadata: int,
    *,
    semantics: QuantForwardSemantics,
    scale: int,
    minimum: int,
    temporary: int,
) -> None:
    """Reconstruct one packed scale/minimum pair into explicit registers."""
    fields = semantics.packed_scale_minimum_fields(group)
    for destination, parts in ((scale, fields.scale), (minimum, fields.minimum)):
        for index, part in enumerate(parts):
            target = destination if index == 0 else temporary
            asm.inst(
                f"v_bfe_u32 v{target}, v{metadata + part.metadata_word}, "
                f"{part.bit_offset}, {part.bit_count}"
            )
            if index:
                asm.inst(
                    f"v_lshl_or_b32 v{destination}, v{temporary}, "
                    f"{part.destination_shift}, v{destination}"
                )
