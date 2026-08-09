"""Backward-specific assembly emission state."""

from enum import Enum

from .kernel_writer_assembly import Assembly


class BackwardKernelWriterError(RuntimeError):
    pass


class BackwardDiagnosticMode(str, Enum):
    WMMA_FLOOR = "wmma_floor"
    DECODE_FLOOR = "decode_floor"


class PendingZeroPairableOp(str, Enum):
    ADD_NC_U32 = "add_nc_u32"
    LSHLREV_B32 = "lshlrev_b32"
    AND_B32 = "and_b32"


class _Assembly(Assembly):
    def __init__(self) -> None:
        super().__init__()
        self._pending_zero_moves: dict[int, list[int]] = {0: [], 1: []}

    def emit_pairable_with_pending_zero(
        self,
        opcode: PendingZeroPairableOp,
        destination: int,
        source0: str | int,
        source1: int,
    ) -> None:
        text = f"v_{opcode.value} v{destination}, {source0}, v{source1}"
        x_parity = 1 - (destination & 1)
        if not self._pending_zero_moves[x_parity]:
            self.inst(text)
            return
        x_destination = self._pending_zero_moves[x_parity].pop()
        self.inst(
            f"v_dual_mov_b32 v{x_destination}, 0 :: "
            f"v_dual_{opcode.value} v{destination}, {source0}, v{source1}"
        )

    def defer_zero_moves(self, registers: range) -> None:
        if any(self._pending_zero_moves.values()):
            raise BackwardKernelWriterError("cannot nest deferred VGPR zero fills")
        for register in reversed(registers):
            self._pending_zero_moves[register & 1].append(register)

    def flush_zero_moves(self) -> None:
        for parity in (0, 1):
            while self._pending_zero_moves[parity]:
                register = self._pending_zero_moves[parity].pop()
                self.inst(f"v_mov_b32 v{register}, 0")
