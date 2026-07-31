import contextlib
import hashlib
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .model import SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class KernelWriterError(RuntimeError):
    pass


class DiagnosticMode(str, Enum):
    WMMA_FLOOR = "wmma_floor"
    DECODE_FLOOR = "decode_floor"


@dataclass(frozen=True)
class RegisterLayout:
    accum: int
    valu_a: int
    valu_b: int
    global_read_b: int
    quant_dm: int
    quant_scale: int
    lds_address: int
    address: int
    temporary: int
    serial: int
    total_vgprs: int
    kernarg: int
    loop_counter: int
    block_offset: int
    input_half: int
    scalar_temporary: int
    total_sgprs: int


class _Assembly:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._pending_zero_moves: dict[int, list[int]] = {0: [], 1: []}

    def comment(self, text: str) -> None:
        self.lines.append(f"// {text}")

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def inst(self, text: str, comment: str = "") -> None:
        dual = self._dual_with_pending_zero(text)
        if dual is not None:
            text = dual
        suffix = f" // {comment}" if comment else ""
        self.lines.append(f"  {text}{suffix}")

    def defer_zero_moves(self, registers: range) -> None:
        if any(self._pending_zero_moves.values()):
            raise KernelWriterError("cannot nest deferred VGPR zero fills")
        for register in reversed(registers):
            self._pending_zero_moves[register & 1].append(register)

    def flush_zero_moves(self) -> None:
        for parity in (0, 1):
            while self._pending_zero_moves[parity]:
                register = self._pending_zero_moves[parity].pop()
                self.inst(f"v_mov_b32 v{register}, 0")

    def _dual_with_pending_zero(self, text: str) -> str | None:
        match = re.fullmatch(
            r"v_(add_nc_u32|lshlrev_b32|and_b32) v(\d+), ([^,]+), (v\d+)",
            text,
        )
        if match is None:
            return None
        y_destination = int(match.group(2))
        x_parity = 1 - (y_destination & 1)
        if not self._pending_zero_moves[x_parity]:
            return None
        x_destination = self._pending_zero_moves[x_parity].pop()
        y_instruction = (
            f"v_dual_{match.group(1)} v{y_destination}, "
            f"{match.group(3)}, {match.group(4)}"
        )
        return f"v_dual_mov_b32 v{x_destination}, 0 :: {y_instruction}"

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class KernelWriterAssembly:
    """Emit an exact gfx1151 dense-MMQ backward solution."""

    def __init__(
        self,
        solution_key: SolutionKey,
        toolchain: Toolchain,
        *,
        diagnostic_mode: DiagnosticMode | None = None,
    ) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise KernelWriterError(f"solution rejected: {details}")
        self.solution_key = solution_key
        self.toolchain = toolchain
        self.diagnostic_mode = diagnostic_mode
        if diagnostic_mode is not None and solution_key.solution.one_lds_buffer != 0:
            raise KernelWriterError("lower-bound diagnostics require 1LDSBuffer=0")
        self.registers = self._allocate_registers()

    def write(self, output: Path) -> str:
        source = self.source()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(output)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def source(self) -> str:
        try:
            import rocisa
            from rocisa import code
            from rocisa.enum import SignatureValueKind as SVK
        except ImportError as error:
            raise KernelWriterError(
                "rocisa is required to generate assembly"
            ) from error

        global_isa = rocisa.rocIsa.getInstance()
        with tempfile.TemporaryDirectory(prefix="ggtensile-rocisa-") as temp:
            with contextlib.chdir(temp):
                global_isa.init(
                    self.solution_key.solution.isa,
                    str(self.toolchain.assembler),
                    False,
                )
        global_isa.setKernel(
            self.solution_key.solution.isa,
            self.solution_key.solution.wavefront_size,
        )

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=self.solution_key.solution.lds_num_bytes,
            sgprWorkGroup=(1, 1, 1),
            vgprWorkItem=1,
            flatWorkGroupSize=self.solution_key.solution.num_threads,
            totalVgprs=self.registers.total_vgprs,
            totalAgprs=0,
            totalSgprs=self.registers.total_sgprs,
        )
        description = (
            f"GGTensile {self.solution_key.problem_type.quant_data_type} "
            "dense MMQ backward"
        )
        if self.diagnostic_mode is not None:
            description += f" {self.diagnostic_mode.value} diagnostic"
        signature.addDescriptionTopic(description)
        signature.addArg("grad_output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("grad_input", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("rows", SVK.SIG_VALUE, "u32")
        signature.addArg("out_features", SVK.SIG_VALUE, "u32")
        signature.addArg("in_features", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileKernel")
        module.add(signature)
        module.add(code.TextBlock(self._body()))
        return str(module)

    def _decoder_rows(self) -> int:
        solution = self.solution_key.solution
        decoder_threads = min(solution.num_threads, 128)
        return (
            solution.depth_u
            * solution.macro_tile1
            // (decoder_threads * solution.decoder_width)
        )

    def _quant_type(self) -> str:
        return self.solution_key.problem_type.quant_data_type

    def _q3_k_full_vopd_decode(self) -> bool:
        size = self.solution_key.problem_size
        return (
            self._quant_type() == "Q3_K"
            and self.solution_key.solution.q3_k_extraction == "packed"
            and self._decoder_rows() <= 2
            and not (
                size.n == 2048
                and size.k == 512
                and (
                    (
                        size.m != 8192
                        and self.solution_key.solution.macro_tile1 != 64
                    )
                    or (
                        size.m == 8192
                        and self.solution_key.solution.macro_tile1 == 64
                    )
                )
            )
        )

    def _payload_register_count(self) -> int:
        return 4 if self._quant_type() == "Q4_K" else 8

    def _uses_extended_a_pointer_state(self) -> bool:
        solution = self.solution_key.solution
        return (
            solution.schedule_iter_alg in (4, 5)
            and solution.matrix_instruction[5] > 2
        )

    def _lds_address_register(self) -> int:
        if self._uses_extended_a_pointer_state():
            return (
                self.registers.address
                + 4
                + self.solution_key.solution.matrix_instruction[5]
            )
        return self.registers.address + 6

    def _quant_shift_register(self) -> int:
        return self._lds_address_register() + 1

    def _q3_low_shift_register(self) -> int:
        if not self._uses_extended_a_pointer_state():
            return self.registers.address + 2
        return self._quant_shift_register() + 1

    def _address_register_count(self) -> int:
        if not self._uses_extended_a_pointer_state():
            return 8
        state_registers = 6 + self.solution_key.solution.matrix_instruction[5]
        q3_extra = 2 * int(self._quant_type() == "Q3_K")
        return state_registers + q3_extra

    def _packed_block_bytes(self) -> int:
        return {
            "Q3_K": 110,
            "Q4_K": 144,
            "Q5_K": 176,
        }[self._quant_type()]

    def _packed_load_count(self) -> int:
        if self._quant_type() == "Q4_K":
            return 5 * self._decoder_rows()
        if self._quant_type() == "Q3_K":
            return 5 * self._decoder_rows()
        return (
            3 if self.solution_key.solution.q5_k_metadata_vector_load else 6
        ) * self._decoder_rows()

    def _lds_row_stride_bytes(self) -> int:
        solution = self.solution_key.solution
        return 2 * (solution.depth_u + solution.lds_pad_b)

    def _allocate_registers(self) -> RegisterLayout:
        from rocisa.enum import RegisterType
        from rocisa.register import RegisterPool

        vgprs = RegisterPool(256, RegisterType.Vgpr, True)
        vgprs.addRange(0, 255, "GGTensile VGPRs")
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        decoder_rows = self._decoder_rows()
        accum = vgprs.checkOutAligned(8 * m_tiles * n_tiles, 8, "accumulators")
        valu_a = vgprs.checkOutAligned(
            8 * m_tiles * self.solution_key.solution.prefetch_global_read,
            4,
            "ValuA",
        )
        valu_b = vgprs.checkOutAligned(
            16 * self.solution_key.solution.prefetch_local_read,
            4,
            "ValuB",
        )
        global_read_b = vgprs.checkOutAligned(
            self._payload_register_count() * decoder_rows,
            4,
            "GlobalReadB payload planes",
        )
        if self._quant_type() == "Q3_K":
            quant_dm = vgprs.checkOut(decoder_rows, "Q3_K d")
            quant_scale = vgprs.checkOut(2 * decoder_rows, "Q3_K scales")
        elif (
            self._quant_type() == "Q5_K"
            and self.solution_key.solution.q5_k_metadata_vector_load
        ):
            quant_dm = vgprs.checkOut(4 * decoder_rows, "Q5_K metadata vector")
            quant_scale = quant_dm + 1
        else:
            quant_dm = vgprs.checkOut(decoder_rows, f"{self._quant_type()} d/m")
            quant_scale = vgprs.checkOut(
                3 * decoder_rows, f"{self._quant_type()} scale/min"
            )
        lds_address = -1
        swizzle = self.solution_key.solution.lds_swizzle_chunk_b
        if swizzle:
            lds_address = vgprs.checkOut(32 // swizzle, "swizzled LDS addresses")
        address_count = self._address_register_count()
        address = vgprs.checkOutAligned(address_count, 2, "addresses")
        temporary_count = max(
            11
            if self._quant_type() == "Q3_K"
            and self.solution_key.solution.q3_k_extraction == "packed"
            else 7,
            3 + 2 * decoder_rows,
        )
        temporary = vgprs.checkOut(temporary_count, "temporaries")
        serial = vgprs.checkOut(1, "Serial")

        sgprs = RegisterPool(64, RegisterType.Sgpr, True)
        sgprs.addRange(5, 63, "GGTensile SGPRs")
        kernarg = sgprs.checkOutAligned(6, 2, "loaded pointer kernargs")
        loop_counter = sgprs.checkOut(1, "LoopCounter")
        block_offset = sgprs.checkOut(1, "PackedBlockOffset")
        input_half = sgprs.checkOut(1, "InputHalf")
        scalar_temporary = sgprs.checkOut(2, "scalar temporaries")

        return RegisterLayout(
            accum=accum,
            valu_a=valu_a,
            valu_b=valu_b,
            global_read_b=global_read_b,
            quant_dm=quant_dm,
            quant_scale=quant_scale,
            lds_address=lds_address,
            address=address,
            temporary=temporary,
            serial=serial,
            total_vgprs=max(
                serial + 1,
                temporary + temporary_count,
                address + address_count,
            ),
            kernarg=kernarg,
            loop_counter=loop_counter,
            block_offset=block_offset,
            input_half=input_half,
            scalar_temporary=scalar_temporary,
            total_sgprs=scalar_temporary + 2,
        )

    def _body(self) -> str:
        asm = _Assembly()
        r = self.registers
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name

        asm.comment("Flatten gfx11 packed workitem X/Y before v0 becomes C storage.")
        asm.inst(f"v_bfe_u32 v{r.serial}, v0, 10, 10")
        asm.inst(f"v_lshlrev_b32 v{r.serial}, 5, v{r.serial}")
        asm.inst(f"v_and_b32 v{r.temporary}, 0x3ff, v0")
        asm.inst(f"v_add_nc_u32 v{r.serial}, v{r.serial}, v{r.temporary}")
        group_m = self.solution_key.solution.work_group_mapping
        asm.comment("Map grouped M launch coordinates to the logical M tile.")
        if group_m == 1:
            asm.inst("s_mov_b32 s2, s4")
        else:
            asm.inst(f"s_mul_i32 s4, s4, {group_m}")
            asm.inst("s_add_u32 s2, s4, s2")
        kernarg = r.kernarg
        asm.inst(f"s_load_dwordx2 s[{kernarg}:{kernarg + 1}], s[0:1], 0x0")
        asm.inst(f"s_load_dwordx2 s[{kernarg + 2}:{kernarg + 3}], s[0:1], 0x8")
        asm.inst(f"s_load_dwordx2 s[{kernarg + 4}:{kernarg + 5}], s[0:1], 0x10")
        asm.inst("s_waitcnt lgkmcnt(0)")
        accumulator_count = (
            8
            * self.solution_key.solution.matrix_instruction[5]
            * self.solution_key.solution.matrix_instruction[6]
        )
        if self.diagnostic_mode != DiagnosticMode.DECODE_FLOOR:
            asm.comment(
                "Pair accumulator zeroing with independent pre-loop address VALU."
            )
            asm.defer_zero_moves(range(r.accum, r.accum + accumulator_count))

        n_per_block = self.solution_key.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_block
        tile_shift = tiles_per_weight_block.bit_length() - 1
        n_shift = n_per_block.bit_length() - 1
        asm.comment("Static packed-row and input-half coordinates.")
        asm.inst(f"s_lshr_b32 s{r.block_offset}, s3, {tile_shift}")
        asm.inst(
            f"s_mul_i32 s{r.block_offset}, s{r.block_offset}, "
            f"{self._packed_block_bytes()}"
        )
        asm.inst(f"s_lshr_b32 s{r.input_half}, s3, {tile_shift - 1}")
        asm.inst(f"s_and_b32 s{r.input_half}, s{r.input_half}, 1")
        asm.inst(f"s_and_b32 s{r.scalar_temporary}, s3, {tiles_per_weight_block - 1}")
        quant_tile_shift = n_shift - 1 if n_per_block == 128 else n_shift
        asm.inst(
            f"s_lshl_b32 s{r.scalar_temporary}, s{r.scalar_temporary}, "
            f"{quant_tile_shift}"
        )
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")

        self._emit_static_thread_coordinates(asm)
        asm.flush_zero_moves()
        store_output = True
        if self.diagnostic_mode == DiagnosticMode.WMMA_FLOOR:
            self._emit_wmma_floor(asm)
        elif self.diagnostic_mode == DiagnosticMode.DECODE_FLOOR:
            self._emit_decode_floor(asm)
            store_output = False
        elif self.solution_key.solution.one_lds_buffer == 0:
            self._emit_decoded_b_pipeline(asm)
        elif self.solution_key.solution.prefetch_packed_weight_next:
            self._emit_packed_weight_pipeline(asm)
        else:
            asm.label(".LDepthULoop")
            if self.solution_key.solution.num_threads > 128:
                asm.comment("Only the first four waves cooperatively decode B.")
                asm.inst(f"v_readfirstlane_b32 s{r.scalar_temporary + 1}, v{r.serial}")
                asm.inst(f"s_cmp_lt_u32 s{r.scalar_temporary + 1}, 128")
                asm.inst("s_cbranch_scc0 .LDecodeReady")
            schedule = self.solution_key.solution.schedule_iter_alg
            self._emit_quant_global_reads(asm, wait_for_reads=schedule not in (4, 5))
            if schedule in (4, 5):
                self._emit_first_a_global_reads(asm)
                a_load_count = (
                    2
                    * self.solution_key.solution.matrix_instruction[5]
                    * self.solution_key.solution.prefetch_global_read
                )
                asm.inst(f"s_waitcnt vmcnt({a_load_count})")
            self._emit_packed_weight_lane_share(asm)
            self._emit_quant_decode(asm)
            if self.solution_key.solution.num_threads > 128:
                asm.label(".LDecodeReady")
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_wmma(asm)
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            asm.inst(
                f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, "
                f"{self.solution_key.solution.depth_u}"
            )
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
            asm.inst("s_cbranch_scc1 .LDepthULoop")
        if store_output:
            self._emit_store(asm)
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _emit_wmma_floor(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        solution = self.solution_key.solution
        a_load_count = (
            2 * solution.matrix_instruction[5] * solution.prefetch_global_read
        )

        asm.comment("Prime one decoded-B tile for the WMMA/A/LDS lower bound.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_quant_decode(asm, label_suffix="WmmaFloorPrime")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

        asm.label(".LWmmaFloorLoop")
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_wmma(asm, pipeline=True)
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst("s_cbranch_scc0 .LWmmaFloorDone")
        self._emit_first_a_global_reads(asm)
        asm.inst("s_branch .LWmmaFloorLoop")
        asm.label(".LWmmaFloorDone")

    def _emit_decode_floor(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        solution = self.solution_key.solution
        asm.comment(
            "Measure packed Q4_K reads, decode, LDS stores, and synchronization."
        )
        asm.label(".LDecodeFloorLoop")
        self._emit_quant_global_reads(asm)
        self._emit_quant_decode(asm, label_suffix="DecodeFloor")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst("s_cbranch_scc1 .LDecodeFloorLoop")

    def _emit_packed_weight_pipeline(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        a_load_count = (
            2
            * self.solution_key.solution.matrix_instruction[5]
            * self.solution_key.solution.prefetch_global_read
        )

        asm.comment("Prime decoded B and both A fragments.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(asm, label_suffix="Initial")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")

        asm.label(".LPackedDepthULoop")
        asm.inst("s_waitcnt vmcnt(0)", "current A before next packed reads")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, 32")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst("s_cbranch_scc0 .LPackedNoPrefetch")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        asm.label(".LPackedNoPrefetch")

        self._emit_wmma(asm)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst("s_cbranch_scc0 .LPackedDepthUDone")

        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_packed_weight_lane_share(asm)
        self._emit_quant_decode(asm, label_suffix="Steady")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst("s_branch .LPackedDepthULoop")
        asm.label(".LPackedDepthUDone")

    def _emit_decoded_b_pipeline(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        solution = self.solution_key.solution
        a_load_count = (
            2 * solution.matrix_instruction[5] * solution.prefetch_global_read
        )
        packed_load_count = self._packed_load_count()

        asm.comment("Prime decoded B0 and current-tile A.")
        self._emit_quant_global_reads(asm, wait_for_reads=False)
        self._emit_first_a_global_reads(asm)
        asm.inst(f"s_waitcnt vmcnt({a_load_count})")
        self._emit_quant_decode(asm, label_suffix="PipelinePrime")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        self._emit_toggle_lds_write_buffer(asm)

        if size.k > solution.depth_u:
            asm.label(".LDecodedBPipelineLoop")
            asm.inst(
                f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, {solution.depth_u}"
            )

            asm.comment("Issue next packed tile behind current-tile A.")
            self._emit_quant_global_reads(asm, wait_for_reads=False)
            asm.inst(f"s_waitcnt vmcnt({packed_load_count})")

            def after_wmma_pair(k_tile: int, n_tile: int) -> None:
                pair = (k_tile // 16) * (solution.matrix_instruction[6] // 2)
                pair += n_tile // 2
                if pair == 0:
                    asm.inst("s_waitcnt vmcnt(0)")
                    self._emit_quant_decode_prepare(asm, label_suffix="PipelineSteady")
                    self._emit_pipeline_lds_read_addresses(asm, 0)
                    if (
                        self._quant_type() == "Q5_K"
                        and solution.q5_k_nibble_shift_hoist
                    ):
                        self._emit_q5_k_nibble_shift(asm)
                self._emit_quant_decode_chunk(asm, pair)
                if pair == 7:
                    self._emit_next_a_half(asm, 0)
                    self._emit_next_a_half(asm, 1)

            self._emit_wmma(asm, pipeline=True, after_pair=after_wmma_pair)
            asm.inst("s_waitcnt lgkmcnt(0)")
            asm.inst("s_barrier")
            self._emit_swap_lds_buffers(asm)
            asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k - solution.depth_u}")
            asm.inst("s_cbranch_scc1 .LDecodedBPipelineLoop")

        asm.label(".LDecodedBPipelineFinal")
        asm.inst("s_waitcnt vmcnt(0)")
        self._emit_wmma(asm, pipeline=True)

    def _emit_next_a_half(self, asm: _Assembly, k_half: int) -> None:
        r = self.registers
        solution = self.solution_key.solution
        m_tiles = solution.matrix_instruction[5]
        pointers = tuple(r.global_read_b + m_tile for m_tile in range(m_tiles))
        if k_half == 0:
            asm.comment("Reload next-tile A0 after current A0 becomes dead.")
            asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
            for m_tile, pointer in enumerate(pointers):
                asm.inst(
                    f"v_add_nc_u32 v{pointer}, s{r.scalar_temporary + 1}, "
                    f"v{r.address + 4 + m_tile}"
                )
        else:
            asm.comment("Reload next-tile A1 after current A1 becomes dead.")
            for pointer in pointers:
                asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
        for m_tile, pointer in enumerate(pointers):
            valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
            asm.inst(
                f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}]"
            )
            asm.inst(
                f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}] offset:16"
            )

    def _emit_toggle_lds_write_buffer(self, asm: _Assembly) -> None:
        r = self.registers
        buffer_bytes = self.solution_key.solution.lds_num_bytes // 2
        for register in range(r.lds_address, r.lds_address + 4):
            asm.inst(f"v_xor_b32 v{register}, {buffer_bytes}, v{register}")

    def _emit_swap_lds_buffers(self, asm: _Assembly) -> None:
        buffer_bytes = self.solution_key.solution.lds_num_bytes // 2
        lds_address = self._lds_address_register()
        asm.inst(f"v_xor_b32 v{lds_address}, {buffer_bytes}, v{lds_address}")
        self._emit_toggle_lds_write_buffer(asm)

    def _emit_pipeline_lds_read_addresses(self, asm: _Assembly, k_tile: int) -> None:
        r = self.registers
        row_stride = self._lds_row_stride_bytes()
        first = r.quant_dm
        second = first + 1
        temporary = r.quant_scale
        lds_address = self._lds_address_register()
        asm.inst(f"v_and_b32 v{temporary}, 15, v{r.serial}")
        self._emit_scale_u32(asm, temporary, row_stride, temporary)
        asm.inst(f"v_add_nc_u32 v{temporary}, v{lds_address}, v{temporary}")
        asm.inst(f"v_and_b32 v{temporary + 1}, 3, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{temporary + 1}, 4, v{temporary + 1}")
        asm.inst(f"v_add_nc_u32 v{first}, v{temporary}, v{temporary + 1}")
        if k_tile % 32:
            asm.inst(f"v_xor_b32 v{first}, 32, v{first}")
        asm.inst(f"v_xor_b32 v{second}, 16, v{first}")

    def _emit_static_thread_coordinates(self, asm: _Assembly) -> None:
        r = self.registers
        a = r.address
        t = r.temporary
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        k_shift = n_tiles.bit_length() - 1
        row_stride = self._lds_row_stride_bytes()
        segment_stride = 16 * row_stride
        lds_address = self._lds_address_register()
        quant_shift = self._quant_shift_register()
        asm.comment("State registers after the A row pointers hold LDS and quant state.")
        asm.inst(f"v_and_b32 v{lds_address}, {n_tiles - 1}, v{r.serial}")
        self._emit_scale_u32(asm, lds_address, segment_stride, lds_address)
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 1, v{t}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, v{lds_address}, v{t}")
        if self._quant_type() == "Q3_K":
            asm.inst(f"v_and_b32 v{quant_shift}, 7, v{r.serial}")
            asm.inst(f"v_lshrrev_b32 v{quant_shift}, 1, v{quant_shift}")
            asm.inst(f"v_lshlrev_b32 v{quant_shift}, 1, v{quant_shift}")
        else:
            asm.inst(f"v_and_b32 v{quant_shift}, 2, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{quant_shift}, 1, v{quant_shift}")
        if self._quant_type() == "Q5_K":
            asm.comment("Derive the Q5_K scale-group index for the high plane.")
            if n_tiles == 8:
                asm.inst(f"v_and_b32 v{t}, 1, s3")
                asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")
                asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 2, 1")
                asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
                asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
                asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 1, 1")
                asm.inst(f"v_add_nc_u32 v{quant_shift}, v{t}, v{t + 1}")
            else:
                asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
                asm.inst(f"v_lshrrev_b32 v{t}, 1, v{t}")
                asm.inst(f"v_and_b32 v{t + 1}, {n_tiles - 1}, s3")
                asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
                asm.inst(f"v_add_nc_u32 v{quant_shift}, v{t}, v{t + 1}")
        swizzle = self.solution_key.solution.lds_swizzle_chunk_b
        if swizzle:
            lds = r.lds_address
            residues = 32 // swizzle
            swizzle_shift = (2 * swizzle).bit_length() - 1
            asm.comment(
                f"Precompute XOR-{swizzle} LDS store bases for "
                f"N row residues 0-{residues - 1}."
            )
            asm.inst(
                f"v_lshrrev_b32 v{t}, {k_shift + swizzle.bit_length() - 1}, v{r.serial}"
            )
            asm.inst(f"v_lshlrev_b32 v{t + 1}, {swizzle_shift}, v{t}")
            asm.inst(f"v_sub_nc_u32 v{t + 1}, v{lds_address}, v{t + 1}")
            for residue in range(residues):
                asm.inst(f"v_xor_b32 v{lds + residue}, {residue}, v{t}")
                asm.inst(
                    f"v_lshlrev_b32 v{lds + residue}, {swizzle_shift}, v{lds + residue}"
                )
                asm.inst(f"v_add_nc_u32 v{lds + residue}, v{t + 1}, v{lds + residue}")
            if self.solution_key.solution.one_lds_buffer == 0:
                asm.comment("Initialize the decoded-B read buffer to LDS0.")
                asm.inst(f"v_mov_b32 v{lds_address}, 0")

        solution = self.solution_key.solution
        if solution.schedule_iter_alg not in (4, 5):
            return
        m_tiles = solution.matrix_instruction[5]
        m_per_wave = 16 * m_tiles
        asm.comment("Precompute A row coordinates shared by every DepthU iteration.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{a + 4}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 2}, {solution.macro_tile0.bit_length() - 1}, s2")
        asm.inst(f"v_add_nc_u32 v{a + 4}, v{a + 4}, v{t + 2}")
        for m_tile in range(1, m_tiles):
            asm.inst(
                f"v_add_nc_u32 v{a + 4 + m_tile}, {16 * m_tile}, v{a + 4}"
            )
        row_stride_a = 2 * self.solution_key.problem_size.k
        for m_tile in range(m_tiles):
            self._emit_scale_u32(
                asm,
                a + 4 + m_tile,
                row_stride_a,
                a + 4 + m_tile,
            )

    def _emit_quant_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        if self._quant_type() == "Q3_K":
            self._emit_q3_k_global_reads(asm, wait_for_reads=wait_for_reads)
        elif self._quant_type() == "Q4_K":
            self._emit_q4_k_global_reads(asm, wait_for_reads=wait_for_reads)
        else:
            self._emit_q5_k_global_reads(asm, wait_for_reads=wait_for_reads)

    def _emit_q3_k_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        n_per_tile = self.solution_key.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        k_shift = self.solution_key.solution.matrix_instruction[6].bit_length() - 1
        k_span = self.solution_key.solution.depth_u // decoder_rows
        packed_row_bytes = self.solution_key.problem_size.n // 256 * 110
        row_delta = k_span * packed_row_bytes
        q_low = r.global_read_b
        q_high = q_low + 4 * decoder_rows
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q3_K block addresses for decoder-owned output rows.")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, {packed_row_bytes}, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        for row in range(1, decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + row}, {row * row_delta}, v{a}")

        asm.comment("Map each lane to a Q3_K 16-value group in the 256-value block.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, {tiles_per_weight_block - 1}, s3")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {n_shift}, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t + 1}, v{t + 1}, v{t}")
        asm.inst(f"v_and_b32 v{t + 2}, 31, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 3}, 31, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 4}, 128, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t + 4}, 2, v{t + 4}")
        asm.inst(f"v_add_nc_u32 v{t + 3}, v{t + 3}, v{t + 4}")
        asm.inst(f"v_lshrrev_b32 v{t + 5}, 4, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 6}, 7, v{t + 5}")
        asm.inst(f"v_and_b32 v{t + 5}, 3, v{t + 5}")

        for row in range(decoder_rows):
            block_address = a + row
            if row:
                asm.inst(f"v_and_b32 v{t + 2}, 31, v{t + 1}")
                asm.inst(f"v_and_b32 v{t + 3}, 31, v{t + 1}")
                asm.inst(f"v_and_b32 v{t + 4}, 128, v{t + 1}")
                asm.inst(f"v_lshrrev_b32 v{t + 4}, 2, v{t + 4}")
                asm.inst(f"v_add_nc_u32 v{t + 3}, v{t + 3}, v{t + 4}")
                asm.inst(f"v_lshrrev_b32 v{t + 5}, 4, v{t + 1}")
                asm.inst(f"v_and_b32 v{t + 6}, 7, v{t + 5}")
                asm.inst(f"v_and_b32 v{t + 5}, 3, v{t + 5}")
            asm.inst(f"v_add_nc_u32 v{t + 2}, v{block_address}, v{t + 2}")
            asm.inst(
                f"global_load_b128 v[{q_high + 4 * row}:{q_high + 4 * row + 3}], "
                f"v{t + 2}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
            )
            asm.inst(f"v_add_nc_u32 v{t + 3}, v{block_address}, v{t + 3}")
            asm.inst(
                f"global_load_b128 v[{q_low + 4 * row}:{q_low + 4 * row + 3}], "
                f"v{t + 3}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:32"
            )
            asm.inst(f"v_add_nc_u32 v{t + 6}, v{block_address}, v{t + 6}")
            asm.inst(
                f"global_load_d16_u8 v{r.quant_scale + 2 * row}, v{t + 6}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:96"
            )
            asm.inst(f"v_add_nc_u32 v{t + 5}, v{block_address}, v{t + 5}")
            asm.inst(
                f"global_load_d16_u8 v{r.quant_scale + 2 * row + 1}, v{t + 5}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:104"
            )
            asm.inst(
                f"global_load_d16_b16 v{r.quant_dm + row}, v{block_address}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:108"
            )
        if wait_for_reads:
            asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q4_k_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        decoder_rows = self._decoder_rows()
        k_shift = n_tiles.bit_length() - 1
        k_span = self.solution_key.solution.depth_u // decoder_rows
        packed_row_bytes = self.solution_key.problem_size.n // 256 * 144
        row_delta = k_span * packed_row_bytes
        q = r.global_read_b
        dm = r.quant_dm
        scale = r.quant_scale
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q4_K block addresses for decoder-owned output rows.")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, {packed_row_bytes}, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        if decoder_rows == 2:
            asm.inst(f"v_add_nc_u32 v{a + 1}, {row_delta}, v{a}")

        asm.comment("Build quant and scale-byte addresses.")
        direct_quant_mapping = self.solution_key.solution.macro_tile1 == 128
        if direct_quant_mapping:
            asm.inst(f"v_and_b32 v{t}, 1, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
            asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 2, 1")
            asm.inst(f"v_lshl_add_u32 v{t + 1}, v{t + 1}, 5, v{t}")
            asm.inst(f"v_add_nc_u32 v{t + 1}, s{r.scalar_temporary}, v{t + 1}")
        else:
            asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
            asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary}, v{t}")
            asm.inst(f"v_lshrrev_b32 v{t + 1}, 6, v{t}")
            asm.inst(f"v_lshlrev_b32 v{t + 1}, 5, v{t + 1}")
            asm.inst(f"v_and_b32 v{t + 2}, 31, v{t}")
            asm.inst(f"v_add_nc_u32 v{t + 1}, v{t + 1}, v{t + 2}")
        if decoder_rows > 2:
            if direct_quant_mapping:
                asm.inst(f"v_bfe_u32 v{t + 2}, v{r.serial}, 1, 2")
            else:
                asm.inst(f"v_lshrrev_b32 v{t + 2}, 5, v{t}")
                asm.inst(f"v_and_b32 v{t + 2}, 3, v{t + 2}")
            for row in range(decoder_rows):
                block_address = a
                if row:
                    asm.inst(f"v_add_nc_u32 v{a + row}, {row * row_delta}, v{a}")
                    block_address = a + row
                asm.inst(f"v_add_nc_u32 v{t + 3}, v{block_address}, v{t + 1}")
                asm.inst(
                    f"global_load_b128 v[{q + 4 * row}:{q + 4 * row + 3}], "
                    f"v{t + 3}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:16"
                )
                asm.inst(
                    f"global_load_b32 v{dm + row}, "
                    f"v{block_address}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
                )
                asm.inst(f"v_add_nc_u32 v{t + 4}, v{block_address}, v{t + 2}")
                for scale_offset in (4, 8, 12):
                    scale_register = scale + 3 * row + scale_offset // 4 - 1
                    asm.inst(
                        f"global_load_d16_u8 v{scale_register}, "
                        f"v{t + 4}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                        f"offset:{scale_offset}"
                    )
            if wait_for_reads:
                asm.inst("s_waitcnt vmcnt(0)")
            return

        for row in range(decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + 2 + row}, v{a + row}, v{t + 1}")
        if self.solution_key.solution.packed_weight_lane_share == 2:
            asm.comment("Load packed q bytes on one lane from each nibble pair.")
            asm.inst(f"v_and_b32 v{t + 2}, 2, v{r.serial}")
            asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{t + 2}")
            asm.inst(f"s_and_saveexec_b32 s{r.scalar_temporary + 1}, vcc_lo")
        for row in range(decoder_rows):
            asm.inst(
                f"global_load_b128 v[{q + 4 * row}:{q + 4 * row + 3}], "
                f"v{a + 2 + row}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:16"
            )
        if self.solution_key.solution.packed_weight_lane_share == 2:
            asm.inst(f"s_mov_b32 exec_lo, s{r.scalar_temporary + 1}")

        if direct_quant_mapping:
            asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 1, 2")
        else:
            asm.inst(f"v_lshrrev_b32 v{t + 1}, 5, v{t}")
            asm.inst(f"v_and_b32 v{t + 1}, 3, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{a + 2}, v{a}, v{t + 1}")
        for row in range(decoder_rows):
            asm.inst(
                f"global_load_b32 v{dm + row}, v{a + row}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}]"
            )
        for row in range(decoder_rows):
            address_pair = a + 2
            if row:
                asm.inst(f"v_add_nc_u32 v{a + 3}, {row * row_delta}, v{a + 2}")
                address_pair = a + 3
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row}, v{address_pair}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:4"
            )
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row + 1}, v{address_pair}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:8"
            )
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row + 2}, v{address_pair}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:12"
            )
        if wait_for_reads:
            asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q5_k_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        decoder_rows = self._decoder_rows()
        k_shift = n_tiles.bit_length() - 1
        k_span = self.solution_key.solution.depth_u // decoder_rows
        packed_row_bytes = self.solution_key.problem_size.n // 256 * 176
        row_delta = k_span * packed_row_bytes
        q_low = r.global_read_b
        q_high = q_low + 4 * decoder_rows
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q5_K block addresses for decoder-owned output rows.")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, {packed_row_bytes}, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        if decoder_rows == 2:
            asm.inst(f"v_add_nc_u32 v{a + 1}, {row_delta}, v{a}")

        asm.comment("Build Q5_K payload and scale-byte addresses.")
        direct_quant_mapping = self.solution_key.solution.macro_tile1 == 128
        if direct_quant_mapping:
            asm.inst(f"v_and_b32 v{t}, 1, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
            asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 2, 1")
            asm.inst(f"v_lshl_add_u32 v{t + 1}, v{t + 1}, 5, v{t}")
            asm.inst(f"v_add_nc_u32 v{t + 1}, s{r.scalar_temporary}, v{t + 1}")
        else:
            asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
            asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary}, v{t}")
            asm.inst(f"v_lshrrev_b32 v{t + 1}, 6, v{t}")
            asm.inst(f"v_lshlrev_b32 v{t + 1}, 5, v{t + 1}")
            asm.inst(f"v_and_b32 v{t + 2}, 31, v{t}")
            asm.inst(f"v_add_nc_u32 v{t + 1}, v{t + 1}, v{t + 2}")

        if decoder_rows > 2:
            if direct_quant_mapping:
                asm.inst(f"v_bfe_u32 v{t + 2}, v{r.serial}, 1, 2")
            else:
                asm.inst(f"v_lshrrev_b32 v{t + 2}, 5, v{t}")
                asm.inst(f"v_and_b32 v{t + 2}, 3, v{t + 2}")
            for row in range(decoder_rows):
                block_address = a
                if row:
                    asm.inst(f"v_add_nc_u32 v{a + row}, {row * row_delta}, v{a}")
                    block_address = a + row
                asm.inst(f"v_add_nc_u32 v{t + 3}, v{block_address}, v{t}")
                asm.inst(
                    f"global_load_b128 v[{q_high + 4 * row}:{q_high + 4 * row + 3}], "
                    f"v{t + 3}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:16"
                )
                asm.inst(f"v_add_nc_u32 v{t + 3}, v{block_address}, v{t + 1}")
                asm.inst(
                    f"global_load_b128 v[{q_low + 4 * row}:{q_low + 4 * row + 3}], "
                    f"v{t + 3}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:48"
                )
                self._emit_q5_k_metadata_loads(asm, block_address, block_address, row)
            if wait_for_reads:
                asm.inst("s_waitcnt vmcnt(0)")
            return

        for row in range(decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + 2 + row}, v{a + row}, v{t + 1}")
        if self.solution_key.solution.packed_weight_lane_share == 2:
            asm.comment("Load Q5_K payload planes on one lane from each lane pair.")
            asm.inst(f"v_and_b32 v{t + 2}, 2, v{r.serial}")
            asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{t + 2}")
            asm.inst(f"s_and_saveexec_b32 s{r.scalar_temporary + 1}, vcc_lo")
        for row in range(decoder_rows):
            high_offset = t if direct_quant_mapping else t + 2
            asm.inst(f"v_add_nc_u32 v{t + 3}, v{a + row}, v{high_offset}")
            asm.inst(
                f"global_load_b128 v[{q_high + 4 * row}:{q_high + 4 * row + 3}], "
                f"v{t + 3}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:16"
            )
            asm.inst(
                f"global_load_b128 v[{q_low + 4 * row}:{q_low + 4 * row + 3}], "
                f"v{a + 2 + row}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:48"
            )
        if self.solution_key.solution.packed_weight_lane_share == 2:
            asm.inst(f"s_mov_b32 exec_lo, s{r.scalar_temporary + 1}")

        if direct_quant_mapping:
            asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 1, 2")
        else:
            asm.inst(f"v_lshrrev_b32 v{t + 1}, 5, v{t}")
            asm.inst(f"v_and_b32 v{t + 1}, 3, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{a + 2}, v{a}, v{t + 1}")
        for row in range(decoder_rows):
            address_pair = a + 2
            if row:
                asm.inst(f"v_add_nc_u32 v{a + 3}, {row * row_delta}, v{a + 2}")
                address_pair = a + 3
            self._emit_q5_k_metadata_loads(asm, a + row, address_pair, row)
        if wait_for_reads:
            asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q5_k_metadata_loads(
        self,
        asm: _Assembly,
        block_address: int,
        scale_address: int,
        row: int,
    ) -> None:
        r = self.registers
        if self.solution_key.solution.q5_k_metadata_vector_load:
            metadata = r.quant_dm + 4 * row
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], "
                f"v{block_address}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
            )
            return
        scale = r.quant_scale + 3 * row
        asm.inst(
            f"global_load_b32 v{r.quant_dm + row}, v{block_address}, "
            f"s[{r.kernarg + 2}:{r.kernarg + 3}]"
        )
        for scale_offset, scale_register in zip((4, 8, 12), range(3)):
            asm.inst(
                f"global_load_d16_u8 v{scale + scale_register}, "
                f"v{scale_address}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                f"offset:{scale_offset}"
            )

    def _emit_packed_weight_lane_share(self, asm: _Assembly) -> None:
        solution = self.solution_key.solution
        if solution.packed_weight_lane_share == 1:
            return

        r = self.registers
        decoder_rows = self._decoder_rows()
        lane_pair = r.temporary
        shifted = lane_pair + 1
        asm.comment("Replicate packed q bytes across low/high-nibble lane pairs.")
        asm.inst(f"v_and_b32 v{lane_pair}, 2, v{r.serial}")
        asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{lane_pair}")
        for register in range(
            r.global_read_b,
            r.global_read_b + self._payload_register_count() * decoder_rows,
        ):
            asm.inst(
                f"v_mov_b32_dpp v{shifted}, v{register} row_shr:2 "
                "row_mask:0xf bank_mask:0xf"
            )
            asm.inst(f"v_cndmask_b32 v{register}, v{shifted}, v{register}, vcc_lo")

    def _emit_first_a_global_reads(self, asm: _Assembly) -> None:
        r = self.registers
        solution = self.solution_key.solution
        m_tiles = solution.matrix_instruction[5]
        a = r.address

        asm.comment("Prefetch A fragments while Q4_K data is pending.")
        row_pointers = tuple((a + 4 + m_tile, a + m_tile) for m_tile in range(m_tiles))

        asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
        for row_offset, pointer in row_pointers:
            asm.inst(
                f"v_add_nc_u32 v{pointer}, s{r.scalar_temporary + 1}, v{row_offset}"
            )
        for k_half in range(solution.prefetch_global_read):
            if k_half:
                for _, pointer in row_pointers:
                    asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
            for m_tile, (_, pointer) in enumerate(row_pointers):
                valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
                asm.inst(
                    f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                    f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}]"
                )
                asm.inst(
                    f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                    f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}] offset:16"
                )

    def _emit_quant_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        if self._quant_type() == "Q3_K":
            self._emit_q3_k_decode(asm, label_suffix=label_suffix)
        elif self._quant_type() == "Q4_K":
            self._emit_q4_k_decode(asm, label_suffix=label_suffix)
        else:
            self._emit_q5_k_decode(asm, label_suffix=label_suffix)

    def _emit_quant_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        if self._quant_type() == "Q3_K":
            self._emit_q3_k_decode_prepare(asm, label_suffix=label_suffix)
        elif self._quant_type() == "Q4_K":
            self._emit_q4_k_decode_prepare(asm, label_suffix=label_suffix)
        else:
            self._emit_q5_k_decode_prepare(asm, label_suffix=label_suffix)

    def _emit_quant_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        if self._quant_type() == "Q3_K":
            self._emit_q3_k_decode_chunk(asm, chunk)
        elif self._quant_type() == "Q4_K":
            self._emit_q4_k_decode_chunk(asm, chunk)
        else:
            self._emit_q5_k_decode_chunk(asm, chunk)

    def _emit_q3_k_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q3_k_decode_prepare(asm, label_suffix=label_suffix)
        asm.comment("Decode Q3_K 2-bit payload and high mask into LDS.")
        for chunk in range(4 * self._decoder_rows()):
            self._emit_q3_k_decode_chunk(asm, chunk)

    def _emit_q3_k_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        del label_suffix
        r = self.registers
        decoder_rows = self._decoder_rows()
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        n_per_tile = self.solution_key.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        scale = r.quant_scale
        t = r.temporary
        quant_shift = self._quant_shift_register()
        asm.comment("Reconstruct Q3_K signed scale and d in FP32.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, {tiles_per_weight_block - 1}, s3")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {n_shift}, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 4, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 2}, 2, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 2}, 1, v{t + 2}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 3, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 2, v{t + 1}")
        for row in range(decoder_rows):
            low = scale + 2 * row
            high = low + 1
            asm.inst(f"v_bfe_u32 v{low}, v{low}, v{t + 1}, 4")
            asm.inst(f"v_bfe_u32 v{high}, v{high}, v{t + 2}, 2")
            asm.inst(f"v_lshlrev_b32 v{t}, 4, v{high}")
            asm.inst(f"v_or_b32 v{low}, v{low}, v{t}")
            asm.inst(f"v_sub_nc_u32 v{low}, v{low}, 32")
            asm.inst(f"v_cvt_f32_i32_e32 v{low}, v{low}")
        for row in range(decoder_rows):
            low = scale + 2 * row
            d_scaled = t + 3 + row if row < 2 else r.quant_dm + row
            asm.inst(
                f"v_fma_mix_f32 v{d_scaled}, v{r.quant_dm + row}, v{low}, "
                "neg(0) op_sel_hi:[1,0,0]"
            )
        if self._q3_k_full_vopd_decode():
            if decoder_rows == 1:
                asm.inst(f"v_mov_b32 v{t + 9}, v{t + 3}")
            elif decoder_rows == 2:
                asm.inst(
                    f"v_dual_mov_b32 v{t + 9}, v{t + 3} :: "
                    f"v_dual_mov_b32 v{t + 10}, v{t + 4}"
                )
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, {tiles_per_weight_block - 1}, s3")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {n_shift}, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 127, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 5, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_mov_b32 v{self._q3_low_shift_register()}, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{t}")
        asm.inst(f"v_mov_b32 v{quant_shift}, v{t}")
        if self.solution_key.solution.q3_k_extraction == "packed":
            asm.inst(f"v_mov_b32 v{t + 2}, 4.0")
            if self._q3_k_full_vopd_decode():
                asm.inst(f"v_mov_b32 v{t + 1}, 4.0")

    def _emit_q3_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        row_stride = self._lds_row_stride_bytes()
        t = r.temporary
        low = r.global_read_b + 4 * row + first_element // 4
        high = r.global_read_b + 4 * decoder_rows + 4 * row + first_element // 4
        asm.inst(f"v_lshrrev_b32 v{low}, v{self._q3_low_shift_register()}, v{low}")
        asm.inst(f"v_and_b32 v{low}, 0x03030303, v{low}")
        asm.inst(f"v_lshrrev_b32 v{high}, v{self._quant_shift_register()}, v{high}")
        asm.inst(f"v_and_b32 v{high}, 0x01010101, v{high}")
        if self.solution_key.solution.q3_k_extraction == "packed":
            asm.inst(f"v_lshl_or_b32 v{low}, v{high}, 2, v{low}")
        else:
            asm.inst(f"v_xor_b32 v{high}, 0x01010101, v{high}")
            asm.inst(f"v_lshlrev_b32 v{high}, 2, v{high}")

        def emit_lds_store(element: int, value: int, rounding: int) -> None:
            lds_offset = row_stride * element + 32 * row
            lds_address = self._lds_address_register()
            swizzle = self.solution_key.solution.lds_swizzle_chunk_b
            if swizzle:
                residues = 32 // swizzle
                residue = element % residues
                lds_address = r.lds_address + residue
                lds_offset = row_stride * element + 64 * (row // 2)
                if row % 2:
                    lds_offset += 32 if residue < residues // 2 else -32
            self._emit_round_bf16(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )

        if self.solution_key.solution.q3_k_extraction == "scalar":
            for element in range(first_element, first_element + 4):
                value = t + 5
                rounding = t + 6
                d_scaled = t + 3 + row if row < 2 else r.quant_dm + row
                asm.inst(f"v_bfe_u32 v{rounding}, v{high}, {8 * (element % 4)}, 3")
                asm.inst(f"v_cvt_f32_u32_e32 v{rounding}, v{rounding}")
                asm.inst(f"v_bfe_u32 v{value}, v{low}, {8 * (element % 4)}, 2")
                asm.inst(f"v_cvt_f32_u32_e32 v{value}, v{value}")
                asm.inst(f"v_sub_f32 v{value}, v{value}, v{rounding}")
                asm.inst(f"v_mul_f32 v{value}, v{d_scaled}, v{value}")
                emit_lds_store(element, value, rounding)
            return

        d_scaled = t + 3 + row if row < 2 else r.quant_dm + row
        d_scaled_copy = t + 9 + row
        first = first_element
        second = first_element + 1
        third = first_element + 2
        fourth = first_element + 3
        q0 = t + 5
        q1 = t + 8
        full_vopd = self._q3_k_full_vopd_decode()
        q1_low = q1 if full_vopd else (q1 if d_scaled % 2 else t + 7)
        rounding = t + 6

        def emit_pair(first_element: int, second_element: int) -> None:
            asm.inst(f"v_cvt_f32_ubyte{first_element % 4}_e32 v{q0}, v{low}")
            if not full_vopd:
                asm.inst(f"v_sub_f32 v{q0}, v{q0}, v{t + 2}")
            asm.inst(f"v_cvt_f32_ubyte{second_element % 4}_e32 v{q1_low}, v{low}")
            if full_vopd:
                asm.inst(
                    f"v_dual_sub_f32 v{q0}, v{q0}, v{t + 1} :: "
                    f"v_dual_sub_f32 v{q1}, v{q1_low}, v{t + 2}"
                )
                asm.inst(
                    f"v_dual_mul_f32 v{q0}, v{d_scaled}, v{q0} :: "
                    f"v_dual_mul_f32 v{q1}, v{d_scaled_copy}, v{q1}"
                )
            else:
                asm.inst(
                    f"v_dual_mul_f32 v{q0}, v{d_scaled}, v{q0} :: "
                    f"v_dual_sub_f32 v{q1}, v{q1_low}, v{t + 2}"
                )
                asm.inst(f"v_mul_f32 v{q1}, v{d_scaled}, v{q1}")
            emit_lds_store(first_element, q0, rounding)
            emit_lds_store(second_element, q1, rounding)

        emit_pair(first, second)
        emit_pair(third, fourth)

    def _emit_q4_k_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q4_k_decode_prepare(asm, label_suffix=label_suffix)
        asm.comment("Decode 16 nibbles from each packed output row into LDS.")
        for chunk in range(4 * self._decoder_rows()):
            self._emit_q4_k_decode_chunk(asm, chunk)

    def _emit_q4_k_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        scale = r.quant_scale
        t = r.temporary
        asm.comment("Unpack Q4_K six-bit scale/min fields.")
        scale_odd = f".LScaleOdd{label_suffix}"
        scale_ready = f".LScaleReady{label_suffix}"
        asm.inst(f"s_cmp_eq_u32 s{r.input_half}, 0")
        asm.inst(f"s_cbranch_scc0 {scale_odd}")
        for row in range(decoder_rows):
            asm.inst(f"v_and_b32 v{scale + 3 * row}, 0x3f, v{scale + 3 * row}")
            asm.inst(f"v_and_b32 v{scale + 3 * row + 1}, 0x3f, v{scale + 3 * row + 1}")
        asm.inst(f"s_branch {scale_ready}")
        asm.label(scale_odd)
        for row in range(decoder_rows):
            lo = scale + 3 * row
            minimum = lo + 1
            high = lo + 2
            asm.inst(f"v_lshrrev_b32 v{t}, 2, v{lo}")
            asm.inst(f"v_and_b32 v{t}, 0x30, v{t}")
            asm.inst(f"v_and_b32 v{lo}, 0x0f, v{high}")
            asm.inst(f"v_or_b32 v{lo}, v{lo}, v{t}")
            asm.inst(f"v_lshrrev_b32 v{t}, 2, v{minimum}")
            asm.inst(f"v_and_b32 v{t}, 0x30, v{t}")
            asm.inst(f"v_lshrrev_b32 v{minimum}, 4, v{high}")
            asm.inst(f"v_or_b32 v{minimum}, v{minimum}, v{t}")
        asm.label(scale_ready)

        asm.comment("Convert and scale packed FP16 d/dmin values in FP32.")
        for row in range(decoder_rows):
            dm = r.quant_dm + row
            lo = scale + 3 * row
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            asm.inst(f"v_lshrrev_b32 v{min_scaled}, 16, v{dm}")
            asm.inst(f"v_cvt_f32_ubyte0_e32 v{lo}, v{lo}")
            asm.inst(f"v_cvt_f32_ubyte0_e32 v{lo + 1}, v{lo + 1}")
            asm.inst(
                f"v_fma_mix_f32 v{d_scaled}, v{dm}, v{lo}, neg(0) op_sel_hi:[1,0,0]"
            )
            asm.inst(
                f"v_fma_mix_f32 v{min_scaled}, v{min_scaled}, v{lo + 1}, "
                "neg(0) op_sel_hi:[1,0,0]"
            )
        asm.comment("Hoist the Q5_K low-payload nibble-half shift.")
        asm.inst(f"v_and_b32 v{t}, 1, v{self._quant_shift_register()}")
        asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")

    def _emit_q4_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        row_stride = self._lds_row_stride_bytes()
        t = r.temporary
        packed = r.global_read_b + 4 * row + first_element // 4
        asm.inst(f"v_lshrrev_b32 v{packed}, v{self._quant_shift_register()}, v{packed}")
        asm.inst(f"v_and_b32 v{packed}, 0x0f0f0f0f, v{packed}")
        for element in range(first_element, first_element + 4):
            value = t + 1 + 2 * decoder_rows
            rounding = value + 1
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            lds_offset = row_stride * element + 32 * row
            lds_address = self._lds_address_register()
            swizzle = self.solution_key.solution.lds_swizzle_chunk_b
            if swizzle:
                residues = 32 // swizzle
                residue = element % residues
                lds_address = r.lds_address + residue
                lds_offset = row_stride * element + 64 * (row // 2)
                if row % 2:
                    lds_offset += 32 if residue < residues // 2 else -32
            asm.inst(f"v_cvt_f32_ubyte{element % 4}_e32 v{value}, v{packed}")
            asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
            self._emit_round_bf16(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )

    def _emit_q5_k_decode(self, asm: _Assembly, *, label_suffix: str = "") -> None:
        self._emit_q5_k_decode_prepare(asm, label_suffix=label_suffix)
        if self.solution_key.solution.q5_k_nibble_shift_hoist:
            self._emit_q5_k_nibble_shift(asm)
        asm.comment("Decode Q5_K low nibbles and high payload bits into LDS.")
        for chunk in range(4 * self._decoder_rows()):
            self._emit_q5_k_decode_chunk(asm, chunk)

    def _emit_q5_k_nibble_shift(self, asm: _Assembly) -> None:
        r = self.registers
        t = r.temporary
        asm.comment("Hoist the Q5_K low-payload nibble-half shift.")
        asm.inst(f"v_and_b32 v{t}, 1, v{self._quant_shift_register()}")
        asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")

    def _emit_q5_k_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        scale = r.quant_scale
        t = r.temporary
        asm.comment("Unpack Q5_K six-bit scale/min fields.")
        if self.solution_key.solution.q5_k_metadata_vector_load:
            asm.comment("Select the lane-owned Q5_K scale-byte triplet.")
            asm.inst(f"v_bfe_u32 v{t + 1}, v{r.serial}, 1, 2")
            asm.inst(f"v_lshlrev_b32 v{t + 2}, 3, v{t + 1}")
            for row in range(decoder_rows):
                metadata = r.quant_dm + 4 * row
                for offset in (1, 2, 3):
                    asm.inst(
                        f"v_bfe_u32 v{metadata + offset}, "
                        f"v{metadata + offset}, v{t + 2}, 8"
                    )
        scale_odd = f".LScaleOdd{label_suffix}"
        scale_ready = f".LScaleReady{label_suffix}"
        asm.inst(f"s_cmp_eq_u32 s{r.input_half}, 0")
        asm.inst(f"s_cbranch_scc0 {scale_odd}")
        for row in range(decoder_rows):
            lo = (
                r.quant_dm + 4 * row + 1
                if self.solution_key.solution.q5_k_metadata_vector_load
                else scale + 3 * row
            )
            asm.inst(f"v_and_b32 v{lo}, 0x3f, v{lo}")
            asm.inst(f"v_and_b32 v{lo + 1}, 0x3f, v{lo + 1}")
        asm.inst(f"s_branch {scale_ready}")
        asm.label(scale_odd)
        for row in range(decoder_rows):
            lo = (
                r.quant_dm + 4 * row + 1
                if self.solution_key.solution.q5_k_metadata_vector_load
                else scale + 3 * row
            )
            minimum = lo + 1
            high = lo + 2
            asm.inst(f"v_lshrrev_b32 v{t}, 2, v{lo}")
            asm.inst(f"v_and_b32 v{t}, 0x30, v{t}")
            asm.inst(f"v_and_b32 v{lo}, 0x0f, v{high}")
            asm.inst(f"v_or_b32 v{lo}, v{lo}, v{t}")
            asm.inst(f"v_lshrrev_b32 v{t}, 2, v{minimum}")
            asm.inst(f"v_and_b32 v{t}, 0x30, v{t}")
            asm.inst(f"v_lshrrev_b32 v{minimum}, 4, v{high}")
            asm.inst(f"v_or_b32 v{minimum}, v{minimum}, v{t}")
        asm.label(scale_ready)

        asm.comment("Convert and scale Q5_K d/dmin values in FP32.")
        for row in range(decoder_rows):
            if self.solution_key.solution.q5_k_metadata_vector_load:
                dm = r.quant_dm + 4 * row
                lo = dm + 1
            else:
                dm = r.quant_dm + row
                lo = scale + 3 * row
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            asm.inst(f"v_lshrrev_b32 v{min_scaled}, 16, v{dm}")
            asm.inst(f"v_cvt_f32_ubyte0_e32 v{lo}, v{lo}")
            asm.inst(f"v_cvt_f32_ubyte0_e32 v{lo + 1}, v{lo + 1}")
            asm.inst(
                f"v_fma_mix_f32 v{d_scaled}, v{dm}, v{lo}, neg(0) op_sel_hi:[1,0,0]"
            )
            asm.inst(
                f"v_fma_mix_f32 v{min_scaled}, v{min_scaled}, v{lo + 1}, "
                "neg(0) op_sel_hi:[1,0,0]"
            )

    def _emit_q5_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self._decoder_rows()
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        row_stride = self._lds_row_stride_bytes()
        t = r.temporary
        low = r.global_read_b + 4 * row + first_element // 4
        high = r.global_read_b + 4 * decoder_rows + 4 * row + first_element // 4
        if not self.solution_key.solution.q5_k_nibble_shift_hoist:
            asm.inst(f"v_and_b32 v{t}, 1, v{self._quant_shift_register()}")
            asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")
        asm.inst(f"v_lshrrev_b32 v{low}, v{t}, v{low}")
        asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
        asm.inst(f"v_lshrrev_b32 v{high}, v{self._quant_shift_register()}, v{high}")
        asm.inst(f"v_and_b32 v{high}, 0x01010101, v{high}")
        asm.inst(f"v_lshl_or_b32 v{low}, v{high}, 4, v{low}")
        for element in range(first_element, first_element + 4):
            value = t + 1 + 2 * decoder_rows
            rounding = value + 1
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            lds_offset = row_stride * element + 32 * row
            lds_address = self._lds_address_register()
            swizzle = self.solution_key.solution.lds_swizzle_chunk_b
            if swizzle:
                residues = 32 // swizzle
                residue = element % residues
                lds_address = r.lds_address + residue
                lds_offset = row_stride * element + 64 * (row // 2)
                if row % 2:
                    lds_offset += 32 if residue < residues // 2 else -32
            if self.solution_key.solution.q5_k_extraction == "scalar":
                asm.inst(f"v_bfe_u32 v{value}, v{low}, {8 * (element % 4)}, 5")
                asm.inst(f"v_cvt_f32_u32_e32 v{value}, v{value}")
            else:
                asm.inst(f"v_cvt_f32_ubyte{element % 4}_e32 v{value}, v{low}")
            asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
            self._emit_round_bf16(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )

    def _emit_wmma(
        self,
        asm: _Assembly,
        *,
        pipeline: bool = False,
        after_pair: Callable[[int, int], None] | None = None,
    ) -> None:
        r = self.registers
        solution = self.solution_key.solution
        m_tiles = solution.matrix_instruction[5]
        n_tiles = solution.matrix_instruction[6]
        m_per_wave = 16 * m_tiles
        size = self.solution_key.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Load A and issue two DepthU=16 WMMA halves.")
        if solution.schedule_iter_alg not in (4, 5):
            asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
            asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
            asm.inst(f"v_add_nc_u32 v{a + 4}, v{t}, v{t + 1}")
            asm.inst(
                f"v_lshlrev_b32 v{t + 2}, {solution.macro_tile0.bit_length() - 1}, s2"
            )
            asm.inst(f"v_add_nc_u32 v{a + 4}, v{a + 4}, v{t + 2}")
            if m_tiles == 2:
                asm.inst(f"v_add_nc_u32 v{a + 5}, 16, v{a + 4}")
        else:
            asm.comment("Reuse prefetched A pointers across fused B decode.")
        for k_tile in range(0, solution.depth_u, 16):
            if solution.schedule_iter_alg in (4, 5):
                if k_tile and solution.prefetch_global_read == 1:
                    for m_tile in range(m_tiles):
                        pointer = a + m_tile
                        asm.inst(f"v_add_nc_u32 v{pointer}, {2 * k_tile}, v{pointer}")
                        valu_a = r.valu_a + 8 * m_tile
                        asm.inst(
                            f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                            f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}]"
                        )
                        asm.inst(
                            f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                            f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}] offset:16"
                        )
                elif (
                    solution.depth_u == 64
                    and solution.prefetch_global_read == 2
                    and k_tile == 32
                ):
                    for k_half in range(2):
                        for m_tile in range(m_tiles):
                            pointer = a + m_tile
                            asm.inst(f"v_add_nc_u32 v{pointer}, 32, v{pointer}")
                            valu_a = r.valu_a + 8 * (k_half * m_tiles + m_tile)
                            asm.inst(
                                f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                                f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}]"
                            )
                            asm.inst(
                                f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                                f"v{pointer}, s[{r.kernarg}:{r.kernarg + 1}] offset:16"
                            )
            elif m_tiles == 2:
                asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
                for m_tile, row, pointer in (
                    (0, a + 4, a),
                    (1, a + 5, a + 2),
                ):
                    self._emit_scale_u32(asm, t, 2 * size.k, row)
                    asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
                    if k_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
                    self._emit_add_pointer(asm, pointer, r.kernarg, t)
                for m_tile, pointer in ((0, a), (1, a + 2)):
                    valu_a = r.valu_a + 8 * m_tile
                    asm.inst(
                        f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                        f"v[{pointer}:{pointer + 1}], off"
                    )
                    asm.inst(
                        f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                        f"v[{pointer}:{pointer + 1}], off offset:16"
                    )
            else:
                asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
                for m_tile in range(m_tiles):
                    if m_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {16 * m_tile}, v{a + 4}")
                        row = t
                    else:
                        row = a + 4
                    self._emit_scale_u32(asm, t, 2 * size.k, row)
                    asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
                    if k_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
                    self._emit_add_pointer(asm, a, r.kernarg, t)
                    valu_a = r.valu_a + 8 * m_tile
                    asm.inst(
                        f"global_load_b128 v[{valu_a}:{valu_a + 3}], "
                        f"v[{a}:{a + 1}], off"
                    )
                    asm.inst(
                        f"global_load_b128 v[{valu_a + 4}:{valu_a + 7}], "
                        f"v[{a}:{a + 1}], off offset:16"
                    )
            lds_arguments = self._emit_lds_read_arguments(
                asm, k_tile, pipeline=pipeline
            )
            valu_a_base = r.valu_a
            if solution.prefetch_global_read > 1:
                k_half = (k_tile // 16) % solution.prefetch_global_read
                valu_a_base += 8 * m_tiles * k_half
            if self.solution_key.solution.prefetch_local_read == 2:
                self._emit_prefetched_wmma_pairs(
                    asm,
                    n_tiles,
                    lds_arguments,
                )
            else:
                for n_tile in range(0, n_tiles, 2):
                    first = r.valu_b
                    second = r.valu_b + 8
                    pair_lds_arguments = lds_arguments
                    if after_pair is not None and n_tile:
                        pair_lds_arguments = (
                            solution.lds_swizzle_chunk_b,
                            (),
                            r.quant_dm,
                            r.quant_dm + 1,
                            0,
                            self._lds_row_stride_bytes(),
                        )
                    load_count = self._emit_lds_pair(
                        asm,
                        n_tile,
                        first,
                        second,
                        *pair_lds_arguments,
                    )
                    if solution.schedule_iter_alg in (3, 5):
                        trailing_a_loads = (
                            2 * m_tiles
                            if solution.prefetch_global_read == 2 and k_tile == 0
                            else 0
                        )
                        self._emit_sia3_wmma_pair(
                            asm,
                            n_tile,
                            first,
                            second,
                            first_pair=n_tile == 0,
                            pending_second_loads=load_count // 2,
                            valu_a_base=valu_a_base,
                            trailing_a_loads=trailing_a_loads,
                        )
                    else:
                        if solution.schedule_iter_alg == 4:
                            if pipeline or solution.prefetch_packed_weight_next:
                                asm.inst("s_waitcnt lgkmcnt(0)")
                            elif n_tile == 0:
                                pending_a = (
                                    2 * m_tiles * (solution.prefetch_global_read - 1)
                                    if k_tile % (16 * solution.prefetch_global_read)
                                    == 0
                                    else 0
                                )
                                asm.inst(f"s_waitcnt vmcnt({pending_a}) lgkmcnt(0)")
                            else:
                                asm.inst("s_waitcnt lgkmcnt(0)")
                        else:
                            asm.inst("s_waitcnt vmcnt(0) lgkmcnt(0)")
                        self._emit_wmma_pair(
                            asm,
                            n_tile,
                            first,
                            valu_a_base=valu_a_base,
                        )
                        self._emit_wmma_pair(
                            asm,
                            n_tile + 1,
                            second,
                            valu_a_base=valu_a_base,
                        )
                    if after_pair is not None:
                        after_pair(k_tile, n_tile)

    def _emit_lds_read_arguments(
        self,
        asm: _Assembly,
        k_tile: int,
        *,
        pipeline: bool,
    ) -> tuple[int, tuple[int, ...], int, int, int, int]:
        r = self.registers
        solution = self.solution_key.solution
        row_stride = self._lds_row_stride_bytes()
        if pipeline and k_tile:
            self._emit_pipeline_lds_read_addresses(asm, k_tile)
            return (
                solution.lds_swizzle_chunk_b,
                (),
                r.quant_dm,
                r.quant_dm + 1,
                0,
                row_stride,
            )

        t = r.temporary
        asm.inst(f"v_and_b32 v{t}, 15, v{r.serial}")
        self._emit_scale_u32(asm, t, row_stride, t)
        if pipeline:
            asm.inst(
                f"v_add_nc_u32 v{t}, v{self._lds_address_register()}, v{t}"
            )
        swizzle = solution.lds_swizzle_chunk_b
        chunk_addresses: tuple[int, ...] = ()
        if swizzle == 4:
            asm.inst(f"v_and_b32 v{t + 1}, 7, v{r.serial}")
            chunk_addresses = tuple(t + 2 + chunk for chunk in range(4))
            for chunk, chunk_address in enumerate(chunk_addresses):
                logical_chunk = k_tile // 4 + chunk
                asm.inst(f"v_xor_b32 v{chunk_address}, {logical_chunk}, v{t + 1}")
                asm.inst(f"v_lshlrev_b32 v{chunk_address}, 3, v{chunk_address}")
                asm.inst(f"v_add_nc_u32 v{chunk_address}, v{t}, v{chunk_address}")
            first_address = -1
            second_address = -1
            second_chunk_offset = 0
        elif swizzle in (8, 16):
            mask = 32 // swizzle - 1
            byte_shift = (2 * swizzle).bit_length() - 1
            asm.inst(f"v_and_b32 v{t + 1}, {mask}, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t + 1}, {byte_shift}, v{t + 1}")
            asm.inst(f"v_add_nc_u32 v{t + 2}, v{t}, v{t + 1}")
            if k_tile >= 32:
                asm.inst(f"v_add_nc_u32 v{t + 2}, {2 * (k_tile // 32) * 32}, v{t + 2}")
            if k_tile % 32:
                asm.inst(f"v_xor_b32 v{t + 2}, 32, v{t + 2}")
            asm.inst(f"v_xor_b32 v{t + 3}, 16, v{t + 2}")
            first_address = t + 2
            second_address = t + 3
            second_chunk_offset = 0
        else:
            if k_tile:
                asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
            first_address = t
            second_address = t
            second_chunk_offset = 16
        return (
            swizzle,
            chunk_addresses,
            first_address,
            second_address,
            second_chunk_offset,
            row_stride,
        )

    @staticmethod
    def _emit_lds_pair(
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        swizzle: int,
        chunk_addresses: tuple[int, ...],
        first_address: int,
        second_address: int,
        second_chunk_offset: int,
        row_stride: int,
    ) -> int:
        tile_stride = 16 * row_stride
        first_offset = tile_stride * n_tile
        second_offset = first_offset + tile_stride
        if swizzle == 4:
            for chunk, chunk_address in enumerate(chunk_addresses):
                destination = first + 2 * chunk
                asm.inst(
                    f"ds_load_b64 v[{destination}:{destination + 1}], "
                    f"v{chunk_address} offset:{first_offset}"
                )
            for chunk, chunk_address in enumerate(chunk_addresses):
                destination = second + 2 * chunk
                asm.inst(
                    f"ds_load_b64 v[{destination}:{destination + 1}], "
                    f"v{chunk_address} offset:{second_offset}"
                )
            return 8

        asm.inst(
            f"ds_load_b128 v[{first}:{first + 3}], "
            f"v{first_address} offset:{first_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{first + 4}:{first + 7}], "
            f"v{second_address} offset:{first_offset + second_chunk_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{second}:{second + 3}], "
            f"v{first_address} offset:{second_offset}"
        )
        asm.inst(
            f"ds_load_b128 v[{second + 4}:{second + 7}], "
            f"v{second_address} offset:{second_offset + second_chunk_offset}"
        )
        return 4

    def _emit_prefetched_wmma_pairs(
        self,
        asm: _Assembly,
        n_tiles: int,
        lds_arguments: tuple[int, tuple[int, ...], int, int, int, int],
    ) -> None:
        r = self.registers
        n_pairs = n_tiles // 2
        self._emit_lds_pair(
            asm,
            0,
            r.valu_b,
            r.valu_b + 8,
            *lds_arguments,
        )
        for pair in range(n_pairs):
            n_tile = 2 * pair
            buffer = 16 * (pair % 2)
            first = r.valu_b + buffer
            second = first + 8
            pending_loads = 0
            if pair + 1 < n_pairs:
                next_buffer = 16 * ((pair + 1) % 2)
                pending_loads = self._emit_lds_pair(
                    asm,
                    n_tile + 2,
                    r.valu_b + next_buffer,
                    r.valu_b + next_buffer + 8,
                    *lds_arguments,
                )
            self._emit_prefetched_wmma_pair(
                asm,
                n_tile,
                first,
                second,
                first_pair=pair == 0,
                pending_loads=pending_loads,
            )

    def _emit_prefetched_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        *,
        first_pair: bool,
        pending_loads: int,
    ) -> None:
        r = self.registers
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        if first_pair:
            pending_a_loads = 2 * (m_tiles - 1)
            asm.inst(f"s_waitcnt vmcnt({pending_a_loads}) lgkmcnt({pending_loads + 2})")
            self._emit_wmma_instruction(asm, 0, n_tile, r.valu_a, first)
            asm.inst(f"s_waitcnt lgkmcnt({pending_loads})")
            self._emit_wmma_instruction(asm, 0, n_tile + 1, r.valu_a, second)
            for m_tile in range(1, m_tiles):
                pending_a_loads -= 2
                asm.inst(f"s_waitcnt vmcnt({pending_a_loads})")
                valu_a = r.valu_a + 8 * m_tile
                self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
                self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)
            return

        asm.inst(f"s_waitcnt lgkmcnt({pending_loads + 2})")
        self._emit_wmma_pair(asm, n_tile, first)
        asm.inst(f"s_waitcnt lgkmcnt({pending_loads})")
        self._emit_wmma_pair(asm, n_tile + 1, second)

    def _emit_sia3_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        first: int,
        second: int,
        *,
        first_pair: bool,
        pending_second_loads: int,
        valu_a_base: int,
        trailing_a_loads: int,
    ) -> None:
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        if first_pair:
            pending_a_loads = 2 * (m_tiles - 1) + trailing_a_loads
            asm.inst(
                f"s_waitcnt vmcnt({pending_a_loads}) lgkmcnt({pending_second_loads})"
            )
            self._emit_wmma_instruction(asm, 0, n_tile, valu_a_base, first)
            asm.inst("s_waitcnt lgkmcnt(0)")
            self._emit_wmma_instruction(asm, 0, n_tile + 1, valu_a_base, second)
            for m_tile in range(1, m_tiles):
                pending_a_loads -= 2
                asm.inst(f"s_waitcnt vmcnt({pending_a_loads})")
                valu_a = valu_a_base + 8 * m_tile
                self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
                self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)
            return

        asm.inst(f"s_waitcnt lgkmcnt({pending_second_loads})")
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
        asm.inst("s_waitcnt lgkmcnt(0)")
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)

    def _emit_wmma_pair(
        self,
        asm: _Assembly,
        n_tile: int,
        valu_b: int,
        *,
        valu_a_base: int | None = None,
    ) -> None:
        r = self.registers
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        if valu_a_base is None:
            valu_a_base = r.valu_a
        for m_tile in range(m_tiles):
            valu_a = valu_a_base + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, valu_b)

    def _emit_wmma_instruction(
        self,
        asm: _Assembly,
        m_tile: int,
        n_tile: int,
        valu_a: int,
        valu_b: int,
    ) -> None:
        r = self.registers
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        accum = r.accum + (n_tiles * m_tile + n_tile) * 8
        asm.inst(
            f"v_wmma_f32_16x16x16_bf16 v[{accum}:{accum + 7}], "
            f"v[{valu_a}:{valu_a + 7}], v[{valu_b}:{valu_b + 7}], "
            f"v[{accum}:{accum + 7}]"
        )

    def _emit_store(self, asm: _Assembly) -> None:
        r = self.registers
        solution = self.solution_key.solution
        m_tiles = solution.matrix_instruction[5]
        n_tiles = solution.matrix_instruction[6]
        m_per_wave = 16 * m_tiles
        size = self.solution_key.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Map gfx11 physical C fragments to row-major grad_input.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 4, v{r.serial}")
        asm.inst(f"v_and_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {solution.macro_tile0.bit_length() - 1}, s2")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        self._emit_scale_u32(asm, t, 2 * size.n, t)
        asm.inst(
            f"v_lshlrev_b32 v{t + 1}, {(2 * solution.macro_tile1).bit_length() - 1}, s3"
        )
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        if self.solution_key.solution.store_priority_opt:
            asm.inst("s_setprio 1")
        for m_tile in range(m_tiles):
            for element in range(8):
                for n_tile in range(n_tiles):
                    accum = r.accum + (n_tiles * m_tile + n_tile) * 8 + element
                    self._emit_round_bf16(asm, accum, t + 2)
                    asm.inst(
                        f"global_store_d16_hi_b16 v{a}, v{accum}, "
                        f"s[{r.kernarg + 4}:{r.kernarg + 5}] offset:{32 * n_tile}"
                    )
                if not (m_tile == m_tiles - 1 and element == 7):
                    asm.inst(f"v_add_nc_u32 v{a}, {4 * size.n}, v{a}")
        if self.solution_key.solution.store_priority_opt:
            asm.inst("s_setprio 0")

    @staticmethod
    def _emit_round_bf16(
        asm: _Assembly, value_register: int, temporary_register: int
    ) -> None:
        asm.inst(f"v_bfe_u32 v{temporary_register}, v{value_register}, 16, 1")
        asm.inst(
            f"v_add3_u32 v{value_register}, v{temporary_register}, "
            f"v{value_register}, 0x7fff"
        )

    @staticmethod
    def _emit_scale_u32(
        asm: _Assembly,
        destination: int,
        scale: int,
        source: int,
    ) -> None:
        if scale > 0 and scale & (scale - 1) == 0:
            shift = scale.bit_length() - 1
            asm.inst(f"v_lshlrev_b32 v{destination}, {shift}, v{source}")
        else:
            asm.inst(f"v_mul_lo_u32 v{destination}, {scale}, v{source}")

    @staticmethod
    def _emit_add_pointer(
        asm: _Assembly,
        destination: int,
        scalar_pointer: int,
        offset: int,
    ) -> None:
        asm.inst(f"v_add_co_u32 v{destination}, vcc_lo, s{scalar_pointer}, v{offset}")
        asm.inst(
            f"v_add_co_ci_u32_e64 v{destination + 1}, null, "
            f"s{scalar_pointer + 1}, 0, vcc_lo"
        )

    @staticmethod
    def _emit_add_vector_offset(
        asm: _Assembly, destination: int, source: int, offset: int
    ) -> None:
        asm.inst(f"v_add_co_u32 v{destination}, vcc_lo, v{source}, v{offset}")
        asm.inst(
            f"v_add_co_ci_u32_e64 v{destination + 1}, null, v{source + 1}, 0, vcc_lo"
        )

    @staticmethod
    def _emit_add_literal64(
        asm: _Assembly, destination: int, source: int, literal: int
    ) -> None:
        asm.inst(f"v_add_co_u32 v{destination}, vcc_lo, v{source}, {literal}")
        asm.inst(
            f"v_add_co_ci_u32_e64 v{destination + 1}, null, v{source + 1}, 0, vcc_lo"
        )
