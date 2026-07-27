import contextlib
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .model import SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class KernelWriterError(RuntimeError):
    pass


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

    def comment(self, text: str) -> None:
        self.lines.append(f"// {text}")

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def inst(self, text: str, comment: str = "") -> None:
        suffix = f" // {comment}" if comment else ""
        self.lines.append(f"  {text}{suffix}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class KernelWriterAssembly:
    """Emit the exact gfx1151 Q4_K dense-backward pilot solution."""

    def __init__(self, solution_key: SolutionKey, toolchain: Toolchain) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise KernelWriterError(f"solution rejected: {details}")
        self.solution_key = solution_key
        self.toolchain = toolchain
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
        signature.addDescriptionTopic("GGTensile Q4_K dense MMQ backward")
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

    def _allocate_registers(self) -> RegisterLayout:
        from rocisa.enum import RegisterType
        from rocisa.register import RegisterPool

        vgprs = RegisterPool(256, RegisterType.Vgpr, True)
        vgprs.addRange(0, 255, "GGTensile VGPRs")
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        decoder_rows = n_tiles // 4
        accum = vgprs.checkOutAligned(8 * m_tiles * n_tiles, 8, "accumulators")
        valu_a = vgprs.checkOutAligned(8 * m_tiles, 4, "ValuA")
        valu_b = vgprs.checkOutAligned(
            16 * self.solution_key.solution.prefetch_local_read,
            4,
            "ValuB",
        )
        global_read_b = vgprs.checkOutAligned(4 * decoder_rows, 4, "GlobalReadB")
        quant_dm = vgprs.checkOut(decoder_rows, "Q4K dm")
        quant_scale = vgprs.checkOut(3 * decoder_rows, "Q4K scale/min")
        lds_address = -1
        swizzle = self.solution_key.solution.lds_swizzle_chunk_b
        if swizzle:
            lds_address = vgprs.checkOut(32 // swizzle, "swizzled LDS addresses")
        address = vgprs.checkOutAligned(12, 2, "addresses")
        temporary = vgprs.checkOut(7, "temporaries")
        serial = vgprs.checkOut(1, "Serial")

        sgprs = RegisterPool(64, RegisterType.Sgpr, True)
        sgprs.addRange(5, 63, "GGTensile SGPRs")
        kernarg = sgprs.checkOutAligned(10, 2, "loaded kernargs")
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
            total_vgprs=serial + 1,
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
        asm.inst(f"s_load_dwordx4 s[{kernarg + 6}:{kernarg + 9}], s[0:1], 0x18")
        asm.inst("s_waitcnt lgkmcnt(0)")
        accumulator_count = (
            8
            * self.solution_key.solution.matrix_instruction[5]
            * self.solution_key.solution.matrix_instruction[6]
        )
        for register in range(r.accum, r.accum + accumulator_count):
            asm.inst(f"v_mov_b32 v{register}, 0")

        n_per_block = self.solution_key.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_block
        tile_shift = tiles_per_weight_block.bit_length() - 1
        n_shift = n_per_block.bit_length() - 1
        asm.comment("Static packed-row and input-half coordinates.")
        asm.inst(f"s_lshr_b32 s{r.block_offset}, s3, {tile_shift}")
        asm.inst(f"s_mul_i32 s{r.block_offset}, s{r.block_offset}, 144")
        asm.inst(f"s_lshr_b32 s{r.input_half}, s3, {tile_shift - 1}")
        asm.inst(f"s_and_b32 s{r.input_half}, s{r.input_half}, 1")
        asm.inst(f"s_and_b32 s{r.scalar_temporary}, s3, {tiles_per_weight_block - 1}")
        asm.inst(f"s_lshl_b32 s{r.scalar_temporary}, s{r.scalar_temporary}, {n_shift}")
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")

        self._emit_static_thread_coordinates(asm)
        asm.label(".LDepthULoop")
        if self.solution_key.solution.num_threads > 128:
            asm.comment("Only the first four waves cooperatively decode B.")
            asm.inst(f"v_readfirstlane_b32 s{r.scalar_temporary + 1}, v{r.serial}")
            asm.inst(f"s_cmp_lt_u32 s{r.scalar_temporary + 1}, 128")
            asm.inst("s_cbranch_scc0 .LDecodeReady")
        self._emit_q4_k_global_reads(asm)
        self._emit_q4_k_decode(asm)
        if self.solution_key.solution.num_threads > 128:
            asm.label(".LDecodeReady")
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst("buffer_gl0_inv")
        self._emit_wmma(asm)
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(f"s_add_u32 s{r.loop_counter}, s{r.loop_counter}, 32")
        asm.inst(f"s_cmp_lt_u32 s{r.loop_counter}, {size.k}")
        asm.inst("s_cbranch_scc1 .LDepthULoop")
        asm.inst("s_nop 7", "cover final WMMA result latency")
        self._emit_store(asm)
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _emit_static_thread_coordinates(self, asm: _Assembly) -> None:
        r = self.registers
        a = r.address
        t = r.temporary
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        k_shift = n_tiles.bit_length() - 1
        asm.comment("v[address+10] is the LDS row base; +11 is nibble shift.")
        asm.inst(f"v_and_b32 v{a + 10}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{a + 10}, 10, v{a + 10}")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 1, v{t}")
        asm.inst(f"v_add_nc_u32 v{a + 10}, v{a + 10}, v{t}")
        asm.inst(f"v_and_b32 v{a + 11}, 2, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{a + 11}, 1, v{a + 11}")
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
            asm.inst(f"v_sub_nc_u32 v{t + 1}, v{a + 10}, v{t + 1}")
            for residue in range(residues):
                asm.inst(f"v_xor_b32 v{lds + residue}, {residue}, v{t}")
                asm.inst(
                    f"v_lshlrev_b32 v{lds + residue}, {swizzle_shift}, v{lds + residue}"
                )
                asm.inst(f"v_add_nc_u32 v{lds + residue}, v{t + 1}, v{lds + residue}")

    def _emit_q4_k_global_reads(self, asm: _Assembly) -> None:
        r = self.registers
        n_tiles = self.solution_key.solution.matrix_instruction[6]
        decoder_rows = n_tiles // 4
        k_shift = n_tiles.bit_length() - 1
        k_span = 32 // decoder_rows
        row_delta = k_span * 1152
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
        asm.inst(f"v_mul_lo_u32 v{t}, 1152, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        self._emit_add_pointer(asm, a, r.kernarg + 2, t)
        if decoder_rows == 2:
            asm.inst(f"v_add_co_u32 v{a + 2}, vcc_lo, v{a}, {row_delta}")
            asm.inst(f"v_add_co_ci_u32_e64 v{a + 3}, null, v{a + 1}, 0, vcc_lo")

        asm.comment("Build quant and scale-byte addresses.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary}, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 6, v{t}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 5, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 2}, 31, v{t}")
        asm.inst(f"v_add_nc_u32 v{t + 1}, v{t + 1}, v{t + 2}")
        for row in range(decoder_rows):
            self._emit_add_vector_offset(asm, a + 4 + 2 * row, a + 2 * row, t + 1)
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 5, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 3, v{t + 1}")
        self._emit_add_vector_offset(asm, a + 8, a, t + 1)

        for row in range(decoder_rows):
            asm.inst(
                f"global_load_b128 v[{q + 4 * row}:{q + 4 * row + 3}], "
                f"v[{a + 4 + 2 * row}:{a + 5 + 2 * row}], off offset:16"
            )
        for row in range(decoder_rows):
            asm.inst(
                f"global_load_b32 v{dm + row}, v[{a + 2 * row}:{a + 2 * row + 1}], off"
            )
        for row in range(decoder_rows):
            address_pair = a + 8
            if row:
                self._emit_add_literal64(asm, a + 6, a + 8, row * row_delta)
                address_pair = a + 6
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row}, v[{address_pair}:{address_pair + 1}], off offset:4"
            )
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row + 1}, v[{address_pair}:{address_pair + 1}], off offset:8"
            )
            asm.inst(
                f"global_load_d16_u8 v{scale + 3 * row + 2}, v[{address_pair}:{address_pair + 1}], off offset:12"
            )
        asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q4_k_decode(self, asm: _Assembly) -> None:
        r = self.registers
        decoder_rows = self.solution_key.solution.matrix_instruction[6] // 4
        scale = r.quant_scale
        t = r.temporary
        asm.comment("Unpack Q4_K six-bit scale/min fields.")
        asm.inst(f"s_cmp_eq_u32 s{r.input_half}, 0")
        asm.inst("s_cbranch_scc0 .LScaleOdd")
        for row in range(decoder_rows):
            asm.inst(f"v_and_b32 v{scale + 3 * row}, 0x3f, v{scale + 3 * row}")
            asm.inst(f"v_and_b32 v{scale + 3 * row + 1}, 0x3f, v{scale + 3 * row + 1}")
        asm.inst("s_branch .LScaleReady")
        asm.label(".LScaleOdd")
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
        asm.label(".LScaleReady")

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

        asm.comment("Decode 16 nibbles from each packed output row into LDS.")
        for row in range(decoder_rows):
            for element in range(16):
                packed = r.global_read_b + 4 * row + element // 4
                byte_shift = 8 * (element % 4)
                value = t + 5
                rounding = t + 6
                d_scaled = t + 1 + 2 * row
                min_scaled = d_scaled + 1
                lds_offset = 64 * element + 32 * row
                lds_address = r.address + 10
                swizzle = self.solution_key.solution.lds_swizzle_chunk_b
                if swizzle:
                    residues = 32 // swizzle
                    residue = element % residues
                    lds_address = r.lds_address + residue
                    if row:
                        lds_offset = 64 * element + (
                            32 if residue < residues // 2 else -32
                        )
                asm.inst(f"v_bfe_u32 v{value}, v{packed}, {byte_shift}, 8")
                asm.inst(f"v_bfe_u32 v{value}, v{value}, v{r.address + 11}, 4")
                asm.inst(f"v_cvt_f32_ubyte0_e32 v{value}, v{value}")
                asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
                self._emit_round_bf16(asm, value, rounding)
                asm.inst(
                    f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
                )

    def _emit_wmma(self, asm: _Assembly) -> None:
        r = self.registers
        solution = self.solution_key.solution
        m_tiles = solution.matrix_instruction[5]
        n_tiles = solution.matrix_instruction[6]
        m_per_wave = 16 * m_tiles
        size = self.solution_key.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Load A and issue two DepthU=16 WMMA halves.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, {m_per_wave.bit_length() - 1}, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{a + 8}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 2}, {solution.macro_tile0.bit_length() - 1}, s2")
        asm.inst(f"v_add_nc_u32 v{a + 8}, v{a + 8}, v{t + 2}")
        if m_tiles == 2:
            asm.inst(f"v_add_nc_u32 v{a + 9}, 16, v{a + 8}")
        for k_tile in (0, 16):
            asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
            if m_tiles == 2:
                for m_tile, row, pointer in (
                    (0, a + 8, a),
                    (1, a + 9, a + 2),
                ):
                    asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.k}, v{row}")
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
                for m_tile in range(m_tiles):
                    if m_tile:
                        asm.inst(f"v_add_nc_u32 v{t}, {16 * m_tile}, v{a + 8}")
                        row = t
                    else:
                        row = a + 8
                    asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.k}, v{row}")
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
            asm.inst(f"v_and_b32 v{t}, 15, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 6, v{t}")
            swizzle = self.solution_key.solution.lds_swizzle_chunk_b
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
                if k_tile:
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
            lds_arguments = (
                swizzle,
                chunk_addresses,
                first_address,
                second_address,
                second_chunk_offset,
            )
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
                    load_count = self._emit_lds_pair(
                        asm,
                        n_tile,
                        first,
                        second,
                        *lds_arguments,
                    )
                    if self.solution_key.solution.schedule_iter_alg == 3:
                        self._emit_sia3_wmma_pair(
                            asm,
                            n_tile,
                            first,
                            second,
                            first_pair=n_tile == 0,
                            pending_second_loads=load_count // 2,
                        )
                    else:
                        asm.inst("s_waitcnt vmcnt(0) lgkmcnt(0)")
                        self._emit_wmma_pair(asm, n_tile, first)
                        self._emit_wmma_pair(asm, n_tile + 1, second)

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
    ) -> int:
        first_offset = 1024 * n_tile
        second_offset = first_offset + 1024
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
        lds_arguments: tuple[int, tuple[int, ...], int, int, int],
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
    ) -> None:
        r = self.registers
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        if first_pair:
            pending_a_loads = 2 * (m_tiles - 1)
            asm.inst(
                f"s_waitcnt vmcnt({pending_a_loads}) lgkmcnt({pending_second_loads})"
            )
            self._emit_wmma_instruction(asm, 0, n_tile, r.valu_a, first)
            asm.inst("s_waitcnt lgkmcnt(0)")
            self._emit_wmma_instruction(asm, 0, n_tile + 1, r.valu_a, second)
            for m_tile in range(1, m_tiles):
                pending_a_loads -= 2
                asm.inst(f"s_waitcnt vmcnt({pending_a_loads})")
                valu_a = r.valu_a + 8 * m_tile
                self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
                self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)
            return

        asm.inst(f"s_waitcnt lgkmcnt({pending_second_loads})")
        for m_tile in range(m_tiles):
            valu_a = r.valu_a + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile, valu_a, first)
        asm.inst("s_waitcnt lgkmcnt(0)")
        for m_tile in range(m_tiles):
            valu_a = r.valu_a + 8 * m_tile
            self._emit_wmma_instruction(asm, m_tile, n_tile + 1, valu_a, second)

    def _emit_wmma_pair(self, asm: _Assembly, n_tile: int, valu_b: int) -> None:
        r = self.registers
        m_tiles = self.solution_key.solution.matrix_instruction[5]
        for m_tile in range(m_tiles):
            valu_a = r.valu_a + 8 * m_tile
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
        asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.n}, v{t}")
        asm.inst(
            f"v_lshlrev_b32 v{t + 1}, {(2 * solution.macro_tile1).bit_length() - 1}, s3"
        )
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        self._emit_add_pointer(asm, a, r.kernarg + 4, t)
        if self.solution_key.solution.store_priority_opt:
            asm.inst("s_setprio 1")
        for m_tile in range(m_tiles):
            for element in range(8):
                for n_tile in range(n_tiles):
                    accum = r.accum + (n_tiles * m_tile + n_tile) * 8 + element
                    self._emit_round_bf16(asm, accum, t + 2)
                    asm.inst(
                        f"global_store_d16_hi_b16 v[{a}:{a + 1}], v{accum}, off offset:{32 * n_tile}"
                    )
                if not (m_tile == m_tiles - 1 and element == 7):
                    asm.inst(f"v_add_co_u32 v{a}, vcc_lo, v{a}, {4 * size.n}")
                    asm.inst(f"v_add_co_ci_u32_e64 v{a + 1}, null, v{a + 1}, 0, vcc_lo")
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
