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
            sgprWorkGroup=(1, 1, 0),
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

    @staticmethod
    def _allocate_registers() -> RegisterLayout:
        from rocisa.enum import RegisterType
        from rocisa.register import RegisterPool

        vgprs = RegisterPool(256, RegisterType.Vgpr, True)
        vgprs.addRange(0, 255, "GGTensile VGPRs")
        accum = vgprs.checkOutAligned(128, 8, "accumulators")
        valu_a = vgprs.checkOutAligned(16, 4, "ValuA")
        valu_b = vgprs.checkOutAligned(16, 4, "ValuB")
        global_read_b = vgprs.checkOutAligned(8, 4, "GlobalReadB")
        quant_dm = vgprs.checkOut(2, "Q4K dm")
        quant_scale = vgprs.checkOut(6, "Q4K scale/min")
        address = vgprs.checkOutAligned(12, 2, "addresses")
        temporary = vgprs.checkOut(7, "temporaries")
        serial = vgprs.checkOut(1, "Serial")

        sgprs = RegisterPool(64, RegisterType.Sgpr, True)
        sgprs.addRange(4, 63, "GGTensile SGPRs")
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
        asm.inst("s_load_dwordx4 s[4:7], s[0:1], 0x0")
        asm.inst("s_load_dwordx4 s[8:11], s[0:1], 0x10")
        asm.inst("s_load_dwordx2 s[12:13], s[0:1], 0x20")
        asm.inst("s_waitcnt lgkmcnt(0)")
        for register in range(r.accum, r.accum + 128):
            asm.inst(f"v_mov_b32 v{register}, 0")

        asm.comment("Static packed-row and input-half coordinates.")
        asm.inst(f"s_lshr_b32 s{r.block_offset}, s3, 1")
        asm.inst(f"s_mul_i32 s{r.block_offset}, s{r.block_offset}, 144")
        asm.inst(f"s_and_b32 s{r.input_half}, s3, 1")
        asm.inst(f"s_lshl_b32 s{r.scalar_temporary}, s{r.input_half}, 7")
        asm.inst(f"s_mov_b32 s{r.loop_counter}, 0")

        self._emit_static_thread_coordinates(asm)
        asm.label(".LDepthULoop")
        self._emit_q4_k_global_reads(asm)
        self._emit_q4_k_decode(asm)
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
        asm.comment("v[address+10] is the LDS row base; +11 is nibble shift.")
        asm.inst(f"v_and_b32 v{a + 10}, 7, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{a + 10}, 10, v{a + 10}")
        asm.inst(f"v_lshrrev_b32 v{t}, 3, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 1, v{t}")
        asm.inst(f"v_add_nc_u32 v{a + 10}, v{a + 10}, v{t}")
        asm.inst(f"v_and_b32 v{a + 11}, 2, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{a + 11}, 1, v{a + 11}")

    def _emit_q4_k_global_reads(self, asm: _Assembly) -> None:
        r = self.registers
        q = r.global_read_b
        dm = r.quant_dm
        scale = r.quant_scale
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q4_K block addresses for output rows k and k+16.")
        asm.inst(f"v_lshrrev_b32 v{t}, 3, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, 1152, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        self._emit_add_pointer(asm, a, 6, t)
        asm.inst(f"v_add_co_u32 v{a + 2}, vcc_lo, v{a}, 18432")
        asm.inst(f"v_add_co_ci_u32_e64 v{a + 3}, null, v{a + 1}, 0, vcc_lo")

        asm.comment("Build quant and scale-byte addresses.")
        asm.inst(f"v_and_b32 v{t}, 7, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary}, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 6, v{t}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 5, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 2}, 31, v{t}")
        asm.inst(f"v_add_nc_u32 v{t + 1}, v{t + 1}, v{t + 2}")
        self._emit_add_vector_offset(asm, a + 4, a, t + 1)
        self._emit_add_vector_offset(asm, a + 6, a + 2, t + 1)
        asm.inst(f"v_and_b32 v{t + 1}, 7, v{r.serial}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 1, v{t + 1}")
        self._emit_add_vector_offset(asm, a + 8, a, t + 1)

        asm.inst(f"global_load_b128 v[{q}:{q + 3}], v[{a + 4}:{a + 5}], off offset:16")
        asm.inst(
            f"global_load_b128 v[{q + 4}:{q + 7}], v[{a + 6}:{a + 7}], off offset:16"
        )
        asm.inst(f"global_load_b32 v{dm}, v[{a}:{a + 1}], off")
        asm.inst(f"global_load_b32 v{dm + 1}, v[{a + 2}:{a + 3}], off")
        for row, base_address in ((0, a + 8), (1, a + 8)):
            address_pair = base_address if row == 0 else a + 8
            block_delta = 0 if row == 0 else 18432
            if block_delta:
                self._emit_add_literal64(asm, a + 6, a + 8, block_delta)
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
        scale = r.quant_scale
        t = r.temporary
        asm.comment("Unpack Q4_K six-bit scale/min fields.")
        asm.inst(f"s_cmp_eq_u32 s{r.input_half}, 0")
        asm.inst("s_cbranch_scc0 .LScaleOdd")
        for row in range(2):
            asm.inst(f"v_and_b32 v{scale + 3 * row}, 0x3f, v{scale + 3 * row}")
            asm.inst(f"v_and_b32 v{scale + 3 * row + 1}, 0x3f, v{scale + 3 * row + 1}")
        asm.inst("s_branch .LScaleReady")
        asm.label(".LScaleOdd")
        for row in range(2):
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
        for row in range(2):
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
        for row in range(2):
            for element in range(16):
                packed = r.global_read_b + 4 * row + element // 4
                byte_shift = 8 * (element % 4)
                value = t + 5
                rounding = t + 6
                d_scaled = t + 1 + 2 * row
                min_scaled = d_scaled + 1
                lds_offset = 64 * element + 32 * row
                asm.inst(f"v_bfe_u32 v{value}, v{packed}, {byte_shift}, 8")
                asm.inst(f"v_bfe_u32 v{value}, v{value}, v{r.address + 11}, 4")
                asm.inst(f"v_cvt_f32_ubyte0_e32 v{value}, v{value}")
                asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
                self._emit_round_bf16(asm, value, rounding)
                asm.inst(
                    f"ds_store_b16_d16_hi v{r.address + 10}, v{value} offset:{lds_offset}"
                )

    def _emit_wmma(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Load A and issue two DepthU=16 WMMA halves.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 5, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{a + 8}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 2}, 7, s2")
        asm.inst(f"v_add_nc_u32 v{a + 8}, v{a + 8}, v{t + 2}")
        asm.inst(f"v_add_nc_u32 v{a + 9}, 16, v{a + 8}")
        for k_tile in (0, 16):
            asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.k}, v{a + 8}")
            asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
            asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
            if k_tile:
                asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
            self._emit_add_pointer(asm, a, 4, t)
            asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.k}, v{a + 9}")
            asm.inst(f"v_add_nc_u32 v{t}, s{r.scalar_temporary + 1}, v{t}")
            if k_tile:
                asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
            self._emit_add_pointer(asm, a + 2, 4, t)
            asm.inst(
                f"global_load_b128 v[{r.valu_a}:{r.valu_a + 3}], v[{a}:{a + 1}], off"
            )
            asm.inst(
                f"global_load_b128 v[{r.valu_a + 4}:{r.valu_a + 7}], v[{a}:{a + 1}], off offset:16"
            )
            asm.inst(
                f"global_load_b128 v[{r.valu_a + 8}:{r.valu_a + 11}], v[{a + 2}:{a + 3}], off"
            )
            asm.inst(
                f"global_load_b128 v[{r.valu_a + 12}:{r.valu_a + 15}], v[{a + 2}:{a + 3}], off offset:16"
            )
            asm.inst(f"v_and_b32 v{t}, 15, v{r.serial}")
            asm.inst(f"v_lshlrev_b32 v{t}, 6, v{t}")
            if k_tile:
                asm.inst(f"v_add_nc_u32 v{t}, {2 * k_tile}, v{t}")
            for n_tile in range(0, 8, 2):
                first = r.valu_b
                second = r.valu_b + 8
                first_offset = 1024 * n_tile
                second_offset = first_offset + 1024
                asm.inst(
                    f"ds_load_b128 v[{first}:{first + 3}], v{t} offset:{first_offset}"
                )
                asm.inst(
                    f"ds_load_b128 v[{first + 4}:{first + 7}], v{t} offset:{first_offset + 16}"
                )
                asm.inst(
                    f"ds_load_b128 v[{second}:{second + 3}], v{t} offset:{second_offset}"
                )
                asm.inst(
                    f"ds_load_b128 v[{second + 4}:{second + 7}], v{t} offset:{second_offset + 16}"
                )
                asm.inst("s_waitcnt vmcnt(0) lgkmcnt(0)")
                self._emit_wmma_pair(asm, n_tile, first)
                self._emit_wmma_pair(asm, n_tile + 1, second)

    def _emit_wmma_pair(self, asm: _Assembly, n_tile: int, valu_b: int) -> None:
        r = self.registers
        for m_tile, valu_a in ((0, r.valu_a), (1, r.valu_a + 8)):
            accum = r.accum + (8 * m_tile + n_tile) * 8
            asm.inst(
                f"v_wmma_f32_16x16x16_bf16 v[{accum}:{accum + 7}], "
                f"v[{valu_a}:{valu_a + 7}], v[{valu_b}:{valu_b + 7}], "
                f"v[{accum}:{accum + 7}]"
            )

    def _emit_store(self, asm: _Assembly) -> None:
        r = self.registers
        size = self.solution_key.problem_size
        a = r.address
        t = r.temporary
        asm.comment("Map gfx11 physical C fragments to row-major grad_input.")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 5, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 4, v{r.serial}")
        asm.inst(f"v_and_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 7, s2")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_mul_lo_u32 v{t}, {2 * size.n}, v{t}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 8, s3")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 15, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        self._emit_add_pointer(asm, a, 8, t)
        if self.solution_key.solution.store_priority_opt:
            asm.inst("s_setprio 1")
        for m_tile in range(2):
            for element in range(8):
                for n_tile in range(8):
                    accum = r.accum + (8 * m_tile + n_tile) * 8 + element
                    self._emit_round_bf16(asm, accum, t + 2)
                    asm.inst(
                        f"global_store_d16_hi_b16 v[{a}:{a + 1}], v{accum}, off offset:{32 * n_tile}"
                    )
                if not (m_tile == 1 and element == 7):
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
