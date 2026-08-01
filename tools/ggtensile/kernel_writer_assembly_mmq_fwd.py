import contextlib
import hashlib
import tempfile
from pathlib import Path

from .model import DenseForwardSolution, SolutionKey
from .toolchain import Toolchain
from .validation import validate_solution


class ForwardKernelWriterError(RuntimeError):
    pass


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


class DenseForwardKernelWriterAssembly:
    """Emit the strict one-wave Q4_K/Q8_1_DS4 forward control."""

    TOTAL_VGPRS = 88
    TOTAL_VGPRS_REUSE = 164
    TOTAL_VGPRS_BATCH = 194
    TOTAL_SGPRS = 16

    C = 0
    SUM = 8
    WEIGHT_Q = 16
    ACTIVATION_Q = 24
    WEIGHT_METADATA = 32
    RESULT_WEIGHT_ADDRESS = 64
    WEIGHT_Q_ADDRESS = 72
    ACTIVATION_ADDRESS_0 = 73
    ACTIVATION_ADDRESS_1 = 74
    OUTPUT_ADDRESS = 75
    SCALE = 76
    MINIMUM = 77
    SCALED_DM = 78
    ACTIVATION_DS = 79
    WEIGHT_D = 80
    WEIGHT_MIN = 81
    ACTIVATION_D = 82
    ACTIVATION_SUM = 83
    TEMPORARY = 84
    OUTPUT_COLUMN = 85
    SERIAL = 86
    ACTIVATION_ROW = 87

    KERNARG = 4
    LOOP_COUNTER = 10
    SCALAR_TEMPORARY = 11

    def __init__(self, solution_key: SolutionKey, toolchain: Toolchain) -> None:
        reasons = validate_solution(solution_key)
        if reasons:
            details = "; ".join(
                f"{reason.rule_id}: {reason.message}" for reason in reasons
            )
            raise ForwardKernelWriterError(f"solution rejected: {details}")
        if not isinstance(solution_key.solution, DenseForwardSolution):
            raise ForwardKernelWriterError("forward writer requires DenseForwardSolution")
        self.solution_key = solution_key
        self.toolchain = toolchain

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
            raise ForwardKernelWriterError(
                "rocisa is required to generate assembly"
            ) from error

        solution = self.solution_key.solution
        global_isa = rocisa.rocIsa.getInstance()
        with tempfile.TemporaryDirectory(prefix="ggtensile-forward-rocisa-") as temp:
            with contextlib.chdir(temp):
                global_isa.init(solution.isa, str(self.toolchain.assembler), False)
        global_isa.setKernel(solution.isa, solution.wavefront_size)

        signature = code.SignatureBase(
            kernelName=self.solution_key.kernel_name,
            kernArgsVersion=0,
            codeObjectVersion="5",
            groupSegmentSize=solution.lds_num_bytes,
            sgprWorkGroup=(1, 1, 0),
            vgprWorkItem=0,
            flatWorkGroupSize=solution.num_threads,
            totalVgprs=self._total_vgprs(),
            totalAgprs=0,
            totalSgprs=self.TOTAL_SGPRS,
        )
        signature.addDescriptionTopic(
            "GGTensile Q4_K dense MMQ forward, fixed Q8_1_DS4 producer"
        )
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("nrows_weight", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation", SVK.SIG_VALUE, "u32")
        signature.addArg("nrows_activation_padded", SVK.SIG_VALUE, "u32")
        signature.addArg("blocks_per_weight_row", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileForwardKernel")
        module.add(signature)
        module.add(code.TextBlock(self._body()))
        return str(module)

    def _body(self) -> str:
        if self._uses_wave_reuse():
            return self._body_wave_reuse()
        asm = _Assembly()
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name
        row_stride = size.k // 256 * 144
        activation_plane_stride = size.m * 144
        activation_block_stride = 2 * activation_plane_stride

        asm.comment("Load the exact packed-weight, DS4-workspace, and output pointers.")
        asm.inst(f"s_load_dwordx2 s[{self.KERNARG}:{self.KERNARG + 1}], s[0:1], 0x0")
        asm.inst(
            f"s_load_dwordx2 s[{self.KERNARG + 2}:{self.KERNARG + 3}], s[0:1], 0x8"
        )
        asm.inst(
            f"s_load_dwordx2 s[{self.KERNARG + 4}:{self.KERNARG + 5}], s[0:1], 0x10"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")

        asm.comment("Map one wave to an exact 16x16 output tile.")
        asm.inst(f"v_mov_b32 v{self.SERIAL}, v0")
        asm.inst(f"v_and_b32 v{self.OUTPUT_COLUMN}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_COLUMN}, v{self.TEMPORARY}, "
            f"v{self.OUTPUT_COLUMN}"
        )
        asm.inst(f"v_and_b32 v{self.ACTIVATION_ROW}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ROW}, v{self.TEMPORARY}, "
            f"v{self.ACTIVATION_ROW}"
        )
        asm.comment("Build one Q payload row and eight C-fragment metadata rows.")
        asm.inst(f"v_lshrrev_b32 v{self.TEMPORARY}, 4, v{self.SERIAL}")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(f"v_lshlrev_b32 v{self.SCALE}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.SCALE}, "
            f"v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.RESULT_WEIGHT_ADDRESS}, {row_stride}, "
            f"v{self.TEMPORARY}"
        )
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"{2 * element * row_stride}, v{self.RESULT_WEIGHT_ADDRESS}"
            )
        asm.inst(
            f"v_mul_lo_u32 v{self.WEIGHT_Q_ADDRESS}, {row_stride}, "
            f"v{self.OUTPUT_COLUMN}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.ACTIVATION_ADDRESS_0}, 144, "
            f"v{self.ACTIVATION_ROW}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_plane_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        for register in range(self.SUM, self.SUM + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ4KBlockLoop")
        asm.comment("Keep one Q4_K block's dm/scales live across its eight groups.")
        for element in range(8):
            metadata = self.WEIGHT_METADATA + 4 * element
            asm.inst(
                f"global_load_b128 v[{metadata}:{metadata + 3}], "
                f"v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"s[{self.KERNARG}:{self.KERNARG + 1}]"
            )
        for group in range(8):
            self._emit_group(asm, group)
        for element in range(8):
            asm.inst(
                f"v_add_nc_u32 v{self.RESULT_WEIGHT_ADDRESS + element}, 144, "
                f"v{self.RESULT_WEIGHT_ADDRESS + element}"
            )
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_Q_ADDRESS}, 144, "
            f"v{self.WEIGHT_Q_ADDRESS}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_0}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_1}"
        )
        asm.inst(
            f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1"
        )
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {size.k // 256}")
        asm.inst("s_cbranch_scc1 .LForwardQ4KBlockLoop")

        self._emit_store(asm)
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _uses_wave_reuse(self) -> bool:
        return self.solution_key.solution.operand_source in (
            "GlobalWaveReuse",
            "GlobalWaveBatch4",
        )

    def _uses_wave_batch(self) -> bool:
        return self.solution_key.solution.operand_source == "GlobalWaveBatch4"

    def _total_vgprs(self) -> int:
        if self._uses_wave_batch():
            return self.TOTAL_VGPRS_BATCH
        return self.TOTAL_VGPRS_REUSE if self._uses_wave_reuse() else self.TOTAL_VGPRS

    def _body_wave_reuse(self) -> str:
        """Reuse one Q4_K weight tile across eight activation-row tiles per wave."""
        asm = _Assembly()
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name
        row_stride = size.k // 256 * 144
        activation_plane_stride = size.m * 144
        activation_block_stride = 2 * activation_plane_stride
        sum_base = 8
        weight_q = 72
        batch = self._uses_wave_batch()
        activation_q = 80
        metadata = 80 if batch else 88
        result_addresses = 176 if batch else 136
        weight_q_address = 184 if batch else 144
        activation_address_0 = 185 if batch else 145
        activation_address_1 = 186 if batch else 146
        output_address = 187 if batch else 147
        scale = 0
        minimum = 1
        scaled_dm = 2
        activation_ds = 151
        weight_d = 3
        weight_min = 4
        activation_d = 154
        temporary = 188 if batch else 156
        serial = 189 if batch else 157
        output_column = 190 if batch else 158
        wave_column_base = 191 if batch else 159
        activation_row = 192 if batch else 160
        lane = 193 if batch else 161

        asm.comment("Load the exact packed-weight, DS4-workspace, and output pointers.")
        asm.inst(f"s_load_dwordx2 s[{self.KERNARG}:{self.KERNARG + 1}], s[0:1], 0x0")
        asm.inst(
            f"s_load_dwordx2 s[{self.KERNARG + 2}:{self.KERNARG + 3}], s[0:1], 0x8"
        )
        asm.inst(
            f"s_load_dwordx2 s[{self.KERNARG + 4}:{self.KERNARG + 5}], s[0:1], 0x10"
        )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst(f"v_mov_b32 v{serial}, v0")
        asm.inst(f"v_and_b32 v{lane}, 15, v{serial}")
        asm.inst(f"v_lshrrev_b32 v{wave_column_base}, 5, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{wave_column_base}, 4, v{wave_column_base}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(
            f"v_add_nc_u32 v{wave_column_base}, v{temporary}, "
            f"v{wave_column_base}"
        )
        asm.inst(
            f"v_add_nc_u32 v{output_column}, v{wave_column_base}, v{lane}"
        )
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{result_addresses}, {row_stride}, v{temporary}"
        )
        for element in range(1, 8):
            asm.inst(
                f"v_add_nc_u32 v{result_addresses + element}, "
                f"{2 * element * row_stride}, v{result_addresses}"
            )
        asm.inst(f"v_mul_lo_u32 v{weight_q_address}, {row_stride}, v{output_column}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{activation_row}, v{temporary}, v{lane}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{temporary}, v{lane}")
        asm.inst(f"v_mul_lo_u32 v{activation_address_0}, 144, v{temporary}")
        asm.inst(
            f"v_add_nc_u32 v{activation_address_1}, {activation_plane_stride}, "
            f"v{activation_address_0}"
        )
        for register in range(sum_base, sum_base + 64):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        asm.label(".LForwardQ4KWaveReuseBlockLoop")
        for element in range(8):
            asm.inst(
                f"global_load_b128 v[{metadata + 4 * element}:{metadata + 4 * element + 3}], "
                f"v{result_addresses + element}, s[{self.KERNARG}:{self.KERNARG + 1}]"
            )
        asm.inst("s_waitcnt vmcnt(0)")
        for group in range(8):
            group_emitter = (
                self._emit_wave_batch_group
                if batch
                else self._emit_wave_reuse_group
            )
            group_emitter(
                asm,
                group,
                weight_q,
                activation_q,
                metadata,
                activation_address_0,
                activation_address_1,
                weight_q_address,
                sum_base,
                scale,
                minimum,
                scaled_dm,
                activation_ds,
                weight_d,
                weight_min,
                activation_d,
                temporary,
            )
        for element in range(8):
            asm.inst(
                f"v_add_nc_u32 v{result_addresses + element}, 144, "
                f"v{result_addresses + element}"
            )
        asm.inst(f"v_add_nc_u32 v{weight_q_address}, 144, v{weight_q_address}")
        asm.inst(
            f"v_add_nc_u32 v{activation_address_0}, {activation_block_stride}, "
            f"v{activation_address_0}"
        )
        asm.inst(
            f"v_add_nc_u32 v{activation_address_1}, {activation_block_stride}, "
            f"v{activation_address_1}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {size.k // 256}")
        asm.inst("s_cbranch_scc1 .LForwardQ4KWaveReuseBlockLoop")

        asm.comment("Store eight reused activation-row tiles as row-major BF16 output.")
        for tile in range(8):
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {16 * tile}, v{activation_row}"
            )
            asm.inst(
                f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{temporary}"
            )
            asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
            asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}"
            )
            asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
            asm.inst(
                f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}"
            )
            for element in range(8):
                total = sum_base + tile * 8 + element
                asm.inst(f"v_bfe_u32 v{temporary}, v{total}, 16, 1")
                asm.inst(
                    f"v_add3_u32 v{total}, v{temporary}, v{total}, 0x7fff"
                )
                asm.inst(
                    f"global_store_d16_hi_b16 v{output_address}, v{total}, "
                    f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                )
                if element != 7:
                    asm.inst(f"v_add_nc_u32 v{output_address}, 4, v{output_address}")
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _emit_wave_batch_group(
        self,
        asm: _Assembly,
        group: int,
        weight_q: int,
        activation_q: int,
        metadata: int,
        activation_address_0: int,
        activation_address_1: int,
        weight_q_address: int,
        sum_base: int,
        scale: int,
        minimum: int,
        scaled_dm: int,
        activation_ds: int,
        weight_d: int,
        weight_min: int,
        activation_d: int,
        temporary: int,
    ) -> None:
        del activation_q, activation_ds, activation_d
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_ds_offset = 4 * (group % 4)
        activation_address = (
            activation_address_0 if group < 4 else activation_address_1
        )
        c_base = 112
        high_activation_base = 148
        activation_ds_base = 164
        scaled_dm_base = 168
        batch_activation_d = high_activation_base

        asm.comment(
            f"Batch Q4_K group {group} across four activation tiles at a time."
        )
        asm.inst(
            f"global_load_b128 v[{weight_q}:{weight_q + 3}], v{weight_q_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{weight_q + 4}:{weight_q + 7}], v{weight_q_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] offset:{weight_q_offset + 16}"
        )
        asm.inst("s_waitcnt vmcnt(0)")
        for register in range(weight_q, weight_q + 8):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")

        for element in range(8):
            element_metadata = metadata + 4 * element
            self._emit_scale_and_minimum(
                asm,
                group,
                element_metadata,
                scale=scale,
                minimum=minimum,
                temporary=weight_d,
            )
            asm.inst(f"v_cvt_f32_u32 v{weight_d}, v{scale}")
            asm.inst(f"v_cvt_f32_u32 v{weight_min}, v{minimum}")
            asm.inst(f"v_cvt_f16_f32_e32 v{scaled_dm}.l, v{weight_d}")
            asm.inst(f"v_cvt_f16_f32_e32 v{scaled_dm}.h, v{weight_min}")
            asm.inst(f"v_pk_mul_f16 v{scaled_dm}, 0xbc003c00, v{scaled_dm}")
            asm.inst(
                f"v_pk_mul_f16 v{scaled_dm}, v{element_metadata}, v{scaled_dm}"
            )
            asm.inst(f"v_mov_b32 v{scaled_dm_base + element}, v{scaled_dm}")

        clamp = " clamp" if self.solution_key.solution.wmma_clamp else ""
        for tile_start in (0, 4):
            for local_tile in range(4):
                tile = tile_start + local_tile
                low_activation = c_base + 8 * (local_tile + 1)
                high_activation = high_activation_base + 4 * local_tile
                asm.inst(
                    f"v_add_nc_u32 v{temporary}, {16 * tile * 144}, "
                    f"v{activation_address}"
                )
                asm.inst(
                    f"global_load_b128 v[{low_activation}:{low_activation + 3}], "
                    f"v{temporary}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_q_offset}"
                )
                asm.inst(
                    f"global_load_b128 "
                    f"v[{high_activation}:{high_activation + 3}], "
                    f"v{temporary}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_q_offset + 16}"
                )
                asm.inst(
                    f"global_load_b32 v{activation_ds_base + local_tile}, "
                    f"v{temporary}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_ds_offset}"
                )
            asm.inst("s_waitcnt vmcnt(0)")
            for register in range(self.C, self.C + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            for local_tile in range(4):
                c_fragment = c_base + 8 * local_tile
                low_activation = c_base + 8 * (local_tile + 1)
                asm.inst(
                    f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                    f"v[{weight_q}:{weight_q + 3}], "
                    f"v[{low_activation}:{low_activation + 3}], "
                    f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
                )
            for local_tile in range(4):
                c_fragment = c_base + 8 * local_tile
                high_activation = high_activation_base + 4 * local_tile
                asm.inst(
                    f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                    f"v[{weight_q + 4}:{weight_q + 7}], "
                    f"v[{high_activation}:{high_activation + 3}], "
                    f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]{clamp}"
                )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * local_tile
                tile_ds = activation_ds_base + local_tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{batch_activation_d}, "
                        f"v{scaled_dm_base + element}, v{tile_ds}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
                    asm.inst(
                        f"v_cvt_f32_i32 v{temporary}, v{c_fragment + element}"
                    )
                    asm.inst(
                        f"v_fma_f32 v{total}, v{temporary}, "
                        f"v{batch_activation_d}, v{total}"
                    )
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, v{scaled_dm_base + element}, "
                        f"v{tile_ds}, v{total} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )

    def _emit_wave_reuse_group(
        self,
        asm: _Assembly,
        group: int,
        weight_q: int,
        activation_q: int,
        metadata: int,
        activation_address_0: int,
        activation_address_1: int,
        weight_q_address: int,
        sum_base: int,
        scale: int,
        minimum: int,
        scaled_dm: int,
        activation_ds: int,
        weight_d: int,
        weight_min: int,
        activation_d: int,
        temporary: int,
    ) -> None:
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_ds_offset = 4 * (group % 4)
        activation_address = (
            activation_address_0 if group < 4 else activation_address_1
        )
        asm.comment(f"Reused Q4_K group {group} across eight activation tiles.")
        asm.inst(
            f"global_load_b128 v[{weight_q}:{weight_q + 3}], v{weight_q_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{weight_q + 4}:{weight_q + 7}], v{weight_q_address}, "
            f"s[{self.KERNARG}:{self.KERNARG + 1}] offset:{weight_q_offset + 16}"
        )
        asm.inst("s_waitcnt vmcnt(0)")
        for register in range(weight_q, weight_q + 8):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")

        for element in range(8):
            element_metadata = metadata + 4 * element
            self._emit_scale_and_minimum(
                asm,
                group,
                element_metadata,
                scale=scale,
                minimum=minimum,
                temporary=weight_d,
            )
            asm.inst(f"v_cvt_f32_u32 v{weight_d}, v{scale}")
            asm.inst(f"v_cvt_f32_u32 v{weight_min}, v{minimum}")
            asm.inst(f"v_cvt_f16_f32_e32 v{scaled_dm}.l, v{weight_d}")
            asm.inst(f"v_cvt_f16_f32_e32 v{scaled_dm}.h, v{weight_min}")
            asm.inst(
                f"v_pk_mul_f16 v{scaled_dm}, 0xbc003c00, v{scaled_dm}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{scaled_dm}, v{element_metadata}, v{scaled_dm}"
            )
            asm.inst(f"v_mov_b32 v{120 + element}, v{scaled_dm}")

        for tile in range(8):
            tile_address = temporary
            tile_offset = 16 * tile * 144
            asm.inst(
                f"v_add_nc_u32 v{tile_address}, {tile_offset}, "
                f"v{activation_address}"
            )
            asm.inst(
                f"global_load_b128 v[{activation_q}:{activation_q + 3}], "
                f"v{tile_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{activation_q_offset}"
            )
            asm.inst(
                f"global_load_b128 v[{activation_q + 4}:{activation_q + 7}], "
                f"v{tile_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{activation_q_offset + 16}"
            )
            asm.inst(
                f"global_load_b32 v{activation_ds}, v{tile_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{activation_ds_offset}"
            )
            asm.inst("s_waitcnt vmcnt(0)")
            for register in range(self.C, self.C + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            clamp = " clamp" if self.solution_key.solution.wmma_clamp else ""
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
                f"v[{weight_q}:{weight_q + 3}], "
                f"v[{activation_q}:{activation_q + 3}], "
                f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
            )
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
                f"v[{weight_q + 4}:{weight_q + 7}], "
                f"v[{activation_q + 4}:{activation_q + 7}], "
                f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
            )
            for element in range(8):
                total = sum_base + tile * 8 + element
                asm.inst(
                    f"v_fma_mix_f32 v{activation_d}, v{120 + element}, "
                    f"v{activation_ds}, 0 op_sel_hi:[1,1,0]"
                )
                asm.inst(f"v_cvt_f32_i32 v{temporary}, v{self.C + element}")
                asm.inst(
                    f"v_fma_f32 v{total}, v{temporary}, v{activation_d}, v{total}"
                )
                asm.inst(
                    f"v_fma_mix_f32 v{total}, v{120 + element}, "
                    f"v{activation_ds}, v{total} "
                    f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                )

    def _emit_group(self, asm: _Assembly, group: int) -> None:
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_ds_offset = 4 * (group % 4)
        activation_address = (
            self.ACTIVATION_ADDRESS_0 if group < 4 else self.ACTIVATION_ADDRESS_1
        )

        asm.comment(f"Q4_K group {group}: direct nibbles and fixed DS4 payload.")
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_Q}:{self.WEIGHT_Q + 3}], "
            f"v{self.WEIGHT_Q_ADDRESS}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{self.WEIGHT_Q + 4}:{self.WEIGHT_Q + 7}], "
            f"v{self.WEIGHT_Q_ADDRESS}, s[{self.KERNARG}:{self.KERNARG + 1}] "
            f"offset:{weight_q_offset + 16}"
        )
        asm.inst(
            f"global_load_b128 v[{self.ACTIVATION_Q}:{self.ACTIVATION_Q + 3}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_q_offset}"
        )
        asm.inst(
            f"global_load_b128 v[{self.ACTIVATION_Q + 4}:{self.ACTIVATION_Q + 7}], "
            f"v{activation_address}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_q_offset + 16}"
        )
        asm.inst(
            f"global_load_b32 v{self.ACTIVATION_DS}, v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_ds_offset}"
        )
        asm.inst("s_waitcnt vmcnt(0)")

        for register in range(self.WEIGHT_Q, self.WEIGHT_Q + 8):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")
        for register in range(self.C, self.C + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        clamp = " clamp" if self.solution_key.solution.wmma_clamp else ""
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
            f"v[{self.WEIGHT_Q}:{self.WEIGHT_Q + 3}], "
            f"v[{self.ACTIVATION_Q}:{self.ACTIVATION_Q + 3}], "
            f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
        )
        asm.inst(
            f"v_wmma_i32_16x16x16_iu8 v[{self.C}:{self.C + 7}], "
            f"v[{self.WEIGHT_Q + 4}:{self.WEIGHT_Q + 7}], "
            f"v[{self.ACTIVATION_Q + 4}:{self.ACTIVATION_Q + 7}], "
            f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
        )
        self._emit_scaled_accumulate(asm, group)

    def _emit_scale_and_minimum(
        self,
        asm: _Assembly,
        group: int,
        metadata: int,
        *,
        scale: int | None = None,
        minimum: int | None = None,
        temporary: int | None = None,
    ) -> None:
        scale = self.SCALE if scale is None else scale
        minimum = self.MINIMUM if minimum is None else minimum
        temporary = self.TEMPORARY if temporary is None else temporary
        if group < 4:
            bit = 8 * group
            asm.inst(f"v_bfe_u32 v{scale}, v{metadata + 1}, {bit}, 6")
            asm.inst(f"v_bfe_u32 v{minimum}, v{metadata + 2}, {bit}, 6")
            return

        index = group - 4
        bit = 8 * index
        asm.inst(f"v_bfe_u32 v{scale}, v{metadata + 3}, {bit}, 4")
        asm.inst(f"v_bfe_u32 v{temporary}, v{metadata + 1}, {bit + 6}, 2")
        asm.inst(
            f"v_lshl_or_b32 v{scale}, v{temporary}, 4, v{scale}"
        )
        asm.inst(f"v_bfe_u32 v{minimum}, v{metadata + 3}, {bit + 4}, 4")
        asm.inst(f"v_bfe_u32 v{temporary}, v{metadata + 2}, {bit + 6}, 2")
        asm.inst(
            f"v_lshl_or_b32 v{minimum}, v{temporary}, 4, v{minimum}"
        )

    def _emit_scaled_accumulate(self, asm: _Assembly, group: int) -> None:
        asm.comment("Reproduce Q4_K FP16 scale/min construction before FP32 sums.")
        asm.inst(f"v_cvt_f32_f16 v{self.ACTIVATION_D}, v{self.ACTIVATION_DS}")
        asm.inst(
            f"v_cvt_f32_f16 v{self.ACTIVATION_SUM}, v{self.ACTIVATION_DS}.h"
        )
        for element in range(8):
            metadata = self.WEIGHT_METADATA + 4 * element
            self._emit_scale_and_minimum(asm, group, metadata)
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_D}, v{self.SCALE}")
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_MIN}, v{self.MINIMUM}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.l, v{self.WEIGHT_D}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.h, v{self.WEIGHT_MIN}")
            asm.inst(
                f"v_pk_mul_f16 v{self.SCALED_DM}, 0xbc003c00, v{self.SCALED_DM}"
            )
            asm.inst(
                f"v_pk_mul_f16 v{self.SCALED_DM}, v{metadata}, v{self.SCALED_DM}"
            )
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_D}, v{self.SCALED_DM}")
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_MIN}, v{self.SCALED_DM}.h")
            c = self.C + element
            total = self.SUM + element
            asm.inst(f"v_cvt_f32_i32 v{self.TEMPORARY}, v{c}")
            asm.inst(
                f"v_mul_f32 v{self.TEMPORARY}, v{self.WEIGHT_D}, "
                f"v{self.TEMPORARY}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{self.TEMPORARY}, "
                f"v{self.ACTIVATION_D}, v{total}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{self.WEIGHT_MIN}, "
                f"v{self.ACTIVATION_SUM}, v{total}"
            )

    def _emit_store(self, asm: _Assembly) -> None:
        size = self.solution_key.problem_size
        asm.comment("Store the gfx11 J-major C fragments as row-major BF16 output.")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 15, v{self.SERIAL}")
        asm.inst(f"v_lshlrev_b32 v{self.OUTPUT_ADDRESS}, 4, s3")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, v{self.OUTPUT_ADDRESS}, "
            f"v{self.TEMPORARY}"
        )
        asm.inst(
            f"v_mul_lo_u32 v{self.OUTPUT_ADDRESS}, {2 * size.n}, "
            f"v{self.OUTPUT_ADDRESS}"
        )
        asm.inst(f"v_lshrrev_b32 v{self.TEMPORARY}, 4, v{self.SERIAL}")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, "
            f"v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, v{self.OUTPUT_ADDRESS}, "
            f"v{self.TEMPORARY}"
        )
        for element in range(8):
            total = self.SUM + element
            asm.inst(
                f"v_bfe_u32 v{self.TEMPORARY}, v{total}, 16, 1"
            )
            asm.inst(
                f"v_add3_u32 v{total}, v{self.TEMPORARY}, v{total}, 0x7fff"
            )
            asm.inst(
                f"global_store_d16_hi_b16 v{self.OUTPUT_ADDRESS}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
            )
            if element != 7:
                asm.inst(
                    f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, 4, "
                    f"v{self.OUTPUT_ADDRESS}"
                )
