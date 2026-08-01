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
            groupSegmentSize=0,
            sgprWorkGroup=(1, 1, 0),
            vgprWorkItem=0,
            flatWorkGroupSize=solution.num_threads,
            totalVgprs=self.TOTAL_VGPRS,
            totalAgprs=0,
            totalSgprs=self.TOTAL_SGPRS,
        )
        signature.addDescriptionTopic(
            "GGTensile Q4_K dense MMQ forward, fixed Q8_1_DS4 producer"
        )
        signature.addArg("packed_weight", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("activations", SVK.SIG_GLOBALBUFFER, "struct", "generic")
        signature.addArg("output", SVK.SIG_GLOBALBUFFER, "bf16", "generic")
        signature.addArg("rows", SVK.SIG_VALUE, "u32")
        signature.addArg("rows_padded", SVK.SIG_VALUE, "u32")
        signature.addArg("in_features", SVK.SIG_VALUE, "u32")
        signature.addArg("out_features", SVK.SIG_VALUE, "u32")

        module = code.Module("GGTensileForwardKernel")
        module.add(signature)
        module.add(code.TextBlock(self._body()))
        return str(module)

    def _body(self) -> str:
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
        self, asm: _Assembly, group: int, metadata: int
    ) -> None:
        if group < 4:
            bit = 8 * group
            asm.inst(f"v_bfe_u32 v{self.SCALE}, v{metadata + 1}, {bit}, 6")
            asm.inst(f"v_bfe_u32 v{self.MINIMUM}, v{metadata + 2}, {bit}, 6")
            return

        index = group - 4
        bit = 8 * index
        asm.inst(f"v_bfe_u32 v{self.SCALE}, v{metadata + 3}, {bit}, 4")
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{metadata + 1}, {bit + 6}, 2")
        asm.inst(
            f"v_lshl_or_b32 v{self.SCALE}, v{self.TEMPORARY}, 4, v{self.SCALE}"
        )
        asm.inst(f"v_bfe_u32 v{self.MINIMUM}, v{metadata + 3}, {bit + 4}, 4")
        asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{metadata + 2}, {bit + 6}, 2")
        asm.inst(
            f"v_lshl_or_b32 v{self.MINIMUM}, v{self.TEMPORARY}, 4, "
            f"v{self.MINIMUM}"
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
