import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import rocisa
from rocisa import code  # ty: ignore[unresolved-import]
from rocisa.enum import SignatureValueKind as SVK  # ty: ignore[unresolved-import]

from .model import ForwardSolution, SolutionKey
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


class ForwardKernelWriterAssembly:
    """Emit strict packed K-quant/Q8_1 F16_D4S4 forward controls."""

    TOTAL_VGPRS = 88
    TOTAL_VGPRS_REUSE = 164
    TOTAL_VGPRS_BATCH = 194
    TOTAL_VGPRS_HIP_STAGED = 239
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
    ACTIVATION_SCALE_SUM = 79
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
        if not isinstance(solution_key.solution, ForwardSolution):
            raise ForwardKernelWriterError("forward writer requires ForwardSolution")
        self.solution_key = solution_key
        self.solution = solution_key.solution
        self.toolchain = toolchain

    def _quant_type(self) -> str:
        return self.solution_key.problem_type.quant_data_type

    def write(self, output: Path) -> str:
        source = self.source()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(output)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def source(self) -> str:
        solution = self.solution
        global_isa = rocisa.rocIsa.getInstance()  # ty: ignore[unresolved-attribute]
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="ggtensile-forward-rocisa-") as temp:
            os.chdir(temp)
            try:
                global_isa.init(solution.isa, str(self.toolchain.assembler), False)
            finally:
                os.chdir(original_directory)
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
            f"GGTensile {self._quant_type()} MMQ forward, fixed Q8_1 F16_D4S4 producer"
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
        if self._uses_hip_staged():
            return self._body_hip_staged()
        if self._uses_wave_reuse():
            return self._body_wave_reuse()
        asm = _Assembly()
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name
        row_stride = size.k // 256 * self.solution.packed_weight_block_bytes
        activation_plane_stride = size.m * 144
        activation_block_stride = 2 * activation_plane_stride

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
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
        asm.inst(f"v_add_nc_u32 v{self.TEMPORARY}, v{self.SCALE}, v{self.TEMPORARY}")
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
            f"v_mul_lo_u32 v{self.ACTIVATION_ADDRESS_0}, 144, v{self.ACTIVATION_ROW}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_plane_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        for register in range(self.SUM, self.SUM + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")

        quant_type = self._quant_type()
        quant_label = quant_type.replace("_", "")
        asm.label(f".LForward{quant_label}BlockLoop")
        asm.comment(
            f"Keep one {quant_type} block's dm/scales live across its eight groups."
        )
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
                f"v_add_nc_u32 v{self.RESULT_WEIGHT_ADDRESS + element}, "
                f"{self.solution.packed_weight_block_bytes}, "
                f"v{self.RESULT_WEIGHT_ADDRESS + element}"
            )
        asm.inst(
            f"v_add_nc_u32 v{self.WEIGHT_Q_ADDRESS}, "
            f"{self.solution.packed_weight_block_bytes}, v{self.WEIGHT_Q_ADDRESS}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_0}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_0}"
        )
        asm.inst(
            f"v_add_nc_u32 v{self.ACTIVATION_ADDRESS_1}, "
            f"{activation_block_stride}, v{self.ACTIVATION_ADDRESS_1}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {size.k // 256}")
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}BlockLoop")

        self._emit_store(asm)
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _emit_hip_stage_decoded_weights(
        self,
        asm: _Assembly,
        *,
        row_stride: int,
        serial: int,
        wave: int,
        temporary: int,
        lds_address: int,
        auxiliary: int,
        metadata_address: int,
        staging_base: int,
    ) -> None:
        weight_lds_base = 18_944
        weight_lds_stride = 304
        quant_type = self._quant_type()
        quant_label = quant_type.replace("_", "")
        q5_k = quant_type == "Q5_K"
        qh_address = auxiliary + 1

        asm.comment(
            f"Cooperatively decode {quant_type} payload into HIP's padded LDS rows."
        )
        asm.inst(f"v_lshrrev_b32 v{temporary}, 3, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{temporary}, {row_stride}, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, s{self.SCALAR_TEMPORARY}, v{temporary}")
        asm.inst(f"v_and_b32 v{auxiliary}, 7, v{serial}")
        if q5_k:
            asm.comment("Build Q5_K high-bit and low-nibble payload addresses.")
            asm.inst(f"v_and_b32 v{qh_address}, 1, v{auxiliary}")
            asm.inst(f"v_mad_u32_u24 v{qh_address}, 16, v{qh_address}, v{temporary}")
            asm.inst(f"v_mad_u32_u24 v{temporary}, 16, v{auxiliary}, v{temporary}")
        else:
            asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{auxiliary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, 16, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")

        asm.inst(f"v_lshrrev_b32 v{lds_address}, 3, v{serial}")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{lds_address}")
        asm.inst(f"v_lshrrev_b32 v{metadata_address}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, v{metadata_address}")
        asm.inst(f"v_and_b32 v{auxiliary}, 1, v{auxiliary}")
        asm.inst(f"v_lshlrev_b32 v{auxiliary}, 4, v{auxiliary}")
        asm.inst(
            f"v_add3_u32 v{lds_address}, v{metadata_address}, "
            f"v{auxiliary}, v{lds_address}"
        )
        asm.inst(f"v_add_nc_u32 v{lds_address}, {weight_lds_base}, v{lds_address}")

        metadata_base = staging_base + 32
        asm.inst(f"v_readfirstlane_b32 s12, v{wave}")
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataLoadDone")
        asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{serial}, v{metadata_address}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {row_stride}, v{metadata_address}")
        asm.inst(
            f"v_add_nc_u32 v{metadata_address}, s{self.SCALAR_TEMPORARY}, "
            f"v{metadata_address}"
        )
        asm.inst(
            f"global_load_b128 v[{metadata_base}:{metadata_base + 3}], "
            f"v{metadata_address}, s[{self.KERNARG}:{self.KERNARG + 1}]"
        )
        asm.label(f".LForward{quant_label}DecodedMetadataLoadDone")

        if q5_k:
            qh_base = staging_base + 36
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                row_qh_base = qh_base + 4 * row_slice
                asm.inst(
                    f"global_load_b128 v[{row_qh_base}:{row_qh_base + 3}], "
                    f"v{qh_address}, s[{self.KERNARG}:{self.KERNARG + 1}] offset:16"
                )
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}] offset:48"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{qh_address}, {16 * row_stride}, v{qh_address}"
                    )
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
            asm.inst(f"v_and_b32 v{qh_address}, 6, v{serial}")
        else:
            for row_slice in range(4):
                raw_base = staging_base + 8 * row_slice
                asm.inst(
                    f"global_load_b128 v[{raw_base}:{raw_base + 3}], "
                    f"v{temporary}, s[{self.KERNARG}:{self.KERNARG + 1}]"
                )
                if row_slice != 3:
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )
        for row_slice in range(4):
            raw_base = staging_base + 8 * row_slice
            asm.inst(f"s_waitcnt vmcnt({6 - 2 * row_slice if q5_k else 3 - row_slice})")
            if q5_k:
                qh_base = staging_base + 36 + 4 * row_slice
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{qh_base + item}, v{qh_address}, "
                        f"v{qh_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_lshrrev_b32 v{raw_base + 4 + item}, 4, v{raw_base + item}"
                    )
                for item in range(4):
                    asm.inst(
                        f"v_and_b32 v{raw_base + item}, 0x0f0f0f0f, v{raw_base + item}"
                    )
                    asm.inst(
                        f"v_and_b32 v{raw_base + 4 + item}, 0x0f0f0f0f, "
                        f"v{raw_base + 4 + item}"
                    )
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    qh = qh_base + item
                    asm.inst(f"v_and_b32 v{auxiliary}, 0x01010101, v{qh}")
                    asm.inst(f"v_lshl_or_b32 v{low}, v{auxiliary}, 4, v{low}")
                    asm.inst(f"v_and_b32 v{auxiliary}, 0x02020202, v{qh}")
                    asm.inst(f"v_lshl_or_b32 v{high}, v{auxiliary}, 3, v{high}")
            else:
                for item in range(4):
                    low = raw_base + item
                    high = raw_base + 4 + item
                    asm.inst(f"v_lshrrev_b32 v{high}, 4, v{low}")
                    asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
                    asm.inst(f"v_and_b32 v{high}, 0x0f0f0f0f, v{high}")
            asm.inst(f"ds_write_b128 v{lds_address}, v[{raw_base}:{raw_base + 3}]")
            asm.inst(
                f"ds_write_b128 v{lds_address}, "
                f"v[{raw_base + 4}:{raw_base + 7}] offset:32"
            )
            if row_slice != 3:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, "
                    f"{16 * weight_lds_stride}, v{lds_address}"
                )

        asm.comment(
            f"Compute each packed {quant_type} scale/min pair once per weight row."
        )
        asm.inst("s_cmp_lt_u32 s12, 2")
        asm.inst(f"s_cbranch_scc0 .LForward{quant_label}DecodedMetadataDone")
        asm.inst(f"v_mul_lo_u32 v{lds_address}, {weight_lds_stride}, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{lds_address}, {weight_lds_base + 256}, v{lds_address}"
        )
        if self.solution.metadata_schedule in (
            "IndependentExtraction",
            "IndependentExtractionMetadataAfterLowWmma",
        ):
            # Expose the eight metadata fields before conversion so the
            # conversion/product chains do not serialize on v72:v77.
            for group in range(4):
                bit = 8 * group
                asm.inst(f"v_bfe_u32 v{72 + group}, v145, {bit}, 6")
                asm.inst(f"v_bfe_u32 v{80 + group}, v146, {bit}, 6")
            for packed in range(4):
                group = 4 + packed
                bit = 8 * packed
                asm.inst(f"v_bfe_u32 v{72 + group}, v147, {bit}, 4")
                asm.inst(f"v_bfe_u32 v{80 + group}, v147, {bit + 4}, 4")
                asm.inst(f"v_bfe_u32 v{88 + packed}, v145, {bit + 6}, 2")
                asm.inst(f"v_bfe_u32 v{92 + packed}, v146, {bit + 6}, 2")
            for packed in range(4):
                group = 4 + packed
                asm.inst(
                    f"v_lshl_or_b32 v{72 + group}, v{88 + packed}, 4, v{72 + group}"
                )
                asm.inst(
                    f"v_lshl_or_b32 v{80 + group}, v{92 + packed}, 4, v{80 + group}"
                )
            for group in range(8):
                asm.inst(f"v_cvt_f16_u16_e32 v{96 + group}.l, v{72 + group}.l")
            for group in range(8):
                asm.inst(f"v_cvt_f16_u16_e32 v{96 + group}.h, v{80 + group}.l")
            for group in range(8):
                asm.inst(f"v_pk_mul_f16 v{96 + group}, 0xbc003c00, v{96 + group}")
            for group in range(8):
                asm.inst(f"v_pk_mul_f16 v{96 + group}, v{metadata_base}, v{96 + group}")
            for group in range(8):
                asm.inst(
                    f"ds_write_b32 v{lds_address}, v{96 + group} offset:{4 * group}"
                )
        else:
            for group in range(8):
                self._emit_scale_and_minimum(
                    asm,
                    group,
                    metadata_base,
                    scale=72,
                    minimum=73,
                    temporary=74,
                )
                if self.solution.metadata_conversion == "DirectFloat16Unsigned16":
                    asm.inst("v_cvt_f16_u16_e32 v77.l, v72.l")
                    asm.inst("v_cvt_f16_u16_e32 v77.h, v73.l")
                else:
                    asm.inst("v_cvt_f32_u32 v75, v72")
                    asm.inst("v_cvt_f32_u32 v76, v73")
                    asm.inst("v_cvt_f16_f32_e32 v77.l, v75")
                    asm.inst("v_cvt_f16_f32_e32 v77.h, v76")
                asm.inst("v_pk_mul_f16 v77, 0xbc003c00, v77")
                asm.inst(f"v_pk_mul_f16 v77, v{metadata_base}, v77")
                asm.inst(f"ds_write_b32 v{lds_address}, v77 offset:{4 * group}")
        asm.label(f".LForward{quant_label}DecodedMetadataDone")

    def _uses_wave_reuse(self) -> bool:
        return self.solution.operand_source in (
            "GlobalWaveReuse",
            "GlobalWaveBatch4",
        )

    def _uses_hip_staged(self) -> bool:
        return self.solution.operand_source in (
            "HipStagedBatch8",
            "HipDecodedStagedBatch8",
        )

    def _uses_retained_decoded_schedule(self) -> bool:
        solution = self.solution
        return (
            isinstance(solution, ForwardSolution)
            and solution.operand_source == "HipDecodedStagedBatch8"
            and solution.lds_address_hoist == "WeightMetadata"
            and solution.activation_addressing == "MadU24"
            and solution.metadata_conversion == "DirectFloat16Unsigned16"
        )

    def _total_vgprs(self) -> int:
        if self._uses_hip_staged():
            return self.TOTAL_VGPRS_HIP_STAGED
        if self.solution.operand_source == "GlobalWaveBatch4":
            return self.TOTAL_VGPRS_BATCH
        return self.TOTAL_VGPRS_REUSE if self._uses_wave_reuse() else self.TOTAL_VGPRS

    def _body_hip_staged(self) -> str:
        """Mirror HIP's cooperative operand staging and eight-WMMA batches."""
        asm = _Assembly()
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name
        decoded = self.solution.operand_source == "HipDecodedStagedBatch8"
        row_stride = size.k // 256 * self.solution.packed_weight_block_bytes
        activation_plane_stride = size.m * 144
        activation_lds_base = 512 if decoded else 8192
        sum_base = 8
        weight_q = 72
        metadata = 80
        c_base = 112
        low_activation_last = 176
        high_activation_base = 180
        activation_scale_sum_base = 212
        scaled_dm_base = 220
        temporary = 228
        lds_address = 229
        auxiliary = 230
        metadata_address = 232
        activation_plane_address = 233
        output_column = 234
        wave_column_base = 235
        wave = 236
        lane = 237
        serial = 238
        retained_decoded = self._uses_retained_decoded_schedule()
        quant_label = self._quant_type().replace("_", "")

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
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
        asm.inst(f"v_lshrrev_b32 v{wave}, 5, v{serial}")
        asm.inst(f"v_lshlrev_b32 v{wave_column_base}, 4, v{wave}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 6, s2")
        asm.inst(f"v_add_nc_u32 v{wave_column_base}, v{temporary}, v{wave_column_base}")
        if retained_decoded:
            asm.inst(f"v_lshlrev_b32 v{output_column}, 4, v{wave}")
            asm.inst(f"v_add_nc_u32 v{output_column}, v{lane}, v{output_column}")
            asm.inst(f"v_mul_lo_u32 v{output_column}, 304, v{output_column}")
            asm.inst(f"v_add_nc_u32 v{output_column}, 18944, v{output_column}")
        else:
            asm.inst(f"v_add_nc_u32 v{output_column}, v{wave_column_base}, v{lane}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(f"v_mul_lo_u32 v{activation_plane_address}, 144, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 2, v{serial}")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, v{temporary}, "
            f"v{activation_plane_address}"
        )
        if self.solution.accumulator_initialization == "VopdPair":
            for register in range(self.C, sum_base + 64, 2):
                x_source = "0" if register < sum_base else "v0"
                y_source = "0" if register < sum_base else "v1"
                asm.inst(
                    f"v_dual_mov_b32 v{register}, {x_source} :: "
                    f"v_dual_mov_b32 v{register + 1}, {y_source}"
                )
        else:
            for register in range(self.C, self.C + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            for register in range(sum_base, sum_base + 64):
                asm.inst(f"v_mov_b32 v{register}, v0")
        asm.inst(f"s_mov_b32 s{self.LOOP_COUNTER}, 0")
        asm.inst(f"s_mov_b32 s{self.SCALAR_TEMPORARY}, 0")
        if retained_decoded:
            asm.inst("s_mov_b32 s15, 512")

        asm.label(f".LForward{quant_label}HipStagedBlockLoop")
        if decoded:
            self._emit_hip_stage_decoded_weights(
                asm,
                row_stride=row_stride,
                serial=serial,
                wave=wave,
                temporary=temporary,
                lds_address=lds_address,
                auxiliary=auxiliary,
                metadata_address=metadata_address,
                staging_base=c_base,
            )
        else:
            asm.comment("Load the eight per-result Q4_K metadata records.")
            asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
            asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
            asm.inst(f"v_mul_lo_u32 v{metadata_address}, {row_stride}, v{temporary}")
            asm.inst(
                f"v_add_nc_u32 v{metadata_address}, s{self.SCALAR_TEMPORARY}, "
                f"v{metadata_address}"
            )
            for element in range(8):
                asm.inst(
                    f"global_load_b128 v[{metadata + 4 * element}:"
                    f"{metadata + 4 * element + 3}], v{metadata_address}, "
                    f"s[{self.KERNARG}:{self.KERNARG + 1}]"
                )
                if element != 7:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata_address}, {2 * row_stride}, "
                        f"v{metadata_address}"
                    )

            asm.comment("Cooperatively stage the raw packed Q4_K payload.")
            asm.inst(f"v_lshrrev_b32 v{temporary}, 3, v{serial}")
            asm.inst(f"v_lshlrev_b32 v{metadata_address}, 6, s2")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{metadata_address}, v{temporary}")
            asm.inst(f"v_mul_lo_u32 v{temporary}, {row_stride}, v{temporary}")
            asm.inst(
                f"v_add_nc_u32 v{temporary}, s{self.SCALAR_TEMPORARY}, v{temporary}"
            )
            asm.inst(f"v_and_b32 v{auxiliary}, 7, v{serial}")
            asm.inst(f"v_lshlrev_b32 v{auxiliary}, 4, v{auxiliary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, 16, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{auxiliary}, v{temporary}")
            asm.inst(f"v_lshrrev_b32 v{lds_address}, 3, v{serial}")
            asm.inst(f"v_lshlrev_b32 v{lds_address}, 7, v{lds_address}")
            asm.inst(f"v_add_nc_u32 v{lds_address}, v{auxiliary}, v{lds_address}")
            for pair in range(2):
                first = c_base
                second = c_base + 4
                asm.inst(
                    f"global_load_b128 v[{first}:{first + 3}], v{temporary}, "
                    f"s[{self.KERNARG}:{self.KERNARG + 1}]"
                )
                asm.inst(f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}")
                asm.inst(
                    f"global_load_b128 v[{second}:{second + 3}], v{temporary}, "
                    f"s[{self.KERNARG}:{self.KERNARG + 1}]"
                )
                asm.inst("s_waitcnt vmcnt(0)")
                asm.inst(
                    f"ds_write_b128 v{lds_address}, v[{first}:{first + 3}] "
                    f"offset:{pair * 4096}"
                )
                asm.inst(
                    f"ds_write_b128 v{lds_address}, v[{second}:{second + 3}] "
                    f"offset:{pair * 4096 + 2048}"
                )
                if pair == 0:
                    asm.inst(
                        f"v_add_nc_u32 v{temporary}, {16 * row_stride}, v{temporary}"
                    )

        if retained_decoded:
            asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
            asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
            asm.inst(f"v_lshlrev_b32 v{metadata_address}, 4, v{wave}")
            asm.inst(
                f"v_add_nc_u32 v{metadata_address}, v{temporary}, v{metadata_address}"
            )
            asm.inst(f"v_mul_lo_u32 v{metadata_address}, 304, v{metadata_address}")
            asm.inst(f"v_add_nc_u32 v{metadata_address}, 19200, v{metadata_address}")

        self._emit_hip_stage_activation(
            asm,
            activation_plane_address=activation_plane_address,
            serial=serial,
            temporary=temporary,
            lds_address=lds_address,
            activation_plane_stride=activation_plane_stride,
            staging_base=c_base,
            lds_base=activation_lds_base,
        )
        if decoded:
            self._emit_hip_decoded_group_loop(
                asm,
                group_base=0,
                weight_q=weight_q,
                metadata=metadata,
                c_base=c_base,
                low_activation_last=low_activation_last,
                high_activation_base=high_activation_base,
                activation_scale_sum_base=activation_scale_sum_base,
                scaled_dm_base=scaled_dm_base,
                sum_base=sum_base,
                wave=wave,
                lane=lane,
                temporary=temporary,
                lds_address=lds_address,
                serial=serial,
                weight_lds_base_address=output_column,
                metadata_lds_base_address=metadata_address,
            )
        else:
            for group in range(4):
                self._emit_hip_staged_group(
                    asm,
                    group,
                    weight_q=weight_q,
                    metadata=metadata,
                    c_base=c_base,
                    low_activation_last=low_activation_last,
                    high_activation_base=high_activation_base,
                    activation_scale_sum_base=activation_scale_sum_base,
                    scaled_dm_base=scaled_dm_base,
                    sum_base=sum_base,
                    wave=wave,
                    lane=lane,
                    temporary=temporary,
                    lds_address=lds_address,
                    serial=serial,
                )
        asm.inst("s_barrier")

        self._emit_hip_stage_activation(
            asm,
            activation_plane_address=activation_plane_address,
            serial=serial,
            temporary=temporary,
            lds_address=lds_address,
            activation_plane_stride=activation_plane_stride,
            staging_base=c_base,
            lds_base=activation_lds_base,
        )
        if decoded:
            self._emit_hip_decoded_group_loop(
                asm,
                group_base=4,
                weight_q=weight_q,
                metadata=metadata,
                c_base=c_base,
                low_activation_last=low_activation_last,
                high_activation_base=high_activation_base,
                activation_scale_sum_base=activation_scale_sum_base,
                scaled_dm_base=scaled_dm_base,
                sum_base=sum_base,
                wave=wave,
                lane=lane,
                temporary=temporary,
                lds_address=lds_address,
                serial=serial,
                weight_lds_base_address=output_column,
                metadata_lds_base_address=metadata_address,
            )
        else:
            for group in range(4, 8):
                self._emit_hip_staged_group(
                    asm,
                    group,
                    weight_q=weight_q,
                    metadata=metadata,
                    c_base=c_base,
                    low_activation_last=low_activation_last,
                    high_activation_base=high_activation_base,
                    activation_scale_sum_base=activation_scale_sum_base,
                    scaled_dm_base=scaled_dm_base,
                    sum_base=sum_base,
                    wave=wave,
                    lane=lane,
                    temporary=temporary,
                    lds_address=lds_address,
                    serial=serial,
                )
        asm.inst("s_barrier")
        asm.inst(
            f"s_add_u32 s{self.SCALAR_TEMPORARY}, s{self.SCALAR_TEMPORARY}, "
            f"{self.solution.packed_weight_block_bytes}"
        )
        asm.inst(f"s_add_u32 s{self.LOOP_COUNTER}, s{self.LOOP_COUNTER}, 1")
        asm.inst(f"s_cmp_lt_u32 s{self.LOOP_COUNTER}, {size.k // 256}")
        asm.inst(f"s_cbranch_scc1 .LForward{quant_label}HipStagedBlockLoop")

        asm.comment("Store the 128x64 row-major BF16 output tile.")
        if retained_decoded:
            self._emit_hip_decoded_store(
                asm,
                size_n=size.n,
                sum_base=sum_base,
                temporary=temporary,
                auxiliary=auxiliary,
                metadata_address=metadata_address,
                wave_column_base=wave_column_base,
                lane=lane,
                serial=serial,
            )
        else:
            asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
            asm.inst(f"v_add_nc_u32 v{auxiliary}, v{temporary}, v{lane}")
            for tile in range(8):
                asm.inst(f"v_add_nc_u32 v{temporary}, {16 * tile}, v{auxiliary}")
                asm.inst(
                    f"v_mul_lo_u32 v{metadata_address}, {2 * size.n}, v{temporary}"
                )
                asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
                asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
                asm.inst(
                    f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}"
                )
                asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
                asm.inst(
                    f"v_add_nc_u32 v{metadata_address}, v{metadata_address}, "
                    f"v{temporary}"
                )
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(f"v_bfe_u32 v{temporary}, v{total}, 16, 1")
                    asm.inst(f"v_add3_u32 v{total}, v{temporary}, v{total}, 0x7fff")
                    asm.inst(
                        f"global_store_d16_hi_b16 v{metadata_address}, v{total}, "
                        f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
                    )
                    if element != 7:
                        asm.inst(
                            f"v_add_nc_u32 v{metadata_address}, 4, v{metadata_address}"
                        )
        asm.inst("s_endpgm")
        asm.lines.append(f".L{name}_end:")
        asm.lines.append(f".size {name}, .L{name}_end - {name}")
        return asm.text()

    def _emit_hip_decoded_store(
        self,
        asm: _Assembly,
        *,
        size_n: int,
        sum_base: int,
        temporary: int,
        auxiliary: int,
        metadata_address: int,
        wave_column_base: int,
        lane: int,
        serial: int,
    ) -> None:
        solution = self.solution
        assert isinstance(solution, ForwardSolution)
        scheduled = (
            solution.epilogue_tiles_ahead != 8
            or solution.epilogue_dependency_width != 1
            or solution.epilogue_priority != 0
        )

        if not scheduled:
            for total in range(sum_base, sum_base + 64):
                asm.inst(f"v_bfe_u32 v{temporary}, v{total}, 16, 1")
                asm.inst(f"v_add3_u32 v{total}, v{temporary}, v{total}, 0x7fff")
            self._emit_hip_decoded_store_address(
                asm,
                size_n=size_n,
                temporary=temporary,
                auxiliary=auxiliary,
                metadata_address=metadata_address,
                wave_column_base=wave_column_base,
                lane=lane,
                serial=serial,
            )
            for tile in range(8):
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata_address}, {32 * size_n}, "
                        f"v{metadata_address}"
                    )
                self._emit_hip_decoded_store_tile(
                    asm,
                    tile=tile,
                    sum_base=sum_base,
                    metadata_address=metadata_address,
                )
            return

        if solution.epilogue_priority:
            asm.inst(f"s_setprio {solution.epilogue_priority}")
        self._emit_hip_decoded_store_address(
            asm,
            size_n=size_n,
            temporary=temporary,
            auxiliary=auxiliary,
            metadata_address=metadata_address,
            wave_column_base=wave_column_base,
            lane=lane,
            serial=serial,
        )

        tiles_ahead = solution.epilogue_tiles_ahead
        dependency_width = solution.epilogue_dependency_width
        for first_tile in range(0, 8, tiles_ahead):
            tile_count = min(tiles_ahead, 8 - first_tile)
            first_element = 8 * first_tile
            element_count = 8 * tile_count
            for batch in range(
                first_element,
                first_element + element_count,
                dependency_width,
            ):
                batch_count = min(
                    dependency_width,
                    first_element + element_count - batch,
                )
                if batch_count == 1:
                    total = sum_base + batch
                    asm.inst(f"v_bfe_u32 v{temporary}, v{total}, 16, 1")
                    asm.inst(f"v_add3_u32 v{total}, v{temporary}, v{total}, 0x7fff")
                    continue
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(f"v_bfe_u32 v{72 + item}, v{total}, 16, 1")
                for item in range(batch_count):
                    total = sum_base + batch + item
                    asm.inst(f"v_add3_u32 v{total}, v{72 + item}, v{total}, 0x7fff")
            for relative_tile in range(tile_count):
                tile = first_tile + relative_tile
                if tile:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata_address}, {32 * size_n}, "
                        f"v{metadata_address}"
                    )
                self._emit_hip_decoded_store_tile(
                    asm,
                    tile=tile,
                    sum_base=sum_base,
                    metadata_address=metadata_address,
                )

    def _emit_hip_decoded_store_address(
        self,
        asm: _Assembly,
        *,
        size_n: int,
        temporary: int,
        auxiliary: int,
        metadata_address: int,
        wave_column_base: int,
        lane: int,
        serial: int,
    ) -> None:
        asm.inst(f"v_lshlrev_b32 v{temporary}, 7, s3")
        asm.inst(f"v_add_nc_u32 v{auxiliary}, v{temporary}, v{lane}")
        asm.inst(f"v_mul_lo_u32 v{metadata_address}, {2 * size_n}, v{auxiliary}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{metadata_address}, v{metadata_address}, v{temporary}")

    def _emit_hip_decoded_store_tile(
        self,
        asm: _Assembly,
        *,
        tile: int,
        sum_base: int,
        metadata_address: int,
    ) -> None:
        asm.inst("s_clause 7")
        for element in range(8):
            total = sum_base + tile * 8 + element
            offset = f" offset:{4 * element}" if element else ""
            asm.inst(
                f"global_store_d16_hi_b16 v{metadata_address}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]{offset}"
            )

    def _emit_hip_stage_activation(
        self,
        asm: _Assembly,
        *,
        activation_plane_address: int,
        serial: int,
        temporary: int,
        lds_address: int,
        activation_plane_stride: int,
        staging_base: int,
        lds_base: int,
    ) -> None:
        asm.comment("Cooperatively stage one contiguous 128-row Q8_1 F16_D4S4 plane.")
        for chunk in range(5):
            chunk_start = 8 * chunk
            chunk_count = min(8, 36 - chunk_start)
            asm.inst(
                f"v_add_nc_u32 v{temporary}, {4096 * chunk}, "
                f"v{activation_plane_address}"
            )
            for item in range(chunk_count):
                register = staging_base + chunk_start + item
                asm.inst(
                    f"global_load_b32 v{register}, v{temporary}, "
                    f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{512 * item}"
                )
        asm.inst(f"v_lshlrev_b32 v{lds_address}, 2, v{serial}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, {lds_base}, v{lds_address}")
        for item in range(0, 36, 2):
            asm.inst(f"s_waitcnt vmcnt({34 - item})")
            asm.inst(
                f"ds_write2st64_b32 v{lds_address}, v{staging_base + item}, "
                f"v{staging_base + item + 1} offset0:{2 * item} "
                f"offset1:{2 * (item + 1)}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
        asm.inst("s_barrier")
        asm.inst(
            f"v_add_nc_u32 v{activation_plane_address}, "
            f"{activation_plane_stride}, v{activation_plane_address}"
        )

    def _emit_hip_staged_group(
        self,
        asm: _Assembly,
        group: int,
        *,
        weight_q: int,
        metadata: int,
        c_base: int,
        low_activation_last: int,
        high_activation_base: int,
        activation_scale_sum_base: int,
        scaled_dm_base: int,
        sum_base: int,
        wave: int,
        lane: int,
        temporary: int,
        lds_address: int,
        serial: int,
    ) -> None:
        q_offset = 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_scale_sum_offset = 4 * (group % 4)
        asm.comment(f"HIP-shaped staged Q4_K group {group}.")
        asm.inst(f"v_lshlrev_b32 v{temporary}, 11, v{wave}")
        asm.inst(f"v_lshlrev_b32 v{lds_address}, 7, v{lane}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, v{temporary}, v{lds_address}")
        asm.inst(
            f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address} "
            f"offset:{q_offset}"
        )
        asm.inst(
            f"ds_read_b128 v[{weight_q + 4}:{weight_q + 7}], v{lds_address} "
            f"offset:{q_offset + 16}"
        )
        asm.inst(f"v_mul_lo_u32 v{lds_address}, 144, v{lane}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, 8192, v{lds_address}")
        for tile in range(8):
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            high_activation = high_activation_base + 4 * tile
            tile_offset = 16 * tile * 144
            asm.inst(
                f"ds_read_b128 v[{low_activation}:{low_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + activation_q_offset}"
            )
            asm.inst(
                f"ds_read_b128 v[{high_activation}:{high_activation + 3}], "
                f"v{lds_address} "
                f"offset:{tile_offset + activation_q_offset + 16}"
            )
            asm.inst(
                f"ds_read_b32 v{activation_scale_sum_base + tile}, v{lds_address} "
                f"offset:{tile_offset + activation_scale_sum_offset}"
            )
        asm.inst("s_waitcnt lgkmcnt(0)")
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
                scale=112,
                minimum=113,
                temporary=117,
            )
            asm.inst("v_cvt_f32_u32 v114, v112")
            asm.inst("v_cvt_f32_u32 v115, v113")
            asm.inst("v_cvt_f16_f32_e32 v116.l, v114")
            asm.inst("v_cvt_f16_f32_e32 v116.h, v115")
            asm.inst("v_pk_mul_f16 v116, 0xbc003c00, v116")
            asm.inst(f"v_pk_mul_f16 v116, v{element_metadata}, v116")
            asm.inst(f"v_mov_b32 v{scaled_dm_base + element}, v116")
        self._emit_hip_staged_accumulate(
            asm,
            weight_q=weight_q,
            c_base=c_base,
            low_activation_last=low_activation_last,
            high_activation_base=high_activation_base,
            activation_scale_sum_base=activation_scale_sum_base,
            scaled_dm_base=scaled_dm_base,
            sum_base=sum_base,
            overlap_lds=False,
        )

    def _emit_hip_decoded_group_loop(
        self,
        asm: _Assembly,
        *,
        group_base: int,
        weight_q: int,
        metadata: int,
        c_base: int,
        low_activation_last: int,
        high_activation_base: int,
        activation_scale_sum_base: int,
        scaled_dm_base: int,
        sum_base: int,
        wave: int,
        lane: int,
        temporary: int,
        lds_address: int,
        serial: int,
        weight_lds_base_address: int,
        metadata_lds_base_address: int,
    ) -> None:
        quant_type = self._quant_type()
        quant_label = quant_type.replace("_", "")
        label = f".LForward{quant_label}DecodedGroupLoop{group_base}"
        asm.comment(
            f"Roll decoded {quant_type} groups {group_base} through {group_base + 3}."
        )
        asm.inst("s_mov_b32 s13, 0")
        asm.label(label)

        asm.inst("s_lshl_b32 s14, s13, 5")
        if self.solution.lds_address_hoist == "WeightMetadata":
            if group_base:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, {32 * group_base}, "
                    f"v{weight_lds_base_address}"
                )
                asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{lds_address}")
            else:
                asm.inst(
                    f"v_add_nc_u32 v{lds_address}, s14, v{weight_lds_base_address}"
                )
        else:
            asm.inst(f"v_lshlrev_b32 v{temporary}, 4, v{wave}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{lane}, v{temporary}")
            asm.inst(f"v_mul_lo_u32 v{lds_address}, 304, v{temporary}")
            asm.inst(
                f"v_add_nc_u32 v{lds_address}, {18944 + 32 * group_base}, "
                f"v{lds_address}"
            )
            asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{lds_address}")
        asm.inst(f"ds_read_b128 v[{weight_q}:{weight_q + 3}], v{lds_address}")
        asm.inst(
            f"ds_read_b128 v[{weight_q + 4}:{weight_q + 7}], v{lds_address} offset:16"
        )

        if self.solution.activation_addressing == "MadU24":
            asm.inst(f"v_mad_u32_u24 v{metadata}, 144, v{lane}, s15")
        else:
            asm.inst(f"v_mul_lo_u32 v{metadata}, 144, v{lane}")
            asm.inst(f"v_add_nc_u32 v{metadata}, 512, v{metadata}")
        asm.inst(f"v_add_nc_u32 v{lds_address}, s14, v{metadata}")
        for tile in range(8):
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            high_activation = high_activation_base + 4 * tile
            tile_offset = 16 * tile * 144
            asm.inst(
                f"ds_read_b128 v[{low_activation}:{low_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + 16}"
            )
            asm.inst(
                f"ds_read_b128 v[{high_activation}:{high_activation + 3}], "
                f"v{lds_address} offset:{tile_offset + 32}"
            )

        def emit_metadata_reads() -> None:
            asm.inst("s_lshl_b32 s14, s13, 2")
            asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            for tile in range(0, 8, 2):
                asm.inst(
                    f"ds_read2st64_b32 v[{activation_scale_sum_base + tile}:"
                    f"{activation_scale_sum_base + tile + 1}], v{metadata} "
                    f"offset0:{9 * tile} offset1:{9 * (tile + 1)}"
                )

            if self.solution.lds_address_hoist == "WeightMetadata":
                if group_base:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata}, {4 * group_base}, "
                        f"v{metadata_lds_base_address}"
                    )
                    asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
                else:
                    asm.inst(
                        f"v_add_nc_u32 v{metadata}, s14, v{metadata_lds_base_address}"
                    )
            else:
                asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
                asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
                asm.inst(f"v_lshlrev_b32 v{metadata}, 4, v{wave}")
                asm.inst(f"v_add_nc_u32 v{metadata}, v{temporary}, v{metadata}")
                asm.inst(f"v_mul_lo_u32 v{metadata}, 304, v{metadata}")
                asm.inst(
                    f"v_add_nc_u32 v{metadata}, {18944 + 256 + 4 * group_base}, "
                    f"v{metadata}"
                )
                asm.inst(f"v_add_nc_u32 v{metadata}, s14, v{metadata}")
            for pair in range(1, 4):
                asm.inst(f"v_add_nc_u32 v{metadata + pair}, {1216 * pair}, v{metadata}")
            for element in range(0, 8, 2):
                asm.inst(
                    f"ds_read2_b32 v[{scaled_dm_base + element}:"
                    f"{scaled_dm_base + element + 1}], "
                    f"v{metadata + element // 2} offset0:0 offset1:152"
                )

        deferred_metadata = self.solution.metadata_schedule in (
            "MetadataAfterLowWmma",
            "IndependentExtractionMetadataAfterLowWmma",
        )
        if not deferred_metadata:
            emit_metadata_reads()
        self._emit_hip_staged_accumulate(
            asm,
            weight_q=weight_q,
            c_base=c_base,
            low_activation_last=low_activation_last,
            high_activation_base=high_activation_base,
            activation_scale_sum_base=activation_scale_sum_base,
            scaled_dm_base=scaled_dm_base,
            sum_base=sum_base,
            overlap_lds=True,
            deferred_metadata_emitter=(
                emit_metadata_reads if deferred_metadata else None
            ),
        )
        asm.inst("s_add_u32 s13, s13, 1")
        asm.inst("s_cmp_lt_u32 s13, 4")
        asm.inst(f"s_cbranch_scc1 {label}")

    def _emit_hip_staged_accumulate(
        self,
        asm: _Assembly,
        *,
        weight_q: int,
        c_base: int,
        low_activation_last: int,
        high_activation_base: int,
        activation_scale_sum_base: int,
        scaled_dm_base: int,
        sum_base: int,
        overlap_lds: bool,
        deferred_metadata_emitter: Callable[[], None] | None = None,
    ) -> None:
        clamp = " clamp" if self.solution.wmma_clamp else ""
        for tile in range(8):
            if overlap_lds:
                first_wait = 15 if deferred_metadata_emitter is not None else 23
                asm.inst(f"s_waitcnt lgkmcnt({first_wait - 2 * tile})")
            c_fragment = c_base + 8 * tile
            low_activation = (
                c_base + 8 * (tile + 1) if tile < 7 else low_activation_last
            )
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_q}:{weight_q + 3}], "
                f"v[{low_activation}:{low_activation + 3}], "
                f"v[{self.C}:{self.C + 7}] neg_lo:[1,1,0]{clamp}"
            )
        if deferred_metadata_emitter is not None:
            deferred_metadata_emitter()
        for tile in range(8):
            if overlap_lds and tile == 7:
                asm.inst("s_waitcnt lgkmcnt(8)")
            c_fragment = c_base + 8 * tile
            high_activation = high_activation_base + 4 * tile
            asm.inst(
                f"v_wmma_i32_16x16x16_iu8 v[{c_fragment}:{c_fragment + 7}], "
                f"v[{weight_q + 4}:{weight_q + 7}], "
                f"v[{high_activation}:{high_activation + 3}], "
                f"v[{c_fragment}:{c_fragment + 7}] neg_lo:[1,1,0]{clamp}"
            )
        if overlap_lds:
            asm.inst("s_waitcnt lgkmcnt(0)")

        product_base = low_activation_last
        for tile_start in (0, 4):
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_fma_mix_f32 v{product}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(8):
                    asm.inst(
                        f"v_cvt_f32_i32 v{c_fragment + element}, "
                        f"v{c_fragment + element}"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                c_fragment = c_base + 8 * tile
                for element in range(0, 8, 2):
                    total = sum_base + tile * 8 + element
                    product = product_base + 8 * local_tile + element
                    asm.inst(
                        f"v_dual_fmac_f32 v{total}, v{product}, "
                        f"v{c_fragment + element} :: "
                        f"v_dual_fmac_f32 v{total + 1}, v{product + 1}, "
                        f"v{c_fragment + element + 1}"
                    )
            for local_tile in range(4):
                tile = tile_start + local_tile
                tile_scale_sum = activation_scale_sum_base + tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, v{total} "
                        f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                    )

    def _body_wave_reuse(self) -> str:
        """Reuse one Q4_K weight tile across eight activation-row tiles per wave."""
        asm = _Assembly()
        size = self.solution_key.problem_size
        name = self.solution_key.kernel_name
        row_stride = size.k // 256 * self.solution.packed_weight_block_bytes
        activation_plane_stride = size.m * 144
        activation_block_stride = 2 * activation_plane_stride
        sum_base = 8
        weight_q = 72
        batch = self.solution.operand_source == "GlobalWaveBatch4"
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
        activation_scale_sum = 151
        weight_d = 3
        weight_min = 4
        activation_d = 154
        temporary = 188 if batch else 156
        serial = 189 if batch else 157
        output_column = 190 if batch else 158
        wave_column_base = 191 if batch else 159
        activation_row = 192 if batch else 160
        lane = 193 if batch else 161

        asm.comment(
            "Load the exact packed-weight, Q8_1 F16_D4S4 workspace, and output pointers."
        )
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
        asm.inst(f"v_add_nc_u32 v{wave_column_base}, v{temporary}, v{wave_column_base}")
        asm.inst(f"v_add_nc_u32 v{output_column}, v{wave_column_base}, v{lane}")
        asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
        asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
        asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
        asm.inst(f"v_mul_lo_u32 v{result_addresses}, {row_stride}, v{temporary}")
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
                self._emit_wave_batch_group if batch else self._emit_wave_reuse_group
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
                activation_scale_sum,
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
            asm.inst(f"v_add_nc_u32 v{temporary}, {16 * tile}, v{activation_row}")
            asm.inst(f"v_mul_lo_u32 v{output_address}, {2 * size.n}, v{temporary}")
            asm.inst(f"v_lshrrev_b32 v{temporary}, 4, v{serial}")
            asm.inst(f"v_and_b32 v{temporary}, 1, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{temporary}, v{wave_column_base}, v{temporary}")
            asm.inst(f"v_lshlrev_b32 v{temporary}, 1, v{temporary}")
            asm.inst(f"v_add_nc_u32 v{output_address}, v{output_address}, v{temporary}")
            for element in range(8):
                total = sum_base + tile * 8 + element
                asm.inst(f"v_bfe_u32 v{temporary}, v{total}, 16, 1")
                asm.inst(f"v_add3_u32 v{total}, v{temporary}, v{total}, 0x7fff")
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
        activation_scale_sum: int,
        weight_d: int,
        weight_min: int,
        activation_d: int,
        temporary: int,
    ) -> None:
        del activation_q, activation_scale_sum, activation_d
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_scale_sum_offset = 4 * (group % 4)
        activation_address = activation_address_0 if group < 4 else activation_address_1
        c_base = 112
        high_activation_base = 148
        activation_scale_sum_base = 164
        scaled_dm_base = 168
        batch_activation_d = high_activation_base

        asm.comment(f"Batch Q4_K group {group} across four activation tiles at a time.")
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
            asm.inst(f"v_pk_mul_f16 v{scaled_dm}, v{element_metadata}, v{scaled_dm}")
            asm.inst(f"v_mov_b32 v{scaled_dm_base + element}, v{scaled_dm}")

        clamp = " clamp" if self.solution.wmma_clamp else ""
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
                    f"global_load_b32 v{activation_scale_sum_base + local_tile}, "
                    f"v{temporary}, s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                    f"offset:{activation_scale_sum_offset}"
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
                tile_scale_sum = activation_scale_sum_base + local_tile
                for element in range(8):
                    total = sum_base + tile * 8 + element
                    asm.inst(
                        f"v_fma_mix_f32 v{batch_activation_d}, "
                        f"v{scaled_dm_base + element}, v{tile_scale_sum}, 0 "
                        f"op_sel_hi:[1,1,0]"
                    )
                    asm.inst(f"v_cvt_f32_i32 v{temporary}, v{c_fragment + element}")
                    asm.inst(
                        f"v_fma_f32 v{total}, v{temporary}, "
                        f"v{batch_activation_d}, v{total}"
                    )
                    asm.inst(
                        f"v_fma_mix_f32 v{total}, v{scaled_dm_base + element}, "
                        f"v{tile_scale_sum}, v{total} "
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
        activation_scale_sum: int,
        weight_d: int,
        weight_min: int,
        activation_d: int,
        temporary: int,
    ) -> None:
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_scale_sum_offset = 4 * (group % 4)
        activation_address = activation_address_0 if group < 4 else activation_address_1
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
            asm.inst(f"v_pk_mul_f16 v{scaled_dm}, 0xbc003c00, v{scaled_dm}")
            asm.inst(f"v_pk_mul_f16 v{scaled_dm}, v{element_metadata}, v{scaled_dm}")
            asm.inst(f"v_mov_b32 v{120 + element}, v{scaled_dm}")

        for tile in range(8):
            tile_address = temporary
            tile_offset = 16 * tile * 144
            asm.inst(
                f"v_add_nc_u32 v{tile_address}, {tile_offset}, v{activation_address}"
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
                f"global_load_b32 v{activation_scale_sum}, v{tile_address}, "
                f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
                f"offset:{activation_scale_sum_offset}"
            )
            asm.inst("s_waitcnt vmcnt(0)")
            for register in range(self.C, self.C + 8):
                asm.inst(f"v_mov_b32 v{register}, 0")
            clamp = " clamp" if self.solution.wmma_clamp else ""
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
                    f"v{activation_scale_sum}, 0 op_sel_hi:[1,1,0]"
                )
                asm.inst(f"v_cvt_f32_i32 v{temporary}, v{self.C + element}")
                asm.inst(f"v_fma_f32 v{total}, v{temporary}, v{activation_d}, v{total}")
                asm.inst(
                    f"v_fma_mix_f32 v{total}, v{120 + element}, "
                    f"v{activation_scale_sum}, v{total} "
                    f"op_sel:[1,1,0] op_sel_hi:[1,1,0]"
                )

    def _emit_group(self, asm: _Assembly, group: int) -> None:
        weight_q_offset = 16 + 32 * (group // 2)
        activation_q_offset = 16 + 32 * (group % 4)
        activation_scale_sum_offset = 4 * (group % 4)
        activation_address = (
            self.ACTIVATION_ADDRESS_0 if group < 4 else self.ACTIVATION_ADDRESS_1
        )

        asm.comment(
            f"Q4_K group {group}: direct nibbles and fixed Q8_1 F16_D4S4 block."
        )
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
            f"global_load_b32 v{self.ACTIVATION_SCALE_SUM}, v{activation_address}, "
            f"s[{self.KERNARG + 2}:{self.KERNARG + 3}] "
            f"offset:{activation_scale_sum_offset}"
        )
        asm.inst("s_waitcnt vmcnt(0)")

        for register in range(self.WEIGHT_Q, self.WEIGHT_Q + 8):
            if group & 1:
                asm.inst(f"v_lshrrev_b32 v{register}, 4, v{register}")
            asm.inst(f"v_and_b32 v{register}, 0x0f0f0f0f, v{register}")
        for register in range(self.C, self.C + 8):
            asm.inst(f"v_mov_b32 v{register}, 0")
        clamp = " clamp" if self.solution.wmma_clamp else ""
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
        asm.inst(f"v_lshl_or_b32 v{scale}, v{temporary}, 4, v{scale}")
        asm.inst(f"v_bfe_u32 v{minimum}, v{metadata + 3}, {bit + 4}, 4")
        asm.inst(f"v_bfe_u32 v{temporary}, v{metadata + 2}, {bit + 6}, 2")
        asm.inst(f"v_lshl_or_b32 v{minimum}, v{temporary}, 4, v{minimum}")

    def _emit_scaled_accumulate(self, asm: _Assembly, group: int) -> None:
        asm.comment("Reproduce Q4_K FP16 scale/min construction before FP32 sums.")
        asm.inst(f"v_cvt_f32_f16 v{self.ACTIVATION_D}, v{self.ACTIVATION_SCALE_SUM}")
        asm.inst(
            f"v_cvt_f32_f16 v{self.ACTIVATION_SUM}, v{self.ACTIVATION_SCALE_SUM}.h"
        )
        for element in range(8):
            metadata = self.WEIGHT_METADATA + 4 * element
            self._emit_scale_and_minimum(asm, group, metadata)
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_D}, v{self.SCALE}")
            asm.inst(f"v_cvt_f32_u32 v{self.WEIGHT_MIN}, v{self.MINIMUM}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.l, v{self.WEIGHT_D}")
            asm.inst(f"v_cvt_f16_f32_e32 v{self.SCALED_DM}.h, v{self.WEIGHT_MIN}")
            asm.inst(f"v_pk_mul_f16 v{self.SCALED_DM}, 0xbc003c00, v{self.SCALED_DM}")
            asm.inst(f"v_pk_mul_f16 v{self.SCALED_DM}, v{metadata}, v{self.SCALED_DM}")
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_D}, v{self.SCALED_DM}")
            asm.inst(f"v_cvt_f32_f16 v{self.WEIGHT_MIN}, v{self.SCALED_DM}.h")
            c = self.C + element
            total = self.SUM + element
            asm.inst(f"v_cvt_f32_i32 v{self.TEMPORARY}, v{c}")
            asm.inst(
                f"v_mul_f32 v{self.TEMPORARY}, v{self.WEIGHT_D}, v{self.TEMPORARY}"
            )
            asm.inst(
                f"v_fma_f32 v{total}, v{self.TEMPORARY}, v{self.ACTIVATION_D}, v{total}"
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
            f"v_mul_lo_u32 v{self.OUTPUT_ADDRESS}, {2 * size.n}, v{self.OUTPUT_ADDRESS}"
        )
        asm.inst(f"v_lshrrev_b32 v{self.TEMPORARY}, 4, v{self.SERIAL}")
        asm.inst(f"v_and_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY + 1}, 4, s2")
        asm.inst(
            f"v_add_nc_u32 v{self.TEMPORARY}, v{self.TEMPORARY + 1}, v{self.TEMPORARY}"
        )
        asm.inst(f"v_lshlrev_b32 v{self.TEMPORARY}, 1, v{self.TEMPORARY}")
        asm.inst(
            f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, v{self.OUTPUT_ADDRESS}, "
            f"v{self.TEMPORARY}"
        )
        for element in range(8):
            total = self.SUM + element
            asm.inst(f"v_bfe_u32 v{self.TEMPORARY}, v{total}, 16, 1")
            asm.inst(f"v_add3_u32 v{total}, v{self.TEMPORARY}, v{total}, 0x7fff")
            asm.inst(
                f"global_store_d16_hi_b16 v{self.OUTPUT_ADDRESS}, v{total}, "
                f"s[{self.KERNARG + 4}:{self.KERNARG + 5}]"
            )
            if element != 7:
                asm.inst(
                    f"v_add_nc_u32 v{self.OUTPUT_ADDRESS}, 4, v{self.OUTPUT_ADDRESS}"
                )
