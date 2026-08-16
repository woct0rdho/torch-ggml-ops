"""Packed-weight readers and decoders for MMQ backward lowering."""

from .kernel_writer_assembly import emit_bf16_rne
from .mmq_bwd_emission import _Assembly
from .mmq_bwd_physical import BackwardPhysicalPlan, BackwardRegisterPlan
from .mmq_bwd_spec import BackwardExtraction, DerivedBackwardState


class BackwardQuantLowering:
    """Substantial quant reader/decoder implementation shared by the pipeline."""

    state: DerivedBackwardState
    physical: BackwardPhysicalPlan
    registers: BackwardRegisterPlan

    def _emit_quant_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        quant_type = self.state.contract.quant_type
        if quant_type == "Q3_K":
            self._emit_q3_k_global_reads(asm, wait_for_reads=wait_for_reads)
        elif quant_type == "Q4_K":
            self._emit_q4_k_global_reads(asm, wait_for_reads=wait_for_reads)
        elif quant_type == "Q5_K":
            self._emit_q5_k_global_reads(asm, wait_for_reads=wait_for_reads)
        elif quant_type == "Q6_K":
            self._emit_q6_k_global_reads(asm, wait_for_reads=wait_for_reads)
        else:
            self._emit_q8_0_global_reads(asm, wait_for_reads=wait_for_reads)

    def _emit_q6_k_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        n_tiles = self.state.solution.matrix_instruction[6]
        n_per_tile = self.state.solution.macro_tile1
        quant_format = self.state.contract.quant_format
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        k_shift = n_tiles.bit_length() - 1
        k_span = self.state.solution.depth_u // decoder_rows
        packed_row_bytes = (
            self.state.contract.problem_size.n
            // quant_format.block_values
            * quant_format.block_bytes
        )
        row_delta = k_span * packed_row_bytes
        q_low = r.global_read_b
        q_high = q_low + 4 * decoder_rows
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q6_K block addresses for decoder-owned output rows.")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, {packed_row_bytes}, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        for row in range(1, decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + row}, {row * row_delta}, v{a}")

        asm.comment("Map each lane to one Q6_K 16-value scale group.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, {tiles_per_weight_block - 1}, s3")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {n_shift}, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 1}, 63, v{t}")
        asm.inst(f"v_lshrrev_b32 v{t + 2}, 7, v{t}")
        asm.inst(f"v_lshl_add_u32 v{t + 1}, v{t + 2}, 6, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 3}, 31, v{t}")
        asm.inst(f"v_lshl_add_u32 v{t + 3}, v{t + 2}, 5, v{t + 3}")
        asm.inst(f"v_lshrrev_b32 v{t + 4}, 4, v{t}")
        payload_address = t + 5
        lane_share = self.state.spec.pipeline.packed_weight_lane_share
        if lane_share == 1:
            for row in range(decoder_rows):
                block_address = a + row
                asm.inst(f"v_add_nc_u32 v{payload_address}, v{block_address}, v{t + 1}")
                asm.inst(
                    f"global_load_b128 v[{q_low + 4 * row}:{q_low + 4 * row + 3}], "
                    f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
                )
                asm.inst(f"v_add_nc_u32 v{payload_address}, v{block_address}, v{t + 3}")
                asm.inst(
                    f"global_load_b128 v[{q_high + 4 * row}:{q_high + 4 * row + 3}], "
                    f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                    "offset:128"
                )
        else:
            asm.comment("Load unique Q6_K low planes on every lane.")
            for row in range(decoder_rows):
                block_address = a + row
                asm.inst(f"v_add_nc_u32 v{payload_address}, v{block_address}, v{t + 1}")
                asm.inst(
                    f"global_load_b128 v[{q_low + 4 * row}:{q_low + 4 * row + 3}], "
                    f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
                )
            asm.comment("Load shared Q6_K high planes on lane-pair owners.")
            asm.inst(f"v_and_b32 v{t + 6}, 2, v{r.serial}")
            asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{t + 6}")
            asm.inst(f"s_and_saveexec_b32 s{r.scalar_temporary + 1}, vcc_lo")
            for row in range(decoder_rows):
                block_address = a + row
                asm.inst(f"v_add_nc_u32 v{payload_address}, v{block_address}, v{t + 3}")
                asm.inst(
                    f"global_load_b128 v[{q_high + 4 * row}:{q_high + 4 * row + 3}], "
                    f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                    "offset:128"
                )
            asm.inst(f"s_mov_b32 exec_lo, s{r.scalar_temporary + 1}")
        for row in range(decoder_rows):
            block_address = a + row
            asm.inst(f"v_add_nc_u32 v{payload_address}, v{block_address}, v{t + 4}")
            asm.inst(
                f"global_load_d16_u8 v{r.quant_scale + row}, "
                f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                "offset:192"
            )
            asm.inst(
                f"global_load_d16_b16 v{r.quant_dm + row}, v{block_address}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}] offset:208"
            )
        if wait_for_reads:
            asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q8_0_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        n_tiles = self.state.solution.matrix_instruction[6]
        quant_format = self.state.contract.quant_format
        k_shift = n_tiles.bit_length() - 1
        k_span = self.state.solution.depth_u // decoder_rows
        packed_row_bytes = (
            self.state.contract.problem_size.n
            // quant_format.block_values
            * quant_format.block_bytes
        )
        row_delta = k_span * packed_row_bytes
        q = r.global_read_b
        a = r.address
        t = r.temporary
        loop = r.loop_counter
        block = r.block_offset

        asm.comment("Build Q8_0 block addresses for decoder-owned output rows.")
        asm.inst(f"v_lshrrev_b32 v{t}, {k_shift}, v{r.serial}")
        asm.inst(f"v_add_nc_u32 v{t}, s{loop}, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t}, {packed_row_bytes}, v{t}")
        asm.inst(f"v_add_nc_u32 v{t}, s{block}, v{t}")
        asm.inst(f"v_mov_b32 v{a}, v{t}")
        for row in range(1, decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + row}, {row * row_delta}, v{a}")

        asm.comment("Map each lane to a Q8_0 16-value block half.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 1, v{t}")
        asm.inst(f"v_mul_lo_u32 v{t + 1}, {quant_format.block_bytes}, v{t + 1}")
        asm.inst(f"v_and_b32 v{t + 2}, 1, v{t}")
        asm.inst(f"v_lshlrev_b32 v{t + 2}, 4, v{t + 2}")
        asm.inst(f"v_add_nc_u32 v{t + 2}, 2, v{t + 2}")
        payload_address = t + 3
        for row in range(decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + row}, v{a + row}, v{t + 1}")
            asm.inst(f"v_add_nc_u32 v{payload_address}, v{a + row}, v{t + 2}")
            if self.state.spec.decode.q8.extraction is not BackwardExtraction.scalar:
                asm.inst(
                    f"global_load_b128 v[{q + 4 * row}:{q + 4 * row + 3}], "
                    f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}]"
                )
            else:
                for index in range(4):
                    asm.inst(
                        f"global_load_b32 v{q + 4 * row + index}, "
                        f"v{payload_address}, s[{r.kernarg + 2}:{r.kernarg + 3}] "
                        f"offset:{4 * index}"
                    )
            asm.inst(
                f"global_load_d16_b16 v{r.quant_dm + row}, v{a + row}, "
                f"s[{r.kernarg + 2}:{r.kernarg + 3}]"
            )
        if wait_for_reads:
            asm.inst("s_waitcnt vmcnt(0)")

    def _emit_q3_k_global_reads(
        self,
        asm: _Assembly,
        *,
        wait_for_reads: bool = True,
    ) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        n_tiles = self.state.solution.matrix_instruction[6]
        n_per_tile = self.state.solution.macro_tile1
        quant_format = self.state.contract.quant_format
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        k_shift = self.state.solution.matrix_instruction[6].bit_length() - 1
        k_span = self.state.solution.depth_u // decoder_rows
        packed_row_bytes = (
            self.state.contract.problem_size.n
            // quant_format.block_values
            * quant_format.block_bytes
        )
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
        n_tiles = self.state.solution.matrix_instruction[6]
        decoder_rows = self.physical.decoder.rows
        quant_format = self.state.contract.quant_format
        k_shift = n_tiles.bit_length() - 1
        k_span = self.state.solution.depth_u // decoder_rows
        packed_row_bytes = (
            self.state.contract.problem_size.n
            // quant_format.block_values
            * quant_format.block_bytes
        )
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
        direct_quant_mapping = self.state.solution.macro_tile1 == 128
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
            asm.inst(f"v_bfe_u32 v{t + 2}, v{r.serial}, 1, 2")
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
        if self.state.spec.pipeline.packed_weight_lane_share == 2:
            asm.comment("Load packed q bytes on one lane from each nibble pair.")
            asm.inst(f"v_and_b32 v{t + 2}, 2, v{r.serial}")
            asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{t + 2}")
            asm.inst(f"s_and_saveexec_b32 s{r.scalar_temporary + 1}, vcc_lo")
        for row in range(decoder_rows):
            asm.inst(
                f"global_load_b128 v[{q + 4 * row}:{q + 4 * row + 3}], "
                f"v{a + 2 + row}, s[{r.kernarg + 2}:{r.kernarg + 3}] offset:16"
            )
        if self.state.spec.pipeline.packed_weight_lane_share == 2:
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
        n_tiles = self.state.solution.matrix_instruction[6]
        decoder_rows = self.physical.decoder.rows
        quant_format = self.state.contract.quant_format
        k_shift = n_tiles.bit_length() - 1
        k_span = self.state.solution.depth_u // decoder_rows
        packed_row_bytes = (
            self.state.contract.problem_size.n
            // quant_format.block_values
            * quant_format.block_bytes
        )
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
        direct_quant_mapping = self.state.solution.macro_tile1 == 128
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
            asm.inst(f"v_bfe_u32 v{t + 2}, v{r.serial}, 1, 2")
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
                asm.inst(f"v_add_nc_u32 v{t + 4}, v{block_address}, v{t + 2}")
                self._emit_q5_k_metadata_loads(asm, block_address, t + 4, row)
            if wait_for_reads:
                asm.inst("s_waitcnt vmcnt(0)")
            return

        for row in range(decoder_rows):
            asm.inst(f"v_add_nc_u32 v{a + 2 + row}, v{a + row}, v{t + 1}")
        if self.state.spec.pipeline.packed_weight_lane_share == 2:
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
        if self.state.spec.pipeline.packed_weight_lane_share == 2:
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
        if self.state.spec.decode.q5.vector_metadata_load:
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
        if self.state.spec.pipeline.packed_weight_lane_share == 1:
            return

        r = self.registers
        decoder_rows = self.physical.decoder.rows
        lane_pair = r.temporary
        shifted = lane_pair + 1
        asm.comment("Replicate packed q bytes across low/high-nibble lane pairs.")
        asm.inst(f"v_and_b32 v{lane_pair}, 2, v{r.serial}")
        asm.inst(f"v_cmp_eq_u32_e32 vcc_lo, 0, v{lane_pair}")
        first_payload = r.global_read_b
        if self.state.contract.quant_type == "Q6_K":
            first_payload += 4 * decoder_rows
        for register in range(
            first_payload,
            r.global_read_b
            + self.physical.decoder.payload_register_count * decoder_rows,
        ):
            asm.inst(
                f"v_mov_b32_dpp v{shifted}, v{register} row_shr:2 "
                "row_mask:0xf bank_mask:0xf"
            )
            asm.inst(f"v_cndmask_b32 v{register}, v{shifted}, v{register}, vcc_lo")

    def _emit_first_a_global_reads(self, asm: _Assembly) -> None:
        r = self.registers
        solution = self.state.solution
        m_tiles = solution.matrix_instruction[5]
        a = r.address

        asm.comment("Prefetch A fragments while Q4_K data is pending.")
        row_pointers = tuple((a + 4 + m_tile, a + m_tile) for m_tile in range(m_tiles))

        asm.inst(f"s_lshl_b32 s{r.scalar_temporary + 1}, s{r.loop_counter}, 1")
        for row_offset, pointer in row_pointers:
            asm.inst(
                f"v_add_nc_u32 v{pointer}, s{r.scalar_temporary + 1}, v{row_offset}"
            )
        for k_half in range(self.state.spec.pipeline.global_read_prefetch):
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
        quant_type = self.state.contract.quant_type
        if quant_type == "Q3_K":
            self._emit_q3_k_decode(asm, label_suffix=label_suffix)
        elif quant_type == "Q4_K":
            self._emit_q4_k_decode(asm, label_suffix=label_suffix)
        elif quant_type == "Q5_K":
            self._emit_q5_k_decode(asm, label_suffix=label_suffix)
        elif quant_type == "Q6_K":
            self._emit_q6_k_decode(asm, label_suffix=label_suffix)
        else:
            self._emit_q8_0_decode(asm, label_suffix=label_suffix)

    def _emit_quant_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        quant_type = self.state.contract.quant_type
        if quant_type == "Q3_K":
            self._emit_q3_k_decode_prepare(asm, label_suffix=label_suffix)
        elif quant_type == "Q4_K":
            self._emit_q45_metadata_prepare(
                asm, label_suffix=label_suffix, quant_type="Q4_K"
            )
            self._emit_q45_nibble_shift(asm)
        elif quant_type == "Q5_K":
            self._emit_q45_metadata_prepare(
                asm, label_suffix=label_suffix, quant_type="Q5_K"
            )
        elif quant_type == "Q6_K":
            self._emit_q6_k_decode_prepare(asm, label_suffix=label_suffix)
        else:
            self._emit_q8_0_decode_prepare(asm, label_suffix=label_suffix)

    def _emit_quant_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        quant_type = self.state.contract.quant_type
        if quant_type == "Q3_K":
            self._emit_q3_k_decode_chunk(asm, chunk)
        elif quant_type == "Q4_K":
            self._emit_q4_k_decode_chunk(asm, chunk)
        elif quant_type == "Q5_K":
            self._emit_q5_k_decode_chunk(asm, chunk)
        elif quant_type == "Q6_K":
            self._emit_q6_k_decode_chunk(asm, chunk)
        else:
            self._emit_q8_0_decode_chunk(asm, chunk)

    def _emit_q6_k_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q6_k_decode_prepare(asm, label_suffix=label_suffix)
        asm.comment("Decode Q6_K low/high payload planes into LDS.")
        for chunk in range(4 * self.physical.decoder.rows):
            self._emit_q6_k_decode_chunk(asm, chunk)

    def _emit_q6_k_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        del label_suffix
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        n_tiles = self.state.solution.matrix_instruction[6]
        n_per_tile = self.state.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        t = r.temporary
        low_shift = self.physical.address.q3_low_shift
        high_shift = self.physical.address.quant_shift
        asm.comment("Derive Q6_K payload shifts and signed scaled d values.")
        asm.inst(f"v_and_b32 v{t}, {n_tiles - 1}, v{r.serial}")
        asm.inst(f"v_lshlrev_b32 v{t}, 4, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, {tiles_per_weight_block - 1}, s3")
        asm.inst(f"v_lshlrev_b32 v{t + 1}, {n_shift}, v{t + 1}")
        asm.inst(f"v_add_nc_u32 v{t}, v{t}, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 6, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 1, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{low_shift}, 2, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t + 1}, 5, v{t}")
        asm.inst(f"v_and_b32 v{t + 1}, 3, v{t + 1}")
        asm.inst(f"v_lshlrev_b32 v{high_shift}, 1, v{t + 1}")
        for row in range(decoder_rows):
            scale = r.quant_scale + row
            d_scaled = t + 1 + 2 * row
            asm.inst(f"v_bfe_i32 v{scale}, v{scale}, 0, 8")
            asm.inst(f"v_cvt_f32_i32_e32 v{scale}, v{scale}")
            asm.inst(
                f"v_fma_mix_f32 v{d_scaled}, v{r.quant_dm + row}, v{scale}, "
                "neg(0) op_sel_hi:[1,0,0]"
            )

    def _emit_q6_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        k_span = self.state.solution.depth_u // decoder_rows
        t = r.temporary
        low = r.global_read_b + 4 * row + first_element // 4
        high = r.global_read_b + 4 * decoder_rows + 4 * row + first_element // 4
        asm.inst(f"v_lshrrev_b32 v{low}, v{self.physical.address.q3_low_shift}, v{low}")
        asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
        asm.inst(
            f"v_lshrrev_b32 v{high}, v{self.physical.address.quant_shift}, v{high}"
        )
        asm.inst(f"v_and_b32 v{high}, 0x03030303, v{high}")
        asm.inst(f"v_lshl_or_b32 v{low}, v{high}, 4, v{low}")
        extraction = self.state.spec.decode.q6.extraction
        d_scaled = t + 1 + 2 * row
        value = t + 1 + 2 * decoder_rows
        if extraction is BackwardExtraction.packed_vopd:
            second_value = value + 1
            rounding = value + 2
            alternate_d = value + 3
            first_subtrahend = alternate_d + 1
            second_subtrahend = alternate_d + 2
            if first_element == 0:
                asm.inst(f"v_mov_b32 v{alternate_d}, v{d_scaled}")
                if row == 0:
                    asm.inst(f"v_mov_b32 v{first_subtrahend}, 32.0")
                    asm.inst(f"v_mov_b32 v{second_subtrahend}, 32.0")
            elements = (
                (first_element, value, first_element + 1, second_value),
                (first_element + 2, value, first_element + 3, second_value),
            )
        else:
            rounding = value + 1
            elements = tuple(
                (element, value, None, None)
                for element in range(first_element, first_element + 4)
            )
        for element, element_value, second_element, second_value in elements:
            if extraction is BackwardExtraction.scalar:
                asm.inst(f"v_bfe_u32 v{element_value}, v{low}, {8 * (element % 4)}, 6")
                asm.inst(f"v_cvt_f32_u32_e32 v{element_value}, v{element_value}")
            else:
                asm.inst(f"v_cvt_f32_ubyte{element % 4}_e32 v{element_value}, v{low}")
            if second_element is None:
                asm.inst(f"v_sub_f32 v{element_value}, v{element_value}, 32.0")
                asm.inst(f"v_mul_f32 v{element_value}, v{d_scaled}, v{element_value}")
            else:
                asm.inst(
                    f"v_cvt_f32_ubyte{second_element % 4}_e32 v{second_value}, v{low}"
                )
                asm.inst(
                    f"v_dual_sub_f32 v{element_value}, v{element_value}, "
                    f"v{first_subtrahend} :: v_dual_sub_f32 v{second_value}, "
                    f"v{second_value}, v{second_subtrahend}"
                )
                asm.inst(
                    f"v_dual_mul_f32 v{element_value}, v{d_scaled}, "
                    f"v{element_value} :: v_dual_mul_f32 v{second_value}, "
                    f"v{alternate_d}, v{second_value}"
                )
            for output_element, output_value in (
                (element, element_value),
                (second_element, second_value),
            ):
                if output_element is None or output_value is None:
                    continue
                lds_address, lds_offset = self.physical.lds.decoded_store_location(
                    self.registers, self.physical.address, output_element, row, k_span
                )
                emit_bf16_rne(asm, output_value, rounding)
                asm.inst(
                    f"ds_store_b16_d16_hi v{lds_address}, v{output_value} "
                    f"offset:{lds_offset}"
                )

    def _emit_q8_0_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q8_0_decode_prepare(asm, label_suffix=label_suffix)
        asm.comment("Decode Q8_0 signed int8 payload and scale into LDS.")
        for chunk in range(4 * self.physical.decoder.rows):
            self._emit_q8_0_decode_chunk(asm, chunk)

    def _emit_q8_0_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        del label_suffix
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        t = r.temporary
        asm.comment("Convert Q8_0 FP16 scales to FP32.")
        for row in range(decoder_rows):
            asm.inst(f"v_cvt_f32_f16 v{t + 1 + 2 * row}, v{r.quant_dm + row}")

    def _emit_q8_0_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        k_span = self.state.solution.depth_u // decoder_rows
        t = r.temporary
        packed = r.global_read_b + 4 * row + first_element // 4
        d_scaled = t + 1 + 2 * row
        value = t + 1 + 2 * decoder_rows
        rounding = value + 1
        lds_address = self.physical.address.lds
        if self.state.spec.decode.q8.extraction is BackwardExtraction.packed_vopd:
            rounding = value + 2
            alternate_d = value + 3
            if first_element == 0:
                asm.inst(f"v_mov_b32 v{alternate_d}, v{d_scaled}")
            pairs = (
                (first_element, value, first_element + 1, value + 1),
                (first_element + 2, value, first_element + 3, value + 1),
            )
        else:
            pairs = tuple(
                (element, value, None, None)
                for element in range(first_element, first_element + 4)
            )
        for element, element_value, second_element, second_value in pairs:
            asm.inst(f"v_bfe_i32 v{element_value}, v{packed}, {8 * (element % 4)}, 8")
            asm.inst(f"v_cvt_f32_i32_e32 v{element_value}, v{element_value}")
            if second_element is None:
                asm.inst(f"v_mul_f32 v{element_value}, v{d_scaled}, v{element_value}")
            else:
                asm.inst(
                    f"v_bfe_i32 v{second_value}, v{packed}, "
                    f"{8 * (second_element % 4)}, 8"
                )
                asm.inst(f"v_cvt_f32_i32_e32 v{second_value}, v{second_value}")
                asm.inst(
                    f"v_dual_mul_f32 v{element_value}, v{d_scaled}, "
                    f"v{element_value} :: v_dual_mul_f32 v{second_value}, "
                    f"v{alternate_d}, v{second_value}"
                )
            for output_element, output_value in (
                (element, element_value),
                (second_element, second_value),
            ):
                if output_element is None or output_value is None:
                    continue
                lds_address, lds_offset = self.physical.lds.decoded_store_location(
                    self.registers, self.physical.address, output_element, row, k_span
                )
                emit_bf16_rne(asm, output_value, rounding)
                asm.inst(
                    f"ds_store_b16_d16_hi v{lds_address}, v{output_value} "
                    f"offset:{lds_offset}"
                )

    def _emit_q3_k_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q3_k_decode_prepare(asm, label_suffix=label_suffix)
        asm.comment("Decode Q3_K 2-bit payload and high mask into LDS.")
        for chunk in range(4 * self.physical.decoder.rows):
            self._emit_q3_k_decode_chunk(asm, chunk)

    def _emit_q3_k_decode_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
    ) -> None:
        del label_suffix
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        n_tiles = self.state.solution.matrix_instruction[6]
        n_per_tile = self.state.solution.macro_tile1
        tiles_per_weight_block = 256 // n_per_tile
        n_shift = n_per_tile.bit_length() - 1
        scale = r.quant_scale
        t = r.temporary
        quant_shift = self.physical.address.quant_shift
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
        if self.physical.decoder.q3_full_vopd:
            if decoder_rows == 1:
                asm.inst(f"v_mov_b32 v{t + 9}, v{t + 3}")
            else:
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
        asm.inst(f"v_mov_b32 v{self.physical.address.q3_low_shift}, v{t + 1}")
        asm.inst(f"v_lshrrev_b32 v{t}, 5, v{t}")
        asm.inst(f"v_mov_b32 v{quant_shift}, v{t}")
        if self.state.spec.decode.q3.extraction is BackwardExtraction.packed:
            asm.inst(f"v_mov_b32 v{t + 2}, 4.0")
            if self.physical.decoder.q3_full_vopd:
                asm.inst(f"v_mov_b32 v{t + 1}, 4.0")

    def _emit_q3_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        k_span = self.state.solution.depth_u // decoder_rows
        t = r.temporary
        low = r.global_read_b + 4 * row + first_element // 4
        high = r.global_read_b + 4 * decoder_rows + 4 * row + first_element // 4
        asm.inst(f"v_lshrrev_b32 v{low}, v{self.physical.address.q3_low_shift}, v{low}")
        asm.inst(f"v_and_b32 v{low}, 0x03030303, v{low}")
        asm.inst(
            f"v_lshrrev_b32 v{high}, v{self.physical.address.quant_shift}, v{high}"
        )
        asm.inst(f"v_and_b32 v{high}, 0x01010101, v{high}")
        if self.state.spec.decode.q3.extraction is BackwardExtraction.packed:
            asm.inst(f"v_lshl_or_b32 v{low}, v{high}, 2, v{low}")
        else:
            asm.inst(f"v_xor_b32 v{high}, 0x01010101, v{high}")
            asm.inst(f"v_lshlrev_b32 v{high}, 2, v{high}")

        def emit_lds_store(element: int, value: int, rounding: int) -> None:
            lds_address, lds_offset = self.physical.lds.decoded_store_location(
                self.registers, self.physical.address, element, row, k_span
            )
            emit_bf16_rne(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )

        if self.state.spec.decode.q3.extraction is BackwardExtraction.scalar:
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
        full_vopd = self.physical.decoder.q3_full_vopd
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

    def _emit_q45_metadata_prepare(
        self,
        asm: _Assembly,
        *,
        label_suffix: str,
        quant_type: str,
    ) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        scale = r.quant_scale
        t = r.temporary
        is_q5 = quant_type == "Q5_K"
        vector_metadata = is_q5 and self.state.spec.decode.q5.vector_metadata_load
        asm.comment(f"Unpack {quant_type} six-bit scale/min fields.")
        if vector_metadata:
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
            lo = r.quant_dm + 4 * row + 1 if vector_metadata else scale + 3 * row
            asm.inst(f"v_and_b32 v{lo}, 0x3f, v{lo}")
            asm.inst(f"v_and_b32 v{lo + 1}, 0x3f, v{lo + 1}")
        asm.inst(f"s_branch {scale_ready}")
        asm.label(scale_odd)
        for row in range(decoder_rows):
            lo = r.quant_dm + 4 * row + 1 if vector_metadata else scale + 3 * row
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

        conversion = (
            "Convert and scale Q5_K d/dmin values in FP32."
            if is_q5
            else "Convert and scale packed FP16 d/dmin values in FP32."
        )
        asm.comment(conversion)
        for row in range(decoder_rows):
            if vector_metadata:
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

    def _emit_q4_k_decode(
        self,
        asm: _Assembly,
        *,
        label_suffix: str = "",
    ) -> None:
        self._emit_q45_metadata_prepare(
            asm, label_suffix=label_suffix, quant_type="Q4_K"
        )
        self._emit_q45_nibble_shift(asm)
        asm.comment("Decode 16 nibbles from each packed output row into LDS.")
        for chunk in range(4 * self.physical.decoder.rows):
            self._emit_q4_k_decode_chunk(asm, chunk)

    def _emit_q4_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        k_span = self.state.solution.depth_u // decoder_rows
        t = r.temporary
        packed = r.global_read_b + 4 * row + first_element // 4
        asm.inst(
            f"v_lshrrev_b32 v{packed}, v{self.physical.address.quant_shift}, v{packed}"
        )
        asm.inst(f"v_and_b32 v{packed}, 0x0f0f0f0f, v{packed}")
        for element in range(first_element, first_element + 4):
            value = t + 1 + 2 * decoder_rows
            rounding = value + 1
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            lds_address, lds_offset = self.physical.lds.decoded_store_location(
                self.registers, self.physical.address, element, row, k_span
            )
            asm.inst(f"v_cvt_f32_ubyte{element % 4}_e32 v{value}, v{packed}")
            asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
            emit_bf16_rne(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )

    def _emit_q5_k_decode(self, asm: _Assembly, *, label_suffix: str = "") -> None:
        self._emit_q45_metadata_prepare(
            asm, label_suffix=label_suffix, quant_type="Q5_K"
        )
        if self.state.spec.decode.q5.hoists_nibble_shift:
            self._emit_q45_nibble_shift(asm)
        asm.comment("Decode Q5_K low nibbles and high payload bits into LDS.")
        for chunk in range(4 * self.physical.decoder.rows):
            self._emit_q5_k_decode_chunk(asm, chunk)

    def _emit_q45_nibble_shift(self, asm: _Assembly) -> None:
        r = self.registers
        t = r.temporary
        asm.comment("Hoist the Q5_K low-payload nibble-half shift.")
        asm.inst(f"v_and_b32 v{t}, 1, v{self.physical.address.quant_shift}")
        asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")

    def _emit_q5_k_decode_chunk(self, asm: _Assembly, chunk: int) -> None:
        r = self.registers
        decoder_rows = self.physical.decoder.rows
        row = chunk // 4
        first_element = 4 * (chunk % 4)
        k_span = self.state.solution.depth_u // decoder_rows
        t = r.temporary
        low = r.global_read_b + 4 * row + first_element // 4
        high = r.global_read_b + 4 * decoder_rows + 4 * row + first_element // 4
        if not self.state.spec.decode.q5.hoists_nibble_shift:
            asm.inst(f"v_and_b32 v{t}, 1, v{self.physical.address.quant_shift}")
            asm.inst(f"v_lshlrev_b32 v{t}, 2, v{t}")
        asm.inst(f"v_lshrrev_b32 v{low}, v{t}, v{low}")
        asm.inst(f"v_and_b32 v{low}, 0x0f0f0f0f, v{low}")
        asm.inst(
            f"v_lshrrev_b32 v{high}, v{self.physical.address.quant_shift}, v{high}"
        )
        asm.inst(f"v_and_b32 v{high}, 0x01010101, v{high}")
        asm.inst(f"v_lshl_or_b32 v{low}, v{high}, 4, v{low}")
        for element in range(first_element, first_element + 4):
            value = t + 1 + 2 * decoder_rows
            rounding = value + 1
            d_scaled = t + 1 + 2 * row
            min_scaled = d_scaled + 1
            lds_address, lds_offset = self.physical.lds.decoded_store_location(
                self.registers, self.physical.address, element, row, k_span
            )
            if self.state.spec.decode.q5.extraction is BackwardExtraction.scalar:
                asm.inst(f"v_bfe_u32 v{value}, v{low}, {8 * (element % 4)}, 5")
                asm.inst(f"v_cvt_f32_u32_e32 v{value}, v{value}")
            else:
                asm.inst(f"v_cvt_f32_ubyte{element % 4}_e32 v{value}, v{low}")
            asm.inst(f"v_fma_f32 v{value}, v{d_scaled}, v{value}, -v{min_scaled}")
            emit_bf16_rne(asm, value, rounding)
            asm.inst(
                f"ds_store_b16_d16_hi v{lds_address}, v{value} offset:{lds_offset}"
            )
