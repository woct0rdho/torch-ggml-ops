#pragma once

#include "grouped_mmq_backward_tiled_common.cuh"

namespace torch_ggml_ops::ck {

// Staged paired grouped-backward bodies.
//
// Every pair family shares one structure: an N64 x M128 tile per workgroup
// (four waves of 32 rows), a 32-wide contraction stage, two weight tiles per
// stage (one per projection), STAGES LDS buffers with a single plain barrier
// per stage, clamped activation rows, and one vectorised packed row load per
// thread, projection and stage. Optional inactive-wave suppression lets waves
// whose first row is past the task end skip their activation loads, matrix
// work and stores while still taking part in the shared decode.

struct grouped_backward_pair_decoder_q3_k {
    static constexpr ggml_type type = GGML_TYPE_Q3_K;
    static constexpr int out_features = 512;
    static constexpr int in_features = 2048;
    static constexpr int blocks_per_weight_row = 8;
    static constexpr int padding = GROUPED_BACKWARD_TILED_Q3_PADDING;
    static constexpr int swizzle = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q3_K *>(
            packed_row)[block_index];
        const int low_chunk = value_index >> 7;
        const int low_shift = 2 * ((value_index & 127) >> 5);
        const int low_byte = low_chunk * 32 + (value_index & 31);
        const int high_shift = value_index >> 5;
        const int high_byte = value_index & 31;
        const uint4 low =
            *reinterpret_cast<const uint4 *>(block.qs + low_byte);
        const uint4 high =
            *reinterpret_cast<const uint4 *>(block.hmask + high_byte);
        decode_backward_tile_q3_preloaded(
            block, value_index, low, high, values);
    }
};

struct grouped_backward_pair_decoder_iq2_s {
    static constexpr ggml_type type = GGML_TYPE_IQ2_S;
    static constexpr int out_features = 512;
    static constexpr int in_features = 2048;
    static constexpr int blocks_per_weight_row = 8;
    static constexpr int padding = 0;
    static constexpr int swizzle = GROUPED_BACKWARD_TILED_IQ2_PAIR_SWIZZLE;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        decode_iq2_s_group_16(packed_row, block_index, value_index, values);
    }
};

struct grouped_backward_pair_decoder_iq2_xxs {
    static constexpr ggml_type type = GGML_TYPE_IQ2_XXS;
    static constexpr int out_features = 2048;
    static constexpr int in_features = 4096;
    static constexpr int blocks_per_weight_row = 16;
    static constexpr int padding = 0;
    static constexpr int swizzle = 4;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_iq2_xxs *>(
            packed_row)[block_index];
        const int group = value_index >> 5;
        const int first_sub_group = (value_index & 31) >> 3;
        const int q_word = 2 * group;
        // The two 32-bit code words of a 16-value segment are adjacent.
        uint2 packed;
        __builtin_memcpy(&packed, block.qs + 2 * q_word, sizeof(packed));
        const uint32_t grid_data = packed.x;
        const uint32_t sign_data = packed.y;
        const int first_grid_index = (grid_data >> (8 * first_sub_group)) & 0xff;
        const int second_grid_index =
            (grid_data >> (8 * (first_sub_group + 1))) & 0xff;
        const uint64_t first_grid = iq2xxs_grid[first_grid_index];
        const uint64_t second_grid = iq2xxs_grid[second_grid_index];
        const uint8_t first_signs = static_cast<uint8_t>(
            unpack_ksigns(sign_data >> (7 * first_sub_group)));
        const uint8_t second_signs = static_cast<uint8_t>(
            unpack_ksigns(sign_data >> (7 * (first_sub_group + 1))));
        const float db = fp16_to_fp32(block.d) *
            (0.5f + static_cast<float>(sign_data >> 28)) * 0.25f;
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int local = index & 7;
            const uint64_t grid = index < 8 ? first_grid : second_grid;
            const uint8_t signs = index < 8 ? first_signs : second_signs;
            const int magnitude =
                static_cast<int>((grid >> (8 * local)) & 0xff);
            const int sign = ((signs >> local) & 0x01) == 0 ? 1 : -1;
            values[index] = __float2bfloat16(
                db * static_cast<float>(magnitude * sign));
        }
    }
};

static constexpr int GROUPED_BACKWARD_PAIR_STAGED_N_TILES = 4;
static constexpr int GROUPED_BACKWARD_PAIR_STAGED_N =
    GROUPED_BACKWARD_PAIR_STAGED_N_TILES * BACKWARD_N_PER_TILE;

template <typename Decoder, int M_TILES, int STAGES, bool SKIP_INACTIVE_M>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_staged_tile(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_expert_weight,
        const char * __restrict__ second_expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        backward_shared_b_tile<
            GROUPED_BACKWARD_PAIR_STAGED_N,
            GROUPED_BACKWARD_TILED_K,
            Decoder::padding,
            Decoder::swizzle> (*tiles)[2],
        int row_start,
        int row_end,
        int input_column_start) {
    constexpr int K = GROUPED_BACKWARD_TILED_K;
    constexpr int OUT_FEATURES = Decoder::out_features;
    constexpr int IN_FEATURES = Decoder::in_features;
    constexpr int ITERATIONS = OUT_FEATURES / K;
    constexpr int N_TILES = GROUPED_BACKWARD_PAIR_STAGED_N_TILES;
    constexpr int packed_row_bytes =
        Decoder::blocks_per_weight_row * gguf_block_bytes<Decoder::type>();
    using tile_type = backward_shared_b_tile<
        GROUPED_BACKWARD_PAIR_STAGED_N,
        K,
        Decoder::padding,
        Decoder::swizzle>;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = row_start + wave * M_TILES * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators[M_TILES][N_TILES];

    const int column_group = threadIdx.x % N_TILES;
    const int k_index = threadIdx.x / N_TILES;
    const int local_input_column = BACKWARD_N_PER_TILE * column_group;
    const int input_column = input_column_start + local_input_column;
    const int block_index = input_column / QK_K;
    const int value_index = input_column % QK_K;
    const int a_row_base = wave_row_start + c_row(lane);
    const bool wave_has_rows = wave_row_start < row_end;

    auto decode_stage = [&](
                                int iteration,
                                tile_type & first_tile,
                                tile_type & second_tile) {
        const int64_t offset = static_cast<int64_t>(iteration * K + k_index) *
            packed_row_bytes;
        __hip_bfloat16 values[BACKWARD_N_PER_TILE];
        Decoder::decode(
            first_expert_weight + offset, block_index, value_index, values);
#pragma unroll
        for (int index = 0; index < BACKWARD_N_PER_TILE; ++index) {
            first_tile[(local_input_column + index) * K + k_index] =
                values[index];
        }
        Decoder::decode(
            second_expert_weight + offset, block_index, value_index, values);
#pragma unroll
        for (int index = 0; index < BACKWARD_N_PER_TILE; ++index) {
            second_tile[(local_input_column + index) * K + k_index] =
                values[index];
        }
    };

    auto load_a = [&](
                          const __hip_bfloat16 * grad_output,
                          int iteration,
                          bf16_fragment (&a)[2][M_TILES]) {
        if (SKIP_INACTIVE_M && !wave_has_rows) {
            return;
        }
#pragma unroll
        for (int k_tile = 0; k_tile < K; k_tile += 16) {
#pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                const int row = min(
                    a_row_base + m_tile * BACKWARD_M_PER_TILE, row_end - 1);
                __hip_bfloat16 * fragment = fragment_data(a[k_tile / 16][m_tile]);
                const int64_t offset = static_cast<int64_t>(row) * OUT_FEATURES +
                    iteration * K + k_tile;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    fragment[k] = grad_output[offset + k];
                }
            }
        }
    };

    auto multiply = [&](tile_type & tile, bf16_fragment (&a)[2][M_TILES]) {
        if (SKIP_INACTIVE_M && !wave_has_rows) {
            return;
        }
#pragma unroll
        for (int k_tile = 0; k_tile < K; k_tile += 16) {
#pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; n_tile += 2) {
                bf16_fragment b_first{};
                bf16_fragment b_second{};
                tile.load_fragment_vector(
                    b_first,
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane),
                    k_tile);
                tile.load_fragment_vector(
                    b_second,
                    (n_tile + 1) * BACKWARD_N_PER_TILE + c_row(lane),
                    k_tile);
#pragma unroll
                for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                    wmma_f32_16x16x16_bf16(
                        accumulators[m_tile][n_tile],
                        a[k_tile / 16][m_tile],
                        b_first);
                }
#pragma unroll
                for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                    wmma_f32_16x16x16_bf16(
                        accumulators[m_tile][n_tile + 1],
                        a[k_tile / 16][m_tile],
                        b_second);
                }
            }
        }
    };

    bf16_fragment a[2][M_TILES];
    decode_stage(0, tiles[0][0], tiles[0][1]);
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        __builtin_amdgcn_s_barrier();
        if (iteration + 1 < ITERATIONS) {
            decode_stage(
                iteration + 1,
                tiles[(iteration + 1) % STAGES][0],
                tiles[(iteration + 1) % STAGES][1]);
        }
        load_a(first_grad_output, iteration, a);
        multiply(tiles[iteration % STAGES][0], a);
        load_a(second_grad_output, iteration, a);
        multiply(tiles[iteration % STAGES][1], a);
    }

#pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
#pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const int output_row = wave_row_start +
                    m_tile * BACKWARD_M_PER_TILE +
                    c_column(lane, element);
                const int output_column = input_column_start +
                    n_tile * BACKWARD_N_PER_TILE + c_row(lane);
                if (output_row < row_end) {
                    grad_input[
                        static_cast<int64_t>(output_row) * IN_FEATURES +
                        output_column] = __float2bfloat16(
                            accumulators[m_tile][n_tile].values[element]);
                }
            }
        }
    }
}

template <typename Decoder, int M_TILES, int STAGES, bool SKIP_INACTIVE_M>
static __device__ __forceinline__ void grouped_mmq_pair_grad_input_staged_body(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_packed_weight,
        const char * __restrict__ second_packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int64_t bytes_per_expert) {
    constexpr int M_PER_BLOCK =
        M_TILES * BACKWARD_M_PER_TILE * BACKWARD_WAVES;
    using tile_type = backward_shared_b_tile<
        GROUPED_BACKWARD_PAIR_STAGED_N,
        GROUPED_BACKWARD_TILED_K,
        Decoder::padding,
        Decoder::swizzle>;
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > rows) {
        return;
    }
    const int input_column_start = blockIdx.x * GROUPED_BACKWARD_PAIR_STAGED_N;
    const char * first_expert_weight =
        first_packed_weight + expert * bytes_per_expert;
    const char * second_expert_weight =
        second_packed_weight + expert * bytes_per_expert;
    __shared__ tile_type tiles[STAGES][2];

    for (int block_row_start = row_begin; block_row_start < row_end;
         block_row_start += M_PER_BLOCK) {
        grouped_mmq_pair_grad_input_staged_tile<
            Decoder,
            M_TILES,
            STAGES,
            SKIP_INACTIVE_M>(
            first_grad_output,
            second_grad_output,
            first_expert_weight,
            second_expert_weight,
            grad_input,
            tiles,
            block_row_start,
            row_end,
            input_column_start);
    }
}

} // namespace torch_ggml_ops::ck
