#pragma once

#include "grouped_mmq_backward_tiled_common.cuh"

namespace torch_ggml_ops::ck {

struct grouped_backward_row_task_decoder_q4_k {
    static constexpr ggml_type type = GGML_TYPE_Q4_K;
    static constexpr int out_features = 2048;
    static constexpr int in_features = 512;
    static constexpr int blocks_per_weight_row = 2;
    static constexpr int swizzle = 16;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q4_K *>(
            packed_row)[block_index];
        const int group = value_index >> 5;
        const int byte = (group >> 1) * 32 + (value_index & 31);
        const uint4 quants =
            *reinterpret_cast<const uint4 *>(block.qs + byte);
        decode_backward_tile_q4_preloaded(block, value_index, quants, values);
    }
};

struct grouped_backward_row_task_decoder_q5_k {
    static constexpr ggml_type type = GGML_TYPE_Q5_K;
    static constexpr int out_features = 2048;
    static constexpr int in_features = 512;
    static constexpr int blocks_per_weight_row = 2;
    static constexpr int swizzle = 8;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q5_K *>(
            packed_row)[block_index];
        const int group = value_index >> 5;
        const int low_byte = (group >> 1) * 32 + (value_index & 31);
        const int high_byte = value_index & 31;
        const uint4 low =
            *reinterpret_cast<const uint4 *>(block.qs + low_byte);
        const uint4 high =
            *reinterpret_cast<const uint4 *>(block.qh + high_byte);
        decode_backward_tile_q5_preloaded<false>(
            block, value_index, low, high, values);
    }
};

struct grouped_backward_row_task_decoder_iq2_s {
    static constexpr ggml_type type = GGML_TYPE_IQ2_S;
    static constexpr int out_features = 2048;
    static constexpr int in_features = 512;
    static constexpr int blocks_per_weight_row = 2;
    static constexpr int swizzle = 16;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        decode_iq2_s_group_16(packed_row, block_index, value_index, values);
    }
};

struct grouped_backward_row_task_decoder_q2_k {
    static constexpr ggml_type type = GGML_TYPE_Q2_K;
    static constexpr int out_features = 4096;
    static constexpr int in_features = 2048;
    static constexpr int blocks_per_weight_row = 8;
    static constexpr int swizzle = 16;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q2_K *>(
            packed_row)[block_index];
        const int group = value_index >> 4;
        const int group_in_half = group & 7;
        const int half = group >> 3;
        const int shift = (group_in_half >> 1) * 2;
        const int byte = half * 32 + (group_in_half & 1) * 16;
        const uint8_t scale_min = block.scales[group];
        const float scale = fp16_to_fp32(block.d) *
            static_cast<float>(scale_min & 0x0f);
        const float minimum = fp16_to_fp32(block.dmin) *
            static_cast<float>(scale_min >> 4);
        const uint4 quants = load_uint4_unaligned(block.qs + byte);
        const auto * quant_bytes = reinterpret_cast<const uint8_t *>(&quants);
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int quant = (quant_bytes[index] >> shift) & 0x03;
            values[index] = __float2bfloat16(
                scale * static_cast<float>(quant) - minimum);
        }
    }
};

struct grouped_backward_row_task_decoder_q2_0_sw8 {
    static constexpr ggml_type type = GGML_TYPE_Q2_0;
    static constexpr int out_features = 2560;
    static constexpr int in_features = 640;
    static constexpr int blocks_per_weight_row = 10;
    static constexpr int swizzle = 8;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q2_0 *>(
            packed_row)[block_index];
        const float scale = fp16_to_fp32(block.d);
        uint32_t word;
        __builtin_memcpy(&word, block.qs + (value_index >> 2), sizeof(word));
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int quant = (word >> (2 * index)) & 0x03;
            values[index] = __float2bfloat16(
                scale * static_cast<float>(quant - 1));
        }
    }
};


struct grouped_backward_row_task_decoder_q2_0_sw4 {
    static constexpr ggml_type type = GGML_TYPE_Q2_0;
    static constexpr int out_features = 2560;
    static constexpr int in_features = 640;
    static constexpr int blocks_per_weight_row = 10;
    static constexpr int swizzle = 4;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q2_0 *>(
            packed_row)[block_index];
        const float scale = fp16_to_fp32(block.d);
        uint32_t word;
        __builtin_memcpy(&word, block.qs + (value_index >> 2), sizeof(word));
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int quant = (word >> (2 * index)) & 0x03;
            values[index] = __float2bfloat16(
                scale * static_cast<float>(quant - 1));
        }
    }
};

struct grouped_backward_row_task_decoder_q2_0 {
    static constexpr ggml_type type = GGML_TYPE_Q2_0;
    static constexpr int out_features = 2560;
    static constexpr int in_features = 640;
    static constexpr int blocks_per_weight_row = 10;
    static constexpr int swizzle = 16;
    static constexpr int swizzle_grain = 0;

    static __device__ __forceinline__ void decode(
            const char * packed_row,
            int block_index,
            int value_index,
            __hip_bfloat16 * values) {
        const auto & block = reinterpret_cast<const block_q2_0 *>(
            packed_row)[block_index];
        const float scale = fp16_to_fp32(block.d);
        // Four consecutive codes per payload byte, lowest bits first, so one
        // sixteen-value segment is one unaligned 32-bit word.
        uint32_t word;
        __builtin_memcpy(&word, block.qs + (value_index >> 2), sizeof(word));
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int quant = (word >> (2 * index)) & 0x03;
            values[index] = __float2bfloat16(
                scale * static_cast<float>(quant - 1));
        }
    }
};

struct grouped_backward_row_task_decoder_q2_k_g4
    : grouped_backward_row_task_decoder_q2_k {
    static constexpr int swizzle_grain = 4;
};

struct grouped_backward_row_task_decoder_q2_0_sw8_g4
    : grouped_backward_row_task_decoder_q2_0_sw8 {
    static constexpr int swizzle_grain = 4;
};

static constexpr int GROUPED_BACKWARD_ROW_TASK_STAGED_N_TILES = 4;
static constexpr int GROUPED_BACKWARD_ROW_TASK_STAGED_M_TILES_PER_WAVE = 2;
static constexpr int GROUPED_BACKWARD_ROW_TASK_STAGED_N =
    GROUPED_BACKWARD_ROW_TASK_STAGED_N_TILES * BACKWARD_N_PER_TILE;
static constexpr int GROUPED_BACKWARD_ROW_TASK_STAGED_M =
    GROUPED_BACKWARD_ROW_TASK_STAGED_M_TILES_PER_WAVE * BACKWARD_M_PER_TILE *
    BACKWARD_WAVES;

// `K_STAGE` and `N_TILES` carry the same meaning as in the paired body: the
// contraction depth per barrier and the number of sixteen-column tiles the
// workgroup owns, with `N_TILES * K_STAGE` decode slots and `N_TILES * K_STAGE`
// shared values per stage. The deployed single shape is `32` by `4`.
template <
    typename Decoder,
    int STAGES,
    bool SKIP_INACTIVE_M = false,
    int K_STAGE = GROUPED_BACKWARD_TILED_K,
    int N_TILES = GROUPED_BACKWARD_ROW_TASK_STAGED_N_TILES,
    bool A_BEFORE_BARRIER = false,
    int THREADS = BACKWARD_THREADS>
static __device__ __forceinline__ void grouped_mmq_grad_input_row_task_staged_tile(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ expert_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        backward_shared_b_tile<
            N_TILES * BACKWARD_N_PER_TILE,
            K_STAGE,
            0,
            Decoder::swizzle,
            Decoder::swizzle_grain> * tiles,
        int row_start,
        int row_end,
        int input_column_start) {
    constexpr int K = K_STAGE;
    constexpr int OUT_FEATURES = Decoder::out_features;
    constexpr int IN_FEATURES = Decoder::in_features;
    constexpr int ITERATIONS = OUT_FEATURES / K;
    constexpr int M_TILES = GROUPED_BACKWARD_ROW_TASK_STAGED_M_TILES_PER_WAVE;
    constexpr int COLUMNS = N_TILES * BACKWARD_N_PER_TILE;
    constexpr int packed_row_bytes =
        Decoder::blocks_per_weight_row * gguf_block_bytes<Decoder::type>();
    using tile_type = backward_shared_b_tile<
        COLUMNS,
        K,
        0,
        Decoder::swizzle,
        Decoder::swizzle_grain>;
    const int wave = threadIdx.x / BACKWARD_WAVE_SIZE;
    const int lane = threadIdx.x % BACKWARD_WAVE_SIZE;
    const int wave_row_start = row_start + wave * M_TILES * BACKWARD_M_PER_TILE;
    f32_accumulator accumulators[M_TILES][N_TILES];

    constexpr int DECODE_SLOTS = N_TILES * K;
    constexpr int DECODE_PASSES = (DECODE_SLOTS + THREADS - 1) / THREADS;
    const int a_row_base = wave_row_start + c_row(lane);
    // Waves whose first row is already past the task end still take part in the
    // shared decode, but they need no activation load, matrix work or store.
    const bool wave_has_rows = wave_row_start < row_end;

    auto decode_slot = [&](int slot, int iteration, tile_type & tile) {
        const int column_group = slot % N_TILES;
        const int k_index = slot / N_TILES;
        const int local_input_column = BACKWARD_N_PER_TILE * column_group;
        const int input_column = input_column_start + local_input_column;
        // The packed block width follows from the geometry the decoder
        // declares: 256 values for the K-quant layouts and 64 for Q2_0.
        constexpr int BLOCK_VALUES =
            IN_FEATURES / Decoder::blocks_per_weight_row;
        const int block_index = input_column / BLOCK_VALUES;
        const int value_index = input_column % BLOCK_VALUES;
        const char * packed_row = expert_weight +
            static_cast<int64_t>(iteration * K + k_index) * packed_row_bytes;
        __hip_bfloat16 values[BACKWARD_N_PER_TILE];
        Decoder::decode(packed_row, block_index, value_index, values);
#pragma unroll
        for (int index = 0; index < BACKWARD_N_PER_TILE; ++index) {
            tile[(local_input_column + index) * K + k_index] = values[index];
        }
    };

    auto decode_stage = [&](int iteration, tile_type & tile) {
#pragma unroll
        for (int pass = 0; pass < DECODE_PASSES; ++pass) {
            const int slot = threadIdx.x + pass * THREADS;
            if constexpr (DECODE_SLOTS % THREADS == 0) {
                decode_slot(slot, iteration, tile);
            } else if (slot < DECODE_SLOTS) {
                decode_slot(slot, iteration, tile);
            }
        }
    };

    auto load_a = [&](int iteration, bf16_fragment (&a)[K / 16][M_TILES]) {
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

    auto multiply = [&](tile_type & tile, bf16_fragment (&a)[K / 16][M_TILES]) {
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

    bf16_fragment a[K / 16][M_TILES];
    decode_stage(0, tiles[0]);
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        // The gradient fragments are private to the wave, so their loads can be
        // issued before the stage barrier and the barrier wait then covers the
        // global-memory latency instead of following it.
        if constexpr (A_BEFORE_BARRIER) {
            load_a(iteration, a);
        }
        __builtin_amdgcn_s_barrier();
        if (iteration + 1 < ITERATIONS) {
            decode_stage(iteration + 1, tiles[(iteration + 1) % STAGES]);
        }
        if constexpr (!A_BEFORE_BARRIER) {
            load_a(iteration, a);
        }
        multiply(tiles[iteration % STAGES], a);
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

template <
    typename Decoder,
    int STAGES,
    bool SKIP_INACTIVE_M = false,
    int K_STAGE = GROUPED_BACKWARD_TILED_K,
    int N_TILES = GROUPED_BACKWARD_ROW_TASK_STAGED_N_TILES,
    bool A_BEFORE_BARRIER = false,
    int THREADS = BACKWARD_THREADS>
static __device__ __forceinline__ void grouped_mmq_grad_input_row_task_staged_body(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int32_t * __restrict__ task_count,
        const int32_t * __restrict__ task_experts,
        const int32_t * __restrict__ task_row_starts,
        const int32_t * __restrict__ task_row_ends,
        int64_t bytes_per_expert) {
    const int task = blockIdx.y;
    if (task >= task_count[0]) {
        return;
    }
    const int expert = task_experts[task];
    const int row_start = task_row_starts[task];
    const int row_end = task_row_ends[task];
    constexpr int COLUMNS = N_TILES * BACKWARD_N_PER_TILE;
    const int input_column_start = blockIdx.x * COLUMNS;
    const char * expert_weight = packed_weight +
        static_cast<int64_t>(expert) * bytes_per_expert;
    __shared__ backward_shared_b_tile<
        COLUMNS,
        K_STAGE,
        0,
        Decoder::swizzle,
        Decoder::swizzle_grain> tiles[STAGES];
    grouped_mmq_grad_input_row_task_staged_tile<
        Decoder,
        STAGES,
        SKIP_INACTIVE_M,
        K_STAGE,
        N_TILES,
        A_BEFORE_BARRIER,
        THREADS>(
        grad_output,
        expert_weight,
        grad_input,
        tiles,
        row_start,
        row_end,
        input_column_start);
}

} // namespace torch_ggml_ops::ck
