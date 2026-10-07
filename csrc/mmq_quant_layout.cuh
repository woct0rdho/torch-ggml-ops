#pragma once

#include <hip/hip_runtime.h>

#include "vendor/llama_cpp/common.cuh"
#include <torch/csrc/stable/library.h>

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

int64_t packed_block_bytes(int64_t quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q2_0: return sizeof(block_q2_0);
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default:
            STD_TORCH_CHECK(false, "unsupported quant_type: ", quant_type);
    }
}

int64_t packed_block_values(int64_t quant_type) {
    switch (quant_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL: return QK4_0;
        case GGML_TYPE_Q2_0: return 2 * QK4_0;
        default: return QK_K;
    }
}

int64_t packed_row_bytes(int64_t quant_type, int64_t in_features) {
    return (in_features / packed_block_values(quant_type)) *
        packed_block_bytes(quant_type);
}

} // namespace
