#pragma once

#include "generated/mmq_bundle_table.cuh"

#include <hip/hip_runtime_api.h>

#include <string>

namespace torch_ggml_ops::mmq_bundle::detail {

[[noreturn]] void fail(const std::string & message);

void launch_kernel(
    MMQKernelIndex index,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    hipStream_t stream,
    void ** arguments);

} // namespace torch_ggml_ops::mmq_bundle::detail
