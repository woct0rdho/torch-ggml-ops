// torch.utils.cpp_extension disables HIP half operators/conversions by default.
// llama.cpp MMQ deliberately uses the native half/half2 arithmetic surface.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include "mmq_bundle.h"
#include "vendor/llama_cpp/common.cuh"

#include <hip/hip_runtime.h>
#include <Python.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

#include "mmq_dense_validation.cuh"
#include "mmq_grouped_validation.cuh"
#include "mmq_fixed_validation.cuh"
#include "mmq_dense_fixed_routes.cuh"
#include "mmq_grouped_routes.cuh"
#include "mmq_grouped_pair_routes.cuh"

} // namespace

STABLE_TORCH_LIBRARY(torch_ggml_ops, m) {
    m.def("_mmq_launch(Tensor input, Tensor packed_weight, int quant_type, int out_features, "
          "Tensor(a!) output, Tensor(b!) workspace) -> ()");
    m.def("_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, int quant_type, int in_features, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_fixed_grouped_mmq_launch(Tensor input, Tensor packed_weight, Tensor(a!) output, "
          "Tensor(b!) workspace) -> ()");
    m.def("_fixed_grouped_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_grouped_mmq_launch(Tensor input, Tensor packed_weight, Tensor expert_indices, "
          "Tensor expert_offsets, int quant_type, int out_features, Tensor(a!) output, "
          "Tensor(b!) workspace) -> ()");
    m.def("_grouped_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, "
          "Tensor expert_indices, Tensor expert_offsets, int quant_type, int in_features, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_grouped_mmq_pair_launch(Tensor input, Tensor first_packed_weight, "
          "Tensor second_packed_weight, Tensor expert_indices, Tensor expert_offsets, "
          "int quant_type, int out_features, Tensor(a!) first_output, "
          "Tensor(b!) second_output, Tensor(c!) workspace, Tensor(d!) task_count, "
          "Tensor(e!) task_experts, Tensor(f!) task_row_starts, Tensor(g!) task_row_ends) -> ()");
    m.def("_grouped_mmq_pair_grad_input_launch(Tensor first_grad_output, Tensor second_grad_output, "
          "Tensor first_packed_weight, Tensor second_packed_weight, Tensor expert_indices, "
          "Tensor expert_offsets, int quant_type, int in_features, Tensor(a!) grad_input) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(torch_ggml_ops, CUDA, m) {
    m.impl("_mmq_launch", TORCH_BOX(&mmq_launch_cuda));
    m.impl("_mmq_grad_input_launch", TORCH_BOX(&mmq_grad_input_launch_cuda));
    m.impl("_fixed_grouped_mmq_launch", TORCH_BOX(&fixed_grouped_mmq_launch_cuda));
    m.impl("_fixed_grouped_mmq_grad_input_launch", TORCH_BOX(&fixed_grouped_mmq_grad_input_launch_cuda));
    m.impl("_grouped_mmq_launch", TORCH_BOX(&grouped_mmq_launch_cuda));
    m.impl("_grouped_mmq_grad_input_launch", TORCH_BOX(&grouped_mmq_grad_input_launch_cuda));
    m.impl("_grouped_mmq_pair_launch", TORCH_BOX(&grouped_mmq_pair_launch_cuda));
    m.impl("_grouped_mmq_pair_grad_input_launch", TORCH_BOX(&grouped_mmq_pair_grad_input_launch_cuda));
}

PyMODINIT_FUNC PyInit__C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
    };
    return PyModule_Create(&module);
}
