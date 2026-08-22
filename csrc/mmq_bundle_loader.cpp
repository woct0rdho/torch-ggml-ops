#include "mmq_bundle_loader.h"

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch_ggml_ops::mmq_bundle::detail {
namespace {

struct LoadedKernel {
    std::vector<std::uint8_t> image;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

std::mutex loaded_kernels_mutex;
std::map<std::pair<int, MMQKernelIndex>, std::unique_ptr<LoadedKernel>> loaded_kernels;

void check_hip(hipError_t status, const std::string & operation) {
    if (status != hipSuccess) {
        fail(operation + " failed: " + hipGetErrorString(status));
    }
}

void bundle_path_anchor() {}

std::filesystem::path bundle_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&bundle_path_anchor), &info) == 0 ||
        info.dli_fname == nullptr) {
        fail("dladdr could not locate the extension shared object");
    }
    return std::filesystem::path(info.dli_fname).parent_path() /
        "kernels" / "gfx1151";
}

std::vector<std::uint8_t> read_artifact(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail("cannot open kernel artifact " + path.string());
    }
    const std::streamoff end = file.tellg();
    if (end < 0) {
        fail("cannot determine kernel artifact size " + path.string());
    }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char *>(image.data()), end)) {
        fail("cannot read kernel artifact " + path.string());
    }
    return image;
}

LoadedKernel & resolve_kernel(MMQKernelIndex index) {
    int device = -1;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    const auto cache_key = std::make_pair(device, index);
    std::lock_guard<std::mutex> lock(loaded_kernels_mutex);
    const auto found = loaded_kernels.find(cache_key);
    if (found != loaded_kernels.end()) {
        return *found->second;
    }

    const char * symbol = mmq_kernel_symbol(index);
    std::filesystem::path path = bundle_directory() / symbol;
    path += ".hsaco";
    auto loaded = std::make_unique<LoadedKernel>();
    loaded->image = read_artifact(path);
    check_hip(
        hipModuleLoadData(&loaded->module, loaded->image.data()),
        "hipModuleLoadData for " + path.string());
    const hipError_t function_status = hipModuleGetFunction(
        &loaded->function, loaded->module, symbol);
    if (function_status != hipSuccess) {
        (void)hipModuleUnload(loaded->module);
        fail("hipModuleGetFunction failed for " + std::string(symbol) +
             ": " + hipGetErrorString(function_status));
    }
    LoadedKernel & result = *loaded;
    loaded_kernels.emplace(cache_key, std::move(loaded));
    return result;
}

} // namespace

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("MMQ gfx1151 bundle: " + message);
}

void launch_kernel(
        MMQKernelIndex index,
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        unsigned int block_x,
        unsigned int block_y,
        unsigned int block_z,
        hipStream_t stream,
        void ** arguments) {
    LoadedKernel & loaded = resolve_kernel(index);
    const hipError_t status = hipModuleLaunchKernel(
        loaded.function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        0,
        stream,
        arguments,
        nullptr);
    if (status != hipSuccess) {
        fail("hipModuleLaunchKernel failed for " +
             std::string(mmq_kernel_symbol(index)) + ": " +
             hipGetErrorString(status));
    }
}

} // namespace torch_ggml_ops::mmq_bundle::detail
