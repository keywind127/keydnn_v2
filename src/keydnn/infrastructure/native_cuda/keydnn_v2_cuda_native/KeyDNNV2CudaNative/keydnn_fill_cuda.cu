#include "keydnn_fill_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// ----------------------------
// Error helper (C ABI returns int)
// ----------------------------
static inline int keydnn_cuda_ok(cudaError_t st) {
    return (st == cudaSuccess) ? 0 : static_cast<int>(st);
}

// ----------------------------
// Fill kernel
// ----------------------------
template <typename T>
__global__ void fill_kernel(T* __restrict__ y, std::int64_t n, T value) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x
        + static_cast<std::int64_t>(threadIdx.x);
    if (idx < n) {
        y[idx] = value;
    }
}

// ----------------------------
// Launch helper
// ----------------------------
template <typename T>
static inline int fill_cuda_impl(T* y, std::int64_t numel, T value) {
    if (!y) return -1;
    if (numel < 0) return -2;
    if (numel == 0) return 0;

    // Clear any stale error from prior CUDA work on this host thread.
    (void)cudaGetLastError();

    const int block = 256;
    const int grid = static_cast<int>((numel + block - 1) / block);

    fill_kernel<T> << <grid, block >> > (y, numel, value);

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return static_cast<int>(st);

    st = cudaDeviceSynchronize();
    return static_cast<int>(st);
}


// ----------------------------
// Exported C ABI
// ----------------------------
int keydnn_cuda_fill_f32(float* y, std::int64_t numel, float value) {
    return fill_cuda_impl<float>(y, numel, value);
}

int keydnn_cuda_fill_f64(double* y, std::int64_t numel, double value) {
    return fill_cuda_impl<double>(y, numel, value);
}
