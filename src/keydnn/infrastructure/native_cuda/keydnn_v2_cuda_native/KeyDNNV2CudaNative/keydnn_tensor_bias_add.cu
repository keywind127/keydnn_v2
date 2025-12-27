#include "keydnn_tensor_arithmetic.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

    constexpr int kBlock = 256;

    inline std::int64_t ceil_div_i64(std::int64_t a, std::int64_t b) {
        return (a + b - 1) / b;
    }

    inline int status_from_cuda(cudaError_t e) {
        return (e == cudaSuccess) ? 0 : int(e);
    }

    // Launch helper for 1D grids (n threads)
    template <typename KernelFn, typename... Args>
    inline int launch_1d(std::int64_t n, KernelFn k, Args... args) {
        if (n <= 0) return 0;

        (void)cudaGetLastError();
        dim3 block(kBlock);
        dim3 grid((unsigned)ceil_div_i64(n, kBlock));
        k << <grid, block >> > (args..., n);
        return status_from_cuda(cudaGetLastError());
    }

    template <typename T>
    __global__ void bias_add_kernel_2d(const T* x, const T* b, T* y, std::int64_t out, std::int64_t n) {
        // n == batch*out
        std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            std::int64_t j = i % out;
            y[i] = x[i] + b[j];
        }
    }

    template <typename T>
    __global__ void bias_add_inplace_kernel_2d(T* y, const T* b, std::int64_t out, std::int64_t n) {
        std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            std::int64_t j = i % out;
            y[i] = y[i] + b[j];
        }
    }

} // namespace

extern "C" {

    int keydnn_cuda_bias_add_f32(
        const float* x,
        const float* b,
        float* y,
        std::int64_t batch,
        std::int64_t out
    ) {
        if (!x || !b || !y) return int(cudaErrorInvalidDevicePointer);
        if (batch < 0 || out < 0) return int(cudaErrorInvalidValue);
        const std::int64_t n = batch * out;
        return launch_1d(n, bias_add_kernel_2d<float>, x, b, y, out);
    }

    int keydnn_cuda_bias_add_f64(
        const double* x,
        const double* b,
        double* y,
        std::int64_t batch,
        std::int64_t out
    ) {
        if (!x || !b || !y) return int(cudaErrorInvalidDevicePointer);
        if (batch < 0 || out < 0) return int(cudaErrorInvalidValue);
        const std::int64_t n = batch * out;
        return launch_1d(n, bias_add_kernel_2d<double>, x, b, y, out);
    }

    int keydnn_cuda_bias_add_inplace_f32(
        float* y,
        const float* b,
        std::int64_t batch,
        std::int64_t out
    ) {
        if (!y || !b) return int(cudaErrorInvalidDevicePointer);
        if (batch < 0 || out < 0) return int(cudaErrorInvalidValue);
        const std::int64_t n = batch * out;
        return launch_1d(n, bias_add_inplace_kernel_2d<float>, y, b, out);
    }

    int keydnn_cuda_bias_add_inplace_f64(
        double* y,
        const double* b,
        std::int64_t batch,
        std::int64_t out
    ) {
        if (!y || !b) return int(cudaErrorInvalidDevicePointer);
        if (batch < 0 || out < 0) return int(cudaErrorInvalidValue);
        const std::int64_t n = batch * out;
        return launch_1d(n, bias_add_inplace_kernel_2d<double>, y, b, out);
    }

} // extern "C"
