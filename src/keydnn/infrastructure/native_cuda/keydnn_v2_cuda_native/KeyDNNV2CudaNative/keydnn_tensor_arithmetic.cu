// keydnn_tensor_arithmetic.cu
#include "keydnn_tensor_arithmetic.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

    constexpr int kBlock = 256;

    inline int64_t ceil_div_i64(int64_t a, int64_t b) {
        return (a + b - 1) / b;
    }

    inline int status_from_cuda(cudaError_t e) {
        return (e == cudaSuccess) ? 0 : int(e);
    }

    template <typename T>
    __global__ void fill_kernel(T* x, int64_t n, T value) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) x[i] = value;
    }

    template <typename T>
    __global__ void neg_kernel(const T* x, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = -x[i];
    }

    template <typename T>
    __global__ void add_kernel(const T* a, const T* b, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] + b[i];
    }

    template <typename T>
    __global__ void sub_kernel(const T* a, const T* b, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] - b[i];
    }

    template <typename T>
    __global__ void mul_kernel(const T* a, const T* b, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] * b[i];
    }

    template <typename T>
    __global__ void div_kernel(const T* a, const T* b, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] / b[i];
    }

    __global__ void gt_kernel_f32(const float* a, const float* b, float* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }

    __global__ void gt_kernel_f64_out_f32(const double* a, const double* b, float* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }

    template <typename KernelFn, typename... Args>
    inline int launch_1d(int64_t n, KernelFn k, Args... args) {
        if (n <= 0) return 0;

        // Clear any stale error so this call's status reflects THIS launch.
        (void)cudaGetLastError();

        dim3 block(kBlock);
        dim3 grid((unsigned)ceil_div_i64(n, kBlock));
        k << <grid, block >> > (args..., n);

        return status_from_cuda(cudaGetLastError());
    }

    template <typename T>
    __global__ void add_scalar_kernel(const T* a, T alpha, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] + alpha;
    }

    template <typename T>
    __global__ void sub_scalar_kernel(const T* a, T alpha, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] - alpha;
    }

    template <typename T>
    __global__ void mul_scalar_kernel(const T* a, T alpha, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] * alpha;
    }

    template <typename T>
    __global__ void div_scalar_kernel(const T* a, T alpha, T* y, int64_t n) {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a[i] / alpha;
    }


} // namespace

extern "C" {

    // ----------------------------
    // Neg
    // ----------------------------
    int keydnn_cuda_neg_f32(const float* x, float* y, int64_t n) {
        if (!x || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, neg_kernel<float>, x, y);
    }

    int keydnn_cuda_neg_f64(const double* x, double* y, int64_t n) {
        if (!x || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, neg_kernel<double>, x, y);
    }

    // ----------------------------
    // Add
    // ----------------------------
    int keydnn_cuda_add_f32(const float* a, const float* b, float* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, add_kernel<float>, a, b, y);
    }

    int keydnn_cuda_add_f64(const double* a, const double* b, double* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, add_kernel<double>, a, b, y);
    }

    // ----------------------------
    // Sub
    // ----------------------------
    int keydnn_cuda_sub_f32(const float* a, const float* b, float* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, sub_kernel<float>, a, b, y);
    }

    int keydnn_cuda_sub_f64(const double* a, const double* b, double* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, sub_kernel<double>, a, b, y);
    }

    // ----------------------------
    // Div
    // ----------------------------
    int keydnn_cuda_div_f32(const float* a, const float* b, float* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, div_kernel<float>, a, b, y);
    }

    int keydnn_cuda_div_f64(const double* a, const double* b, double* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, div_kernel<double>, a, b, y);
    }

    // ----------------------------
    // GT
    // ----------------------------
    int keydnn_cuda_gt_f32(const float* a, const float* b, float* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, gt_kernel_f32, a, b, y);
    }

    int keydnn_cuda_gt_f64(const double* a, const double* b, float* y, int64_t n) {
        if (!a || !b || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, gt_kernel_f64_out_f32, a, b, y);
    }

    // ----------------------------
    // Add scalar
    // ----------------------------
    int keydnn_cuda_add_scalar_f32(const float* a, float alpha, float* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, add_scalar_kernel<float>, a, alpha, y);
    }

    int keydnn_cuda_add_scalar_f64(const double* a, double alpha, double* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, add_scalar_kernel<double>, a, alpha, y);
    }

    // ----------------------------
    // Sub scalar
    // ----------------------------
    int keydnn_cuda_sub_scalar_f32(const float* a, float alpha, float* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, sub_scalar_kernel<float>, a, alpha, y);
    }

    int keydnn_cuda_sub_scalar_f64(const double* a, double alpha, double* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, sub_scalar_kernel<double>, a, alpha, y);
    }

    // ----------------------------
    // Div scalar
    // ----------------------------
    int keydnn_cuda_div_scalar_f32(const float* a, float alpha, float* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, div_scalar_kernel<float>, a, alpha, y);
    }

    int keydnn_cuda_div_scalar_f64(const double* a, double alpha, double* y, int64_t n) {
        if (!a || !y) return int(cudaErrorInvalidDevicePointer);
        return launch_1d(n, div_scalar_kernel<double>, a, alpha, y);
    }


} // extern "C"
