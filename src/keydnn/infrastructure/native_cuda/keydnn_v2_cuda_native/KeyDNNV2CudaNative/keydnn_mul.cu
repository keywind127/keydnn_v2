#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>

// ============================================================
// Kernels
// ============================================================

template <typename T>
__global__ void mul_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ y,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        y[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void mul_scalar_kernel(
    const T* __restrict__ a,
    T alpha,
    T* __restrict__ y,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        y[i] = a[i] * alpha;
    }
}

// ============================================================
// Launch helpers (NO cudaDeviceSynchronize here)
// ============================================================

template <typename T>
static int mul_launch(const T* a, const T* b, T* y, int numel) {
    if (numel < 0) return 1;
    if (numel == 0) return 0;
    if (!a || !b || !y) return 2;

    // Clear any stale CUDA error
    (void)cudaGetLastError();

    int block = 256;
    int grid = (numel + block - 1) / block;
    mul_kernel<T> << <grid, block >> > (a, b, y, numel);

    cudaError_t st = cudaGetLastError();
    return (st == cudaSuccess) ? 0 : 3;
}

template <typename T>
static int mul_scalar_launch(const T* a, T alpha, T* y, int numel) {
    if (numel < 0) return 1;
    if (numel == 0) return 0;
    if (!a || !y) return 2;

    (void)cudaGetLastError();

    int block = 256;
    int grid = (numel + block - 1) / block;
    mul_scalar_kernel<T> << <grid, block >> > (a, alpha, y, numel);

    cudaError_t st = cudaGetLastError();
    return (st == cudaSuccess) ? 0 : 3;
}

// ============================================================
// C ABI exports
// ============================================================

extern "C" int keydnn_cuda_mul_f32(
    const float* a,
    const float* b,
    float* y,
    int numel
) {
    return mul_launch<float>(a, b, y, numel);
}

extern "C" int keydnn_cuda_mul_f64(
    const double* a,
    const double* b,
    double* y,
    int numel
) {
    return mul_launch<double>(a, b, y, numel);
}

extern "C" int keydnn_cuda_mul_scalar_f32(
    const float* a,
    float alpha,
    float* y,
    int numel
) {
    return mul_scalar_launch<float>(a, alpha, y, numel);
}

extern "C" int keydnn_cuda_mul_scalar_f64(
    const double* a,
    double alpha,
    double* y,
    int numel
) {
    return mul_scalar_launch<double>(a, alpha, y, numel);
}
