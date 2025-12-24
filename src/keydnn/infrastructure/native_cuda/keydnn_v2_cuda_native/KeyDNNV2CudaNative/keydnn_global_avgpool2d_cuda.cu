#include "keydnn_global_avgpool2d_cuda.hpp"

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
// NCHW index helper (same style as CPU)
// ----------------------------
static inline __host__ __device__ std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return static_cast<std::size_t>((((n * C + c) * H + h) * W + w));
}

// ----------------------------
// Forward kernel
// One thread per (n,c) computes mean over H*W
// y layout is NCHW with H_out=W_out=1 => y[n,c,0,0]
// ----------------------------
template <typename T>
__global__ void global_avgpool2d_forward_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int N, int C,
    int H, int W
) {
    static_assert(std::is_floating_point<T>::value, "global_avgpool2d_forward_kernel requires floating point T");

    const int nc = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C;
    if (nc >= total) return;

    const int n = nc / C;
    const int c = nc - n * C;

    T sum = static_cast<T>(0);
    const int HW = H * W;

    // Simple reduction (portable, not optimized)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const std::size_t off = idx4_nchw(n, c, h, w, C, H, W);
            sum += x[off];
        }
    }

    const T denom = static_cast<T>(HW);
    // y is (N,C,1,1): index is equivalent to ((n*C + c)*1 + 0)*1 + 0
    y[static_cast<std::size_t>(nc)] = sum / denom;
}

// ----------------------------
// Backward kernel
// One thread per input element (n,c,h,w)
// grad_x = grad_out[n,c,0,0] / (H*W)
// ----------------------------
template <typename T>
__global__ void global_avgpool2d_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    int N, int C,
    int H, int W
) {
    static_assert(std::is_floating_point<T>::value, "global_avgpool2d_backward_kernel requires floating point T");

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    const int HW = H * W;
    const T scale = static_cast<T>(1) / static_cast<T>(HW);

    // Map flat idx to (n,c,h,w)
    const int w = idx % W;
    const int t1 = idx / W;
    const int h = t1 % H;
    const int t2 = t1 / H;
    const int c = t2 % C;
    const int n = t2 / C;

    // grad_out stored as (N,C,1,1), flattened as N*C
    const int nc = n * C + c;
    const T go = grad_out[static_cast<std::size_t>(nc)];

    grad_x[static_cast<std::size_t>(idx)] = go * scale;
}

// ----------------------------
// Launch helpers (template impl)
// ----------------------------
template <typename T>
static inline int global_avgpool2d_forward_cuda_impl(
    const T* x,
    T* y,
    int N, int C,
    int H, int W
) {
    if (!x || !y) return -1;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return -2;

    const int total = N * C;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    global_avgpool2d_forward_kernel<T> << <grid, block >> > (
        x, y, N, C, H, W
        );

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

template <typename T>
static inline int global_avgpool2d_backward_cuda_impl(
    const T* grad_out,
    T* grad_x,
    int N, int C,
    int H, int W
) {
    if (!grad_out || !grad_x) return -1;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return -2;

    const int total = N * C * H * W;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    global_avgpool2d_backward_kernel<T> << <grid, block >> > (
        grad_out, grad_x, N, C, H, W
        );

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// Exported_attach: C ABI functions
// ----------------------------
int keydnn_cuda_global_avgpool2d_forward_f32(
    const float* x,
    float* y,
    int N, int C,
    int H, int W
) {
    return global_avgpool2d_forward_cuda_impl<float>(x, y, N, C, H, W);
}

int keydnn_cuda_global_avgpool2d_forward_f64(
    const double* x,
    double* y,
    int N, int C,
    int H, int W
) {
    return global_avgpool2d_forward_cuda_impl<double>(x, y, N, C, H, W);
}

int keydnn_cuda_global_avgpool2d_backward_f32(
    const float* grad_out,
    float* grad_x,
    int N, int C,
    int H, int W
) {
    return global_avgpool2d_backward_cuda_impl<float>(grad_out, grad_x, N, C, H, W);
}

int keydnn_cuda_global_avgpool2d_backward_f64(
    const double* grad_out,
    double* grad_x,
    int N, int C,
    int H, int W
) {
    return global_avgpool2d_backward_cuda_impl<double>(grad_out, grad_x, N, C, H, W);
}
