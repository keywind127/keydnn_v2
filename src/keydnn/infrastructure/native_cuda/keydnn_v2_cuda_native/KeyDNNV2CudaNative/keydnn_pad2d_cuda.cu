#include "keydnn_pad2d_cuda.hpp"

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
// NCHW index helper
// ----------------------------
static inline __host__ __device__ std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return static_cast<std::size_t>((((n * C + c) * H + h) * W + w));
}

// ----------------------------
// Pad kernel: one thread per output element
// ----------------------------
template <typename T>
__global__ void pad2d_kernel(
    const T* __restrict__ x,
    T* __restrict__ y_pad,
    int N, int C,
    int H, int W,
    int H_pad, int W_pad,
    int p_h, int p_w,
    T pad_value
) {
    static_assert(std::is_floating_point<T>::value, "pad2d_kernel requires floating point T");

    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int w_pad = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_pad = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || c >= C) return;
    if (h_pad >= H_pad || w_pad >= W_pad) return;

    const int h_in = h_pad - p_h;
    const int w_in = w_pad - p_w;

    const std::size_t out_off = idx4_nchw(n, c, h_pad, w_pad, C, H_pad, W_pad);

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        const std::size_t in_off = idx4_nchw(n, c, h_in, w_in, C, H, W);
        y_pad[out_off] = x[in_off];
    }
    else {
        y_pad[out_off] = pad_value;
    }
}

// ----------------------------
// Crop kernel: one thread per output element
// ----------------------------
template <typename T>
__global__ void crop2d_kernel(
    const T* __restrict__ x_pad,
    T* __restrict__ y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
) {
    static_assert(std::is_floating_point<T>::value, "crop2d_kernel requires floating point T");

    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || c >= C) return;
    if (h >= H || w >= W) return;

    const int h_pad = h + p_h;
    const int w_pad = w + p_w;

    if (h_pad < 0 || h_pad >= H_pad || w_pad < 0 || w_pad >= W_pad) return;

    const std::size_t in_off = idx4_nchw(n, c, h_pad, w_pad, C, H_pad, W_pad);
    const std::size_t out_off = idx4_nchw(n, c, h, w, C, H, W);
    y[out_off] = x_pad[in_off];
}

// ----------------------------
// Launch helpers
// ----------------------------
template <typename T>
static inline int pad2d_cuda_impl(
    const T* x,
    T* y_pad,
    int N, int C,
    int H, int W,
    int p_h, int p_w,
    T pad_value
) {
    if (!x || !y_pad) return -1;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return -2;
    if (p_h < 0 || p_w < 0) return -3;

    const int H_pad = H + 2 * p_h;
    const int W_pad = W + 2 * p_w;
    if (H_pad <= 0 || W_pad <= 0) return -4;

    dim3 block(16, 16, 1);
    dim3 grid(
        (W_pad + block.x - 1) / block.x,
        (H_pad + block.y - 1) / block.y,
        N * C
    );

    pad2d_kernel<T> << <grid, block >> > (
        x, y_pad,
        N, C,
        H, W,
        H_pad, W_pad,
        p_h, p_w,
        pad_value
        );

    return keydnn_cuda_ok(cudaGetLastError());
}

template <typename T>
static inline int crop2d_cuda_impl(
    const T* x_pad,
    T* y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
) {
    if (!x_pad || !y) return -1;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return -2;
    if (H_pad <= 0 || W_pad <= 0) return -3;
    if (p_h < 0 || p_w < 0) return -4;

    dim3 block(16, 16, 1);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        N * C
    );

    crop2d_kernel<T> << <grid, block >> > (
        x_pad, y,
        N, C,
        H_pad, W_pad,
        p_h, p_w,
        H, W
        );

    return keydnn_cuda_ok(cudaGetLastError());
}

// ----------------------------
// Exported C ABI
// ----------------------------
int keydnn_cuda_pad2d_f32(
    const float* x,
    float* y_pad,
    int N, int C,
    int H, int W,
    int p_h, int p_w,
    float pad_value
) {
    return pad2d_cuda_impl<float>(x, y_pad, N, C, H, W, p_h, p_w, pad_value);
}

int keydnn_cuda_pad2d_f64(
    const double* x,
    double* y_pad,
    int N, int C,
    int H, int W,
    int p_h, int p_w,
    double pad_value
) {
    return pad2d_cuda_impl<double>(x, y_pad, N, C, H, W, p_h, p_w, pad_value);
}

int keydnn_cuda_crop2d_f32(
    const float* x_pad,
    float* y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
) {
    return crop2d_cuda_impl<float>(x_pad, y, N, C, H_pad, W_pad, p_h, p_w, H, W);
}

int keydnn_cuda_crop2d_f64(
    const double* x_pad,
    double* y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
) {
    return crop2d_cuda_impl<double>(x_pad, y, N, C, H_pad, W_pad, p_h, p_w, H, W);
}
