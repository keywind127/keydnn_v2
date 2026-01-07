#include "keydnn_conv2d_transpose_cuda.hpp"

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
// Index helpers (same style as CPU)
// ----------------------------
static inline __host__ __device__ std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return static_cast<std::size_t>((((n * C + c) * H + h) * W + w));
}

static inline __host__ __device__ std::size_t idx4_iohw(
    int ci, int co, int kh, int kw,
    int C_out, int K_h, int K_w
) {
    return static_cast<std::size_t>((((ci * C_out + co) * K_h + kh) * K_w + kw));
}

// ----------------------------
// Forward (gather)
// One thread per output element (n,co,oh,ow)
// ----------------------------
template <typename T>
__global__ void conv2d_transpose_forward_gather_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w,
    const T* __restrict__ b, // may be null
    T* __restrict__ y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    static_assert(std::is_floating_point<T>::value,
        "conv2d_transpose_forward_gather_kernel requires floating point T");

    const int nco = blockIdx.z;
    const int n = nco / C_out;
    const int co = nco - n * C_out;

    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || co >= C_out) return;
    if (oh >= H_out || ow >= W_out) return;

    T acc = static_cast<T>(0);

    // y[n,co,oh,ow] = sum_{ci,kh,kw} x[n,ci,hi,wi] * w[ci,co,kh,kw]
    // where hi = (oh + pad_h - kh) / s_h if divisible
    //       wi = (ow + pad_w - kw) / s_w if divisible
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K_h; ++kh) {
            const int th = oh + pad_h - kh;
            if (th < 0) continue;
            if (s_h <= 0) continue;

            if (th % s_h != 0) continue;
            const int hi = th / s_h;
            if (hi < 0 || hi >= H_in) continue;

            for (int kw = 0; kw < K_w; ++kw) {
                const int tw = ow + pad_w - kw;
                if (tw < 0) continue;
                if (s_w <= 0) continue;

                if (tw % s_w != 0) continue;
                const int wi = tw / s_w;
                if (wi < 0 || wi >= W_in) continue;

                const std::size_t x_off = idx4_nchw(n, ci, hi, wi, C_in, H_in, W_in);
                const std::size_t w_off = idx4_iohw(ci, co, kh, kw, C_out, K_h, K_w);
                acc += x[x_off] * w[w_off];
            }
        }
    }

    if (b) acc += b[co];

    const std::size_t y_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
    y[y_off] = acc;
}

// ----------------------------
// Backward: grad_x (gather)
// One thread per input element (n,ci,hi,wi)
// ----------------------------
template <typename T>
__global__ void conv2d_transpose_backward_gradx_kernel(
    const T* __restrict__ w,
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    static_assert(std::is_floating_point<T>::value,
        "conv2d_transpose_backward_gradx_kernel requires floating point T");

    const int nci = blockIdx.z;
    const int n = nci / C_in;
    const int ci = nci - n * C_in;

    const int wi = blockIdx.x * blockDim.x + threadIdx.x;
    const int hi = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || ci >= C_in) return;
    if (hi >= H_in || wi >= W_in) return;

    // For fixed (n,ci,hi,wi):
    // out_h = hi*s_h - pad_h + kh
    // out_w = wi*s_w - pad_w + kw
    T acc = static_cast<T>(0);

    const int base_oh = hi * s_h - pad_h;
    const int base_ow = wi * s_w - pad_w;

    for (int co = 0; co < C_out; ++co) {
        for (int kh = 0; kh < K_h; ++kh) {
            const int oh = base_oh + kh;
            if (oh < 0 || oh >= H_out) continue;

            for (int kw = 0; kw < K_w; ++kw) {
                const int ow = base_ow + kw;
                if (ow < 0 || ow >= W_out) continue;

                const std::size_t go_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
                const std::size_t w_off = idx4_iohw(ci, co, kh, kw, C_out, K_h, K_w);
                acc += grad_out[go_off] * w[w_off];
            }
        }
    }

    const std::size_t gx_off = idx4_nchw(n, ci, hi, wi, C_in, H_in, W_in);
    grad_x[gx_off] = acc;
}

// ----------------------------
// Backward: grad_w (gather)
// One thread per weight element (ci,co,kh,kw)
// ----------------------------
template <typename T>
__global__ void conv2d_transpose_backward_gradw_kernel(
    const T* __restrict__ x,
    const T* __restrict__ grad_out,
    T* __restrict__ grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    static_assert(std::is_floating_point<T>::value,
        "conv2d_transpose_backward_gradw_kernel requires floating point T");

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = C_in * C_out * K_h * K_w;
    if (tid >= total) return;

    int t = tid;
    const int kw = t % K_w; t /= K_w;
    const int kh = t % K_h; t /= K_h;
    const int co = t % C_out; t /= C_out;
    const int ci = t; // remaining

    // grad_w[ci,co,kh,kw] = sum_{n,hi,wi} x[n,ci,hi,wi] * grad_out[n,co,oh,ow]
    // where oh = hi*s_h - pad_h + kh, ow = wi*s_w - pad_w + kw
    T acc = static_cast<T>(0);

    for (int n = 0; n < N; ++n) {
        for (int hi = 0; hi < H_in; ++hi) {
            const int oh = hi * s_h - pad_h + kh;
            if (oh < 0 || oh >= H_out) continue;

            for (int wi = 0; wi < W_in; ++wi) {
                const int ow = wi * s_w - pad_w + kw;
                if (ow < 0 || ow >= W_out) continue;

                const std::size_t x_off = idx4_nchw(n, ci, hi, wi, C_in, H_in, W_in);
                const std::size_t go_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
                acc += x[x_off] * grad_out[go_off];
            }
        }
    }

    const std::size_t gw_off = idx4_iohw(ci, co, kh, kw, C_out, K_h, K_w);
    grad_w[gw_off] = acc;
}

// ----------------------------
// Launch helpers (template impl)
// ----------------------------
template <typename T>
static inline int conv2d_transpose_forward_cuda_impl(
    const T* x,
    const T* w,
    const T* b,
    T* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    if (!x || !w || !y) return -1;
    if (C_in < 0 || H_in < 0 || W_in < 0) return -2;
    if (K_h <= 0 || K_w <= 0) return -4;
    if (s_h <= 0 || s_w <= 0) return -5;

    // Match Conv2D behavior: zero-sized outputs are legal no-ops.
    // (Also covers cases where N==0 or C_out==0.)
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) {
        return 0;
    }

    // Keep strict checks for negative dims
    if (N < 0 || C_out < 0 || H_out < 0 || W_out < 0) return -3;

    dim3 block(16, 16, 1);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C_out
    );

    conv2d_transpose_forward_gather_kernel<T> << <grid, block >> > (
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
        );

    cudaError_t st = cudaGetLastError();
    return keydnn_cuda_ok(st);
}


template <typename T>
static inline int conv2d_transpose_backward_cuda_impl(
    const T* x,
    const T* w,
    const T* grad_out,
    T* grad_x,
    T* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    if (!x || !w || !grad_out || !grad_x || !grad_w) return -1;
    if (C_in < 0 || H_in < 0 || W_in < 0) return -2;
    if (K_h <= 0 || K_w <= 0) return -4;
    if (s_h <= 0 || s_w <= 0) return -5;

    // Legal no-op when grad_out has zero spatial size or empty batch/channels.
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) {
        return 0;
    }

    if (N < 0 || C_out < 0 || H_out < 0 || W_out < 0) return -3;

    // grad_x kernel: (N*C_in) planes over (H_in, W_in)
    {
        dim3 block(16, 16, 1);
        dim3 grid(
            (W_in + block.x - 1) / block.x,
            (H_in + block.y - 1) / block.y,
            N * C_in
        );

        conv2d_transpose_backward_gradx_kernel<T> << <grid, block >> > (
            w, grad_out, grad_x,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            s_h, s_w,
            pad_h, pad_w
            );

        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) return keydnn_cuda_ok(st);
    }

    // grad_w kernel: 1D over total weights
    {
        const int total = C_in * C_out * K_h * K_w;
        const int block = 256;
        const int grid = (total + block - 1) / block;

        conv2d_transpose_backward_gradw_kernel<T> << <grid, block >> > (
            x, grad_out, grad_w,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            K_h, K_w,
            s_h, s_w,
            pad_h, pad_w
            );

        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) return keydnn_cuda_ok(st);
    }

    // IMPORTANT: surface runtime failures (invalid device function, etc.)
    cudaError_t st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// Exported C ABI functions
// ----------------------------
int keydnn_cuda_conv2d_transpose_forward_f32(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return conv2d_transpose_forward_cuda_impl<float>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

int keydnn_cuda_conv2d_transpose_forward_f64(
    const double* x,
    const double* w,
    const double* b,
    double* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return conv2d_transpose_forward_cuda_impl<double>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

int keydnn_cuda_conv2d_transpose_backward_f32(
    const float* x,
    const float* w,
    const float* grad_out,
    float* grad_x,
    float* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return conv2d_transpose_backward_cuda_impl<float>(
        x, w, grad_out,
        grad_x, grad_w,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

int keydnn_cuda_conv2d_transpose_backward_f64(
    const double* x,
    const double* w,
    const double* grad_out,
    double* grad_x,
    double* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return conv2d_transpose_backward_cuda_impl<double>(
        x, w, grad_out,
        grad_x, grad_w,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}
