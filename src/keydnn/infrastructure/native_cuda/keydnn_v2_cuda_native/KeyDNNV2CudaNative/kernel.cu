#include "keydnn_cuda_utils.hpp"
#include "keydnn_maxpool2d_cuda.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

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
// Device-safe -infinity helpers (avoid std::numeric_limits in device code)
// ----------------------------
template <typename T>
__device__ __forceinline__ T neg_inf();

template <>
__device__ __forceinline__ float neg_inf<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ double neg_inf<double>() {
    return -CUDART_INF;
}

// ----------------------------
// atomicAdd for double (portable; avoids __double_as_longlong intrinsics)
// ----------------------------
__device__ inline double atomicAdd_double_fallback(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    // atomicCAS loop for double on older architectures (e.g., sm_52)
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;

        union {
            unsigned long long int ull;
            double d;
        } cur, next;

        cur.ull = assumed;
        next.d = cur.d + val;

        old = atomicCAS(addr_as_ull, assumed, next.ull);
    } while (assumed != old);

    union {
        unsigned long long int ull;
        double d;
    } out;
    out.ull = old;
    return out.d;
#endif
}

template <typename T>
__device__ inline void atomic_add(T* addr, T val);

template <>
__device__ inline void atomic_add<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template <>
__device__ inline void atomic_add<double>(double* addr, double val) {
    atomicAdd_double_fallback(addr, val);
}

// ----------------------------
// Forward kernel
// One thread per output element (n,c,i,j)
// ----------------------------
template <typename T>
__global__ void maxpool2d_forward_kernel(
    const T* __restrict__ x_pad,
    T* __restrict__ y,
    std::int64_t* __restrict__ argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point<T>::value, "maxpool2d_forward_kernel requires floating point T");

    const int nc = (int)blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // W_out
    const int i = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y; // H_out

    if (n >= N || c >= C) return;
    if (i >= H_out || j >= W_out) return;

    const int h0 = i * s_h;
    const int w0 = j * s_w;

    T best = neg_inf<T>();
    int best_h = h0;
    int best_w = w0;

    // Scan window within padded plane
    for (int ph = 0; ph < k_h; ++ph) {
        const int h = h0 + ph;
        for (int pw = 0; pw < k_w; ++pw) {
            const int w = w0 + pw;
            const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
            const T v = x_pad[in_off];

            // Tie-break: first occurrence (row-major) -> strict ">"
            if (v > best) {
                best = v;
                best_h = h;
                best_w = w;
            }
        }
    }

    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
    y[out_off] = best;
    argmax_idx[out_off] = (std::int64_t)best_h * (std::int64_t)W_pad + (std::int64_t)best_w;
}

// ----------------------------
// Backward kernel
// One thread per output element, atomic add into grad_x_pad
// ----------------------------
template <typename T>
__global__ void maxpool2d_backward_kernel(
    const T* __restrict__ grad_out,
    const std::int64_t* __restrict__ argmax_idx,
    T* __restrict__ grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    static_assert(std::is_floating_point<T>::value, "maxpool2d_backward_kernel requires floating point T");

    const int nc = (int)blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int j = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x; // W_out
    const int i = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y; // H_out

    if (n >= N || c >= C) return;
    if (i >= H_out || j >= W_out) return;

    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
    const std::int64_t idx = argmax_idx[out_off];

    const int h = (int)(idx / (std::int64_t)W_pad);
    const int w = (int)(idx % (std::int64_t)W_pad);

    if (h < 0 || h >= H_pad || w < 0 || w >= W_pad) return;

    const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
    const T go = grad_out[out_off];

    atomic_add<T>(&grad_x_pad[in_off], go);
}

// ----------------------------
// Launch helpers (template impl)
// ----------------------------
template <typename T>
static inline int maxpool2d_forward_cuda_impl(
    const T* x_pad,
    T* y,
    std::int64_t* argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    if (!x_pad || !y || !argmax_idx) return -1;
    if (N <= 0 || C <= 0 || H_pad <= 0 || W_pad <= 0) return -2;
    if (H_out <= 0 || W_out <= 0 || k_h <= 0 || k_w <= 0) return -3;
    if (s_h <= 0 || s_w <= 0) return -4;

    dim3 block(16, 16, 1);
    dim3 grid(
        (unsigned int)((W_out + (int)block.x - 1) / (int)block.x),
        (unsigned int)((H_out + (int)block.y - 1) / (int)block.y),
        (unsigned int)(N * C)
    );

    maxpool2d_forward_kernel<T> << <grid, block >> > (
        x_pad, y, argmax_idx,
        N, C, H_pad, W_pad, H_out, W_out,
        k_h, k_w, s_h, s_w
        );

    return keydnn_cuda_ok(cudaGetLastError());
}

template <typename T>
static inline int maxpool2d_backward_cuda_impl(
    const T* grad_out,
    const std::int64_t* argmax_idx,
    T* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    if (!grad_out || !argmax_idx || !grad_x_pad) return -1;
    if (N <= 0 || C <= 0) return -2;
    if (H_out <= 0 || W_out <= 0 || H_pad <= 0 || W_pad <= 0) return -3;

    dim3 block(16, 16, 1);
    dim3 grid(
        (unsigned int)((W_out + (int)block.x - 1) / (int)block.x),
        (unsigned int)((H_out + (int)block.y - 1) / (int)block.y),
        (unsigned int)(N * C)
    );

    maxpool2d_backward_kernel<T> << <grid, block >> > (
        grad_out, argmax_idx, grad_x_pad,
        N, C, H_out, W_out, H_pad, W_pad
        );

    return keydnn_cuda_ok(cudaGetLastError());
}

// ----------------------------
// Exported C ABI functions
// ----------------------------
int keydnn_cuda_maxpool2d_forward_f32(
    const float* x_pad,
    float* y,
    std::int64_t* argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    return maxpool2d_forward_cuda_impl<float>(
        x_pad, y, argmax_idx,
        N, C, H_pad, W_pad, H_out, W_out,
        k_h, k_w, s_h, s_w
    );
}

int keydnn_cuda_maxpool2d_forward_f64(
    const double* x_pad,
    double* y,
    std::int64_t* argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    return maxpool2d_forward_cuda_impl<double>(
        x_pad, y, argmax_idx,
        N, C, H_pad, W_pad, H_out, W_out,
        k_h, k_w, s_h, s_w
    );
}

int keydnn_cuda_maxpool2d_backward_f32(
    const float* grad_out,
    const std::int64_t* argmax_idx,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    return maxpool2d_backward_cuda_impl<float>(
        grad_out, argmax_idx, grad_x_pad,
        N, C, H_out, W_out, H_pad, W_pad
    );
}

int keydnn_cuda_maxpool2d_backward_f64(
    const double* grad_out,
    const std::int64_t* argmax_idx,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    return maxpool2d_backward_cuda_impl<double>(
        grad_out, argmax_idx, grad_x_pad,
        N, C, H_out, W_out, H_pad, W_pad
    );
}
