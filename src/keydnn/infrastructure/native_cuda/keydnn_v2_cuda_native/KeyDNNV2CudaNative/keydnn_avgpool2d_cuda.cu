#include "keydnn_avgpool2d_cuda.hpp"

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
// atomicAdd wrapper
// (avoid CAS fallback; require HW atomicAdd for double)
// ----------------------------
template <typename T>
__device__ inline void atomic_add(T* addr, T val);

template <>
__device__ inline void atomic_add<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template <>
__device__ inline void atomic_add<double>(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    // No CAS fallback here by request (avoid advanced intrinsics).
    // This build target cannot safely support double atomic adds.
    (void)addr;
    (void)val;
#endif
}

// ----------------------------
// Forward kernel
// One thread per output element (n,c,i,j)
// ----------------------------
template <typename T>
__global__ void avgpool2d_forward_kernel(
    const T* __restrict__ x_pad,
    T* __restrict__ y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point<T>::value, "avgpool2d_forward_kernel requires floating point T");

    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int j = blockIdx.x * blockDim.x + threadIdx.x; // W_out
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // H_out

    if (n >= N || c >= C) return;
    if (i >= H_out || j >= W_out) return;

    const int h0 = i * s_h;
    const int w0 = j * s_w;

    T sum = static_cast<T>(0);
    for (int ph = 0; ph < k_h; ++ph) {
        const int h = h0 + ph;
        for (int pw = 0; pw < k_w; ++pw) {
            const int w = w0 + pw;
            const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
            sum += x_pad[in_off];
        }
    }

    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
    const T denom = static_cast<T>(k_h * k_w);
    y[out_off] = sum / denom;
}

// ----------------------------
// Backward kernel
// One thread per output element, scatter with atomic add
// ----------------------------
template <typename T>
__global__ void avgpool2d_backward_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point<T>::value, "avgpool2d_backward_kernel requires floating point T");

    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int j = blockIdx.x * blockDim.x + threadIdx.x; // W_out
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // H_out

    if (n >= N || c >= C) return;
    if (i >= H_out || j >= W_out) return;

    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
    const T denom = static_cast<T>(k_h * k_w);
    const T go = grad_out[out_off] / denom;

    const int h0 = i * s_h;
    const int w0 = j * s_w;

    for (int ph = 0; ph < k_h; ++ph) {
        const int h = h0 + ph;
        for (int pw = 0; pw < k_w; ++pw) {
            const int w = w0 + pw;
            const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
            atomic_add<T>(&grad_x_pad[in_off], go);
        }
    }
}

// ----------------------------
// Backward kernel (gather, no atomics)
// One thread per input element (n,c,h,w) accumulates contributions
// from all output windows that cover it.
// ----------------------------
template <typename T>
__global__ void avgpool2d_backward_gather_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point<T>::value, "avgpool2d_backward_gather_kernel requires floating point T");

    const int nc = blockIdx.z;
    const int n = nc / C;
    const int c = nc - n * C;

    const int w = blockIdx.x * blockDim.x + threadIdx.x; // W_pad
    const int h = blockIdx.y * blockDim.y + threadIdx.y; // H_pad

    if (n >= N || c >= C) return;
    if (h >= H_pad || w >= W_pad) return;

    // For a given input (h,w), find all output (i,j) such that:
    // h in [i*s_h, i*s_h + k_h - 1] and w in [j*s_w, j*s_w + k_w - 1]
    //
    // => i in [ceil((h-(k_h-1))/s_h), floor(h/s_h)]
    // => j in [ceil((w-(k_w-1))/s_w), floor(w/s_w)]

    const int i_min = (h - (k_h - 1) + (s_h - 1)) / s_h; // ceil
    const int i_max = h / s_h;                           // floor
    const int j_min = (w - (k_w - 1) + (s_w - 1)) / s_w; // ceil
    const int j_max = w / s_w;                           // floor

    const int ii0 = (i_min < 0) ? 0 : i_min;
    const int ii1 = (i_max >= H_out) ? (H_out - 1) : i_max;
    const int jj0 = (j_min < 0) ? 0 : j_min;
    const int jj1 = (j_max >= W_out) ? (W_out - 1) : j_max;

    T acc = static_cast<T>(0);
    const T denom = static_cast<T>(k_h * k_w);

    for (int i = ii0; i <= ii1; ++i) {
        const int h0 = i * s_h;
        if (h < h0 || h >= h0 + k_h) continue;

        for (int j = jj0; j <= jj1; ++j) {
            const int w0 = j * s_w;
            if (w < w0 || w >= w0 + k_w) continue;

            const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
            acc += grad_out[out_off] / denom;
        }
    }

    const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
    grad_x_pad[in_off] += acc;
}


// ----------------------------
// Launch helpers (template impl)
// ----------------------------
template <typename T>
static inline int avgpool2d_forward_cuda_impl(
    const T* x_pad,
    T* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    if (!x_pad || !y) return -1;
    if (N <= 0 || C <= 0) return -2;
    if (H_pad <= 0 || W_pad <= 0 || H_out <= 0 || W_out <= 0) return -3;
    if (k_h <= 0 || k_w <= 0 || s_h <= 0 || s_w <= 0) return -4;

    dim3 block(16, 16, 1);
    dim3 grid(
        (W_out + block.x - 1) / block.x,
        (H_out + block.y - 1) / block.y,
        N * C
    );

    avgpool2d_forward_kernel<T> << <grid, block >> > (
        x_pad, y,
        N, C,
        H_pad, W_pad,
        H_out, W_out,
        k_h, k_w,
        s_h, s_w
        );

    const cudaError_t st = cudaGetLastError();
    return keydnn_cuda_ok(st);
}

template <typename T>
static inline int avgpool2d_backward_cuda_impl(
    const T* grad_out,
    T* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    if (!grad_out || !grad_x_pad) return -1;
    if (N <= 0 || C <= 0) return -2;
    if (H_out <= 0 || W_out <= 0 || H_pad <= 0 || W_pad <= 0) return -3;
    if (k_h <= 0 || k_w <= 0 || s_h <= 0 || s_w <= 0) return -4;

    // Use gather to avoid atomic requirements (portable for float64 too).
    dim3 block(16, 16, 1);
    dim3 grid(
        (W_pad + block.x - 1) / block.x,
        (H_pad + block.y - 1) / block.y,
        N * C
    );

    avgpool2d_backward_gather_kernel<T> << <grid, block >> > (
        grad_out, grad_x_pad,
        N, C,
        H_out, W_out,
        H_pad, W_pad,
        k_h, k_w,
        s_h, s_w
        );

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    // IMPORTANT: surface runtime failures (e.g., invalid device function)
    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}


// ----------------------------
// Exported C ABI functions
// ----------------------------
int keydnn_cuda_avgpool2d_forward_f32(
    const float* x_pad,
    float* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    return avgpool2d_forward_cuda_impl<float>(
        x_pad, y,
        N, C,
        H_pad, W_pad,
        H_out, W_out,
        k_h, k_w,
        s_h, s_w
    );
}

int keydnn_cuda_avgpool2d_forward_f64(
    const double* x_pad,
    double* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    return avgpool2d_forward_cuda_impl<double>(
        x_pad, y,
        N, C,
        H_pad, W_pad,
        H_out, W_out,
        k_h, k_w,
        s_h, s_w
    );
}

int keydnn_cuda_avgpool2d_backward_f32(
    const float* grad_out,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    return avgpool2d_backward_cuda_impl<float>(
        grad_out, grad_x_pad,
        N, C,
        H_out, W_out,
        H_pad, W_pad,
        k_h, k_w,
        s_h, s_w
    );
}

int keydnn_cuda_avgpool2d_backward_f64(
    const double* grad_out,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
    // Device-side arch guard doesn't apply here (host compilation),
    // but keeping the style explicit: double backward needs atomicAdd(double).
    // If you compile for < sm_60 you should reject at runtime; we do it in-kernel.
#endif

    // We reject sm_<60 at runtime by building for >=60, or by
    // checking kernel error after launch (invalid device function).
    return avgpool2d_backward_cuda_impl<double>(
        grad_out, grad_x_pad,
        N, C,
        H_out, W_out,
        H_pad, W_pad,
        k_h, k_w,
        s_h, s_w
    );
}
