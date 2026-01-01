// keydnn_cuda_conv2d.cu
#include "keydnn_cuda_conv2d.hpp"
#include <cuda_runtime.h>

// ---- atomicAdd helpers ----

__device__ __forceinline__ float keydnn_atomic_add(float* addr, float v) {
    return atomicAdd(addr, v);
}

__device__ __forceinline__ double keydnn_atomic_add(double* addr, double v) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, v);
#else
    // Fallback for sm_52/sm_5x: emulate atomicAdd(double) using atomicCAS
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(addr);

    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double old_val = __longlong_as_double((long long)assumed);
        double new_val = old_val + v;
        old = atomicCAS(
            address_as_ull,
            assumed,
            (unsigned long long int)__double_as_longlong(new_val)
        );
    } while (assumed != old);

    return __longlong_as_double((long long)old);
#endif
}


static inline int _validate_common(
    const void* x_pad,
    const void* w,
    const void* y_or_grad_out,
    int N, int C_in, int H_pad, int W_pad,
    int C_out, int H_out, int W_out,
    int K_h, int K_w, int s_h, int s_w
) {
    if (!x_pad || !w || !y_or_grad_out) return (int)cudaErrorInvalidValue;
    if (N < 0 || C_in < 0 || H_pad < 0 || W_pad < 0) return (int)cudaErrorInvalidValue;
    if (C_out < 0 || H_out < 0 || W_out < 0) return (int)cudaErrorInvalidValue;
    if (K_h <= 0 || K_w <= 0) return (int)cudaErrorInvalidValue;
    if (s_h <= 0 || s_w <= 0) return (int)cudaErrorInvalidValue;
    // Quick-exit: empty output -> nothing to do
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;
    return 0;
}

static inline __device__ __forceinline__ std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return (std::size_t)((((n * C + c) * H + h) * W + w));
}

static inline __device__ __forceinline__ std::size_t idx4_oihw(
    int co, int ci, int kh, int kw,
    int C_in, int K_h, int K_w
) {
    return (std::size_t)((((co * C_in + ci) * K_h + kh) * K_w + kw));
}

template <typename T>
__global__ void conv2d_forward_kernel(
    const T* __restrict__ x_pad,
    const T* __restrict__ w,
    const T* __restrict__ b, // may be nullptr
    T* __restrict__ y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const std::size_t tid = (std::size_t)blockIdx.x * (std::size_t)blockDim.x + (std::size_t)threadIdx.x;
    const std::size_t total = (std::size_t)N * (std::size_t)C_out * (std::size_t)H_out * (std::size_t)W_out;
    if (tid >= total) return;

    // Unflatten tid -> (n, co, i, j)
    std::size_t t = tid;
    const int j = (int)(t % (std::size_t)W_out); t /= (std::size_t)W_out;
    const int i = (int)(t % (std::size_t)H_out); t /= (std::size_t)H_out;
    const int co = (int)(t % (std::size_t)C_out); t /= (std::size_t)C_out;
    const int n = (int)t;

    const int h0 = i * s_h;
    const int w0 = j * s_w;

    T acc = (T)0;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K_h; ++kh) {
            const int h = h0 + kh;
            for (int kw = 0; kw < K_w; ++kw) {
                const int ww = w0 + kw;

                const std::size_t in_off = idx4_nchw(n, ci, h, ww, C_in, H_pad, W_pad);
                const std::size_t w_off = idx4_oihw(co, ci, kh, kw, C_in, K_h, K_w);

                acc += x_pad[in_off] * w[w_off];
            }
        }
    }

    if (b) acc += b[co];

    const std::size_t out_off = idx4_nchw(n, co, i, j, C_out, H_out, W_out);
    y[out_off] = acc;
}

template <typename T>
__global__ void conv2d_backward_kernel(
    const T* __restrict__ x_pad,
    const T* __restrict__ w,
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x_pad,
    T* __restrict__ grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const std::size_t tid = (std::size_t)blockIdx.x * (std::size_t)blockDim.x + (std::size_t)threadIdx.x;
    const std::size_t total = (std::size_t)N * (std::size_t)C_out * (std::size_t)H_out * (std::size_t)W_out;
    if (tid >= total) return;

    // Unflatten tid -> (n, co, i, j)
    std::size_t t = tid;
    const int j = (int)(t % (std::size_t)W_out); t /= (std::size_t)W_out;
    const int i = (int)(t % (std::size_t)H_out); t /= (std::size_t)H_out;
    const int co = (int)(t % (std::size_t)C_out); t /= (std::size_t)C_out;
    const int n = (int)t;

    const std::size_t go_off = idx4_nchw(n, co, i, j, C_out, H_out, W_out);
    const T go = grad_out[go_off];

    const int h0 = i * s_h;
    const int w0 = j * s_w;

    // Mirrors your CPU logic:
    // grad_w[co,ci,kh,kw] += go * x_pad[n,ci,h0+kh,w0+kw]
    // grad_x_pad[n,ci,h0+kh,w0+kw] += go * w[co,ci,kh,kw]
    //
    // Parallel threads overlap heavily -> atomic adds on both grad buffers.
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K_h; ++kh) {
            const int h = h0 + kh;
            for (int kw = 0; kw < K_w; ++kw) {
                const int ww = w0 + kw;

                const std::size_t in_off = idx4_nchw(n, ci, h, ww, C_in, H_pad, W_pad);
                const std::size_t w_off = idx4_oihw(co, ci, kh, kw, C_in, K_h, K_w);

                const T xv = x_pad[in_off];
                const T wv = w[w_off];

                keydnn_atomic_add(&grad_w[w_off], go * xv);
                keydnn_atomic_add(&grad_x_pad[in_off], go * wv);

            }
        }
    }
}

template <typename T>
static int launch_conv2d_forward(
    const T* x_pad,
    const T* w,
    const T* b,
    T* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const int v = _validate_common(x_pad, w, y, N, C_in, H_pad, W_pad, C_out, H_out, W_out, K_h, K_w, s_h, s_w);
    if (v != 0) return v;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    (void)cudaGetLastError(); // clear sticky

    const std::size_t total = (std::size_t)N * (std::size_t)C_out * (std::size_t)H_out * (std::size_t)W_out;

    // Reasonable default. You can tune later.
    const int threads = 256;
    const int blocks = (int)((total + (std::size_t)threads - 1) / (std::size_t)threads);

    conv2d_forward_kernel<T> << <blocks, threads >> > (
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
        );

    cudaError_t cu = cudaPeekAtLastError();
    if (cu != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)cu;
    }
    return 0;
}

template <typename T>
static int launch_conv2d_backward(
    const T* x_pad,
    const T* w,
    const T* grad_out,
    T* grad_x_pad,
    T* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const int v = _validate_common(x_pad, w, grad_out, N, C_in, H_pad, W_pad, C_out, H_out, W_out, K_h, K_w, s_h, s_w);
    if (v != 0) return v;
    if (!grad_x_pad || !grad_w) return (int)cudaErrorInvalidValue;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    (void)cudaGetLastError(); // clear sticky

    const std::size_t total = (std::size_t)N * (std::size_t)C_out * (std::size_t)H_out * (std::size_t)W_out;

    const int threads = 256;
    const int blocks = (int)((total + (std::size_t)threads - 1) / (std::size_t)threads);

    conv2d_backward_kernel<T> << <blocks, threads >> > (
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
        );

    cudaError_t cu = cudaPeekAtLastError();
    if (cu != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)cu;
    }
    return 0;
}

// -------------------- Exports --------------------

extern "C" int keydnn_cuda_conv2d_forward_f32(
    const float* x_pad,
    const float* w,
    const float* b,
    float* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_forward<float>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_forward_f64(
    const double* x_pad,
    const double* w,
    const double* b,
    double* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_forward<double>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_backward_f32(
    const float* x_pad,
    const float* w,
    const float* grad_out,
    float* grad_x_pad,
    float* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_backward<float>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_backward_f64(
    const double* x_pad,
    const double* w,
    const double* grad_out,
    double* grad_x_pad,
    double* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_backward<double>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}
