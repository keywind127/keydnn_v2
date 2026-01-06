// keydnn_conv2d_transpose.cpp
#include "keydnn_conv2d_transpose.hpp"

#include <cstddef>
#include <type_traits>

#if defined(_OPENMP)
  #include <omp.h>
#endif

static inline std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return static_cast<std::size_t>(
        (((n * C + c) * H + h) * W + w)
    );
}

static inline std::size_t idx4_iohw(
    int ci, int co, int kh, int kw,
    int C_out, int K_h, int K_w
) {
    // (((ci * C_out + co) * K_h + kh) * K_w + kw)
    return static_cast<std::size_t>(
        (((ci * C_out + co) * K_h + kh) * K_w + kw)
    );
}

template <typename T>
static inline void conv2d_transpose_forward_impl(
    const T* x,
    const T* w,
    const T* b, // may be null
    T* y,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    static_assert(std::is_floating_point_v<T>, "conv2d_transpose_forward_impl requires floating point T");

    if (!x || !w || !y) return;
    if (N <= 0 || C_in <= 0 || H_in <= 0 || W_in <= 0) return;
    if (C_out <= 0 || H_out <= 0 || W_out <= 0) return;
    if (K_h <= 0 || K_w <= 0) return;
    if (s_h <= 0 || s_w <= 0) return;

    // Parallelize over batch only: each thread owns y[n, ...] => no write races.
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; ++n) {
        // Scatter contributions from each input element to output grid.
        for (int ci = 0; ci < C_in; ++ci) {
            for (int hi = 0; hi < H_in; ++hi) {
                for (int wi = 0; wi < W_in; ++wi) {
                    const std::size_t x_off = idx4_nchw(n, ci, hi, wi, C_in, H_in, W_in);
                    const T xv = x[x_off];

                    const int base_oh = hi * s_h - pad_h;
                    const int base_ow = wi * s_w - pad_w;

                    for (int co = 0; co < C_out; ++co) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            const int oh = base_oh + kh;
                            if (oh < 0 || oh >= H_out) continue;

                            for (int kw = 0; kw < K_w; ++kw) {
                                const int ow = base_ow + kw;
                                if (ow < 0 || ow >= W_out) continue;

                                const std::size_t w_off = idx4_iohw(ci, co, kh, kw, C_out, K_h, K_w);
                                const std::size_t y_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
                                y[y_off] += xv * w[w_off];
                            }
                        }
                    }
                }
            }
        }

        // Bias add (safe; still only touches y[n,...] owned by this thread).
        if (b) {
            for (int co = 0; co < C_out; ++co) {
                const T bias = b[co];
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        const std::size_t y_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
                        y[y_off] += bias;
                    }
                }
            }
        }
    }
}

template <typename T>
static inline void conv2d_transpose_backward_impl(
    const T* x,
    const T* w,
    const T* grad_out,
    T* grad_x,
    T* grad_w,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    static_assert(std::is_floating_point_v<T>, "conv2d_transpose_backward_impl requires floating point T");

    if (!x || !w || !grad_out || !grad_x || !grad_w) return;
    if (N <= 0 || C_in <= 0 || H_in <= 0 || W_in <= 0) return;
    if (C_out <= 0 || H_out <= 0 || W_out <= 0) return;
    if (K_h <= 0 || K_w <= 0) return;
    if (s_h <= 0 || s_w <= 0) return;

    // Parallelize over batch: each thread owns grad_x[n,...] => no races for grad_x.
    // grad_w is shared across threads => atomic adds.
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; ++n) {
        for (int ci = 0; ci < C_in; ++ci) {
            for (int hi = 0; hi < H_in; ++hi) {
                for (int wi = 0; wi < W_in; ++wi) {
                    const std::size_t x_off = idx4_nchw(n, ci, hi, wi, C_in, H_in, W_in);
                    const T xv = x[x_off];

                    const int base_oh = hi * s_h - pad_h;
                    const int base_ow = wi * s_w - pad_w;

                    T gx_acc = static_cast<T>(0);

                    for (int co = 0; co < C_out; ++co) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            const int oh = base_oh + kh;
                            if (oh < 0 || oh >= H_out) continue;

                            for (int kw = 0; kw < K_w; ++kw) {
                                const int ow = base_ow + kw;
                                if (ow < 0 || ow >= W_out) continue;

                                const std::size_t go_off = idx4_nchw(n, co, oh, ow, C_out, H_out, W_out);
                                const T go = grad_out[go_off];

                                const std::size_t w_off = idx4_iohw(ci, co, kh, kw, C_out, K_h, K_w);

                                // grad_x accumulation for this (n,ci,hi,wi)
                                gx_acc += go * w[w_off];

                                // grad_w shared across threads => atomic add
                                #if defined(_OPENMP)
                                #pragma omp atomic
                                #endif
                                grad_w[w_off] += xv * go;
                            }
                        }
                    }

                    // write back grad_x (no races due to batch-parallelization)
                    grad_x[x_off] += gx_acc;
                }
            }
        }
    }
}

// -------------------- Exports --------------------

void keydnn_conv2d_transpose_forward_f32(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    conv2d_transpose_forward_impl<float>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w,
        pad_h, pad_w
    );
}

void keydnn_conv2d_transpose_forward_f64(
    const double* x,
    const double* w,
    const double* b,
    double* y,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    conv2d_transpose_forward_impl<double>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w,
        pad_h, pad_w
    );
}

void keydnn_conv2d_transpose_backward_f32(
    const float* x,
    const float* w,
    const float* grad_out,
    float* grad_x,
    float* grad_w,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    conv2d_transpose_backward_impl<float>(
        x, w, grad_out,
        grad_x, grad_w,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w,
        pad_h, pad_w
    );
}

void keydnn_conv2d_transpose_backward_f64(
    const double* x,
    const double* w,
    const double* grad_out,
    double* grad_x,
    double* grad_w,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w,
    int pad_h,
    int pad_w
) {
    conv2d_transpose_backward_impl<double>(
        x, w, grad_out,
        grad_x, grad_w,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w,
        pad_h, pad_w
    );
}
