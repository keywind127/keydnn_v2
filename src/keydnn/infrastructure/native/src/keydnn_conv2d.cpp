// keydnn_conv2d.cpp
#include "keydnn_conv2d.hpp"

#include <cstddef>
#include <type_traits>

static inline std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    return static_cast<std::size_t>(
        (((n * C + c) * H + h) * W + w)
    );
}

static inline std::size_t idx4_oihw(
    int co, int ci, int kh, int kw,
    int C_in, int K_h, int K_w
) {
    // (((co * C_in + ci) * K_h + kh) * K_w + kw)
    return static_cast<std::size_t>(
        (((co * C_in + ci) * K_h + kh) * K_w + kw)
    );
}

template <typename T>
static inline void conv2d_forward_impl(
    const T* x_pad,
    const T* w,
    const T* b, // may be null
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
    static_assert(std::is_floating_point_v<T>, "conv2d_forward_impl requires floating point T");

    // Basic argument sanity (cheap guards; feel free to remove in release builds)
    if (!x_pad || !w || !y) return;
    if (N <= 0 || C_in <= 0 || H_pad <= 0 || W_pad <= 0) return;
    if (C_out <= 0 || H_out <= 0 || W_out <= 0) return;
    if (K_h <= 0 || K_w <= 0) return;
    if (s_h <= 0 || s_w <= 0) return;

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int i = 0; i < H_out; ++i) {
                const int h0 = i * s_h;
                for (int j = 0; j < W_out; ++j) {
                    const int w0 = j * s_w;

                    T acc = static_cast<T>(0);

                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            const int h = h0 + kh;
                            for (int kw = 0; kw < K_w; ++kw) {
                                const int ww = w0 + kw;

                                const std::size_t in_off = idx4_nchw(n, ci, h, ww, C_in, H_pad, W_pad);
                                const std::size_t w_off  = idx4_oihw(co, ci, kh, kw, C_in, K_h, K_w);

                                acc += x_pad[in_off] * w[w_off];
                            }
                        }
                    }

                    const std::size_t out_off = idx4_nchw(n, co, i, j, C_out, H_out, W_out);
                    y[out_off] = acc;
                }
            }

            // bias add matches Python version (after spatial loop for this (n,co))
            if (b) {
                const T bias = b[co];
                for (int i = 0; i < H_out; ++i) {
                    for (int j = 0; j < W_out; ++j) {
                        const std::size_t out_off = idx4_nchw(n, co, i, j, C_out, H_out, W_out);
                        y[out_off] += bias;
                    }
                }
            }
        }
    }
}

template <typename T>
static inline void conv2d_backward_impl(
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
    static_assert(std::is_floating_point_v<T>, "conv2d_backward_impl requires floating point T");

    // Basic argument sanity (cheap guards; feel free to remove in release builds)
    if (!x_pad || !w || !grad_out || !grad_x_pad || !grad_w) return;
    if (N <= 0 || C_in <= 0 || H_pad <= 0 || W_pad <= 0) return;
    if (C_out <= 0 || H_out <= 0 || W_out <= 0) return;
    if (K_h <= 0 || K_w <= 0) return;
    if (s_h <= 0 || s_w <= 0) return;

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int i = 0; i < H_out; ++i) {
                const int h0 = i * s_h;
                for (int j = 0; j < W_out; ++j) {
                    const int w0 = j * s_w;

                    const std::size_t go_off = idx4_nchw(n, co, i, j, C_out, H_out, W_out);
                    const T go = grad_out[go_off];

                    // grad_w[co] += go * x_pad[n, :, h0:h0+K_h, w0:w0+K_w]
                    // grad_x_pad[n, :, h0:h0+K_h, w0:w0+K_w] += go * w[co]
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < K_h; ++kh) {
                            const int h = h0 + kh;
                            for (int kw = 0; kw < K_w; ++kw) {
                                const int ww = w0 + kw;

                                const std::size_t in_off = idx4_nchw(n, ci, h, ww, C_in, H_pad, W_pad);
                                const std::size_t w_off  = idx4_oihw(co, ci, kh, kw, C_in, K_h, K_w);

                                grad_w[w_off] += go * x_pad[in_off];
                                grad_x_pad[in_off] += go * w[w_off];
                            }
                        }
                    }
                }
            }
        }
    }
}

// -------------------- Exports --------------------

void keydnn_conv2d_backward_f32(
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
    conv2d_backward_impl<float>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

void keydnn_conv2d_backward_f64(
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
    conv2d_backward_impl<double>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

// -------------------- Exports --------------------

void keydnn_conv2d_forward_f32(
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
    conv2d_forward_impl<float>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

void keydnn_conv2d_forward_f64(
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
    conv2d_forward_impl<double>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}
