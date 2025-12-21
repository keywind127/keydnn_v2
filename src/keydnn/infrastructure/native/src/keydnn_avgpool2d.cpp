#include "keydnn_avgpool2d.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

static inline std::size_t idx4_nchw(
    int n, int c, int h, int w,
    int C, int H, int W
) {
    // (((n * C + c) * H + h) * W + w)
    return static_cast<std::size_t>(
        (((n * C + c) * H + h) * W + w)
    );
}

template <typename T>
static inline void avgpool2d_forward_impl(
    const T* x_pad,
    T* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point_v<T>, "avgpool2d_forward_impl requires floating point T");

    if (!x_pad || !y) return;
    if (N <= 0 || C <= 0) return;
    if (H_pad <= 0 || W_pad <= 0 || H_out <= 0 || W_out <= 0) return;
    if (k_h <= 0 || k_w <= 0 || s_h <= 0 || s_w <= 0) return;

    const T denom = static_cast<T>(k_h * k_w);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                const int h0 = i * s_h;
                for (int j = 0; j < W_out; ++j) {
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
                    y[out_off] = sum / denom;
                }
            }
        }
    }
}

template <typename T>
static inline void avgpool2d_backward_impl(
    const T* grad_out,
    T* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    static_assert(std::is_floating_point_v<T>, "avgpool2d_backward_impl requires floating point T");

    if (!grad_out || !grad_x_pad) return;
    if (N <= 0 || C <= 0) return;
    if (H_out <= 0 || W_out <= 0 || H_pad <= 0 || W_pad <= 0) return;
    if (k_h <= 0 || k_w <= 0 || s_h <= 0 || s_w <= 0) return;

    const T denom = static_cast<T>(k_h * k_w);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                const int h0 = i * s_h;
                for (int j = 0; j < W_out; ++j) {
                    const int w0 = j * s_w;

                    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
                    const T go = grad_out[out_off] / denom;

                    for (int ph = 0; ph < k_h; ++ph) {
                        const int h = h0 + ph;
                        for (int pw = 0; pw < k_w; ++pw) {
                            const int w = w0 + pw;
                            const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
                            grad_x_pad[in_off] += go;
                        }
                    }
                }
            }
        }
    }
}


void keydnn_avgpool2d_forward_f32(
    const float* x_pad,
    float* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    avgpool2d_forward_impl<float>(x_pad, y, N, C, H_pad, W_pad, H_out, W_out, k_h, k_w, s_h, s_w);
}

void keydnn_avgpool2d_forward_f64(
    const double* x_pad,
    double* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
) {
    avgpool2d_forward_impl<double>(x_pad, y, N, C, H_pad, W_pad, H_out, W_out, k_h, k_w, s_h, s_w);
}

void keydnn_avgpool2d_backward_f32(
    const float* grad_out,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    avgpool2d_backward_impl<float>(grad_out, grad_x_pad, N, C, H_out, W_out, H_pad, W_pad, k_h, k_w, s_h, s_w);
}

void keydnn_avgpool2d_backward_f64(
    const double* grad_out,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
) {
    avgpool2d_backward_impl<double>(grad_out, grad_x_pad, N, C, H_out, W_out, H_pad, W_pad, k_h, k_w, s_h, s_w);
}
