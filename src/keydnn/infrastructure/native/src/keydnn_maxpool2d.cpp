#include "keydnn_maxpool2d.hpp"

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
static inline void maxpool2d_forward_impl(
    const T* x_pad,
    T* y,
    std::int64_t* argmax_idx,
    int N,
    int C,
    int H_pad,
    int W_pad,
    int H_out,
    int W_out,
    int k_h,
    int k_w,
    int s_h,
    int s_w
) {
    static_assert(std::is_floating_point_v<T>, "maxpool2d_forward_impl requires floating point T");

    // Basic argument sanity (cheap guards; feel free to remove in release builds)
    if (!x_pad || !y || !argmax_idx) return;
    if (N <= 0 || C <= 0 || H_pad <= 0 || W_pad <= 0) return;
    if (H_out <= 0 || W_out <= 0 || k_h <= 0 || k_w <= 0) return;
    if (s_h <= 0 || s_w <= 0) return;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                const int h0 = i * s_h;
                for (int j = 0; j < W_out; ++j) {
                    const int w0 = j * s_w;

                    // Scan window (k_h x k_w) within padded plane
                    T best = -std::numeric_limits<T>::infinity();
                    int best_flat = 0;

                    for (int ph = 0; ph < k_h; ++ph) {
                        const int h = h0 + ph;
                        for (int pw = 0; pw < k_w; ++pw) {
                            const int w = w0 + pw;

                            const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
                            const T v = x_pad[in_off];

                            const int flat = ph * k_w + pw;
                            // Tie-breaking matches NumPy argmax (first occurrence in row-major order)
                            if (v > best) {
                                best = v;
                                best_flat = flat;
                            }
                        }
                    }

                    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
                    y[out_off] = best;

                    const int best_ph = best_flat / k_w;
                    const int best_pw = best_flat % k_w;
                    const int best_h = h0 + best_ph;
                    const int best_w = w0 + best_pw;

                    // flattened spatial index into padded HxW plane
                    argmax_idx[out_off] = static_cast<std::int64_t>(best_h) * W_pad + best_w;
                }
            }
        }
    }
}

template <typename T>
static inline void maxpool2d_backward_impl(
    const T* grad_out,
    const std::int64_t* argmax_idx,
    T* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    static_assert(std::is_floating_point_v<T>, "maxpool2d_backward_impl requires floating point T");

    if (!grad_out || !argmax_idx || !grad_x_pad) return;
    if (N <= 0 || C <= 0) return;
    if (H_out <= 0 || W_out <= 0 || H_pad <= 0 || W_pad <= 0) return;

    // Accumulate grad_out into grad_x_pad at argmax locations
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    const std::size_t out_off = idx4_nchw(n, c, i, j, C, H_out, W_out);
                    const std::int64_t idx = argmax_idx[out_off];

                    // idx = h * W_pad + w
                    const int h = static_cast<int>(idx / W_pad);
                    const int w = static_cast<int>(idx % W_pad);

                    if (h < 0 || h >= H_pad || w < 0 || w >= W_pad) {
                        continue; // guard against corrupted indices
                    }

                    const std::size_t in_off = idx4_nchw(n, c, h, w, C, H_pad, W_pad);
                    grad_x_pad[in_off] += grad_out[out_off];
                }
            }
        }
    }
}

// -------------------- Exports --------------------

void keydnn_maxpool2d_backward_f32(
    const float* grad_out,
    const std::int64_t* argmax_idx,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    maxpool2d_backward_impl<float>(grad_out, argmax_idx, grad_x_pad, N, C, H_out, W_out, H_pad, W_pad);
}

void keydnn_maxpool2d_backward_f64(
    const double* grad_out,
    const std::int64_t* argmax_idx,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
) {
    maxpool2d_backward_impl<double>(grad_out, argmax_idx, grad_x_pad, N, C, H_out, W_out, H_pad, W_pad);
}

void keydnn_maxpool2d_forward_f32(
    const float* x_pad,
    float* y,
    std::int64_t* argmax_idx,
    int N,
    int C,
    int H_pad,
    int W_pad,
    int H_out,
    int W_out,
    int k_h,
    int k_w,
    int s_h,
    int s_w
) {
    maxpool2d_forward_impl<float>(
        x_pad, y, argmax_idx,
        N, C, H_pad, W_pad, H_out, W_out,
        k_h, k_w, s_h, s_w
    );
}

void keydnn_maxpool2d_forward_f64(
    const double* x_pad,
    double* y,
    std::int64_t* argmax_idx,
    int N,
    int C,
    int H_pad,
    int W_pad,
    int H_out,
    int W_out,
    int k_h,
    int k_w,
    int s_h,
    int s_w
) {
    maxpool2d_forward_impl<double>(
        x_pad, y, argmax_idx,
        N, C, H_pad, W_pad, H_out, W_out,
        k_h, k_w, s_h, s_w
    );
}
