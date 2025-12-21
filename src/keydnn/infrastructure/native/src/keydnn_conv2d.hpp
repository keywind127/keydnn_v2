// keydnn_conv2d.hpp
#pragma once

#include <cstdint>

#if defined(_WIN32)
  #define KEYDNN_DLL_EXPORT __declspec(dllexport)
#else
  #define KEYDNN_DLL_EXPORT
#endif

#ifdef __cplusplus
  #define KEYDNN_EXTERN_C extern "C"
#else
  #define KEYDNN_EXTERN_C
#endif

#define KEYDNN_EXPORT KEYDNN_EXTERN_C KEYDNN_DLL_EXPORT

// x_pad: float32/float64, shape (N, C_in, H_pad, W_pad), contiguous NCHW
// w:     float32/float64, shape (C_out, C_in, K_h, K_w), contiguous OIHW
// b:     float32/float64, shape (C_out,), contiguous (may be null)
// y:     float32/float64, shape (N, C_out, H_out, W_out), contiguous NCHW
//
// Computes:
//   y[n, co, i, j] = sum_{ci,kh,kw} x_pad[n,ci,i*s_h+kh,j*s_w+kw] * w[co,ci,kh,kw]
// then if b != null:
//   y[n, co, :, :] += b[co]
//
// Notes:
// - Assumes x_pad is already padded (caller handles padding).
// - This kernel is correctness-first (naive loops).

KEYDNN_EXPORT void keydnn_conv2d_forward_f32(
    const float* x_pad,
    const float* w,
    const float* b,  // may be nullptr
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
);

KEYDNN_EXPORT void keydnn_conv2d_forward_f64(
    const double* x_pad,
    const double* w,
    const double* b,  // may be nullptr
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
);

// x_pad:      float32/float64, shape (N, C_in, H_pad, W_pad), contiguous NCHW
// w:          float32/float64, shape (C_out, C_in, K_h, K_w), contiguous OIHW
// grad_out:   float32/float64, shape (N, C_out, H_out, W_out), contiguous NCHW
// grad_x_pad: float32/float64, shape (N, C_in, H_pad, W_pad), contiguous NCHW (written in-place, accumulated)
// grad_w:     float32/float64, shape (C_out, C_in, K_h, K_w), contiguous OIHW (written in-place, accumulated)
//
// Computes gradients mirroring the Python reference loop:
//
// for n,co,i,j:
//   go = grad_out[n,co,i,j]
//   grad_w[co] += go * x_pad[n,:,h0:h0+K_h,w0:w0+K_w]
//   grad_x_pad[n,:,h0:h0+K_h,w0:w0+K_w] += go * w[co]
//
// Notes:
// - grad_x_pad and grad_w are assumed zero-initialized by caller (like NumPy zeros_like).
// - This function does not compute grad_b; Python side keeps grad_b = sum(grad_out).

KEYDNN_EXPORT void keydnn_conv2d_backward_f32(
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
);

KEYDNN_EXPORT void keydnn_conv2d_backward_f64(
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
);
