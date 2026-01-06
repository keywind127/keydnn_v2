// keydnn_conv2d_transpose.hpp
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

// -------------------- ConvTranspose2d (Transposed Conv2d) --------------------
//
// x:     float32/float64, shape (N, C_in,  H_in,  W_in),  contiguous NCHW
// w:     float32/float64, shape (C_in, C_out, K_h, K_w),  contiguous IOHW
// b:     float32/float64, shape (C_out,), contiguous (may be null)
// y:     float32/float64, shape (N, C_out, H_out, W_out), contiguous NCHW
//
// Computes (scatter-style):
//   out_h = h_in * s_h + kh - pad_h
//   out_w = w_in * s_w + kw - pad_w
//   y[n, co, out_h, out_w] += x[n, ci, h_in, w_in] * w[ci, co, kh, kw]
// then if b != null:
//   y[n, co, :, :] += b[co]
//
// Notes:
// - Assumes `y` is zero-initialized by the caller if you want pure accumulation semantics.
//   (This mirrors common implementations where the kernel "writes into" y by adding contributions.)
// - `pad_h/pad_w` correspond to the transpose-conv padding (i.e., output cropping offset).
// - Output_padding is handled by the caller via H_out/W_out choice; this kernel just bounds-checks.
//
// Weight layout for transpose conv is typically (C_in, C_out, K_h, K_w), i.e. IOHW.

KEYDNN_EXPORT void keydnn_conv2d_transpose_forward_f32(
    const float* x,
    const float* w,
    const float* b,  // may be nullptr
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
);

KEYDNN_EXPORT void keydnn_conv2d_transpose_forward_f64(
    const double* x,
    const double* w,
    const double* b,  // may be nullptr
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
);

// x:        float32/float64, shape (N, C_in,  H_in,  W_in),  contiguous NCHW
// w:        float32/float64, shape (C_in, C_out, K_h, K_w),  contiguous IOHW
// grad_out: float32/float64, shape (N, C_out, H_out, W_out), contiguous NCHW
// grad_x:   float32/float64, shape (N, C_in,  H_in,  W_in),  contiguous NCHW (written in-place, accumulated)
// grad_w:   float32/float64, shape (C_in, C_out, K_h, K_w),  contiguous IOHW (written in-place, accumulated)
//
// Backprop rules (scatter-style):
//   out_h = h_in * s_h + kh - pad_h
//   out_w = w_in * s_w + kw - pad_w
//   if in bounds:
//     grad_x[n,ci,h_in,w_in] += grad_out[n,co,out_h,out_w] * w[ci,co,kh,kw]
//     grad_w[ci,co,kh,kw]    += x[n,ci,h_in,w_in]        * grad_out[n,co,out_h,out_w]
//
// Notes:
// - grad_x and grad_w are assumed zero-initialized by caller (like NumPy zeros_like).
// - This function does not compute grad_b; Python side typically does grad_b = sum(grad_out).

KEYDNN_EXPORT void keydnn_conv2d_transpose_backward_f32(
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
);

KEYDNN_EXPORT void keydnn_conv2d_transpose_backward_f64(
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
);
