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

// ------------------------------------------------------------
// CUDA ConvTranspose2D (NCHW) - C ABI
// ------------------------------------------------------------
// All pointers are *device pointers* allocated by keydnn_cuda_malloc.
// Return value:
//   0  : success
//   >0 : cudaError_t code (cast to int)
//   <0 : KeyDNN argument guard
//
// Forward:
//   x: float32/float64, (N, C_in, H_in, W_in), contiguous NCHW
//   w: float32/float64, (C_in, C_out, K_h, K_w), contiguous IOHW
//   b: float32/float64, (C_out,), contiguous (may be null)
//   y: float32/float64, (N, C_out, H_out, W_out), contiguous NCHW
//      (written, not accumulated; i.e., kernel computes full output)
//
// Backward:
//   x:        (N, C_in, H_in, W_in)
//   w:        (C_in, C_out, K_h, K_w)
//   grad_out: (N, C_out, H_out, W_out)
//   grad_x:   (N, C_in, H_in, W_in)  (written, not accumulated)
//   grad_w:   (C_in, C_out, K_h, K_w) (written, not accumulated)
//
// Notes:
// - This CUDA implementation uses gather-style kernels to avoid atomicAdd.
// - Does not compute grad_b; Python side can do grad_b = sum(grad_out).

KEYDNN_EXPORT int keydnn_cuda_conv2d_transpose_forward_f32(
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

KEYDNN_EXPORT int keydnn_cuda_conv2d_transpose_forward_f64(
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

KEYDNN_EXPORT int keydnn_cuda_conv2d_transpose_backward_f32(
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

KEYDNN_EXPORT int keydnn_cuda_conv2d_transpose_backward_f64(
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
