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
// CUDA AvgPool2D (NCHW) - C ABI
// ------------------------------------------------------------
// All pointers are device pointers allocated by keydnn_cuda_malloc.
// Return value:
//   0  : success
//   >0 : cudaError_t code (cast to int)
//   <0 : KeyDNN argument guard / unsupported arch
//
// Forward:
//   x_pad: float32/float64, (N, C, H_pad, W_pad), contiguous NCHW (zero padded)
//   y:     float32/float64, (N, C, H_out, W_out), contiguous NCHW
//
// Backward:
//   grad_out:   float32/float64, (N, C, H_out, W_out), contiguous NCHW
//   grad_x_pad: float32/float64, (N, C, H_pad, W_pad), contiguous NCHW
//               (must be zero-initialized by caller)
// ------------------------------------------------------------

KEYDNN_EXPORT int keydnn_cuda_avgpool2d_forward_f32(
    const float* x_pad,
    float* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
);

KEYDNN_EXPORT int keydnn_cuda_avgpool2d_forward_f64(
    const double* x_pad,
    double* y,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
);

KEYDNN_EXPORT int keydnn_cuda_avgpool2d_backward_f32(
    const float* grad_out,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
);

KEYDNN_EXPORT int keydnn_cuda_avgpool2d_backward_f64(
    const double* grad_out,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad,
    int k_h, int k_w,
    int s_h, int s_w
);
