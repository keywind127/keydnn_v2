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

// CUDA MaxPool2D forward/backward.
// All pointers are **device pointers**.
//
// x_pad:      float32/float64 device ptr, shape (N, C, H_pad, W_pad), contiguous NCHW
// y:          float32/float64 device ptr, shape (N, C, H_out, W_out), contiguous NCHW
// argmax_idx: int64 device ptr,        shape (N, C, H_out, W_out), contiguous NCHW
//
// argmax_idx stores flattened index into padded spatial plane: h * W_pad + w

KEYDNN_EXPORT int keydnn_cuda_maxpool2d_forward_f32(
    const float* x_pad,
    float* y,
    std::int64_t* argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
);

KEYDNN_EXPORT int keydnn_cuda_maxpool2d_forward_f64(
    const double* x_pad,
    double* y,
    std::int64_t* argmax_idx,
    int N, int C,
    int H_pad, int W_pad,
    int H_out, int W_out,
    int k_h, int k_w,
    int s_h, int s_w
);

// grad_out:   float32/float64 device ptr, shape (N, C, H_out, W_out), contiguous NCHW
// argmax_idx: int64 device ptr,        shape (N, C, H_out, W_out), flattened index into padded plane
// grad_x_pad: float32/float64 device ptr, shape (N, C, H_pad, W_pad), contiguous NCHW
//
// grad_x_pad must be zero-initialized by caller (or use keydnn_cuda_memset).

KEYDNN_EXPORT int keydnn_cuda_maxpool2d_backward_f32(
    const float* grad_out,
    const std::int64_t* argmax_idx,
    float* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
);

KEYDNN_EXPORT int keydnn_cuda_maxpool2d_backward_f64(
    const double* grad_out,
    const std::int64_t* argmax_idx,
    double* grad_x_pad,
    int N, int C,
    int H_out, int W_out,
    int H_pad, int W_pad
);
