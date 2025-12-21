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

// x_pad: float32/float64, shape (N, C, H_pad, W_pad), contiguous NCHW
// y:     float32/float64, shape (N, C, H_out, W_out), contiguous NCHW
// argmax_idx: int64, shape (N, C, H_out, W_out), contiguous NCHW
//
// argmax_idx stores flattened index into padded spatial plane: h * W_pad + w

KEYDNN_EXPORT void keydnn_maxpool2d_forward_f32(
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
);

KEYDNN_EXPORT void keydnn_maxpool2d_forward_f64(
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
);
