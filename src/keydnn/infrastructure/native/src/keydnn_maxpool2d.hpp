#pragma once

#include <cstdint>

#if defined(_WIN32)
  #define KEYDNN_EXPORT extern "C" __declspec(dllexport)
#else
  #define KEYDNN_EXPORT extern "C"
#endif

// x_pad: float32, shape (N, C, H_pad, W_pad), contiguous NCHW
// y:     float32, shape (N, C, H_out, W_out), contiguous NCHW
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
