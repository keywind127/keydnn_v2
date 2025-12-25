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

// -----------------------------------------------------------------------------
// CUDA Pad/Crop utilities for NCHW tensors.
//
// These utilities exist to support pool2d CUDA ops without host round-trips:
// - pad2d:  create padded tensor on device using a scalar pad_value
// - crop2d: extract the unpadded region from a padded tensor on device
//
// All tensors are contiguous NCHW.
//
// pad2d:
//   x:     (N, C, H, W)
//   y_pad: (N, C, H+2p_h, W+2p_w)
//
// crop2d:
//   x_pad: (N, C, H_pad, W_pad)
//   y:     (N, C, H, W)
// -----------------------------------------------------------------------------

// ----------------------------
// Pad2D forward
// ----------------------------
KEYDNN_EXPORT int keydnn_cuda_pad2d_f32(
    const float* x,
    float* y_pad,
    int N, int C,
    int H, int W,
    int p_h, int p_w,
    float pad_value
);

KEYDNN_EXPORT int keydnn_cuda_pad2d_f64(
    const double* x,
    double* y_pad,
    int N, int C,
    int H, int W,
    int p_h, int p_w,
    double pad_value
);

// ----------------------------
// Crop2D forward (extract unpadded region)
// ----------------------------
KEYDNN_EXPORT int keydnn_cuda_crop2d_f32(
    const float* x_pad,
    float* y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
);

KEYDNN_EXPORT int keydnn_cuda_crop2d_f64(
    const double* x_pad,
    double* y,
    int N, int C,
    int H_pad, int W_pad,
    int p_h, int p_w,
    int H, int W
);
