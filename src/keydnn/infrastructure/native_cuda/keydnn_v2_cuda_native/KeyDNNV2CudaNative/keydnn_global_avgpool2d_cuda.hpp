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
// CUDA GlobalAvgPool2D (NCHW) exports
// -----------------------------------------------------------------------------
//
// Forward:
//   x: (N, C, H, W) contiguous NCHW on device
//   y: (N, C, 1, 1) contiguous NCHW on device
//
// Backward:
//   grad_out: (N, C, 1, 1) contiguous NCHW on device
//   grad_x:   (N, C, H, W) contiguous NCHW on device (caller may memset to 0;
//             this kernel overwrites all elements, so zero-init is not required)
//
// All functions return 0 on success, or a CUDA error code (cudaError_t cast to int).

KEYDNN_EXPORT int keydnn_cuda_global_avgpool2d_forward_f32(
    const float* x,
    float* y,
    int N, int C,
    int H, int W
);

KEYDNN_EXPORT int keydnn_cuda_global_avgpool2d_forward_f64(
    const double* x,
    double* y,
    int N, int C,
    int H, int W
);

KEYDNN_EXPORT int keydnn_cuda_global_avgpool2d_backward_f32(
    const float* grad_out,
    float* grad_x,
    int N, int C,
    int H, int W
);

KEYDNN_EXPORT int keydnn_cuda_global_avgpool2d_backward_f64(
    const double* grad_out,
    double* grad_x,
    int N, int C,
    int H, int W
);
