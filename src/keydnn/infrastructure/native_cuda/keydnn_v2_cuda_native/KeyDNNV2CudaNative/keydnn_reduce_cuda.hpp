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
// CUDA reductions / elementwise fills (contiguous, row-major flattening)
// -----------------------------------------------------------------------------
//
// All tensors are assumed contiguous in memory.
// We model the tensor as a flat array of length `numel` unless noted.
//
// sum_all:
//   x: (numel)
//   y: scalar (single value)
//
// mean_all:
//   x: (numel)
//   y: scalar
//
// sum_backward_fill:
//   grad_out: scalar
//   grad_x: (numel) filled with grad_out
//
// mean_backward_fill:
//   grad_out: scalar
//   grad_x: (numel) filled with grad_out / numel
//
// max_axis (2D specialization):
//   x: (rows, cols) contiguous
//   y: (rows) if axis=1 OR (cols) if axis=0
//   idx: int64 indices, same shape as y; indices are along reduced axis
//
// max_axis_backward (2D):
//   grad_out: same shape as y
//   idx: same shape as y (int64)
//   grad_x: (rows, cols) output (caller must memset to 0 before calling)
//   scatter grad_out to argmax positions
//
// All functions return 0 on success, else cudaError_t cast to int (or negative for argument validation).

// ---- sum / mean (all-elements) ----
KEYDNN_EXPORT int keydnn_cuda_sum_all_f32(const float* x, float* y, int64_t numel);
KEYDNN_EXPORT int keydnn_cuda_sum_all_f64(const double* x, double* y, int64_t numel);

KEYDNN_EXPORT int keydnn_cuda_mean_all_f32(const float* x, float* y, int64_t numel);
KEYDNN_EXPORT int keydnn_cuda_mean_all_f64(const double* x, double* y, int64_t numel);

KEYDNN_EXPORT int keydnn_cuda_sum_backward_fill_f32(const float* grad_out, float* grad_x, int64_t numel);
KEYDNN_EXPORT int keydnn_cuda_sum_backward_fill_f64(const double* grad_out, double* grad_x, int64_t numel);

KEYDNN_EXPORT int keydnn_cuda_mean_backward_fill_f32(const float* grad_out, float* grad_x, int64_t numel);
KEYDNN_EXPORT int keydnn_cuda_mean_backward_fill_f64(const double* grad_out, double* grad_x, int64_t numel);

// ---- max (2D only, axis=0 or axis=1) ----
KEYDNN_EXPORT int keydnn_cuda_max_axis2d_forward_f32(
    const float* x, float* y, int64_t* idx,
    int rows, int cols, int axis
);
KEYDNN_EXPORT int keydnn_cuda_max_axis2d_forward_f64(
    const double* x, double* y, int64_t* idx,
    int rows, int cols, int axis
);

KEYDNN_EXPORT int keydnn_cuda_max_axis2d_backward_f32(
    const float* grad_out, const int64_t* idx, float* grad_x,
    int rows, int cols, int axis
);
KEYDNN_EXPORT int keydnn_cuda_max_axis2d_backward_f64(
    const double* grad_out, const int64_t* idx, double* grad_x,
    int rows, int cols, int axis
);
