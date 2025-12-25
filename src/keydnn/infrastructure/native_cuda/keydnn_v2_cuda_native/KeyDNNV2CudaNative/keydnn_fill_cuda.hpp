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
// CUDA Fill (C ABI)
// ------------------------------------------------------------
// Fill a contiguous device buffer of length `numel` with a scalar value.
// Pointers are device pointers allocated by keydnn_cuda_malloc.
//
// Return value:
//   0  : success
//   >0 : cudaError_t code (cast to int)
//   <0 : KeyDNN argument guard / validation failure
//
// Notes
// -----
// - For zeros, prefer keydnn_cuda_memset(ptr, 0, nbytes) (already exported).
// - For ones (and arbitrary constants), use these fill kernels.
// ------------------------------------------------------------

KEYDNN_EXPORT int keydnn_cuda_fill_f32(
    float* y,
    std::int64_t numel,
    float value
);

KEYDNN_EXPORT int keydnn_cuda_fill_f64(
    double* y,
    std::int64_t numel,
    double value
);
