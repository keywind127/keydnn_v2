// keydnn_cuda_stack.hpp
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
// CUDA Stack (C ABI) ¡X u64 pointer-array ABI
// ------------------------------------------------------------
// We reduce axis insertion to (pre, post):
//   pre  = prod(in_shape[:axis])
//   post = prod(in_shape[axis:])
//
// Inputs treated as contiguous [pre, post].
//
// Output shape: [pre, K, post] contiguous.
//
// Pointer arrays:
// - xs_u64_dev  : device pointer to uint64[K], each entry is a device pointer to T buffer
// - dxs_u64_dev : device pointer to uint64[K], each entry is a device pointer to T buffer
//
// Return value:
//   0  : success
//   >0 : cudaError_t code (cast to int)
//   <0 : KeyDNN argument guard failure
// ------------------------------------------------------------

// Upload K uint64 values from host -> device.
// dst_dev_u64 must be device memory of size K*sizeof(uint64_t).
KEYDNN_EXPORT int keydnn_cuda_upload_u64_array(
    std::uint64_t* dst_dev_u64,
    const std::uint64_t* src_host_u64,
    std::int64_t K
);

// Forward: y[pre*K*post] = stack(xs[0..K-1], axis)
KEYDNN_EXPORT int keydnn_cuda_stack_fwd_u64_f32(
    const std::uint64_t* xs_u64_dev,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    float* y
);

KEYDNN_EXPORT int keydnn_cuda_stack_fwd_u64_f64(
    const std::uint64_t* xs_u64_dev,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    double* y
);

// Backward: dxs[k][pre*post] = unstack(dy[pre*K*post], axis)
KEYDNN_EXPORT int keydnn_cuda_stack_bwd_u64_f32(
    const float* dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* dxs_u64_dev
);

KEYDNN_EXPORT int keydnn_cuda_stack_bwd_u64_f64(
    const double* dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* dxs_u64_dev
);


// Add near the bottom of keydnn_cuda_stack.hpp (or a common header if you prefer)

KEYDNN_EXPORT void keydnn_cuda_debug_set_enabled(int enabled);

// Copy the last debug message into `buf` (null-terminated).
// Returns number of chars copied (excluding null).
KEYDNN_EXPORT int keydnn_cuda_debug_get_last(char* buf, int buf_bytes);
