#pragma once

#include <cstddef>
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

// ----------------------------
// CUDA utility API (ctypes-friendly)
// ----------------------------
//
// We expose "device pointer handles" as uintptr_t so Python can store them
// (e.g., in an int64) and pass them back later.
//
// NOTE: These functions assume a CUDA-capable build and runtime availability.
//       They return 0 on success, non-zero on failure (simple C ABI).

KEYDNN_EXPORT int keydnn_cuda_set_device(int device);

KEYDNN_EXPORT int keydnn_cuda_malloc(std::uintptr_t* out_dev_ptr, std::size_t nbytes);
KEYDNN_EXPORT int keydnn_cuda_free(std::uintptr_t dev_ptr);

KEYDNN_EXPORT int keydnn_cuda_memcpy_h2d(std::uintptr_t dst_dev_ptr, const void* src_host_ptr, std::size_t nbytes);
KEYDNN_EXPORT int keydnn_cuda_memcpy_d2h(void* dst_host_ptr, std::uintptr_t src_dev_ptr, std::size_t nbytes);

KEYDNN_EXPORT int keydnn_cuda_memset(std::uintptr_t dev_ptr, int value, std::size_t nbytes);
KEYDNN_EXPORT int keydnn_cuda_synchronize();

// Convenience helpers: allocate + copy from host (NumPy CPU buffer) into device.
// Returns device pointer handle in out_dev_ptr.
KEYDNN_EXPORT int keydnn_cuda_from_host_f32(std::uintptr_t* out_dev_ptr, const float* src_host, std::size_t count);
KEYDNN_EXPORT int keydnn_cuda_from_host_f64(std::uintptr_t* out_dev_ptr, const double* src_host, std::size_t count);
