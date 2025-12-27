// keydnn_tensor_arithmetic.hpp
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define KEYDNN_CUDA_API __declspec(dllexport)
#else
#define KEYDNN_CUDA_API __attribute__((visibility("default")))
#endif

extern "C" {


	// ----------------------------
	// Unary: neg
	// ----------------------------
	KEYDNN_CUDA_API int keydnn_cuda_neg_f32(const float* x, float* y, int64_t n);
	KEYDNN_CUDA_API int keydnn_cuda_neg_f64(const double* x, double* y, int64_t n);

	// ----------------------------
	// Binary: add/sub/mul/div (elementwise, same shape)
	// ----------------------------
	KEYDNN_CUDA_API int keydnn_cuda_add_f32(const float* a, const float* b, float* y, int64_t n);
	KEYDNN_CUDA_API int keydnn_cuda_add_f64(const double* a, const double* b, double* y, int64_t n);

	KEYDNN_CUDA_API int keydnn_cuda_sub_f32(const float* a, const float* b, float* y, int64_t n);
	KEYDNN_CUDA_API int keydnn_cuda_sub_f64(const double* a, const double* b, double* y, int64_t n);

	KEYDNN_CUDA_API int keydnn_cuda_div_f32(const float* a, const float* b, float* y, int64_t n);
	KEYDNN_CUDA_API int keydnn_cuda_div_f64(const double* a, const double* b, double* y, int64_t n);

	// ----------------------------
	// Compare: gt (outputs 1/0)
	// Notes:
	// - Output is float32 to match your current CPU semantics (astype(np.float32)).
	// - This is intentionally "no-grad" in Python.
	// ----------------------------
	KEYDNN_CUDA_API int keydnn_cuda_gt_f32(const float* a, const float* b, float* y, int64_t n);
	KEYDNN_CUDA_API int keydnn_cuda_gt_f64(const double* a, const double* b, float* y, int64_t n);

} // extern "C"
