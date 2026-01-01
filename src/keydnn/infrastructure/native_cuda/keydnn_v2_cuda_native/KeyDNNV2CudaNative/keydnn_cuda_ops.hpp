#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KEYDNN_CUDA_API __declspec(dllexport)
#else
#define KEYDNN_CUDA_API
#endif

extern "C" {

	// Return 0 on success, non-zero on failure (match your fill style).

	KEYDNN_CUDA_API int keydnn_cuda_memcpy_d2d(void* dst, const void* src, std::size_t nbytes);

	// 2D transpose: out[c, r] = in[r, c]
	// x: (rows, cols) contiguous row-major
	// y: (cols, rows) contiguous row-major
	KEYDNN_CUDA_API int keydnn_cuda_transpose2d_f32(const float* x, float* y, int rows, int cols);
	KEYDNN_CUDA_API int keydnn_cuda_transpose2d_f64(const double* x, double* y, int rows, int cols);

	// Matmul: C = A @ B
	// A: (M,K), B: (K,N), C: (M,N), row-major
	KEYDNN_CUDA_API int keydnn_cuda_matmul_f32(const float* A, const float* B, float* C, int M, int N, int K);
	KEYDNN_CUDA_API int keydnn_cuda_matmul_f64(const double* A, const double* B, double* C, int M, int N, int K);

	// Unary exp: y = exp(x), length numel
	KEYDNN_CUDA_API int keydnn_cuda_exp_f32(const float* x, float* y, int numel);
	KEYDNN_CUDA_API int keydnn_cuda_exp_f64(const double* x, double* y, int numel);

	// Elementwise mul: y = a * b, length numel
	KEYDNN_CUDA_API int keydnn_cuda_mul_f32(const float* a, const float* b, float* y, int numel);
	KEYDNN_CUDA_API int keydnn_cuda_mul_f64(const double* a, const double* b, double* y, int numel);

	// Elementwise mul scalar: y = a * alpha, length numel
	KEYDNN_CUDA_API int keydnn_cuda_mul_scalar_f32(const float* a, float alpha, float* y, int numel);
	KEYDNN_CUDA_API int keydnn_cuda_mul_scalar_f64(const double* a, double alpha, double* y, int numel);

	// Elementwise mul in-place: a *= b, length numel
	KEYDNN_CUDA_API int keydnn_cuda_mul_inplace_f32(float* a, const float* b, int numel);
	KEYDNN_CUDA_API int keydnn_cuda_mul_inplace_f64(double* a, const double* b, int numel);

	// Elementwise mul scalar in-place: a *= alpha, length numel
	KEYDNN_CUDA_API int keydnn_cuda_mul_scalar_inplace_f32(float* a, float alpha, int numel);
	KEYDNN_CUDA_API int keydnn_cuda_mul_scalar_inplace_f64(double* a, double alpha, int numel);



} // extern "C"
