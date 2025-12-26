// keydnn_cuda_matmul.cu
#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

static int _ensure_cublas_handle() {
    if (g_handle) return 0;
    cublasStatus_t st = cublasCreate(&g_handle);
    if (st != CUBLAS_STATUS_SUCCESS) return 1;
    // Optional: use default stream (0). If you later use streams, set it here.
    // cublasSetStream(g_handle, 0);
    return 0;
}

static int _cublas_to_int(cublasStatus_t st) {
    // Keep it simple: non-zero error code for your C ABI.
    // You can map more precisely if you want.
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : 2;
}

// Row-major GEMM using cuBLAS (column-major):
// If C = A(MxK) @ B(KxN) in row-major,
// then C^T(NxM) = B^T(NxK) @ A^T(KxM) in column-major.
// So we compute: C_col(NxM) = B_col(NxK) * A_col(KxM)
// which corresponds to: cublasGemm(handle, N, M, K, B, A, C) with opN/opN.
static int gemm_rowmajor_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M < 0 || N < 0 || K < 0) return (int)cudaErrorInvalidValue;
    if (M == 0 || N == 0) return 0;
    if (!A || !B || !C) return (int)cudaErrorInvalidValue;

    // Clear sticky CUDA error from any previous op
    (void)cudaGetLastError();

    if (_ensure_cublas_handle() != 0) return 10;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // We want column-major GEMM:
    // C_col is NxM, B_col is NxK, A_col is KxM
    // lda = leading dim of B_col = N
    // ldb = leading dim of A_col = K
    // ldc = leading dim of C_col = N
    cublasStatus_t st = cublasSgemm(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        // Also clear any CUDA error state to avoid poisoning next kernels
        (void)cudaGetLastError();
        return _cublas_to_int(st);
    }

    // Check if the GEMM launch left a CUDA error
    cudaError_t cu = cudaPeekAtLastError();
    if (cu != cudaSuccess) {
        (void)cudaGetLastError(); // clear
        return (int)cu;
    }

    return 0;
}

static int gemm_rowmajor_f64(const double* A, const double* B, double* C, int M, int N, int K) {
    if (M < 0 || N < 0 || K < 0) return (int)cudaErrorInvalidValue;
    if (M == 0 || N == 0) return 0;
    if (!A || !B || !C) return (int)cudaErrorInvalidValue;

    (void)cudaGetLastError();

    if (_ensure_cublas_handle() != 0) return 10;

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t st = cublasDgemm(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        (void)cudaGetLastError();
        return _cublas_to_int(st);
    }

    cudaError_t cu = cudaPeekAtLastError();
    if (cu != cudaSuccess) {
        (void)cudaGetLastError();
        return (int)cu;
    }

    return 0;
}

extern "C" int keydnn_cuda_matmul_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    return gemm_rowmajor_f32(A, B, C, M, N, K);
}

extern "C" int keydnn_cuda_matmul_f64(const double* A, const double* B, double* C, int M, int N, int K) {
    return gemm_rowmajor_f64(A, B, C, M, N, K);
}
