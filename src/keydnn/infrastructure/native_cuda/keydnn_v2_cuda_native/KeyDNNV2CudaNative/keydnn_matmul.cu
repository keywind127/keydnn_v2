#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K) {
    // C[m,n] = sum_k A[m,k]*B[k,n]
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        T acc = (T)0;
        for (int k = 0; k < K; ++k) {
            acc += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = acc;
    }
}

template <typename T>
static int matmul_launch(const T* A, const T* B, T* C, int M, int N, int K) {
    if (M < 0 || N < 0 || K < 0) return 1;
    if (M == 0 || N == 0) return 0;
    if (!A || !B || !C) return 2;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<T> << <grid, block >> > (A, B, C, M, N, K);
    cudaError_t st = cudaGetLastError();
    return (st == cudaSuccess) ? 0 : 3;
}

extern "C" int keydnn_cuda_matmul_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    return matmul_launch<float>(A, B, C, M, N, K);
}
extern "C" int keydnn_cuda_matmul_f64(const double* A, const double* B, double* C, int M, int N, int K) {
    return matmul_launch<double>(A, B, C, M, N, K);
}
