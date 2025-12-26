#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
    int M, int N, int K) {
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

static int _is_device_ptr(const void* p) {
    if (!p) return (int)cudaErrorInvalidValue;

    cudaPointerAttributes attr;
#if CUDART_VERSION >= 10000
    cudaError_t st = cudaPointerGetAttributes(&attr, p);
    if (st != cudaSuccess) return (int)st;
    // For modern CUDA: attr.type is one of cudaMemoryTypeDevice / Host / Managed
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        return (int)cudaErrorInvalidValue;
    }
#else
    cudaError_t st = cudaPointerGetAttributes(&attr, p);
    if (st != cudaSuccess) return (int)st;
    if (attr.memoryType != cudaMemoryTypeDevice) {
        return (int)cudaErrorInvalidValue;
    }
#endif
    return 0;
}

template <typename T>
static int matmul_launch(const T* A, const T* B, T* C, int M, int N, int K) {
    // shape sanity
    if (M < 0 || N < 0 || K < 0) return (int)cudaErrorInvalidValue;
    if (M == 0 || N == 0) return 0;
    if (!A || !B || !C) return (int)cudaErrorInvalidValue;

    // clear any prior sticky error (important!)
    (void)cudaGetLastError();

    // validate pointers are device/managed pointers
    int stp = 0;
    stp = _is_device_ptr((const void*)A); if (stp) return stp;
    stp = _is_device_ptr((const void*)B); if (stp) return stp;
    stp = _is_device_ptr((const void*)C); if (stp) return stp;

    dim3 block(16, 16);
    dim3 grid((unsigned)((N + block.x - 1) / block.x),
        (unsigned)((M + block.y - 1) / block.y));

    matmul_kernel<T> << <grid, block >> > (A, B, C, M, N, K);

    cudaError_t st = cudaPeekAtLastError();
    if (st != cudaSuccess) return (int)st;

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) return (int)st;

    return 0;
}

extern "C" int keydnn_cuda_matmul_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    return matmul_launch<float>(A, B, C, M, N, K);
}
extern "C" int keydnn_cuda_matmul_f64(const double* A, const double* B, double* C, int M, int N, int K) {
    return matmul_launch<double>(A, B, C, M, N, K);
}
