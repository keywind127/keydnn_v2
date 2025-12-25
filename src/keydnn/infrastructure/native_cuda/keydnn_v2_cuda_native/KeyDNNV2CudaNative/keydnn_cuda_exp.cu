#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ T my_exp(T x);
template <>
__device__ float my_exp<float>(float x) { return expf(x); }
template <>
__device__ double my_exp<double>(double x) { return exp(x); }

template <typename T>
__global__ void exp_kernel(const T* __restrict__ x, T* __restrict__ y, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) y[i] = my_exp<T>(x[i]);
}

template <typename T>
static int exp_launch(const T* x, T* y, int numel) {
    if (numel < 0) return 1;
    if (numel == 0) return 0;
    if (!x || !y) return 2;

    int block = 256;
    int grid = (numel + block - 1) / block;
    exp_kernel<T> << <grid, block >> > (x, y, numel);
    cudaError_t st = cudaGetLastError();
    return (st == cudaSuccess) ? 0 : 3;
}

extern "C" int keydnn_cuda_exp_f32(const float* x, float* y, int numel) {
    return exp_launch<float>(x, y, numel);
}
extern "C" int keydnn_cuda_exp_f64(const double* x, double* y, int numel) {
    return exp_launch<double>(x, y, numel);
}
