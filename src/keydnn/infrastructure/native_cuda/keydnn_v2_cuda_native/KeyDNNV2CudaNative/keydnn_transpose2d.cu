#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>

template <typename T>
__global__ void transpose2d_kernel(const T* __restrict__ x, T* __restrict__ y, int rows, int cols) {
    // x[r, c] -> y[c, r]
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && c < cols) {
        y[c * rows + r] = x[r * cols + c];
    }
}

template <typename T>
static int transpose2d_launch(const T* x, T* y, int rows, int cols) {
    if (rows < 0 || cols < 0) return 1;
    if (rows == 0 || cols == 0) return 0;
    if (!x || !y) return 2;

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose2d_kernel<T> << <grid, block >> > (x, y, rows, cols);
    cudaError_t st = cudaGetLastError();
    return (st == cudaSuccess) ? 0 : 3;
}

extern "C" int keydnn_cuda_transpose2d_f32(const float* x, float* y, int rows, int cols) {
    return transpose2d_launch<float>(x, y, rows, cols);
}
extern "C" int keydnn_cuda_transpose2d_f64(const double* x, double* y, int rows, int cols) {
    return transpose2d_launch<double>(x, y, rows, cols);
}
