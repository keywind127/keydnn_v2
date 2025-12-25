#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>

extern "C" int keydnn_cuda_memcpy_d2d(void* dst, const void* src, std::size_t nbytes) {
    if (nbytes == 0) return 0;
    if (!dst || !src) return 1;
    cudaError_t st = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice);
    return (st == cudaSuccess) ? 0 : 2;
}
