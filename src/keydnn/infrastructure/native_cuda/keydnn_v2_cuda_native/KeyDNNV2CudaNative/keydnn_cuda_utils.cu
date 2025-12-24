#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KEYDNN_DLL_EXPORT __declspec(dllexport)
#else
#define KEYDNN_DLL_EXPORT
#endif

extern "C" {

    KEYDNN_DLL_EXPORT int keydnn_cuda_set_device(int device) {
        cudaError_t st = cudaSetDevice(device);
        return (st == cudaSuccess) ? 0 : (int)st;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_malloc(std::uint64_t* out_ptr, std::size_t nbytes) {
        if (!out_ptr) return -1;
        void* p = 0;
        cudaError_t st = cudaMalloc(&p, nbytes);
        if (st != cudaSuccess) {
            *out_ptr = 0;
            return (int)st;
        }
        *out_ptr = (std::uint64_t)(uintptr_t)p;
        return 0;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_free(std::uint64_t ptr) {
        if (ptr == 0) return 0;
        cudaError_t st = cudaFree((void*)(uintptr_t)ptr);
        return (st == cudaSuccess) ? 0 : (int)st;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_memcpy_h2d(std::uint64_t dst_dev, const void* src_host, std::size_t nbytes) {
        cudaError_t st = cudaMemcpy((void*)(uintptr_t)dst_dev, src_host, nbytes, cudaMemcpyHostToDevice);
        return (st == cudaSuccess) ? 0 : (int)st;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_memcpy_d2h(void* dst_host, std::uint64_t src_dev, std::size_t nbytes) {
        cudaError_t st = cudaMemcpy(dst_host, (void*)(uintptr_t)src_dev, nbytes, cudaMemcpyDeviceToHost);
        return (st == cudaSuccess) ? 0 : (int)st;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_memset(std::uint64_t dev_ptr, int value, std::size_t nbytes) {
        cudaError_t st = cudaMemset((void*)(uintptr_t)dev_ptr, value, nbytes);
        return (st == cudaSuccess) ? 0 : (int)st;
    }

    KEYDNN_DLL_EXPORT int keydnn_cuda_synchronize() {
        cudaError_t st = cudaDeviceSynchronize();
        return (st == cudaSuccess) ? 0 : (int)st;
    }

} // extern "C"
