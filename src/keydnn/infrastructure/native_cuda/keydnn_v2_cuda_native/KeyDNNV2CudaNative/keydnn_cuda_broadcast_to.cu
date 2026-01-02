#include "keydnn_cuda_ops.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace {

    constexpr int MAX_DIMS = 8;

    constexpr int ST_OK = 0;
    constexpr int ST_BAD_PTR = 1;
    constexpr int ST_CUDA_ERR = 2;
    constexpr int ST_BAD_DIMS = 3;
    constexpr int ST_BAD_SHAPE = 4;

    struct BroadcastMeta {
        int ndim;                       // == out_ndim
        int64_t out_shape[MAX_DIMS];
        int64_t out_strides[MAX_DIMS];  // row-major strides in elements
        int64_t in_shape[MAX_DIMS];     // padded to out_ndim
        int64_t in_strides[MAX_DIMS];   // padded strides in elements (0 stride ok for dim==1, but we compute normal)
        int64_t out_numel;
    };

    inline bool compute_row_major_strides(const int64_t* shape, int ndim, int64_t* strides) {
        // strides in elements
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            strides[i] = s;
            int64_t d = shape[i];
            if (d < 0) return false;
            s *= (d == 0 ? 1 : d); // keep strides consistent even if dim==0
        }
        return true;
    }

    inline int64_t compute_numel(const int64_t* shape, int ndim) {
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            int64_t d = shape[i];
            if (d == 0) return 0;
            n *= d;
        }
        return n;
    }

    inline bool validate_broadcast(const int64_t* in_padded, const int64_t* out_shape, int ndim) {
        for (int i = 0; i < ndim; ++i) {
            const int64_t sd = in_padded[i];
            const int64_t td = out_shape[i];
            if (sd == td) continue;
            if (sd == 1 && td >= 1) continue;
            if ((sd == 1 && td == 0) || (sd == 0 && td == 0)) continue;
            return false;
        }
        return true;
    }

    template <typename T>
    __global__ void broadcast_to_kernel(const T* __restrict__ x, T* __restrict__ y, BroadcastMeta meta) {
        int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
        if (idx >= meta.out_numel) return;

        // Map linear idx -> multi-index using out_strides
        int64_t rem = idx;
        int64_t in_offset = 0;

#pragma unroll
        for (int i = 0; i < MAX_DIMS; ++i) {
            if (i >= meta.ndim) break;
            const int64_t stride = meta.out_strides[i];
            int64_t coord = 0;
            if (stride != 0) {
                coord = rem / stride;
                rem -= coord * stride;
            }
            // broadcast: if in_shape[i]==1 -> coord = 0
            if (meta.in_shape[i] != 1) {
                in_offset += coord * meta.in_strides[i];
            }
        }

        y[idx] = x[in_offset];
    }

    template <typename T>
    int launch_broadcast_to(
        const T* x, T* y,
        const int64_t* in_shape, int in_ndim,
        const int64_t* out_shape, int out_ndim
    ) {
        if (!x || !y || !in_shape || !out_shape) return ST_BAD_PTR;
        if (out_ndim < 0 || in_ndim < 0) return ST_BAD_DIMS;
        if (out_ndim > MAX_DIMS) return ST_BAD_DIMS;
        if (in_ndim > out_ndim) return ST_BAD_SHAPE;

        BroadcastMeta meta{};
        meta.ndim = out_ndim;

        // copy out_shape
        for (int i = 0; i < out_ndim; ++i) {
            meta.out_shape[i] = out_shape[i];
            if (meta.out_shape[i] < 0) return ST_BAD_SHAPE;
        }

        // build padded in_shape: left pad with ones
        int pad = out_ndim - in_ndim;
        for (int i = 0; i < pad; ++i) meta.in_shape[i] = 1;
        for (int i = 0; i < in_ndim; ++i) {
            meta.in_shape[pad + i] = in_shape[i];
            if (meta.in_shape[pad + i] < 0) return ST_BAD_SHAPE;
        }

        // broadcast compatibility
        if (!validate_broadcast(meta.in_shape, meta.out_shape, out_ndim)) return ST_BAD_SHAPE;

        // compute strides
        if (!compute_row_major_strides(meta.out_shape, out_ndim, meta.out_strides)) return ST_BAD_SHAPE;

        // in strides: compute from padded in_shape (contiguous assumption)
        if (!compute_row_major_strides(meta.in_shape, out_ndim, meta.in_strides)) return ST_BAD_SHAPE;

        meta.out_numel = compute_numel(meta.out_shape, out_ndim);
        if (meta.out_numel == 0) return ST_OK; // empty output is trivially ok

        // launch
        constexpr int THREADS = 256;
        int64_t blocks64 = (meta.out_numel + THREADS - 1) / THREADS;
        // grid.x is int, clamp
        int blocks = (blocks64 > INT32_MAX) ? INT32_MAX : (int)blocks64;

        broadcast_to_kernel<T> << <blocks, THREADS >> > (x, y, meta);
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess) return ST_CUDA_ERR;

        return ST_OK;
    }

} // namespace

extern "C" int keydnn_cuda_broadcast_to_f32(
    const float* x, float* y,
    const int64_t * in_shape, int in_ndim,
    const int64_t * out_shape, int out_ndim
) {
    return launch_broadcast_to<float>(x, y, in_shape, in_ndim, out_shape, out_ndim);
}

extern "C" int keydnn_cuda_broadcast_to_f64(
    const double* x, double* y,
    const int64_t * in_shape, int in_ndim,
    const int64_t * out_shape, int out_ndim
) {
    return launch_broadcast_to<double>(x, y, in_shape, in_ndim, out_shape, out_ndim);
}
