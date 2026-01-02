#include "keydnn_reduce_cuda.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <limits>

#include <math_constants.h>   // CUDART_INF_F / CUDART_INF
#include <stdint.h>           // uint64_t

// ----------------------------
// Error helper (C ABI returns int)
// ----------------------------
static inline int keydnn_cuda_ok(cudaError_t st) {
    return (st == cudaSuccess) ? 0 : static_cast<int>(st);
}

// ----------------------------
// Device-safe -infinity (avoid std::numeric_limits<T>::infinity() in kernels)
// ----------------------------
template <typename T>
__device__ __forceinline__ T neg_inf();

template <>
__device__ __forceinline__ float neg_inf<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ double neg_inf<double>() {
    return -CUDART_INF;
}

// ----------------------------
// atomicAdd for double on < sm_60 (CAS-based fallback)
// ----------------------------
__device__ __forceinline__ double atomicAdd_double_fallback(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* addr_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + val;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(sum));
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

// ----------------------------
// sum_all kernel (simple atomic reduction)
// ----------------------------
template <typename T>
__global__ void sum_all_kernel(const T* __restrict__ x, T* __restrict__ out, int64_t numel) {
    static_assert(std::is_floating_point<T>::value, "sum_all_kernel requires floating point T");
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    if constexpr (std::is_same<T, double>::value) {
        atomicAdd_double_fallback(reinterpret_cast<double*>(out), static_cast<double>(x[i]));
    }
    else {
        atomicAdd(out, x[i]); // float path
    }
}

template <typename T>
static inline int sum_all_impl(const T* x, T* y, int64_t numel) {
    if (!x || !y) return -1;
    if (numel <= 0) return -2;

    cudaError_t st;

    // y is device pointer to a single scalar; clear to 0
    st = cudaMemset(y, 0, sizeof(T));
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    const int block = 256;
    const int grid = static_cast<int>((numel + block - 1) / block);

    sum_all_kernel<T> << <grid, block >> > (x, y, numel);

    st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// mean_all: sum_all then scale
// ----------------------------
template <typename T>
__global__ void scale_scalar_kernel(T* __restrict__ y, T scale) {
    y[0] = y[0] * scale;
}

template <typename T>
static inline int mean_all_impl(const T* x, T* y, int64_t numel) {
    int st = sum_all_impl<T>(x, y, numel);
    if (st != 0) return st;

    const T scale = static_cast<T>(1) / static_cast<T>(numel);
    scale_scalar_kernel<T> << <1, 1 >> > (y, scale);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return keydnn_cuda_ok(e);

    e = cudaDeviceSynchronize();
    return keydnn_cuda_ok(e);
}

// ----------------------------
// backward fills for sum/mean
// grad_out is a device scalar pointer
// ----------------------------
template <typename T>
__global__ void fill_with_scalar_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    int64_t numel,
    T scale
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    grad_x[i] = grad_out[0] * scale;
}

template <typename T>
static inline int fill_backward_impl(const T* grad_out, T* grad_x, int64_t numel, T scale) {
    if (!grad_out || !grad_x) return -1;
    if (numel <= 0) return -2;

    const int block = 256;
    const int grid = static_cast<int>((numel + block - 1) / block);

    fill_with_scalar_kernel<T> << <grid, block >> > (grad_out, grad_x, numel, scale);

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// max axis 2D forward
// axis=1: reduce over cols for each row -> y[rows], idx[rows]
// axis=0: reduce over rows for each col -> y[cols], idx[cols]
// ----------------------------
template <typename T>
__global__ void max_axis2d_forward_axis1_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t* __restrict__ idx,
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    // FIX: device-safe -inf (avoid std::numeric_limits<T>::infinity() in device code)
    T best = neg_inf<T>();
    int64_t best_i = 0;

    const int base = r * cols;
    for (int c = 0; c < cols; ++c) {
        T v = x[base + c];
        if (v > best) {
            best = v;
            best_i = static_cast<int64_t>(c);
        }
    }
    y[r] = best;
    idx[r] = best_i;
}

template <typename T>
__global__ void max_axis2d_forward_axis0_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t* __restrict__ idx,
    int rows, int cols
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;

    // FIX: device-safe -inf (avoid std::numeric_limits<T>::infinity() in device code)
    T best = neg_inf<T>();
    int64_t best_i = 0;

    for (int r = 0; r < rows; ++r) {
        T v = x[r * cols + c];
        if (v > best) {
            best = v;
            best_i = static_cast<int64_t>(r);
        }
    }
    y[c] = best;
    idx[c] = best_i;
}

template <typename T>
static inline int max_axis2d_forward_impl(const T* x, T* y, int64_t* idx, int rows, int cols, int axis) {
    if (!x || !y || !idx) return -1;
    if (rows <= 0 || cols <= 0) return -2;
    if (!(axis == 0 || axis == 1)) return -3;

    const int block = 256;
    if (axis == 1) {
        const int grid = (rows + block - 1) / block;
        max_axis2d_forward_axis1_kernel<T> << <grid, block >> > (x, y, idx, rows, cols);
    }
    else {
        const int grid = (cols + block - 1) / block;
        max_axis2d_forward_axis0_kernel<T> << <grid, block >> > (x, y, idx, rows, cols);
    }

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);
    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// max axis 2D backward (scatter)
// caller must memset grad_x to 0 before calling
// axis=1: for each row r, col = idx[r]; grad_x[r, col] += grad_out[r]
// axis=0: for each col c, row = idx[c]; grad_x[row, c] += grad_out[c]
// ----------------------------
template <typename T>
__global__ void max_axis2d_backward_axis1_kernel(
    const T* __restrict__ grad_out,
    const int64_t* __restrict__ idx,
    T* __restrict__ grad_x,
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    int64_t c = idx[r];
    if (c < 0 || c >= cols) return;
    grad_x[r * cols + static_cast<int>(c)] += grad_out[r];
}

template <typename T>
__global__ void max_axis2d_backward_axis0_kernel(
    const T* __restrict__ grad_out,
    const int64_t* __restrict__ idx,
    T* __restrict__ grad_x,
    int rows, int cols
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;

    int64_t r = idx[c];
    if (r < 0 || r >= rows) return;
    grad_x[static_cast<int>(r) * cols + c] += grad_out[c];
}

template <typename T>
static inline int max_axis2d_backward_impl(const T* grad_out, const int64_t* idx, T* grad_x, int rows, int cols, int axis) {
    if (!grad_out || !idx || !grad_x) return -1;
    if (rows <= 0 || cols <= 0) return -2;
    if (!(axis == 0 || axis == 1)) return -3;

    const int block = 256;
    if (axis == 1) {
        const int grid = (rows + block - 1) / block;
        max_axis2d_backward_axis1_kernel<T> << <grid, block >> > (grad_out, idx, grad_x, rows, cols);
    }
    else {
        const int grid = (cols + block - 1) / block;
        max_axis2d_backward_axis0_kernel<T> << <grid, block >> > (grad_out, idx, grad_x, rows, cols);
    }

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);
    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// sum axis 2D forward
// axis=1: reduce cols for each row -> y[rows]
// axis=0: reduce rows for each col -> y[cols]
// ----------------------------
template <typename T>
__global__ void sum_axis2d_forward_axis1_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int rows, int cols
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    const int base = r * cols;
    T acc = static_cast<T>(0);
    for (int c = 0; c < cols; ++c) {
        acc += x[base + c];
    }
    y[r] = acc;
}

template <typename T>
__global__ void sum_axis2d_forward_axis0_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int rows, int cols
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;

    T acc = static_cast<T>(0);
    for (int r = 0; r < rows; ++r) {
        acc += x[r * cols + c];
    }
    y[c] = acc;
}

template <typename T>
static inline int sum_axis2d_forward_impl(
    const T* x, T* y,
    int rows, int cols, int axis
) {
    if (!x || !y) return -1;
    if (rows <= 0 || cols <= 0) return -2;
    if (!(axis == 0 || axis == 1)) return -3;

    const int block = 256;
    if (axis == 1) {
        const int grid = (rows + block - 1) / block;
        sum_axis2d_forward_axis1_kernel<T> << <grid, block >> > (x, y, rows, cols);
    }
    else {
        const int grid = (cols + block - 1) / block;
        sum_axis2d_forward_axis0_kernel<T> << <grid, block >> > (x, y, rows, cols);
    }

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);
    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}

// ----------------------------
// sum axis 2D backward (broadcast)
// axis=1: grad_x[r,c] = grad_out[r]
// axis=0: grad_x[r,c] = grad_out[c]
// ----------------------------
template <typename T>
__global__ void sum_axis2d_backward_axis1_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    int rows, int cols
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (i >= n) return;

    int r = static_cast<int>(i / cols);
    grad_x[i] = grad_out[r];
}

template <typename T>
__global__ void sum_axis2d_backward_axis0_kernel(
    const T* __restrict__ grad_out,
    T* __restrict__ grad_x,
    int rows, int cols
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (i >= n) return;

    int c = static_cast<int>(i % cols);
    grad_x[i] = grad_out[c];
}

template <typename T>
static inline int sum_axis2d_backward_impl(
    const T* grad_out, T* grad_x,
    int rows, int cols, int axis
) {
    if (!grad_out || !grad_x) return -1;
    if (rows <= 0 || cols <= 0) return -2;
    if (!(axis == 0 || axis == 1)) return -3;

    const int block = 256;
    const int64_t n = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    const int grid = static_cast<int>((n + block - 1) / block);

    if (axis == 1) {
        sum_axis2d_backward_axis1_kernel<T> << <grid, block >> > (grad_out, grad_x, rows, cols);
    }
    else {
        sum_axis2d_backward_axis0_kernel<T> << <grid, block >> > (grad_out, grad_x, rows, cols);
    }

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_ok(st);
    st = cudaDeviceSynchronize();
    return keydnn_cuda_ok(st);
}


// ----------------------------
// sum_to_shape (general unbroadcast reduction)
// ----------------------------

#ifndef KEYDNN_SUM_TO_SHAPE_MAX_NDIM
#define KEYDNN_SUM_TO_SHAPE_MAX_NDIM 8
#endif

template <typename T>
__global__ void sum_to_shape_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t numel_in,
    int ndim,
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ out_strides
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel_in) return;

    // Convert linear index i -> multi-index using in_strides,
    // and compute output linear index by collapsing reduced dims to 0.
    int64_t rem = i;
    int64_t out_off = 0;

#pragma unroll
    for (int d = 0; d < KEYDNN_SUM_TO_SHAPE_MAX_NDIM; ++d) {
        if (d >= ndim) break;

        const int64_t stride = in_strides[d];
        const int64_t idx = (stride > 0) ? (rem / stride) : 0;
        rem = (stride > 0) ? (rem - idx * stride) : rem;

        const int64_t od = out_shape[d];
        const int64_t oidx = (od == 1) ? 0 : idx;

        out_off += oidx * out_strides[d];
    }

    // Atomic add into y[out_off]
    if constexpr (std::is_same<T, double>::value) {
        atomicAdd_double_fallback(reinterpret_cast<double*>(y + out_off), static_cast<double>(x[i]));
    }
    else {
        atomicAdd(y + out_off, x[i]);
    }
}

template <typename T>
static inline int sum_to_shape_impl(
    const T* x,
    T* y,
    const int64_t* in_shape_h,
    const int64_t* out_shape_h,
    int ndim
) {
    static_assert(std::is_floating_point<T>::value, "sum_to_shape_impl requires floating point T");

    if (!x || !y || !in_shape_h || !out_shape_h) return -1;
    if (ndim <= 0) return -2;
    if (ndim > KEYDNN_SUM_TO_SHAPE_MAX_NDIM) return -3;

    // Validate shapes + compute numel
    int64_t in_shape[KEYDNN_SUM_TO_SHAPE_MAX_NDIM];
    int64_t out_shape[KEYDNN_SUM_TO_SHAPE_MAX_NDIM];
    int64_t in_strides[KEYDNN_SUM_TO_SHAPE_MAX_NDIM];
    int64_t out_strides[KEYDNN_SUM_TO_SHAPE_MAX_NDIM];

    // Copy + validate
    int64_t numel_in = 1;
    int64_t numel_out = 1;

    for (int d = 0; d < ndim; ++d) {
        const int64_t id = in_shape_h[d];
        const int64_t od = out_shape_h[d];
        if (id < 0 || od < 0) return -4; // invalid dims
        // Broadcast reduction rule: od must be 1 or equal to id
        if (!(od == 1 || od == id)) return -5;

        in_shape[d] = id;
        out_shape[d] = od;

        // numel with zero-dim allowed
        numel_in = (numel_in == 0 || id == 0) ? 0 : (numel_in * id);
        numel_out = (numel_out == 0 || od == 0) ? 0 : (numel_out * od);
    }

    // If output is empty, just zero it and return (nothing to accumulate).
    // (Caller expects valid output even for empty tensors.)
    if (numel_out == 0) {
        cudaError_t st = cudaMemset(y, 0, 0); // no-op; keep style consistent
        (void)st;
        return 0;
    }

    // Build row-major strides (contiguous)
    // stride[d] = product(shape[d+1:])
    // If any dim is 0, numel is 0 and kernel will no-op after memset.
    {
        int64_t s = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            in_strides[d] = s;
            const int64_t id = in_shape[d];
            s = (id == 0) ? 0 : (s * id);
        }
    }
    {
        int64_t s = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            out_strides[d] = s;
            const int64_t od = out_shape[d];
            s = (od == 0) ? 0 : (s * od);
        }
    }

    cudaError_t st;

    // Zero output buffer before atomic accumulation
    st = cudaMemset(y, 0, static_cast<size_t>(numel_out) * sizeof(T));
    if (st != cudaSuccess) return keydnn_cuda_ok(st);

    // If input is empty, done after memset
    if (numel_in == 0) {
        st = cudaDeviceSynchronize();
        return keydnn_cuda_ok(st);
    }

    // Upload shapes/strides to device (small arrays)
    int64_t* d_in_shape = nullptr, * d_out_shape = nullptr, * d_in_strides = nullptr, * d_out_strides = nullptr;

    st = cudaMalloc(reinterpret_cast<void**>(&d_in_shape), sizeof(int64_t) * ndim);
    if (st != cudaSuccess) return keydnn_cuda_ok(st);
    st = cudaMalloc(reinterpret_cast<void**>(&d_out_shape), sizeof(int64_t) * ndim);
    if (st != cudaSuccess) { cudaFree(d_in_shape); return keydnn_cuda_ok(st); }
    st = cudaMalloc(reinterpret_cast<void**>(&d_in_strides), sizeof(int64_t) * ndim);
    if (st != cudaSuccess) { cudaFree(d_in_shape); cudaFree(d_out_shape); return keydnn_cuda_ok(st); }
    st = cudaMalloc(reinterpret_cast<void**>(&d_out_strides), sizeof(int64_t) * ndim);
    if (st != cudaSuccess) { cudaFree(d_in_shape); cudaFree(d_out_shape); cudaFree(d_in_strides); return keydnn_cuda_ok(st); }

    st = cudaMemcpy(d_in_shape, in_shape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) goto cleanup;
    st = cudaMemcpy(d_out_shape, out_shape, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) goto cleanup;
    st = cudaMemcpy(d_in_strides, in_strides, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) goto cleanup;
    st = cudaMemcpy(d_out_strides, out_strides, sizeof(int64_t) * ndim, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((numel_in + block - 1) / block);
        sum_to_shape_kernel<T> << <grid, block >> > (
            x, y,
            numel_in,
            ndim,
            d_in_shape,
            d_in_strides,
            d_out_shape,
            d_out_strides
            );
    }

    st = cudaGetLastError();
    if (st != cudaSuccess) goto cleanup;

    st = cudaDeviceSynchronize();

cleanup:
    cudaFree(d_in_shape);
    cudaFree(d_out_shape);
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);

    return keydnn_cuda_ok(st);
}




// ----------------------------
// Exported C ABI functions
// ----------------------------
int keydnn_cuda_sum_all_f32(const float* x, float* y, int64_t numel) {
    return sum_all_impl<float>(x, y, numel);
}
int keydnn_cuda_sum_all_f64(const double* x, double* y, int64_t numel) {
    return sum_all_impl<double>(x, y, numel);
}

int keydnn_cuda_mean_all_f32(const float* x, float* y, int64_t numel) {
    return mean_all_impl<float>(x, y, numel);
}
int keydnn_cuda_mean_all_f64(const double* x, double* y, int64_t numel) {
    return mean_all_impl<double>(x, y, numel);
}

int keydnn_cuda_sum_backward_fill_f32(const float* grad_out, float* grad_x, int64_t numel) {
    return fill_backward_impl<float>(grad_out, grad_x, numel, 1.0f);
}
int keydnn_cuda_sum_backward_fill_f64(const double* grad_out, double* grad_x, int64_t numel) {
    return fill_backward_impl<double>(grad_out, grad_x, numel, 1.0);
}

int keydnn_cuda_mean_backward_fill_f32(const float* grad_out, float* grad_x, int64_t numel) {
    const float scale = 1.0f / static_cast<float>(numel);
    return fill_backward_impl<float>(grad_out, grad_x, numel, scale);
}
int keydnn_cuda_mean_backward_fill_f64(const double* grad_out, double* grad_x, int64_t numel) {
    const double scale = 1.0 / static_cast<double>(numel);
    return fill_backward_impl<double>(grad_out, grad_x, numel, scale);
}

int keydnn_cuda_max_axis2d_forward_f32(const float* x, float* y, int64_t* idx, int rows, int cols, int axis) {
    return max_axis2d_forward_impl<float>(x, y, idx, rows, cols, axis);
}
int keydnn_cuda_max_axis2d_forward_f64(const double* x, double* y, int64_t* idx, int rows, int cols, int axis) {
    return max_axis2d_forward_impl<double>(x, y, idx, rows, cols, axis);
}

int keydnn_cuda_max_axis2d_backward_f32(const float* grad_out, const int64_t* idx, float* grad_x, int rows, int cols, int axis) {
    return max_axis2d_backward_impl<float>(grad_out, idx, grad_x, rows, cols, axis);
}
int keydnn_cuda_max_axis2d_backward_f64(const double* grad_out, const int64_t* idx, double* grad_x, int rows, int cols, int axis) {
    return max_axis2d_backward_impl<double>(grad_out, idx, grad_x, rows, cols, axis);
}

int keydnn_cuda_sum_axis2d_forward_f32(const float* x, float* y, int rows, int cols, int axis) {
    return sum_axis2d_forward_impl<float>(x, y, rows, cols, axis);
}
int keydnn_cuda_sum_axis2d_forward_f64(const double* x, double* y, int rows, int cols, int axis) {
    return sum_axis2d_forward_impl<double>(x, y, rows, cols, axis);
}

int keydnn_cuda_sum_axis2d_backward_f32(const float* grad_out, float* grad_x, int rows, int cols, int axis) {
    return sum_axis2d_backward_impl<float>(grad_out, grad_x, rows, cols, axis);
}
int keydnn_cuda_sum_axis2d_backward_f64(const double* grad_out, double* grad_x, int rows, int cols, int axis) {
    return sum_axis2d_backward_impl<double>(grad_out, grad_x, rows, cols, axis);
}

int keydnn_cuda_sum_to_shape_f32(
    const float* x,
    float* y,
    const int64_t* in_shape,
    const int64_t* out_shape,
    int ndim
) {
    return sum_to_shape_impl<float>(x, y, in_shape, out_shape, ndim);
}

int keydnn_cuda_sum_to_shape_f64(
    const double* x,
    double* y,
    const int64_t* in_shape,
    const int64_t* out_shape,
    int ndim
) {
    return sum_to_shape_impl<double>(x, y, in_shape, out_shape, ndim);
}

