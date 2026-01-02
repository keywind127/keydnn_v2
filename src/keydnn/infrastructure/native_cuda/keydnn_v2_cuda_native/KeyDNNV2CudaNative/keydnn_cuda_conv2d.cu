// keydnn_cuda_conv2d.cu  (cuDNN-backed)
#include "keydnn_cuda_conv2d.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>

// -------------------- Error helpers --------------------

static inline int _cuda_to_int(cudaError_t e) { return (int)e; }

// Encode cuDNN failures as a non-zero int distinct from cudaError_t.
// (Your Python just checks "!= 0", so any non-zero is fine.)
static inline int _cudnn_to_int(cudnnStatus_t s) {
    // Reserve a high range so it won't collide with cudaError_t codes.
    return 100000 + (int)s;
}

#define KEYDNN_CUDA_CHECK(expr)            \
    do {                                   \
        cudaError_t _e = (expr);           \
        if (_e != cudaSuccess) return _cuda_to_int(_e); \
    } while (0)

#define KEYDNN_CUDNN_CHECK(expr)           \
    do {                                   \
        cudnnStatus_t _s = (expr);         \
        if (_s != CUDNN_STATUS_SUCCESS) return _cudnn_to_int(_s); \
    } while (0)

static inline int _validate_common(
    const void* x_pad,
    const void* w,
    const void* y_or_grad_out,
    int N, int C_in, int H_pad, int W_pad,
    int C_out, int H_out, int W_out,
    int K_h, int K_w, int s_h, int s_w
) {
    if (!x_pad || !w || !y_or_grad_out) return (int)cudaErrorInvalidValue;
    if (N < 0 || C_in < 0 || H_pad < 0 || W_pad < 0) return (int)cudaErrorInvalidValue;
    if (C_out < 0 || H_out < 0 || W_out < 0) return (int)cudaErrorInvalidValue;
    if (K_h <= 0 || K_w <= 0) return (int)cudaErrorInvalidValue;
    if (s_h <= 0 || s_w <= 0) return (int)cudaErrorInvalidValue;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;
    return 0;
}

// -------------------- cuDNN handle (thread-local) --------------------
// Creating/destroying cudnnHandle every call is expensive; thread_local is a good baseline.

static inline cudnnHandle_t _get_cudnn_handle() {
    static thread_local cudnnHandle_t handle = nullptr;
    if (!handle) {
        // If this fails, we can't return an error easily here; callers should
        // only use after ensuring CUDA runtime is OK. We'll handle in call sites.
        cudnnCreate(&handle);
    }
    return handle;
}

// -------------------- Descriptor helpers --------------------

static inline void _set_tensor4d_nchw(cudnnTensorDescriptor_t desc, cudnnDataType_t dt,
    int N, int C, int H, int W) {
    // NCHW with contiguous strides implied by cudnnSetTensor4dDescriptor
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dt, N, C, H, W);
}

static inline void _set_filter4d_oihw(cudnnFilterDescriptor_t desc, cudnnDataType_t dt,
    int C_out, int C_in, int K_h, int K_w) {
    cudnnSetFilter4dDescriptor(desc, dt, CUDNN_TENSOR_NCHW, C_out, C_in, K_h, K_w);
}

// -------------------- Algo selection helpers (simple) --------------------
// Start simple: query an algorithm each call.
// For best performance, add a cache keyed by (N,C,H,W,Cout,Kh,Kw,stride,dtype) later.

template <typename T>
static int launch_conv2d_forward_cudnn(
    const T* x_pad,
    const T* w,
    const T* b,   // may be nullptr
    T* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const int v = _validate_common(x_pad, w, y, N, C_in, H_pad, W_pad,
        C_out, H_out, W_out, K_h, K_w, s_h, s_w);
    if (v != 0) return v;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    // Use the current CUDA device context; do not change device here.

    cudnnHandle_t handle = _get_cudnn_handle();
    if (!handle) return _cudnn_to_int(CUDNN_STATUS_INTERNAL_ERROR);

    cudnnDataType_t dt = std::is_same<T, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // x_pad is already padded -> set conv padding = 0
    _set_tensor4d_nchw(x_desc, dt, N, C_in, H_pad, W_pad);
    _set_filter4d_oihw(w_desc, dt, C_out, C_in, K_h, K_w);

    KEYDNN_CUDNN_CHECK(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/0, /*pad_w=*/0,
            /*stride_h=*/s_h, /*stride_w=*/s_w,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CROSS_CORRELATION,
            dt
        )
    );

    // y dims are passed in; validate cuDNN agrees (helps catch shape bugs)
    int outN = 0, outC = 0, outH = 0, outW = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &outN, &outC, &outH, &outW
    ));
    if (outN != N || outC != C_out || outH != H_out || outW != W_out) {
        // Mismatch means caller-provided H_out/W_out are inconsistent
        // Use cudaErrorInvalidValue to match your existing style.
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(x_desc);
        return (int)cudaErrorInvalidValue;
    }

    _set_tensor4d_nchw(y_desc, dt, N, C_out, H_out, W_out);

    // Choose algo (v7 API)
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, x_desc, w_desc, conv_desc, y_desc, 1, &returned, &perf
    ));
    cudnnConvolutionFwdAlgo_t algo = perf.algo;

    // Workspace
    size_t ws_bytes = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &ws_bytes
    ));

    void* ws = nullptr;
    if (ws_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));

    const float one_f = 1.0f, zero_f = 0.0f;
    const double one_d = 1.0, zero_d = 0.0;

    const void* alpha = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;
    const void* beta = std::is_same<T, float>::value ? (const void*)&zero_f : (const void*)&zero_d;

    // y = conv(x, w)
    cudnnStatus_t s = cudnnConvolutionForward(
        handle,
        alpha,
        x_desc, x_pad,
        w_desc, w,
        conv_desc, algo,
        ws, ws_bytes,
        beta,
        y_desc, y
    );
    if (ws) cudaFree(ws);
    if (s != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(x_desc);
        return _cudnn_to_int(s);
    }

    // Optional bias add: y += b (broadcast)
    if (b) {
        KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
        _set_tensor4d_nchw(b_desc, dt, 1, C_out, 1, 1);

        const void* alpha2 = alpha;
        const void* beta2 = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;

        // y = alpha2 * b + beta2 * y
        KEYDNN_CUDNN_CHECK(cudnnAddTensor(
            handle,
            alpha2,
            b_desc, b,
            beta2,
            y_desc, y
        ));
        cudnnDestroyTensorDescriptor(b_desc);
    }

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    return 0;
}

template <typename T>
static int launch_conv2d_backward_cudnn(
    const T* x_pad,
    const T* w,
    const T* grad_out,
    T* grad_x_pad,
    T* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    const int v = _validate_common(x_pad, w, grad_out, N, C_in, H_pad, W_pad,
        C_out, H_out, W_out, K_h, K_w, s_h, s_w);
    if (v != 0) return v;
    if (!grad_x_pad || !grad_w) return (int)cudaErrorInvalidValue;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    cudnnHandle_t handle = _get_cudnn_handle();
    if (!handle) return _cudnn_to_int(CUDNN_STATUS_INTERNAL_ERROR);

    cudnnDataType_t dt = std::is_same<T, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t dy_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    _set_tensor4d_nchw(x_desc, dt, N, C_in, H_pad, W_pad);
    _set_tensor4d_nchw(dy_desc, dt, N, C_out, H_out, W_out);
    _set_filter4d_oihw(w_desc, dt, C_out, C_in, K_h, K_w);

    KEYDNN_CUDNN_CHECK(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/0, /*pad_w=*/0,           // already padded input
            /*stride_h=*/s_h, /*stride_w=*/s_w,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CROSS_CORRELATION,
            dt
        )
    );

    // ---- BackwardData: grad_x_pad ----
    cudnnConvolutionBwdDataAlgoPerf_t perf_dx;
    int returned_dx = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, w_desc, dy_desc, conv_desc, x_desc, 1, &returned_dx, &perf_dx
    ));
    cudnnConvolutionBwdDataAlgo_t algo_dx = perf_dx.algo;

    size_t ws_dx_bytes = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, w_desc, dy_desc, conv_desc, x_desc, algo_dx, &ws_dx_bytes
    ));
    void* ws_dx = nullptr;
    if (ws_dx_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws_dx, ws_dx_bytes));

    // ---- BackwardFilter: grad_w ----
    cudnnConvolutionBwdFilterAlgoPerf_t perf_dw;
    int returned_dw = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, x_desc, dy_desc, conv_desc, w_desc, 1, &returned_dw, &perf_dw
    ));
    cudnnConvolutionBwdFilterAlgo_t algo_dw = perf_dw.algo;

    size_t ws_dw_bytes = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, x_desc, dy_desc, conv_desc, w_desc, algo_dw, &ws_dw_bytes
    ));
    void* ws_dw = nullptr;
    if (ws_dw_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws_dw, ws_dw_bytes));

    const float one_f = 1.0f, zero_f = 0.0f;
    const double one_d = 1.0, zero_d = 0.0;
    const void* alpha = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;
    const void* beta0 = std::is_same<T, float>::value ? (const void*)&zero_f : (const void*)&zero_d;

    // IMPORTANT: Your header says grad buffers are "accumulated; caller zero-init".
    // We keep beta=0 (overwrite). If you truly want accumulation, pass beta=1.
    cudnnStatus_t s1 = cudnnConvolutionBackwardData(
        handle,
        alpha,
        w_desc, w,
        dy_desc, grad_out,
        conv_desc, algo_dx,
        ws_dx, ws_dx_bytes,
        beta0,
        x_desc, grad_x_pad
    );

    cudnnStatus_t s2 = cudnnConvolutionBackwardFilter(
        handle,
        alpha,
        x_desc, x_pad,
        dy_desc, grad_out,
        conv_desc, algo_dw,
        ws_dw, ws_dw_bytes,
        beta0,
        w_desc, grad_w
    );

    if (ws_dx) cudaFree(ws_dx);
    if (ws_dw) cudaFree(ws_dw);

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(x_desc);

    if (s1 != CUDNN_STATUS_SUCCESS) return _cudnn_to_int(s1);
    if (s2 != CUDNN_STATUS_SUCCESS) return _cudnn_to_int(s2);
    return 0;
}

// -------------------- Exports (same ABI) --------------------

extern "C" int keydnn_cuda_conv2d_forward_f32(
    const float* x_pad,
    const float* w,
    const float* b,
    float* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_forward_cudnn<float>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_forward_f64(
    const double* x_pad,
    const double* w,
    const double* b,
    double* y,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_forward_cudnn<double>(
        x_pad, w, b, y,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_backward_f32(
    const float* x_pad,
    const float* w,
    const float* grad_out,
    float* grad_x_pad,
    float* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_backward_cudnn<float>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}

extern "C" int keydnn_cuda_conv2d_backward_f64(
    const double* x_pad,
    const double* w,
    const double* grad_out,
    double* grad_x_pad,
    double* grad_w,
    int N,
    int C_in,
    int H_pad,
    int W_pad,
    int C_out,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int s_h,
    int s_w
) {
    return launch_conv2d_backward_cudnn<double>(
        x_pad, w, grad_out,
        grad_x_pad, grad_w,
        N, C_in, H_pad, W_pad,
        C_out, H_out, W_out,
        K_h, K_w, s_h, s_w
    );
}
