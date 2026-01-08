// keydnn_conv2d_transpose_cuda.cu  (cuDNN-backed)
#include "keydnn_conv2d_transpose_cuda.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// -------------------- Error helpers --------------------

static inline int _cuda_to_int(cudaError_t e) { return (int)e; }

// Encode cuDNN failures as a non-zero int distinct from cudaError_t.
static inline int _cudnn_to_int(cudnnStatus_t s) { return 100000 + (int)s; }

#define KEYDNN_CUDA_CHECK(expr)                         \
    do {                                                \
        cudaError_t _e = (expr);                        \
        if (_e != cudaSuccess) return _cuda_to_int(_e); \
    } while (0)

#define KEYDNN_CUDNN_CHECK(expr)                             \
    do {                                                     \
        cudnnStatus_t _s = (expr);                            \
        if (_s != CUDNN_STATUS_SUCCESS) return _cudnn_to_int(_s); \
    } while (0)

static inline int _validate_forward(
    const void* x, const void* w, const void* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w, int s_h, int s_w
) {
    if (!x || !w || !y) return (int)cudaErrorInvalidValue;
    if (N < 0 || C_in < 0 || H_in < 0 || W_in < 0) return (int)cudaErrorInvalidValue;
    if (C_out < 0 || H_out < 0 || W_out < 0) return (int)cudaErrorInvalidValue;
    if (K_h <= 0 || K_w <= 0) return (int)cudaErrorInvalidValue;
    if (s_h <= 0 || s_w <= 0) return (int)cudaErrorInvalidValue;

    // Legal no-op
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;
    return 0;
}

static inline int _validate_backward(
    const void* x, const void* w, const void* grad_out,
    const void* grad_x, const void* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w, int s_h, int s_w
) {
    if (!x || !w || !grad_out || !grad_x || !grad_w) return (int)cudaErrorInvalidValue;
    if (N < 0 || C_in < 0 || H_in < 0 || W_in < 0) return (int)cudaErrorInvalidValue;
    if (C_out < 0 || H_out < 0 || W_out < 0) return (int)cudaErrorInvalidValue;
    if (K_h <= 0 || K_w <= 0) return (int)cudaErrorInvalidValue;
    if (s_h <= 0 || s_w <= 0) return (int)cudaErrorInvalidValue;

    // Legal no-op
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;
    return 0;
}

// -------------------- cuDNN handle (thread-local) --------------------

static inline cudnnHandle_t _get_cudnn_handle() {
    static thread_local cudnnHandle_t handle = nullptr;
    if (!handle) {
        // Best-effort create. If it fails, callers will see null -> internal error.
        cudnnCreate(&handle);
    }
    return handle;
}

// -------------------- Descriptor helpers --------------------

static inline void _set_tensor4d_nchw(
    cudnnTensorDescriptor_t desc, cudnnDataType_t dt,
    int N, int C, int H, int W
) {
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dt, N, C, H, W);
}

// IMPORTANT: we will set filter descriptor as OIHW.
// For transpose-conv (IOHW in your project), we reinterpret:
//   w_iohw(C_in, C_out, Kh, Kw)  ==  filter_oihw(O=C_in, I=C_out, Kh, Kw)
static inline void _set_filter4d_oihw(
    cudnnFilterDescriptor_t desc, cudnnDataType_t dt,
    int O, int I, int K_h, int K_w
) {
    cudnnSetFilter4dDescriptor(desc, dt, CUDNN_TENSOR_NCHW, O, I, K_h, K_w);
}

// -------------------- Algo selection (simple, per-call) --------------------
// For better perf, cache algos/workspace sizes keyed by shape+dtype later.

template <typename T>
static int launch_conv2d_transpose_forward_cudnn(
    const T* x,          // (N, C_in, H_in, W_in)
    const T* w,          // (C_in, C_out, K_h, K_w)  <-- IOHW in project; treated as OIHW(O=C_in,I=C_out)
    const T* b,          // (C_out,) or nullptr
    T* y,                // (N, C_out, H_out, W_out)
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    const int v = _validate_forward(
        x, w, y, N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, s_h, s_w
    );
    if (v != 0) return v;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    cudnnHandle_t handle = _get_cudnn_handle();
    if (!handle) return _cudnn_to_int(CUDNN_STATUS_INTERNAL_ERROR);

    cudnnDataType_t dt = std::is_same<T, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t dy_desc = nullptr; // treat input x as "dy"
    cudnnTensorDescriptor_t dx_desc = nullptr; // output y is "dx"
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // dy = x: (N, C_in, H_in, W_in)
    _set_tensor4d_nchw(dy_desc, dt, N, C_in, H_in, W_in);

    // dx = y: (N, C_out, H_out, W_out)
    _set_tensor4d_nchw(dx_desc, dt, N, C_out, H_out, W_out);

    // w treated as OIHW with O=C_in, I=C_out
    _set_filter4d_oihw(w_desc, dt, /*O=*/C_in, /*I=*/C_out, K_h, K_w);

    KEYDNN_CUDNN_CHECK(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/pad_h, /*pad_w=*/pad_w,
            /*stride_h=*/s_h, /*stride_w=*/s_w,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CROSS_CORRELATION,
            dt
        )
    );

    // NOTE:
    // Some cuDNN distributions (especially split-header setups / older drops)
    // do not expose a "BackwardDataOutputDim" query in the headers you have.
    // We therefore skip explicit dimension validation here and rely on cuDNN
    // to error out if descriptors are inconsistent.

    // Algo (v7)
    cudnnConvolutionBwdDataAlgoPerf_t perf;
    int returned = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, w_desc, dy_desc, conv_desc, dx_desc, 1, &returned, &perf
    ));
    cudnnConvolutionBwdDataAlgo_t algo = perf.algo;

    // Workspace
    size_t ws_bytes = 0;
    KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &ws_bytes
    ));
    void* ws = nullptr;
    if (ws_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));

    const float one_f = 1.0f, zero_f = 0.0f;
    const double one_d = 1.0, zero_d = 0.0;
    const void* alpha = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;
    const void* beta0 = std::is_same<T, float>::value ? (const void*)&zero_f : (const void*)&zero_d;

    // y = dx = backwardData(w, dy=x)
    cudnnStatus_t s = cudnnConvolutionBackwardData(
        handle,
        alpha,
        w_desc, w,
        dy_desc, x,
        conv_desc, algo,
        ws, ws_bytes,
        beta0,
        dx_desc, y
    );

    if (ws) cudaFree(ws);

    if (s != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
        cudnnDestroyTensorDescriptor(dy_desc);
        return _cudnn_to_int(s);
    }

    // bias add: y += b
    if (b) {
        KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
        _set_tensor4d_nchw(b_desc, dt, 1, C_out, 1, 1);

        const void* beta1 = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;

        KEYDNN_CUDNN_CHECK(cudnnAddTensor(
            handle,
            alpha,
            b_desc, b,
            beta1,
            dx_desc, y
        ));

        cudnnDestroyTensorDescriptor(b_desc);
    }

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyTensorDescriptor(dy_desc);
    return 0;
}

template <typename T>
static int launch_conv2d_transpose_backward_cudnn(
    const T* x,          // forward input: (N, C_in, H_in, W_in)
    const T* w,          // forward weight: (C_in, C_out, K_h, K_w)
    const T* grad_out,   // dL/dy: (N, C_out, H_out, W_out)
    T* grad_x,           // dL/dx: (N, C_in, H_in, W_in)
    T* grad_w,           // dL/dw: (C_in, C_out, K_h, K_w)
    T* grad_b,           // dL/db: (C_out,) or nullptr (if you want bias grad)
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    const int v = _validate_backward(
        x, w, grad_out, grad_x, grad_w,
        N, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, s_h, s_w
    );
    if (v != 0) return v;
    if (N == 0 || C_out == 0 || H_out == 0 || W_out == 0) return 0;

    cudnnHandle_t handle = _get_cudnn_handle();
    if (!handle) return _cudnn_to_int(CUDNN_STATUS_INTERNAL_ERROR);

    cudnnDataType_t dt = std::is_same<T, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    // Descriptors:
    // grad_out: (N, C_out, H_out, W_out)
    // grad_x:   (N, C_in,  H_in,  W_in)
    // x:        (N, C_in,  H_in,  W_in)
    // w: filter treated as OIHW with O=C_in, I=C_out
    cudnnTensorDescriptor_t go_desc = nullptr;
    cudnnTensorDescriptor_t gx_desc = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&go_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&gx_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    KEYDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    _set_tensor4d_nchw(go_desc, dt, N, C_out, H_out, W_out);
    _set_tensor4d_nchw(gx_desc, dt, N, C_in, H_in, W_in);
    _set_tensor4d_nchw(x_desc, dt, N, C_in, H_in, W_in);

    _set_filter4d_oihw(w_desc, dt, /*O=*/C_in, /*I=*/C_out, K_h, K_w);

    KEYDNN_CUDNN_CHECK(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            /*pad_h=*/pad_h, /*pad_w=*/pad_w,
            /*stride_h=*/s_h, /*stride_w=*/s_w,
            /*dilation_h=*/1, /*dilation_w=*/1,
            CUDNN_CROSS_CORRELATION,
            dt
        )
    );

    const float one_f = 1.0f, zero_f = 0.0f;
    const double one_d = 1.0, zero_d = 0.0;
    const void* alpha = std::is_same<T, float>::value ? (const void*)&one_f : (const void*)&one_d;
    const void* beta0 = std::is_same<T, float>::value ? (const void*)&zero_f : (const void*)&zero_d;

    // ---- 1) grad_x = dL/dx ----
    {
        cudnnTensorDescriptor_t out_desc = gx_desc;
        cudnnTensorDescriptor_t in_desc = go_desc;

        cudnnConvolutionFwdAlgoPerf_t perf_fwd;
        int returned_fwd = 0;
        KEYDNN_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, in_desc, w_desc, conv_desc, out_desc, 1, &returned_fwd, &perf_fwd
        ));
        cudnnConvolutionFwdAlgo_t algo_fwd = perf_fwd.algo;

        size_t ws_fwd_bytes = 0;
        KEYDNN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, in_desc, w_desc, conv_desc, out_desc, algo_fwd, &ws_fwd_bytes
        ));
        void* ws_fwd = nullptr;
        if (ws_fwd_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws_fwd, ws_fwd_bytes));

        cudnnStatus_t s = cudnnConvolutionForward(
            handle,
            alpha,
            in_desc, grad_out,
            w_desc, w,
            conv_desc, algo_fwd,
            ws_fwd, ws_fwd_bytes,
            beta0,
            out_desc, grad_x
        );

        if (ws_fwd) cudaFree(ws_fwd);
        if (s != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(gx_desc);
            cudnnDestroyTensorDescriptor(go_desc);
            return _cudnn_to_int(s);
        }
    }

    // ---- 2) grad_w ----
    // We want grad_w in IOHW memory, but we expose it to cuDNN as OIHW with:
    //   O = C_in, I = C_out   (so memory matches your IOHW layout).
    //
    // Numerics note:
    // cudnnGetConvolutionBackwardFilterAlgorithm_v7 may pick a very fast algo whose
    // accumulation order can drift in float32 beyond tight unit-test tolerances.
    // Prefer a stable deterministic algo first, then fallback to v7 selection.
    {
        // IMPORTANT mapping to match your NumPy reference:
        //   xDesc  = grad_out  (N, C_out, H_out, W_out)  == I
        //   dyDesc = x         (N, C_in,  H_in,  W_in)   == O
        //
        // This produces grad_w with shape (O=C_in, I=C_out, Kh, Kw).

        cudnnStatus_t s_algo = CUDNN_STATUS_SUCCESS;

        // Try a deterministic / stable algorithm first.
        // (Most cuDNN versions support ALGO_0; if not, we fallback to v7.)
        cudnnConvolutionBwdFilterAlgo_t algo_dw = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

        size_t ws_dw_bytes = 0;
        s_algo = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, go_desc, x_desc, conv_desc, w_desc, algo_dw, &ws_dw_bytes
        );

        if (s_algo != CUDNN_STATUS_SUCCESS) {
            // Fallback: pick algo via v7 API.
            cudnnConvolutionBwdFilterAlgoPerf_t perf_dw;
            int returned_dw = 0;
            KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle,
                /*xDesc=*/go_desc,
                /*dyDesc=*/x_desc,
                conv_desc,
                w_desc,
                1, &returned_dw, &perf_dw
            ));
            algo_dw = perf_dw.algo;

            KEYDNN_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, go_desc, x_desc, conv_desc, w_desc, algo_dw, &ws_dw_bytes
            ));
        }

        void* ws_dw = nullptr;
        if (ws_dw_bytes > 0) KEYDNN_CUDA_CHECK(cudaMalloc(&ws_dw, ws_dw_bytes));

        cudnnStatus_t s = cudnnConvolutionBackwardFilter(
            handle,
            alpha,
            /*xDesc=*/go_desc, grad_out,
            /*dyDesc=*/x_desc, x,
            conv_desc, algo_dw,
            ws_dw, ws_dw_bytes,
            beta0,
            w_desc, grad_w
        );

        if (ws_dw) cudaFree(ws_dw);

        if (s != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(gx_desc);
            cudnnDestroyTensorDescriptor(go_desc);
            return _cudnn_to_int(s);
        }
    }




    // ---- 3) grad_b (optional) ----
    if (grad_b) {
        KEYDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));
        _set_tensor4d_nchw(b_desc, dt, 1, C_out, 1, 1);

        cudnnStatus_t s = cudnnConvolutionBackwardBias(
            handle,
            alpha,
            go_desc, grad_out,
            beta0,
            b_desc, grad_b
        );

        cudnnDestroyTensorDescriptor(b_desc);

        if (s != CUDNN_STATUS_SUCCESS) {
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroyFilterDescriptor(w_desc);
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(gx_desc);
            cudnnDestroyTensorDescriptor(go_desc);
            return _cudnn_to_int(s);
        }
    }

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(gx_desc);
    cudnnDestroyTensorDescriptor(go_desc);
    return 0;
}

// ----------------------------
// Exported C ABI functions
// ----------------------------

extern "C" int keydnn_cuda_conv2d_transpose_forward_f32(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return launch_conv2d_transpose_forward_cudnn<float>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

extern "C" int keydnn_cuda_conv2d_transpose_forward_f64(
    const double* x,
    const double* w,
    const double* b,
    double* y,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return launch_conv2d_transpose_forward_cudnn<double>(
        x, w, b, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

// NOTE: your original ABI for backward did NOT include grad_b.
// If you want bias grad on CUDA-native side, you can either:
//  (A) add new exported symbols *_backward_bias_* OR
//  (B) keep bias grad in Python (sum over grad_out), which you already do in some wrappers.
// Here we keep ABI compatible: compute grad_b only if caller passes a non-null pointer by adding
// a private extension would break ABI, so we omit it here.

extern "C" int keydnn_cuda_conv2d_transpose_backward_f32(
    const float* x,
    const float* w,
    const float* grad_out,
    float* grad_x,
    float* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return launch_conv2d_transpose_backward_cudnn<float>(
        x, w, grad_out,
        grad_x, grad_w,
        /*grad_b=*/nullptr,   // keep ABI
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}

extern "C" int keydnn_cuda_conv2d_transpose_backward_f64(
    const double* x,
    const double* w,
    const double* grad_out,
    double* grad_x,
    double* grad_w,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int s_h, int s_w,
    int pad_h, int pad_w
) {
    return launch_conv2d_transpose_backward_cudnn<double>(
        x, w, grad_out,
        grad_x, grad_w,
        /*grad_b=*/nullptr,   // keep ABI
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        s_h, s_w,
        pad_h, pad_w
    );
}
