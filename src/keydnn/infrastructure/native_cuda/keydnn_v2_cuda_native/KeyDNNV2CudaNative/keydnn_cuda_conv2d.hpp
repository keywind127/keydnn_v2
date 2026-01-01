#pragma once
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define KEYDNN_CUDA_API __declspec(dllexport)
#else
#define KEYDNN_CUDA_API
#endif

extern "C" {

    // Forward: y = conv(x_pad, w) + (b if not null)
    // x_pad: (N, C_in, H_pad, W_pad) contiguous NCHW on device
    // w:     (C_out, C_in, K_h, K_w) contiguous OIHW on device
    // b:     (C_out,) contiguous on device (may be nullptr)
    // y:     (N, C_out, H_out, W_out) contiguous NCHW on device
    KEYDNN_CUDA_API int keydnn_cuda_conv2d_forward_f32(
        const float* x_pad,
        const float* w,
        const float* b,  // may be nullptr
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
    );

    KEYDNN_CUDA_API int keydnn_cuda_conv2d_forward_f64(
        const double* x_pad,
        const double* w,
        const double* b,  // may be nullptr
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
    );

    // Backward: accumulates into grad_x_pad and grad_w
    // x_pad:      (N, C_in, H_pad, W_pad) NCHW on device
    // w:          (C_out, C_in, K_h, K_w) OIHW on device
    // grad_out:   (N, C_out, H_out, W_out) NCHW on device
    // grad_x_pad: (N, C_in, H_pad, W_pad) NCHW on device (accumulated; caller zero-init)
    // grad_w:     (C_out, C_in, K_h, K_w) OIHW on device (accumulated; caller zero-init)
    //
    // Notes:
    // - grad_b is NOT computed here; you already do sum(grad_out) on Python side.
    KEYDNN_CUDA_API int keydnn_cuda_conv2d_backward_f32(
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
    );

    KEYDNN_CUDA_API int keydnn_cuda_conv2d_backward_f64(
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
    );

} // extern "C"
