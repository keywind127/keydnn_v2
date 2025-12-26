// keydnn_cuda_stack.cu
#include "keydnn_cuda_stack.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>

// ----------------------------
// Error helper (C ABI returns int)
// ----------------------------
static inline int keydnn_cuda_ok(cudaError_t st) {
    return (st == cudaSuccess) ? 0 : static_cast<int>(st);
}

// ----------------------------
// Debug logging (opt-in, C ABI getters/setters)
// ----------------------------
static int g_keydnn_cuda_debug_enabled = 0;
static char g_keydnn_cuda_debug_last[2048] = { 0 };

static inline void keydnn_debug_clear() {
    if (!g_keydnn_cuda_debug_enabled) return;
    g_keydnn_cuda_debug_last[0] = '\0';
}

static inline void keydnn_debug_append(const char* fmt, ...) {
    if (!g_keydnn_cuda_debug_enabled) return;

    // find current end
    size_t cur = 0;
    while (cur < sizeof(g_keydnn_cuda_debug_last) && g_keydnn_cuda_debug_last[cur] != '\0') {
        ++cur;
    }
    if (cur >= sizeof(g_keydnn_cuda_debug_last) - 1) return;

    // add newline if not empty
    if (cur > 0 && g_keydnn_cuda_debug_last[cur - 1] != '\n') {
        if (cur < sizeof(g_keydnn_cuda_debug_last) - 1) {
            g_keydnn_cuda_debug_last[cur++] = '\n';
            g_keydnn_cuda_debug_last[cur] = '\0';
        }
    }

    va_list args;
    va_start(args, fmt);
#if defined(_WIN32)
    vsnprintf_s(
        g_keydnn_cuda_debug_last + cur,
        sizeof(g_keydnn_cuda_debug_last) - cur,
        _TRUNCATE,
        fmt,
        args
    );
#else
    vsnprintf(
        g_keydnn_cuda_debug_last + cur,
        sizeof(g_keydnn_cuda_debug_last) - cur,
        fmt,
        args
    );
#endif
    va_end(args);
}

// For compatibility with your existing call sites: "set" == clear + append
static inline void keydnn_debug_set(const char* fmt, ...) {
    if (!g_keydnn_cuda_debug_enabled) return;

    keydnn_debug_clear();

    va_list args;
    va_start(args, fmt);
#if defined(_WIN32)
    vsnprintf_s(
        g_keydnn_cuda_debug_last,
        sizeof(g_keydnn_cuda_debug_last),
        _TRUNCATE,
        fmt,
        args
    );
#else
    vsnprintf(g_keydnn_cuda_debug_last, sizeof(g_keydnn_cuda_debug_last), fmt, args);
#endif
    va_end(args);
}

KEYDNN_EXPORT void keydnn_cuda_debug_set_enabled(int enabled) {
    g_keydnn_cuda_debug_enabled = enabled ? 1 : 0;
    if (g_keydnn_cuda_debug_enabled) {
        keydnn_debug_clear();
        keydnn_debug_append("[KeyDNN CUDA] debug enabled");
    }
}

KEYDNN_EXPORT int keydnn_cuda_debug_get_last(char* buf, int buf_bytes) {
    if (!buf || buf_bytes <= 0) return 0;
    int i = 0;
    for (; i < buf_bytes - 1 && g_keydnn_cuda_debug_last[i] != '\0'; ++i) {
        buf[i] = g_keydnn_cuda_debug_last[i];
    }
    buf[i] = '\0';
    return i;
}

static inline int keydnn_cuda_fail(const char* where, cudaError_t st) {
    // IMPORTANT: append instead of overwrite, so we keep context + failure
    keydnn_debug_append(
        "[KeyDNN CUDA] %s failed: code=%d (%s)",
        where,
        (int)st,
        cudaGetErrorString(st)
    );
    return keydnn_cuda_ok(st);
}

// Returns 0 if OK; otherwise returns a CUDA error code (as int).
static inline int keydnn_debug_check_ptr(const void* p, const char* name) {
    if (!g_keydnn_cuda_debug_enabled) return 0;

    if (p == nullptr) {
        keydnn_debug_append("[KeyDNN CUDA] %s is NULL", name);
        return 0;
    }

    cudaPointerAttributes attr;
    cudaError_t st = cudaPointerGetAttributes(&attr, p);
    if (st != cudaSuccess) {
        keydnn_debug_append(
            "[KeyDNN CUDA] cudaPointerGetAttributes(%s=%p) failed: code=%d (%s)",
            name, p, (int)st, cudaGetErrorString(st)
        );
        return keydnn_cuda_ok(st);
    }

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000)
    const char* t =
        (attr.type == cudaMemoryTypeDevice) ? "device" :
        (attr.type == cudaMemoryTypeHost) ? "host" :
        (attr.type == cudaMemoryTypeManaged) ? "managed" :
        "unknown";
    keydnn_debug_append(
        "[KeyDNN CUDA] %s=%p type=%s device=%d",
        name, p, t, attr.device
    );
#else
    const char* t =
        (attr.memoryType == cudaMemoryTypeDevice) ? "device" :
        (attr.memoryType == cudaMemoryTypeHost) ? "host" :
        "unknown";
    keydnn_debug_append(
        "[KeyDNN CUDA] %s=%p memoryType=%s device=%d",
        name, p, t, attr.device
    );
#endif

    return 0;
}

// Returns 0 if pointer is device/managed (CUDA10+), else returns 1 (invalid value)
// or a CUDA error code if cudaPointerGetAttributes fails.
static inline int keydnn_debug_require_device_or_managed(const void* p, const char* name) {
    if (!g_keydnn_cuda_debug_enabled) return 0;

    if (p == nullptr) {
        keydnn_debug_append("[KeyDNN CUDA] %s is NULL (invalid device pointer)", name);
        return 1; // cudaErrorInvalidValue
    }

    cudaPointerAttributes a{};
    cudaError_t st = cudaPointerGetAttributes(&a, p);
    if (st != cudaSuccess) {
        return keydnn_cuda_fail("cudaPointerGetAttributes(require_device)", st);
    }

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000)
    const bool ok = (a.type == cudaMemoryTypeDevice) || (a.type == cudaMemoryTypeManaged);
#else
    const bool ok = (a.memoryType == cudaMemoryTypeDevice);
#endif

    if (!ok) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000)
        const char* t =
            (a.type == cudaMemoryTypeDevice) ? "device" :
            (a.type == cudaMemoryTypeHost) ? "host" :
            (a.type == cudaMemoryTypeManaged) ? "managed" :
            "unknown";
        keydnn_debug_append("[KeyDNN CUDA] %s=%p is not device/managed (type=%s) -> invalid launch arg", name, p, t);
#else
        const char* t =
            (a.memoryType == cudaMemoryTypeDevice) ? "device" :
            (a.memoryType == cudaMemoryTypeHost) ? "host" :
            "unknown";
        keydnn_debug_append("[KeyDNN CUDA] %s=%p is not device (memoryType=%s) -> invalid launch arg", name, p, t);
#endif
        return 1; // cudaErrorInvalidValue
    }

    return 0;
}

// ----------------------------
// Upload helper: uint64[K] host -> device
// ----------------------------
KEYDNN_EXPORT int keydnn_cuda_upload_u64_array(
    std::uint64_t* dst_dev_u64,
    const std::uint64_t* src_host_u64,
    std::int64_t K
) {
    if (!dst_dev_u64) return -1;
    if (!src_host_u64) return -2;
    if (K < 0) return -3;
    if (K == 0) return 0;

    keydnn_debug_append(
        "[upload_u64] dst_dev_u64=%p src_host_u64=%p K=%lld bytes=%llu",
        (void*)dst_dev_u64,
        (const void*)src_host_u64,
        (long long)K,
        (unsigned long long)(static_cast<size_t>(K) * sizeof(std::uint64_t))
    );

    cudaError_t st = cudaMemcpy(
        dst_dev_u64,
        src_host_u64,
        static_cast<size_t>(K) * sizeof(std::uint64_t),
        cudaMemcpyHostToDevice
    );
    if (st != cudaSuccess) return keydnn_cuda_fail("cudaMemcpy(H2D u64 array)", st);

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) return keydnn_cuda_fail("cudaDeviceSynchronize(upload_u64)", st);

    return 0;
}

// ----------------------------
// Stack forward kernel (gather) ¡X u64 pointer array
// ----------------------------
template <typename T>
__global__ void stack_fwd_u64_kernel(
    const std::uint64_t* __restrict__ xs_u64,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    T* __restrict__ y
) {
    const std::int64_t out_numel = pre * K * post;
    const std::int64_t idx =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
        static_cast<std::int64_t>(threadIdx.x);

    if (idx >= out_numel) return;

    const std::int64_t post_idx = idx % post;
    const std::int64_t tmp = idx / post;
    const std::int64_t k = tmp % K;
    const std::int64_t pre_idx = tmp / K;

    const std::int64_t in_idx = pre_idx * post + post_idx;

    const T* xk = reinterpret_cast<const T*>((uintptr_t)xs_u64[k]);
    y[idx] = xk[in_idx];
}

// ----------------------------
// Stack backward kernel (scatter) ¡X u64 pointer array
// ----------------------------
template <typename T>
__global__ void stack_bwd_u64_kernel(
    const T* __restrict__ dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* __restrict__ dxs_u64
) {
    const std::int64_t out_numel = pre * K * post;
    const std::int64_t idx =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
        static_cast<std::int64_t>(threadIdx.x);

    if (idx >= out_numel) return;

    const std::int64_t post_idx = idx % post;
    const std::int64_t tmp = idx / post;
    const std::int64_t k = tmp % K;
    const std::int64_t pre_idx = tmp / K;

    const std::int64_t in_idx = pre_idx * post + post_idx;

    T* dxk = reinterpret_cast<T*>((uintptr_t)dxs_u64[k]);
    dxk[in_idx] = dy[idx];
}

// ----------------------------
// Launch helpers
// ----------------------------
template <typename T>
static inline int stack_fwd_u64_impl(
    const std::uint64_t* xs_u64_dev,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    T* y
) {
    if (!xs_u64_dev) return -1;
    if (!y) return -2;
    if (K <= 0) return -3;
    if (pre < 0) return -4;
    if (post < 0) return -5;

    const std::int64_t out_numel = pre * K * post;
    if (out_numel == 0) return 0;

    keydnn_debug_append(
        "[stack_fwd_u64_impl] xs_u64_dev=%p y=%p K=%lld pre=%lld post=%lld out_numel=%lld",
        (const void*)xs_u64_dev,
        (void*)y,
        (long long)K,
        (long long)pre,
        (long long)post,
        (long long)out_numel
    );

    const int block = 256;
    const int grid = static_cast<int>((out_numel + block - 1) / block);

    stack_fwd_u64_kernel<T> << <grid, block >> > (xs_u64_dev, K, pre, post, y);

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_fail("stack_fwd_u64_kernel launch", st);

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) return keydnn_cuda_fail("stack_fwd_u64_kernel sync", st);

    return 0;
}

template <typename T>
static inline int stack_bwd_u64_impl(
    const T* dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* dxs_u64_dev
) {
    if (!dy) return -1;
    if (!dxs_u64_dev) return -2;
    if (K <= 0) return -3;
    if (pre < 0) return -4;
    if (post < 0) return -5;

    const std::int64_t out_numel = pre * K * post;
    if (out_numel == 0) return 0;

    const int block = 256;
    const int grid = static_cast<int>((out_numel + block - 1) / block);

    // ---- Debug pointer validation (host-side, pre-launch) ----
    if (g_keydnn_cuda_debug_enabled) {
        int stp = 0;

        // log pointer kinds
        stp = keydnn_debug_check_ptr((const void*)dy, "dy");
        if (stp != 0) return stp;

        stp = keydnn_debug_check_ptr((const void*)dxs_u64_dev, "dxs_u64_dev");
        if (stp != 0) return stp;

        // enforce dy and dxs_u64_dev must be device/managed
        stp = keydnn_debug_require_device_or_managed((const void*)dy, "dy");
        if (stp != 0) return stp;

        stp = keydnn_debug_require_device_or_managed((const void*)dxs_u64_dev, "dxs_u64_dev");
        if (stp != 0) return stp;

        const int n = (K < 4) ? (int)K : 4;
        std::uint64_t host_u64[4] = { 0, 0, 0, 0 };

        cudaError_t st = cudaMemcpy(
            host_u64,
            dxs_u64_dev,
            sizeof(std::uint64_t) * (size_t)n,
            cudaMemcpyDeviceToHost
        );
        if (st != cudaSuccess) return keydnn_cuda_fail("memcpy dxs_u64_dev D2H sample", st);

        for (int i = 0; i < n; ++i) {
            const void* p = (const void*)(uintptr_t)host_u64[i];

            // log pointer kind
            int stx = keydnn_debug_check_ptr(p, "dxs_u64[i]");
            if (stx != 0) return stx;

            // enforce each sampled dx pointer must be device/managed
            stx = keydnn_debug_require_device_or_managed(p, "dxs_u64[i]");
            if (stx != 0) return stx;
        }

        keydnn_debug_append(
            "[stack_bwd_u64_impl] pointers OK | dy=%p dxs_u64_dev=%p K=%lld pre=%lld post=%lld out_numel=%lld grid=%d block=%d sample0=0x%llx",
            (const void*)dy, (const void*)dxs_u64_dev,
            (long long)K, (long long)pre, (long long)post, (long long)out_numel,
            grid, block,
            (unsigned long long)host_u64[0]
        );

        // Clear stale error so a subsequent "invalid argument" is definitely from THIS launch
        cudaGetLastError();
    }
    // ---- End debug validation ----

    stack_bwd_u64_kernel<T> << <grid, block >> > (dy, K, pre, post, dxs_u64_dev);

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return keydnn_cuda_fail("stack_bwd_u64_kernel launch", st);

    st = cudaDeviceSynchronize();
    if (st != cudaSuccess) return keydnn_cuda_fail("stack_bwd_u64_kernel sync", st);

    return 0;
}

// ----------------------------
// Exported C ABI
// ----------------------------
KEYDNN_EXPORT int keydnn_cuda_stack_fwd_u64_f32(
    const std::uint64_t* xs_u64_dev,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    float* y
) {
    return stack_fwd_u64_impl<float>(xs_u64_dev, K, pre, post, y);
}

KEYDNN_EXPORT int keydnn_cuda_stack_fwd_u64_f64(
    const std::uint64_t* xs_u64_dev,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    double* y
) {
    return stack_fwd_u64_impl<double>(xs_u64_dev, K, pre, post, y);
}

KEYDNN_EXPORT int keydnn_cuda_stack_bwd_u64_f32(
    const float* dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* dxs_u64_dev
) {
    return stack_bwd_u64_impl<float>(dy, K, pre, post, dxs_u64_dev);
}

KEYDNN_EXPORT int keydnn_cuda_stack_bwd_u64_f64(
    const double* dy,
    std::int64_t K,
    std::int64_t pre,
    std::int64_t post,
    const std::uint64_t* dxs_u64_dev
) {
    return stack_bwd_u64_impl<double>(dy, K, pre, post, dxs_u64_dev);
}
