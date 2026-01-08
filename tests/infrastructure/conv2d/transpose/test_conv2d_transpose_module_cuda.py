# tests/infrastructure/convolution/transpose/test_conv2d_transpose_module_cuda.py
"""
CUDA unit tests for Conv2dTranspose module.

These tests mirror the existing CPU module tests, but allocate inputs/params on CUDA
so the module dispatches through Conv2dTransposeFn -> conv2d_transpose_cuda_ext.

We validate:
- Forward shape matches CPU reference kernel shape.
- Forward numerics (CUDA -> host) match CPU reference kernel.
- Backward via autograd (loss = y.sum()) matches CPU reference backward under grad_out=ones.
- No-bias behavior.
- output_padding wiring.

Notes
-----
- Requires KeyDNN CUDA native DLL and memcpy symbols.
- We compare CUDA results against CPU reference kernels:
  `conv2d_transpose_forward_cpu` and `conv2d_transpose_backward_cpu`.
- CUDA backward may use atomics -> float32 tolerances are looser.
"""

from __future__ import annotations

import ctypes
import unittest
from typing import Tuple, Optional

import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure.tensor._tensor import Tensor
from src.keydnn.infrastructure.convolution.transpose._conv2d_transpose_module import (
    Conv2dTranspose,
)
from src.keydnn.infrastructure.ops.conv2d_transpose_cpu import (
    conv2d_transpose_forward_cpu,
    conv2d_transpose_backward_cpu,
)

from src.keydnn.infrastructure.ops.pool2d_cuda import (
    _load_cuda_lib,
    cuda_set_device,
    cuda_malloc,
    cuda_free,
)


# ---------------------------
# CUDA test utilities
# ---------------------------
def _get_cuda_device(index: int = 0) -> Device:
    """
    Best-effort helper to obtain a CUDA Device instance across possible Device APIs.
    """
    if hasattr(Device, "cuda") and callable(getattr(Device, "cuda")):
        return Device.cuda(index)  # type: ignore[attr-defined]

    try:
        return Device("cuda", index)  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(f"cuda:{index}")  # type: ignore[call-arg]
    except Exception:
        pass

    try:
        return Device(kind="cuda", index=index)  # type: ignore[call-arg]
    except Exception as e:
        raise RuntimeError(
            "Unable to construct a CUDA Device; update _get_cuda_device() for this repo."
        ) from e


def _bind_memcpy_symbols(lib: ctypes.CDLL) -> Tuple[ctypes._CFuncPtr, ctypes._CFuncPtr]:
    """
    Bind keydnn_cuda_memcpy_h2d and keydnn_cuda_memcpy_d2h from the CUDA DLL.
    """
    if not hasattr(lib, "keydnn_cuda_memcpy_h2d"):
        raise AttributeError("CUDA DLL missing symbol: keydnn_cuda_memcpy_h2d")
    if not hasattr(lib, "keydnn_cuda_memcpy_d2h"):
        raise AttributeError("CUDA DLL missing symbol: keydnn_cuda_memcpy_d2h")

    h2d = lib.keydnn_cuda_memcpy_h2d
    h2d.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    h2d.restype = ctypes.c_int

    d2h = lib.keydnn_cuda_memcpy_d2h
    d2h.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    d2h.restype = ctypes.c_int

    return h2d, d2h


def _h2d(lib: ctypes.CDLL, dst_dev: int, src: np.ndarray) -> None:
    src_c = np.ascontiguousarray(src)
    h2d, _ = _bind_memcpy_symbols(lib)
    st = int(
        h2d(
            ctypes.c_uint64(int(dst_dev)),
            ctypes.c_void_p(src_c.ctypes.data),
            ctypes.c_size_t(src_c.nbytes),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_h2d failed with status={st}")


def _d2h(lib: ctypes.CDLL, src_dev: int, dst: np.ndarray) -> None:
    dst_c = np.ascontiguousarray(dst)
    _, d2h = _bind_memcpy_symbols(lib)
    st = int(
        d2h(
            ctypes.c_void_p(dst_c.ctypes.data),
            ctypes.c_uint64(int(src_dev)),
            ctypes.c_size_t(dst_c.nbytes),
        )
    )
    if st != 0:
        raise RuntimeError(f"keydnn_cuda_memcpy_d2h failed with status={st}")


def _tensor_from_numpy_cuda(
    arr: np.ndarray, *, lib: ctypes.CDLL, cuda_device: Device, requires_grad: bool
) -> Tensor:
    """
    Allocate CUDA storage for `arr` and return a CUDA Tensor wrapping it.
    """
    a = np.ascontiguousarray(arr)
    nbytes = int(a.nbytes)
    dev_ptr = int(cuda_malloc(lib, nbytes if nbytes > 0 else 1))
    try:
        if nbytes > 0:
            _h2d(lib, dev_ptr, a)
    except Exception:
        cuda_free(lib, dev_ptr)
        raise

    return Tensor._from_devptr(
        dev_ptr,
        shape=tuple(int(d) for d in a.shape),
        dtype=a.dtype,
        device=cuda_device,
        requires_grad=requires_grad,
    )


def _to_numpy_from_cuda(t: Tensor, *, lib: ctypes.CDLL) -> np.ndarray:
    """
    Copy CUDA Tensor -> host NumPy array via D2H.
    """
    out = np.empty(tuple(int(d) for d in t.shape), dtype=np.dtype(t.dtype))
    if int(out.nbytes) > 0:
        _d2h(lib, int(t.data), out)
    return out


def _unwrap_param_tensor(p):
    """
    Return the underlying Tensor for a Parameter-like object.

    Supports:
      - Parameter is Tensor-like (has to_numpy/copy_from_numpy)
      - Parameter wraps Tensor in `.data` or `.tensor`
    """
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


def _copy_host_into_cuda_tensor(
    t: Tensor, host: np.ndarray, *, lib: ctypes.CDLL
) -> None:
    """
    Overwrite an existing CUDA Tensor's device buffer with `host` (H2D).
    """
    h = np.ascontiguousarray(host, dtype=np.dtype(t.dtype))
    if tuple(int(d) for d in h.shape) != tuple(int(d) for d in t.shape):
        raise ValueError(
            f"shape mismatch in _copy_host_into_cuda_tensor: {h.shape} vs {t.shape}"
        )
    if int(h.nbytes) > 0:
        _h2d(lib, int(t.data), h)


def _tol(dtype: np.dtype) -> tuple[float, float]:
    """
    Tolerances for comparing CUDA module path vs CPU reference kernels.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        # backward may use atomics -> looser
        return 3e-4, 3e-4
    if dtype == np.float64:
        return 1e-10, 1e-10
    return 3e-4, 3e-4


class TestConv2dTransposeModuleCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Load CUDA DLL; skip if unavailable.
        try:
            cls.lib = _load_cuda_lib()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL not available: {e}") from e

        cls.device_index = 0
        try:
            cuda_set_device(cls.lib, cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Failed to set CUDA device: {e}") from e

        # Ensure memcpy symbols exist; if not, skip (cannot validate numerics).
        try:
            _bind_memcpy_symbols(cls.lib)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA memcpy symbols unavailable: {e}") from e

        # Construct CUDA device object
        try:
            cls.cuda_device = _get_cuda_device(cls.device_index)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to construct CUDA Device: {e}") from e

    def test_conv2d_transpose_module_forward_shape_cuda(self) -> None:
        deconv = Conv2dTranspose(
            in_channels=3,
            out_channels=5,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 2),
            output_padding=(0, 0),
            bias=True,
            device=self.cuda_device,
        )

        x_np = np.random.randn(2, 3, 10, 9).astype(np.float32)
        x = _tensor_from_numpy_cuda(
            x_np, lib=self.lib, cuda_device=self.cuda_device, requires_grad=False
        )

        y = deconv.forward(x)

        # Reference shape from CPU op (use host params)
        w_np = _unwrap_param_tensor(deconv.weight).to_numpy()  # IOHW
        b_np = (
            _unwrap_param_tensor(deconv.bias).to_numpy()
            if deconv.bias is not None
            else None
        )
        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=(2, 1),
            padding=(1, 2),
            output_padding=(0, 0),
        )
        self.assertEqual(y.shape, y_ref.shape)

    def test_conv2d_transpose_module_forward_matches_cpu(self) -> None:
        rng = np.random.default_rng(0)
        dtype = np.float32

        deconv = Conv2dTranspose(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 2),
            stride=(1, 2),
            padding=(1, 0),
            output_padding=(0, 0),
            bias=True,
            device=self.cuda_device,
        )

        # Deterministic params (IOHW)
        w_np = rng.standard_normal((2, 3, 3, 2), dtype=dtype)
        b_np = rng.standard_normal((3,), dtype=dtype)

        w_t = _unwrap_param_tensor(deconv.weight)
        b_t = _unwrap_param_tensor(deconv.bias)

        # Ensure CUDA param storage is overwritten with our deterministic values
        _copy_host_into_cuda_tensor(w_t, w_np, lib=self.lib)
        _copy_host_into_cuda_tensor(b_t, b_np, lib=self.lib)

        # Input
        x_np = rng.standard_normal((1, 2, 5, 4), dtype=dtype)
        x = _tensor_from_numpy_cuda(
            x_np, lib=self.lib, cuda_device=self.cuda_device, requires_grad=False
        )

        y = deconv.forward(x)
        y_host = _to_numpy_from_cuda(y, lib=self.lib)

        y_ref = conv2d_transpose_forward_cpu(
            x_np, w_np, b_np, stride=(1, 2), padding=(1, 0), output_padding=(0, 0)
        )

        rtol, atol = _tol(dtype)
        self.assertEqual(y_host.shape, y_ref.shape)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)

    def test_conv2d_transpose_module_backward_autograd_matches_cpu_cuda(self) -> None:
        """
        Deterministically set weight/bias, run:
          y = deconv(x)
          loss = y.sum()
          loss.backward()
        Compare x.grad and parameter grads with CPU reference under grad_out=ones.
        """
        rng = np.random.default_rng(42)
        dtype = np.float32

        deconv = Conv2dTranspose(
            in_channels=2,
            out_channels=3,
            kernel_size=(3, 2),
            stride=(1, 2),
            padding=(1, 0),
            output_padding=(0, 0),
            bias=True,
            device=self.cuda_device,
        )

        # Deterministic parameters (IOHW)
        w_np = rng.standard_normal((2, 3, 3, 2), dtype=dtype)
        b_np = rng.standard_normal((3,), dtype=dtype)

        w_t = _unwrap_param_tensor(deconv.weight)
        b_t = _unwrap_param_tensor(deconv.bias)
        _copy_host_into_cuda_tensor(w_t, w_np, lib=self.lib)
        _copy_host_into_cuda_tensor(b_t, b_np, lib=self.lib)

        # Input
        x_np = rng.standard_normal((1, 2, 5, 4), dtype=dtype)
        x = _tensor_from_numpy_cuda(
            x_np, lib=self.lib, cuda_device=self.cuda_device, requires_grad=True
        )

        # Forward + backward
        y = deconv.forward(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad, "x.grad should be populated after backward()")

        w_holder = _unwrap_param_tensor(deconv.weight)
        b_holder = _unwrap_param_tensor(deconv.bias)
        self.assertIsNotNone(w_holder.grad, "weight.grad should be populated")
        self.assertIsNotNone(b_holder.grad, "bias.grad should be populated")

        # CPU reference under L = sum(y) -> grad_out = ones
        y_ref = conv2d_transpose_forward_cpu(
            x_np, w_np, b_np, stride=(1, 2), padding=(1, 0), output_padding=(0, 0)
        )
        grad_out_np = np.ones_like(y_ref, dtype=dtype)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            b_np,
            grad_out_np,
            stride=(1, 2),
            padding=(1, 0),
            output_padding=(0, 0),
        )

        gx = _to_numpy_from_cuda(x.grad, lib=self.lib)  # type: ignore[arg-type]
        gw = _to_numpy_from_cuda(w_holder.grad, lib=self.lib)  # type: ignore[arg-type]
        gb = _to_numpy_from_cuda(b_holder.grad, lib=self.lib)  # type: ignore[arg-type]

        rtol, atol = _tol(dtype)
        np.testing.assert_allclose(gx, gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gw, gw_ref, rtol=rtol, atol=atol)
        # bias grad is reduction -> can be a bit tighter
        np.testing.assert_allclose(gb, gb_ref, rtol=1e-4, atol=1e-4)

    def test_conv2d_transpose_module_no_bias_cuda(self) -> None:
        rng = np.random.default_rng(7)
        dtype = np.float32

        deconv = Conv2dTranspose(
            in_channels=2,
            out_channels=2,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 0),
            bias=False,
            device=self.cuda_device,
        )
        self.assertIsNone(deconv.bias)

        x_np = rng.standard_normal((1, 2, 4, 3), dtype=dtype)
        x = _tensor_from_numpy_cuda(
            x_np, lib=self.lib, cuda_device=self.cuda_device, requires_grad=True
        )

        # Deterministic weight (IOHW)
        w_np = rng.standard_normal((2, 2, 3, 3), dtype=dtype)
        _copy_host_into_cuda_tensor(
            _unwrap_param_tensor(deconv.weight), w_np, lib=self.lib
        )

        y = deconv.forward(x)
        y.sum().backward()

        self.assertIsNotNone(x.grad)
        w_holder = _unwrap_param_tensor(deconv.weight)
        self.assertIsNotNone(w_holder.grad)

        # CPU reference (bias None)
        y_ref = conv2d_transpose_forward_cpu(
            x_np, w_np, None, stride=(1, 2), padding=(1, 1), output_padding=(0, 0)
        )
        grad_out_np = np.ones_like(y_ref, dtype=dtype)

        gx_ref, gw_ref, gb_ref = conv2d_transpose_backward_cpu(
            x_np,
            w_np,
            None,
            grad_out_np,
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 0),
        )

        self.assertIsNone(gb_ref)
        gx = _to_numpy_from_cuda(x.grad, lib=self.lib)  # type: ignore[arg-type]
        gw = _to_numpy_from_cuda(w_holder.grad, lib=self.lib)  # type: ignore[arg-type]

        rtol, atol = _tol(dtype)
        np.testing.assert_allclose(gx, gx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(gw, gw_ref, rtol=rtol, atol=atol)

    def test_conv2d_transpose_module_output_padding_shape_cuda(self) -> None:
        """
        Ensure module wires output_padding through to op and produces the same shape as reference.
        """
        rng = np.random.default_rng(1234)
        dtype = np.float32

        deconv = Conv2dTranspose(
            in_channels=1,
            out_channels=2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 0),
            bias=True,
            device=self.cuda_device,
        )

        x_np = rng.standard_normal((1, 1, 4, 5)).astype(dtype)
        x = _tensor_from_numpy_cuda(
            x_np, lib=self.lib, cuda_device=self.cuda_device, requires_grad=False
        )

        w_np = rng.standard_normal((1, 2, 3, 3)).astype(dtype)
        b_np = rng.standard_normal((2,)).astype(dtype)
        _copy_host_into_cuda_tensor(
            _unwrap_param_tensor(deconv.weight), w_np, lib=self.lib
        )
        _copy_host_into_cuda_tensor(
            _unwrap_param_tensor(deconv.bias), b_np, lib=self.lib
        )

        y = deconv.forward(x)
        y_host = _to_numpy_from_cuda(y, lib=self.lib)

        y_ref = conv2d_transpose_forward_cpu(
            x_np,
            w_np,
            b_np,
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 0),
        )

        rtol, atol = _tol(dtype)
        self.assertEqual(y_host.shape, y_ref.shape)
        np.testing.assert_allclose(y_host, y_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
