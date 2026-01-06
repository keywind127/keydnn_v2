import os
import sys
import unittest
from typing import Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError(
        "python-dotenv is required to load .env for native tests. "
        "Install it with: pip install python-dotenv"
    ) from e

# Load repo_root/.env by walking upward from this test file.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# If using MinGW and the produced DLL depends on libstdc++/libgcc,
# prepend MinGW bin to PATH so ctypes can load dependencies.
MINGW_BIN = os.getenv("KEYDNN_MINGW_BIN")
if sys.platform.startswith("win") and MINGW_BIN:
    os.environ["PATH"] = MINGW_BIN + os.pathsep + os.environ.get("PATH", "")

# Import your ctypes wrapper.
# Adjust import path if your module name differs.
from src.keydnn.infrastructure.native.python.conv2d_transpose_ctypes import (
    load_keydnn_native,
    conv2d_transpose_forward_f32_ctypes,
    conv2d_transpose_backward_f32_ctypes,
)


def _ref_conv2d_transpose_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: Optional[np.ndarray],
    *,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    """
    Reference ConvTranspose2d forward matching the C++ kernel contract.

    Layouts:
    - x: (N, C_in,  H_in,  W_in)  NCHW
    - w: (C_in, C_out, K_h, K_w)  IOHW
    - b: (C_out,) or None
    - y: (N, C_out, H_out, W_out) NCHW

    Scatter-style:
      out_h = h_in * s_h + kh - pad_h
      out_w = w_in * s_w + kw - pad_w
      y[n, co, out_h, out_w] += x[n, ci, h_in, w_in] * w[ci, co, kh, kw]
    """
    y = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                for wi in range(W_in):
                    xv = x[n, ci, hi, wi]
                    base_oh = hi * s_h - pad_h
                    base_ow = wi * s_w - pad_w

                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                y[n, co, oh, ow] += xv * w[ci, co, kh, kw]

        if b is not None:
            # Bias add per (n, co)
            for co in range(C_out):
                y[n, co, :, :] += b[co]

    return y


def _ref_conv2d_transpose_backward(
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    N: int,
    C_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    H_out: int,
    W_out: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference ConvTranspose2d backward matching the C++ kernel contract.

    Returns:
    - grad_x: (N, C_in, H_in, W_in)
    - grad_w: (C_in, C_out, K_h, K_w)
    """
    grad_x = np.zeros((N, C_in, H_in, W_in), dtype=x.dtype)
    grad_w = np.zeros((C_in, C_out, K_h, K_w), dtype=x.dtype)

    for n in range(N):
        for ci in range(C_in):
            for hi in range(H_in):
                for wi in range(W_in):
                    xv = x[n, ci, hi, wi]
                    base_oh = hi * s_h - pad_h
                    base_ow = wi * s_w - pad_w

                    for co in range(C_out):
                        for kh in range(K_h):
                            oh = base_oh + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(K_w):
                                ow = base_ow + kw
                                if ow < 0 or ow >= W_out:
                                    continue

                                go = grad_out[n, co, oh, ow]
                                grad_x[n, ci, hi, wi] += go * w[ci, co, kh, kw]
                                grad_w[ci, co, kh, kw] += xv * go

    return grad_x, grad_w


def _out_size_transpose(
    H_in: int,
    W_in: int,
    K_h: int,
    K_w: int,
    s_h: int,
    s_w: int,
    pad_h: int,
    pad_w: int,
    out_pad_h: int,
    out_pad_w: int,
) -> Tuple[int, int]:
    """
    Standard ConvTranspose2d output size formula (no dilation):
      H_out = (H_in - 1) * s_h - 2*pad_h + K_h + out_pad_h
      W_out = (W_in - 1) * s_w - 2*pad_w + K_w + out_pad_w
    """
    H_out = (H_in - 1) * s_h - 2 * pad_h + K_h + out_pad_h
    W_out = (W_in - 1) * s_w - 2 * pad_w + K_w + out_pad_w
    return H_out, W_out


class TestConv2DTransposeCtypesF32(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lib = load_keydnn_native()

    def test_forward_matches_reference_small_with_bias(self) -> None:
        rng = np.random.default_rng(0)

        N = 2
        C_in, C_out = 3, 4
        H_in, W_in = 5, 6
        K_h, K_w = 3, 2
        s_h, s_w = 2, 1
        pad_h, pad_w = 1, 0
        out_pad_h, out_pad_w = 0, 0

        H_out, W_out = _out_size_transpose(
            H_in, W_in, K_h, K_w, s_h, s_w, pad_h, pad_w, out_pad_h, out_pad_w
        )

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)
        b = rng.standard_normal((C_out,), dtype=np.float32)

        # Ensure deterministic check around bounds:
        x[0, 0, 0, 0] = 2.0
        w[0, 0, 0, 0] = 3.0

        y_cpp = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        conv2d_transpose_forward_f32_ctypes(
            self.lib,
            x=x,
            w=w,
            b=b,
            y=y_cpp,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        y_ref = _ref_conv2d_transpose_forward(
            x,
            w,
            b,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        np.testing.assert_allclose(y_cpp, y_ref, rtol=0, atol=0)

    def test_forward_matches_reference_small_no_bias(self) -> None:
        rng = np.random.default_rng(1)

        N = 1
        C_in, C_out = 2, 2
        H_in, W_in = 4, 4
        K_h, K_w = 3, 3
        s_h, s_w = 1, 2
        pad_h, pad_w = 1, 1
        out_pad_h, out_pad_w = 0, 0

        H_out, W_out = _out_size_transpose(
            H_in, W_in, K_h, K_w, s_h, s_w, pad_h, pad_w, out_pad_h, out_pad_w
        )

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)

        y_cpp = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        conv2d_transpose_forward_f32_ctypes(
            self.lib,
            x=x,
            w=w,
            b=None,
            y=y_cpp,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        y_ref = _ref_conv2d_transpose_forward(
            x,
            w,
            None,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        np.testing.assert_allclose(y_cpp, y_ref, rtol=0, atol=0)

    def test_backward_matches_reference_small(self) -> None:
        rng = np.random.default_rng(2)

        N = 2
        C_in, C_out = 2, 3
        H_in, W_in = 4, 5
        K_h, K_w = 2, 3
        s_h, s_w = 2, 2
        pad_h, pad_w = 0, 1
        out_pad_h, out_pad_w = 0, 0

        H_out, W_out = _out_size_transpose(
            H_in, W_in, K_h, K_w, s_h, s_w, pad_h, pad_w, out_pad_h, out_pad_w
        )

        x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
        w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)
        grad_out = rng.standard_normal((N, C_out, H_out, W_out), dtype=np.float32)

        grad_x_cpp = np.zeros((N, C_in, H_in, W_in), dtype=np.float32)
        grad_w_cpp = np.zeros((C_in, C_out, K_h, K_w), dtype=np.float32)

        conv2d_transpose_backward_f32_ctypes(
            self.lib,
            x=x,
            w=w,
            grad_out=grad_out,
            grad_x=grad_x_cpp,
            grad_w=grad_w_cpp,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        grad_x_ref, grad_w_ref = _ref_conv2d_transpose_backward(
            x,
            w,
            grad_out,
            N=N,
            C_in=C_in,
            H_in=H_in,
            W_in=W_in,
            C_out=C_out,
            H_out=H_out,
            W_out=W_out,
            K_h=K_h,
            K_w=K_w,
            s_h=s_h,
            s_w=s_w,
            pad_h=pad_h,
            pad_w=pad_w,
        )

        np.testing.assert_allclose(grad_x_cpp, grad_x_ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad_w_cpp, grad_w_ref, rtol=1e-6, atol=1e-5)

    def test_randomized_shapes_forward_and_backward(self) -> None:
        rng = np.random.default_rng(123)

        for _ in range(10):
            N = int(rng.integers(1, 4))
            C_in = int(rng.integers(1, 5))
            C_out = int(rng.integers(1, 6))
            H_in = int(rng.integers(2, 8))
            W_in = int(rng.integers(2, 8))

            K_h = int(rng.integers(1, 5))
            K_w = int(rng.integers(1, 5))

            s_h = int(rng.integers(1, 4))
            s_w = int(rng.integers(1, 4))

            pad_h = int(rng.integers(0, K_h))  # typical range
            pad_w = int(rng.integers(0, K_w))

            out_pad_h = int(rng.integers(0, min(s_h, 2)))  # keep tiny
            out_pad_w = int(rng.integers(0, min(s_w, 2)))

            H_out, W_out = _out_size_transpose(
                H_in, W_in, K_h, K_w, s_h, s_w, pad_h, pad_w, out_pad_h, out_pad_w
            )
            if H_out <= 0 or W_out <= 0:
                continue

            x = rng.standard_normal((N, C_in, H_in, W_in), dtype=np.float32)
            w = rng.standard_normal((C_in, C_out, K_h, K_w), dtype=np.float32)

            use_bias = bool(rng.integers(0, 2))
            b = rng.standard_normal((C_out,), dtype=np.float32) if use_bias else None

            # Forward
            y_cpp = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
            conv2d_transpose_forward_f32_ctypes(
                self.lib,
                x=x,
                w=w,
                b=b,
                y=y_cpp,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )

            y_ref = _ref_conv2d_transpose_forward(
                x,
                w,
                b,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )

            np.testing.assert_allclose(y_cpp, y_ref, rtol=0, atol=0)

            # Backward
            grad_out = rng.standard_normal((N, C_out, H_out, W_out), dtype=np.float32)
            grad_x_cpp = np.zeros((N, C_in, H_in, W_in), dtype=np.float32)
            grad_w_cpp = np.zeros((C_in, C_out, K_h, K_w), dtype=np.float32)

            conv2d_transpose_backward_f32_ctypes(
                self.lib,
                x=x,
                w=w,
                grad_out=grad_out,
                grad_x=grad_x_cpp,
                grad_w=grad_w_cpp,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )

            grad_x_ref, grad_w_ref = _ref_conv2d_transpose_backward(
                x,
                w,
                grad_out,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )

            np.testing.assert_allclose(grad_x_cpp, grad_x_ref, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(grad_w_cpp, grad_w_ref, rtol=1e-6, atol=1e-5)

    def test_rejects_wrong_dtypes(self) -> None:
        # forward f32 should reject float64 x
        N, C_in, C_out = 1, 1, 1
        H_in, W_in = 3, 3
        K_h, K_w = 3, 3
        s_h, s_w = 1, 1
        pad_h, pad_w = 1, 1
        out_pad_h, out_pad_w = 0, 0
        H_out, W_out = _out_size_transpose(
            H_in, W_in, K_h, K_w, s_h, s_w, pad_h, pad_w, out_pad_h, out_pad_w
        )

        x_f64 = np.zeros((N, C_in, H_in, W_in), dtype=np.float64)
        w_f32 = np.zeros((C_in, C_out, K_h, K_w), dtype=np.float32)
        y_f32 = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

        with self.assertRaises(TypeError):
            conv2d_transpose_forward_f32_ctypes(
                self.lib,
                x=x_f64,  # wrong
                w=w_f32,
                b=None,
                y=y_f32,
                N=N,
                C_in=C_in,
                H_in=H_in,
                W_in=W_in,
                C_out=C_out,
                H_out=H_out,
                W_out=W_out,
                K_h=K_h,
                K_w=K_w,
                s_h=s_h,
                s_w=s_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )


if __name__ == "__main__":
    unittest.main()
