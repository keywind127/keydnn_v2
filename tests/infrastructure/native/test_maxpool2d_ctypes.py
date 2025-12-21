import os
import sys
import unittest
from typing import Tuple

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
# Adjust the import path if your module name differs.
from keydnn.infrastructure.native.python.maxpool2d_ctypes import (
    load_keydnn_native,
    maxpool2d_forward_f32_ctypes,
)


def _ref_maxpool2d_forward_from_xpad(
    x_pad: np.ndarray,
    *,
    N: int,
    C: int,
    H_pad: int,
    W_pad: int,
    H_out: int,
    W_out: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference implementation that matches the C++ kernel's contract:
    - Takes already-padded x_pad (N, C, H_pad, W_pad) float32
    - Returns y float32 and argmax_idx int64
    - argmax_idx stores flattened spatial index (h * W_pad + w) in padded plane
    """
    y = np.empty((N, C, H_out, W_out), dtype=np.float32)
    argmax_idx = np.empty((N, C, H_out, W_out), dtype=np.int64)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                h0 = i * s_h
                for j in range(W_out):
                    w0 = j * s_w

                    patch = x_pad[n, c, h0 : h0 + k_h, w0 : w0 + k_w]
                    flat_idx = int(np.argmax(patch))  # row-major tie-break

                    y[n, c, i, j] = patch.reshape(-1)[flat_idx]

                    ph = flat_idx // k_w
                    pw = flat_idx % k_w
                    h = h0 + ph
                    w = w0 + pw
                    argmax_idx[n, c, i, j] = h * W_pad + w

    return y, argmax_idx


class TestMaxPool2DCtypesForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Loads DLL/so/dylib from the same directory as the wrapper module by default.
        cls.lib = load_keydnn_native()

    def test_forward_matches_reference_small(self) -> None:
        # Small, deterministic case
        rng = np.random.default_rng(0)

        N, C = 2, 3
        H_pad, W_pad = 7, 8
        k_h, k_w = 2, 3
        s_h, s_w = 2, 2

        # Compute output sizes for "valid" windows within the padded plane
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        x_pad = rng.standard_normal((N, C, H_pad, W_pad), dtype=np.float32)

        # Ensure we have some ties to validate deterministic argmax behavior
        x_pad[0, 0, 0, 0] = 5.0
        x_pad[0, 0, 0, 1] = 5.0  # tie in same window; argmax should pick first (0,0)

        y_cpp = np.empty((N, C, H_out, W_out), dtype=np.float32)
        idx_cpp = np.empty((N, C, H_out, W_out), dtype=np.int64)

        maxpool2d_forward_f32_ctypes(
            self.lib,
            x_pad=x_pad,
            y=y_cpp,
            argmax_idx=idx_cpp,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

        y_ref, idx_ref = _ref_maxpool2d_forward_from_xpad(
            x_pad,
            N=N,
            C=C,
            H_pad=H_pad,
            W_pad=W_pad,
            H_out=H_out,
            W_out=W_out,
            k_h=k_h,
            k_w=k_w,
            s_h=s_h,
            s_w=s_w,
        )

        np.testing.assert_allclose(y_cpp, y_ref, rtol=0, atol=0)
        np.testing.assert_array_equal(idx_cpp, idx_ref)

        # Extra checks: idx range is valid
        self.assertTrue(np.all(idx_cpp >= 0))
        self.assertTrue(np.all(idx_cpp < H_pad * W_pad))

    def test_forward_randomized_shapes(self) -> None:
        # Multiple random trials, varying shapes/strides/kernel sizes
        rng = np.random.default_rng(123)

        for _ in range(10):
            N = int(rng.integers(1, 4))
            C = int(rng.integers(1, 5))
            H_pad = int(rng.integers(4, 11))
            W_pad = int(rng.integers(4, 11))

            k_h = int(rng.integers(1, min(5, H_pad) + 1))
            k_w = int(rng.integers(1, min(5, W_pad) + 1))

            s_h = int(rng.integers(1, 4))
            s_w = int(rng.integers(1, 4))

            # Ensure at least one output element
            if H_pad < k_h or W_pad < k_w:
                continue

            H_out = (H_pad - k_h) // s_h + 1
            W_out = (W_pad - k_w) // s_w + 1
            if H_out <= 0 or W_out <= 0:
                continue

            x_pad = rng.standard_normal((N, C, H_pad, W_pad), dtype=np.float32)

            y_cpp = np.empty((N, C, H_out, W_out), dtype=np.float32)
            idx_cpp = np.empty((N, C, H_out, W_out), dtype=np.int64)

            maxpool2d_forward_f32_ctypes(
                self.lib,
                x_pad=x_pad,
                y=y_cpp,
                argmax_idx=idx_cpp,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
            )

            y_ref, idx_ref = _ref_maxpool2d_forward_from_xpad(
                x_pad,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
            )

            np.testing.assert_allclose(y_cpp, y_ref, rtol=0, atol=0)
            np.testing.assert_array_equal(idx_cpp, idx_ref)

    def test_rejects_wrong_dtypes(self) -> None:
        # Ensure wrapper enforces dtype requirements
        N, C, H_pad, W_pad = 1, 1, 5, 5
        k_h, k_w = 2, 2
        s_h, s_w = 2, 2
        H_out = (H_pad - k_h) // s_h + 1
        W_out = (W_pad - k_w) // s_w + 1

        x_pad_f64 = np.zeros((N, C, H_pad, W_pad), dtype=np.float64)
        y = np.empty((N, C, H_out, W_out), dtype=np.float32)
        idx = np.empty((N, C, H_out, W_out), dtype=np.int64)

        with self.assertRaises(TypeError):
            maxpool2d_forward_f32_ctypes(
                self.lib,
                x_pad=x_pad_f64,
                y=y,
                argmax_idx=idx,
                N=N,
                C=C,
                H_pad=H_pad,
                W_pad=W_pad,
                H_out=H_out,
                W_out=W_out,
                k_h=k_h,
                k_w=k_w,
                s_h=s_h,
                s_w=s_w,
            )


if __name__ == "__main__":
    unittest.main()
