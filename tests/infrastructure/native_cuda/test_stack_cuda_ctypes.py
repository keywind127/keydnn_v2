import unittest
import numpy as np

try:
    from src.keydnn.infrastructure.native_cuda.python.stack_ctypes import (
        load_keydnn_cuda_native,
        _get_cuda,
    )
except Exception:
    load_keydnn_cuda_native = None
    _get_cuda = None


def _prod(xs):
    out = 1
    for v in xs:
        out *= int(v)
    return int(out)


def _pre_post(in_shape, axis):
    ndim = len(in_shape)
    if axis < 0:
        axis = axis + (ndim + 1)
    if axis < 0 or axis > ndim:
        raise ValueError("axis out of bounds")
    pre = _prod(in_shape[:axis])
    post = _prod(in_shape[axis:])
    return pre, post


class TestCudaStackCtypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if load_keydnn_cuda_native is None:
            raise unittest.SkipTest(
                "CUDA ctypes wrapper not importable in this environment."
            )
        try:
            cls.lib = load_keydnn_cuda_native()
        except Exception as e:
            raise unittest.SkipTest(f"CUDA native DLL not available: {e}")

        cls.cuda = _get_cuda(cls.lib)
        try:
            cls.cuda.cuda_set_device(0)
        except Exception as e:
            raise unittest.SkipTest(f"CUDA device not usable: {e}")

    def _roundtrip_forward(self, *, in_shape, axis, K, dtype):
        rng = np.random.default_rng(0)

        # create K inputs
        xs = [rng.standard_normal(in_shape).astype(dtype) for _ in range(K)]
        pre, post = _pre_post(in_shape, axis)

        # expected
        expected = np.stack(xs, axis=axis)

        # upload inputs (contiguous)
        x_devs = []
        try:
            for x in xs:
                x_devs.append(self.cuda.cuda_from_host(np.ascontiguousarray(x)))

            # allocate output
            y = np.empty(expected.shape, dtype=dtype)
            y_dev = self.cuda.cuda_malloc(y.nbytes)

            # run forward
            xs_ptrs_dev = self.cuda.stack_forward_cuda(
                xs_dev_ptrs=x_devs,
                y_dev=y_dev,
                pre=pre,
                post=post,
                dtype=dtype,
                sync=True,
            )

            # download
            self.cuda.cuda_memcpy_d2h(y, y_dev)

            # check
            np.testing.assert_allclose(y, expected, rtol=0, atol=0)

        finally:
            # free
            for p in x_devs:
                self.cuda.cuda_free(p)
            if "y_dev" in locals():
                self.cuda.cuda_free(y_dev)
            if "xs_ptrs_dev" in locals():
                self.cuda.cuda_free(xs_ptrs_dev)

    def _roundtrip_backward(self, *, in_shape, axis, K, dtype):
        rng = np.random.default_rng(1)
        pre, post = _pre_post(in_shape, axis)

        # grad_out shape = in_shape[:axis] + (K,) + in_shape[axis:]
        out_shape = tuple(in_shape[:axis]) + (K,) + tuple(in_shape[axis:])
        dy = rng.standard_normal(out_shape).astype(dtype)

        # expected grads: take along axis
        expected_grads = [np.take(dy, i, axis=axis) for i in range(K)]

        # upload dy
        dy_dev = None
        dx_devs = []
        try:
            dy_dev = self.cuda.cuda_from_host(np.ascontiguousarray(dy))

            # allocate each dx and run bwd (overwrite semantics)
            for i in range(K):
                dx = np.empty(in_shape, dtype=dtype)
                dx_dev = self.cuda.cuda_malloc(dx.nbytes)
                dx_devs.append(dx_dev)

            dxs_ptrs_dev = self.cuda.stack_backward_cuda(
                dy_dev=dy_dev,
                dxs_dev_ptrs=dx_devs,
                pre=pre,
                post=post,
                dtype=dtype,
                sync=True,
            )

            # download each dx and compare
            for i in range(K):
                got = np.empty(in_shape, dtype=dtype)
                self.cuda.cuda_memcpy_d2h(got, dx_devs[i])
                np.testing.assert_allclose(got, expected_grads[i], rtol=0, atol=0)

        finally:
            if dy_dev is not None:
                self.cuda.cuda_free(dy_dev)
            for p in dx_devs:
                self.cuda.cuda_free(p)
            if "dxs_ptrs_dev" in locals():
                self.cuda.cuda_free(dxs_ptrs_dev)

    def test_forward_f32_multiple_axes(self):
        in_shape = (2, 3, 4)
        K = 5
        for axis in (0, 1, 2, 3, -1):
            with self.subTest(axis=axis):
                self._roundtrip_forward(
                    in_shape=in_shape, axis=axis, K=K, dtype=np.float32
                )

    def test_forward_f64_multiple_axes(self):
        in_shape = (2, 2, 3)
        K = 3
        for axis in (0, 1, 3, -1):
            with self.subTest(axis=axis):
                self._roundtrip_forward(
                    in_shape=in_shape, axis=axis, K=K, dtype=np.float64
                )

    def test_backward_f32_multiple_axes(self):
        in_shape = (2, 3, 4)
        K = 4
        for axis in (0, 1, 2, 3, -1):
            with self.subTest(axis=axis):
                self._roundtrip_backward(
                    in_shape=in_shape, axis=axis, K=K, dtype=np.float32
                )

    def test_backward_f64_multiple_axes(self):
        in_shape = (2, 2, 3)
        K = 3
        for axis in (0, 2, 3, -1):
            with self.subTest(axis=axis):
                self._roundtrip_backward(
                    in_shape=in_shape, axis=axis, K=K, dtype=np.float64
                )

    def test_forward_rejects_empty_inputs(self):
        # low-level wrapper should reject K=0
        dtype = np.float32
        with self.assertRaises(ValueError):
            self.cuda.stack_forward_cuda(
                xs_dev_ptrs=[],
                y_dev=0,
                pre=1,
                post=1,
                dtype=dtype,
                sync=False,
            )

    def test_backward_rejects_empty_grads(self):
        dtype = np.float32
        with self.assertRaises(ValueError):
            self.cuda.stack_backward_cuda(
                dy_dev=0,
                dxs_dev_ptrs=[],
                pre=1,
                post=1,
                dtype=dtype,
                sync=False,
            )


if __name__ == "__main__":
    unittest.main()
