from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np


class _DummyLib:
    pass


class _MemcpyAliasMixin:
    """
    Shared tests for memcpy alias shims.

    Important:
    In this codebase, avgpool2d_ctypes/global_avgpool2d_ctypes re-export
    memcpy aliases from maxpool2d_ctypes. Therefore the alias *functions*
    may execute in the maxpool2d_ctypes module namespace, so we patch the
    delegates on maxpool2d_ctypes (source of truth), not only on `self.m`.
    """

    MODULE = None
    MODULE_NAME = "<unset>"

    @classmethod
    def setUpClass(cls) -> None:
        # Source-of-truth module where the alias functions are defined.
        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as max_mod

        cls._MAX = max_mod

    def setUp(self) -> None:
        if self.MODULE is None:
            raise RuntimeError("Test misconfigured: MODULE not set")
        self.m = self.MODULE
        self.lib = _DummyLib()

    # ---------------------------
    # Existence / identity
    # ---------------------------

    def test_alias_names_exist_and_are_callables(self) -> None:
        for name in (
            "cudaMemcpyDtoH",
            "cudaMemcpyHtoD",
            "cuda_memcpy_dtoh",
            "cuda_memcpy_htod",
        ):
            self.assertTrue(
                hasattr(self.m, name), f"{self.MODULE_NAME}: missing {name}"
            )
            self.assertTrue(
                callable(getattr(self.m, name)),
                f"{self.MODULE_NAME}: {name} not callable",
            )

    def test_alias_identity(self) -> None:
        self.assertIs(
            self.m.cudaMemcpyHtoD,
            self.m.cuda_memcpy_htod,
            f"{self.MODULE_NAME}: cudaMemcpyHtoD should alias cuda_memcpy_htod",
        )
        self.assertIs(
            self.m.cudaMemcpyDtoH,
            self.m.cuda_memcpy_dtoh,
            f"{self.MODULE_NAME}: cudaMemcpyDtoH should alias cuda_memcpy_dtoh",
        )

    # ---------------------------
    # HtoD shim behavior
    # ---------------------------

    def test_cuda_memcpy_htod_accepts_legacy_4_args_and_validates_nbytes(self) -> None:
        """
        cuda_memcpy_htod must accept legacy signature:
          (lib, dst_dev, src_host, nbytes)
        and validate nbytes when provided.
        """
        max_m = self._MAX
        m = self.m

        called = SimpleNamespace(count=0, lib=None, dst_dev=None, src_host=None)

        orig = max_m.cuda_memcpy_h2d

        def _fake_cuda_memcpy_h2d(fake_lib, dst_dev, src_host):
            called.count += 1
            called.lib = fake_lib
            called.dst_dev = dst_dev
            called.src_host = src_host

        try:
            # Patch where the alias actually delegates
            max_m.cuda_memcpy_h2d = _fake_cuda_memcpy_h2d

            src = np.arange(12, dtype=np.float32).reshape(3, 4).T  # non-contiguous
            self.assertFalse(src.flags["C_CONTIGUOUS"])
            dst_dev = 12345
            nbytes = int(src.nbytes)

            m.cuda_memcpy_htod(self.lib, dst_dev, src, nbytes)

            self.assertEqual(
                called.count,
                1,
                f"{self.MODULE_NAME}: shim did not delegate exactly once",
            )
            self.assertIs(
                called.lib, self.lib, f"{self.MODULE_NAME}: lib not forwarded"
            )
            self.assertEqual(
                called.dst_dev, dst_dev, f"{self.MODULE_NAME}: dst_dev not forwarded"
            )
            self.assertTrue(
                isinstance(called.src_host, np.ndarray)
                and called.src_host.flags["C_CONTIGUOUS"],
                f"{self.MODULE_NAME}: src_host not normalized to contiguous before delegating",
            )
            self.assertEqual(
                called.src_host.dtype,
                src.dtype,
                f"{self.MODULE_NAME}: dtype changed unexpectedly",
            )
            self.assertEqual(
                called.src_host.shape,
                src.shape,
                f"{self.MODULE_NAME}: shape changed unexpectedly",
            )

            with self.assertRaises(ValueError):
                m.cuda_memcpy_htod(self.lib, dst_dev, src, nbytes + 4)

        finally:
            max_m.cuda_memcpy_h2d = orig

    # ---------------------------
    # DtoH shim behavior
    # ---------------------------

    def test_cuda_memcpy_dtoh_accepts_legacy_4_args_and_validates_nbytes(self) -> None:
        """
        cuda_memcpy_dtoh must accept legacy signature:
          (lib, dst_host, src_dev, nbytes)
        and validate nbytes when provided.
        """
        max_m = self._MAX
        m = self.m

        called = SimpleNamespace(count=0, lib=None, dst_host=None, src_dev=None)

        orig = max_m.cuda_memcpy_d2h

        def _fake_cuda_memcpy_d2h(fake_lib, dst_host, src_dev):
            called.count += 1
            called.lib = fake_lib
            called.dst_host = dst_host
            called.src_dev = src_dev

        try:
            max_m.cuda_memcpy_d2h = _fake_cuda_memcpy_d2h

            dst_host = np.empty((2, 3), dtype=np.float64)
            self.assertTrue(dst_host.flags["C_CONTIGUOUS"])
            src_dev = 999
            nbytes = int(dst_host.nbytes)

            m.cuda_memcpy_dtoh(self.lib, dst_host, src_dev, nbytes)

            self.assertEqual(
                called.count,
                1,
                f"{self.MODULE_NAME}: shim did not delegate exactly once",
            )
            self.assertIs(
                called.lib, self.lib, f"{self.MODULE_NAME}: lib not forwarded"
            )
            self.assertIs(
                called.dst_host,
                dst_host,
                f"{self.MODULE_NAME}: dst_host not forwarded by reference",
            )
            self.assertEqual(
                called.src_dev, src_dev, f"{self.MODULE_NAME}: src_dev not forwarded"
            )

            with self.assertRaises(ValueError):
                m.cuda_memcpy_dtoh(self.lib, dst_host, src_dev, nbytes - 8)

        finally:
            max_m.cuda_memcpy_d2h = orig

    def test_cuda_memcpy_dtoh_rejects_noncontiguous_dst_before_delegate(self) -> None:
        max_m = self._MAX
        m = self.m

        base = np.empty((4, 6), dtype=np.float32)
        dst_host = base[:, ::2]  # non-contiguous view
        self.assertFalse(dst_host.flags["C_CONTIGUOUS"])

        orig = max_m.cuda_memcpy_d2h
        called = SimpleNamespace(count=0)

        def _fake_cuda_memcpy_d2h(*args, **kwargs):
            called.count += 1

        try:
            max_m.cuda_memcpy_d2h = _fake_cuda_memcpy_d2h

            with self.assertRaises(ValueError):
                m.cuda_memcpy_dtoh(self.lib, dst_host, 1, int(dst_host.nbytes))

            self.assertEqual(
                called.count,
                0,
                f"{self.MODULE_NAME}: shim should raise before delegating on non-contiguous dst_host",
            )
        finally:
            max_m.cuda_memcpy_d2h = orig

    # ---------------------------
    # C-style alias names should behave the same
    # ---------------------------

    def test_cudaMemcpyHtoD_and_cudaMemcpyDtoH_legacy_signature(self) -> None:
        max_m = self._MAX
        m = self.m

        orig_h2d = max_m.cuda_memcpy_h2d
        orig_d2h = max_m.cuda_memcpy_d2h

        h2d_called = SimpleNamespace(count=0)
        d2h_called = SimpleNamespace(count=0)

        def _fake_h2d(*args, **kwargs):
            h2d_called.count += 1

        def _fake_d2h(*args, **kwargs):
            d2h_called.count += 1

        try:
            max_m.cuda_memcpy_h2d = _fake_h2d
            max_m.cuda_memcpy_d2h = _fake_d2h

            src = np.arange(6, dtype=np.float32)
            dst = np.empty_like(src)

            m.cudaMemcpyHtoD(self.lib, 111, src, int(src.nbytes))
            m.cudaMemcpyDtoH(self.lib, dst, 222, int(dst.nbytes))

            self.assertEqual(
                h2d_called.count,
                1,
                f"{self.MODULE_NAME}: cudaMemcpyHtoD should delegate once",
            )
            self.assertEqual(
                d2h_called.count,
                1,
                f"{self.MODULE_NAME}: cudaMemcpyDtoH should delegate once",
            )
        finally:
            max_m.cuda_memcpy_h2d = orig_h2d
            max_m.cuda_memcpy_d2h = orig_d2h


class TestMaxpool2dCtypesMemcpyAliases(_MemcpyAliasMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from src.keydnn.infrastructure.native_cuda.python import maxpool2d_ctypes as mod

        cls.MODULE = mod
        cls.MODULE_NAME = "maxpool2d_ctypes"


class TestAvgpool2dCtypesMemcpyAliases(_MemcpyAliasMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from src.keydnn.infrastructure.native_cuda.python import avgpool2d_ctypes as mod

        cls.MODULE = mod
        cls.MODULE_NAME = "avgpool2d_ctypes"


class TestGlobalAvgpool2dCtypesMemcpyAliases(_MemcpyAliasMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from src.keydnn.infrastructure.native_cuda.python import (
            global_avgpool2d_ctypes as mod,
        )

        cls.MODULE = mod
        cls.MODULE_NAME = "global_avgpool2d_ctypes"


if __name__ == "__main__":
    unittest.main()
