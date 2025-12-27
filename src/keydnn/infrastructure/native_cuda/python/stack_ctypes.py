"""
ctypes bindings for KeyDNN v2 CUDA Stack kernels.

This module provides low-level Python bindings to the CUDA implementation of
`Tensor.stack` forward/backward via `ctypes`.

It is intentionally "close to the metal":
- device pointers are represented as Python `int` (uintptr_t)
- pointer arrays are passed to kernels as *device* arrays of `uint64_t`
- kernel selection is dtype-specialized (float32 vs float64)
- error reporting can optionally include a native "last debug message"

Expected native exports
-----------------------
- keydnn_cuda_set_device
- keydnn_cuda_malloc / keydnn_cuda_free
- keydnn_cuda_memcpy_h2d / keydnn_cuda_memcpy_d2h
- keydnn_cuda_synchronize
- keydnn_cuda_upload_u64_array
- keydnn_cuda_stack_fwd_u64_f32 / keydnn_cuda_stack_fwd_u64_f64
- keydnn_cuda_stack_bwd_u64_f32 / keydnn_cuda_stack_bwd_u64_f64
- keydnn_cuda_debug_set_enabled
- keydnn_cuda_debug_get_last

Design notes
------------
- Pointer-array arguments are uploaded using `keydnn_cuda_upload_u64_array`,
  which copies a host uint64[K] array into a device uint64[K] allocation.
- `CudaLib` performs lazy, idempotent binding of `argtypes/restype`.
- The functional API at the bottom wraps a cached singleton `CudaLib` to match
  the style of other KeyDNN v2 CUDA wrappers.
"""

from __future__ import annotations

import ctypes
from ctypes import (
    c_int,
    c_size_t,
    c_void_p,
    c_uint64,
    c_int64,
)
from typing import Sequence

import numpy as np

DevPtr = int  # uintptr_t as Python int


# ---------------------------------------------------------------------
# DLL loading (copy your existing path logic; adjust if needed)
# ---------------------------------------------------------------------


def load_keydnn_cuda_native():
    """
    Load the KeyDNN CUDA native shared library (Windows DLL).

    This helper encapsulates:
    - locating the build output DLL relative to this file
    - adding the CUDA runtime `bin` directory to the DLL search path (if available)
    - adding the build folder to the DLL search path
    - returning a loaded `ctypes.CDLL` handle

    Returns
    -------
    ctypes.CDLL
        Loaded native library handle.

    Raises
    ------
    FileNotFoundError
        If the expected DLL path does not exist.
    OSError
        If the OS rejects adding a DLL directory for reasons other than the
        common Windows "path too long" (WinError 206) edge case.
    """
    import os
    import ctypes
    from pathlib import Path

    p = (
        Path(__file__).resolve().parents[2]
        / "native_cuda"
        / "keydnn_v2_cuda_native"
        / "x64"
        / "Debug"
        / "KeyDNNV2CudaNative.dll"
    ).resolve()

    if not p.exists():
        raise FileNotFoundError(f"KeyDNN CUDA native DLL not found at: {p}")

    # Ensure CUDA runtime deps are discoverable
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except OSError as e:
                # Some Windows setups throw WinError 206 here even for normal-looking paths.
                if getattr(e, "winerror", None) != 206:
                    raise

    try:
        os.add_dll_directory(str(p.parent))
    except OSError as e:
        # Same WinError 206 protection for the build folder.
        if getattr(e, "winerror", None) != 206:
            raise

    return ctypes.CDLL(str(p))


# ---------------------------------------------------------------------
# CudaLib wrapper
# ---------------------------------------------------------------------


class CudaLib:
    """
    Thin binding layer around the KeyDNN CUDA native library.

    This object:
    - stores the loaded `ctypes.CDLL` handle
    - lazily binds function signatures (`argtypes` / `restype`)
    - provides small Pythonic helpers for:
      - device management (set device, synchronize)
      - device allocation and H2D/D2H memcpy
      - uploading uint64 pointer arrays to device memory
      - dispatching dtype-specialized stack forward/backward kernels
      - optional native debug integration (last-message retrieval)

    Notes
    -----
    - All device pointers are represented as Python `int` (`DevPtr`).
    - Most methods raise `RuntimeError` on non-zero native status codes, and
      will include `native_debug=...` if the debug exports are present and set.
    """

    def __init__(self, lib: ctypes.CDLL) -> None:
        """
        Create a `CudaLib` wrapper for a loaded native library.

        Parameters
        ----------
        lib : ctypes.CDLL
            Loaded KeyDNN CUDA native library handle.
        """
        self.lib = lib
        self._cuda_utils_bound = False
        self._stack_bound = False
        self._debug_bound = False

        # Make debug opt-in via env var, but default OFF.
        # If you want it always ON during tests, set KEYDNN_CUDA_DEBUG=1
        import os

        self._debug_enabled_default = os.environ.get("KEYDNN_CUDA_DEBUG", "0") not in (
            "0",
            "",
            "false",
            "False",
            "FALSE",
        )

    # ----------------------------
    # binders
    # ----------------------------

    def _bind_cuda_utils(self) -> None:
        """
        Bind argtypes/restype for general CUDA utility exports (idempotent).

        Exports bound here:
        - keydnn_cuda_set_device
        - keydnn_cuda_malloc / keydnn_cuda_free
        - keydnn_cuda_memcpy_h2d / keydnn_cuda_memcpy_d2h
        - keydnn_cuda_synchronize
        """
        if self._cuda_utils_bound:
            return

        lib = self.lib
        lib.keydnn_cuda_set_device.argtypes = [c_int]
        lib.keydnn_cuda_set_device.restype = c_int

        lib.keydnn_cuda_malloc.argtypes = [ctypes.POINTER(c_uint64), c_size_t]
        lib.keydnn_cuda_malloc.restype = c_int

        lib.keydnn_cuda_free.argtypes = [c_uint64]
        lib.keydnn_cuda_free.restype = c_int

        lib.keydnn_cuda_memcpy_h2d.argtypes = [c_uint64, c_void_p, c_size_t]
        lib.keydnn_cuda_memcpy_h2d.restype = c_int

        lib.keydnn_cuda_memcpy_d2h.argtypes = [c_void_p, c_uint64, c_size_t]
        lib.keydnn_cuda_memcpy_d2h.restype = c_int

        lib.keydnn_cuda_synchronize.argtypes = []
        lib.keydnn_cuda_synchronize.restype = c_int

        self._cuda_utils_bound = True

    def _bind_debug(self) -> None:
        """
        Bind native debug exports (idempotent).

        These exports are required by this wrapper's stack binding path, because
        they provide the most actionable error context (a "last debug message").

        Raises
        ------
        AttributeError
            If either required debug export is missing from the loaded library.
        """
        if self._debug_bound:
            return

        lib = self.lib

        missing = []
        if not hasattr(lib, "keydnn_cuda_debug_set_enabled"):
            missing.append("keydnn_cuda_debug_set_enabled")
        if not hasattr(lib, "keydnn_cuda_debug_get_last"):
            missing.append("keydnn_cuda_debug_get_last")

        if missing:
            # Make this fail loud so we stop guessing.
            raise AttributeError(
                "CUDA DLL is missing debug exports: "
                + ", ".join(missing)
                + ". Did you rebuild the correct DLL, and are you loading the correct one?"
            )

        lib.keydnn_cuda_debug_set_enabled.argtypes = [c_int]
        lib.keydnn_cuda_debug_set_enabled.restype = None  # void

        lib.keydnn_cuda_debug_get_last.argtypes = [c_void_p, c_int]
        lib.keydnn_cuda_debug_get_last.restype = c_int

        self._debug_bound = True

    # def _bind_stack(self) -> None:
    #     """
    #     Bind argtypes/restype for stack-related CUDA exports (idempotent).

    #     ABI overview
    #     ------------
    #     - Pointer arrays are uploaded as uint64_t[K] via `keydnn_cuda_upload_u64_array`.
    #     - Stack kernels receive the device pointer to that uint64_t array:
    #       - forward: `xs_u64_dev`
    #       - backward: `dxs_u64_dev`

    #     Binding strategy
    #     ----------------
    #     Pointer parameters are bound as `c_void_p` to avoid fragile pointer casts.
    #     This keeps the wrapper tolerant to `uintptr_t` differences across platforms.

    #     Side effects
    #     ------------
    #     - Binds debug exports and enables debug messages (opt-in semantics in native).
    #     """
    #     if self._stack_bound:
    #         return

    #     lib = self.lib

    #     # int keydnn_cuda_upload_u64_array(uint64_t* dst_dev_u64, const uint64_t* src_host_u64, int64 K)
    #     if not hasattr(lib, "keydnn_cuda_upload_u64_array"):
    #         raise AttributeError(
    #             "CUDA DLL missing symbol: keydnn_cuda_upload_u64_array"
    #         )
    #     lib.keydnn_cuda_upload_u64_array.argtypes = [
    #         c_void_p,  # dst_dev_u64 (device pointer)
    #         c_void_p,  # src_host_u64 (host pointer)
    #         c_int64,  # K
    #     ]
    #     lib.keydnn_cuda_upload_u64_array.restype = c_int

    #     # Forward:
    #     if not hasattr(lib, "keydnn_cuda_stack_fwd_u64_f32"):
    #         raise AttributeError(
    #             "CUDA DLL missing symbol: keydnn_cuda_stack_fwd_u64_f32"
    #         )
    #     lib.keydnn_cuda_stack_fwd_u64_f32.argtypes = [
    #         c_void_p,  # xs_u64_dev (device uint64[K])
    #         c_int64,
    #         c_int64,
    #         c_int64,
    #         c_void_p,  # y (device)
    #     ]
    #     lib.keydnn_cuda_stack_fwd_u64_f32.restype = c_int

    #     if not hasattr(lib, "keydnn_cuda_stack_fwd_u64_f64"):
    #         raise AttributeError(
    #             "CUDA DLL missing symbol: keydnn_cuda_stack_fwd_u64_f64"
    #         )
    #     lib.keydnn_cuda_stack_fwd_u64_f64.argtypes = [
    #         c_void_p,
    #         c_int64,
    #         c_int64,
    #         c_int64,
    #         c_void_p,
    #     ]
    #     lib.keydnn_cuda_stack_fwd_u64_f64.restype = c_int

    #     # Backward:
    #     if not hasattr(lib, "keydnn_cuda_stack_bwd_u64_f32"):
    #         raise AttributeError(
    #             "CUDA DLL missing symbol: keydnn_cuda_stack_bwd_u64_f32"
    #         )
    #     lib.keydnn_cuda_stack_bwd_u64_f32.argtypes = [
    #         c_void_p,  # dy (device)
    #         c_int64,
    #         c_int64,
    #         c_int64,
    #         c_void_p,  # dxs_u64_dev (device uint64[K])
    #     ]
    #     lib.keydnn_cuda_stack_bwd_u64_f32.restype = c_int

    #     if not hasattr(lib, "keydnn_cuda_stack_bwd_u64_f64"):
    #         raise AttributeError(
    #             "CUDA DLL missing symbol: keydnn_cuda_stack_bwd_u64_f64"
    #         )
    #     lib.keydnn_cuda_stack_bwd_u64_f64.argtypes = [
    #         c_void_p,
    #         c_int64,
    #         c_int64,
    #         c_int64,
    #         c_void_p,
    #     ]
    #     lib.keydnn_cuda_stack_bwd_u64_f64.restype = c_int

    #     # Bind debug (optional) and enable if requested
    #     self._bind_debug()
    #     self.cuda_debug_set_enabled(True)

    #     self._stack_bound = True

    # ----------------------------
    # native debug helpers
    # ----------------------------

    def cuda_debug_set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable native debug message collection (if supported).

        Parameters
        ----------
        enabled : bool
            If True, enable native debug messages; otherwise disable.

        Notes
        -----
        This method binds the debug exports on first use. If the export is
        unexpectedly absent after binding, it becomes a no-op (defensive).
        """
        self._bind_debug()
        lib = self.lib
        if not hasattr(lib, "keydnn_cuda_debug_set_enabled"):
            return
        lib.keydnn_cuda_debug_set_enabled(c_int(1 if enabled else 0))

    def cuda_debug_get_last(self) -> str:
        """
        Fetch the last native debug message (if supported).

        Returns
        -------
        str
            The most recent debug string set by the native library, or an empty
            string if not supported or if no message is available.

        Notes
        -----
        The returned string is decoded as UTF-8 with replacement on decode
        errors.
        """
        self._bind_debug()
        lib = self.lib
        if not hasattr(lib, "keydnn_cuda_debug_get_last"):
            return ""

        buf = ctypes.create_string_buffer(2048)
        n = lib.keydnn_cuda_debug_get_last(c_void_p(ctypes.addressof(buf)), c_int(2048))
        if n <= 0:
            return ""
        try:
            return buf.value.decode("utf-8", errors="replace")
        except Exception:
            return str(buf.value)

    def _raise_with_native_debug(self, msg: str, st: int) -> None:
        """
        Raise a RuntimeError that includes the native debug string when available.

        Parameters
        ----------
        msg : str
            Human-readable prefix message.
        st : int
            Native status/error code.

        Raises
        ------
        RuntimeError
            Always raised, with `native_debug=...` appended if present.
        """
        dbg = self.cuda_debug_get_last()
        if dbg:
            raise RuntimeError(f"{msg} status={st} | native_debug={dbg}")
        raise RuntimeError(f"{msg} status={st}")

    # ----------------------------
    # pointer helpers
    # ----------------------------

    @staticmethod
    def _as_dev_ptr(dev_ptr: DevPtr) -> c_void_p:
        """
        Convert a Python `DevPtr` integer into a `ctypes.c_void_p`.

        Parameters
        ----------
        dev_ptr : DevPtr
            Device pointer expressed as a Python int (uintptr_t).

        Returns
        -------
        ctypes.c_void_p
            Pointer value suitable for passing to `ctypes` calls.
        """
        return c_void_p(int(dev_ptr))

    # ----------------------------
    # CUDA utils
    # ----------------------------

    def cuda_set_device(self, device: int = 0) -> None:
        """
        Set the active CUDA device ordinal in the native runtime.

        Parameters
        ----------
        device : int, optional
            CUDA device ordinal. Defaults to 0.

        Raises
        ------
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_set_device(int(device))
        if st != 0:
            self._raise_with_native_debug("keydnn_cuda_set_device failed with", st)

    def cuda_malloc(self, nbytes: int) -> DevPtr:
        """
        Allocate device memory.

        Parameters
        ----------
        nbytes : int
            Number of bytes to allocate.

        Returns
        -------
        DevPtr
            Device pointer to the allocated buffer.

        Raises
        ------
        RuntimeError
            If allocation fails (non-zero status or returned pointer is 0).
        """
        self._bind_cuda_utils()
        out = c_uint64(0)
        st = self.lib.keydnn_cuda_malloc(ctypes.byref(out), c_size_t(int(nbytes)))
        if st != 0 or out.value == 0:
            self._raise_with_native_debug(
                f"keydnn_cuda_malloc failed (nbytes={nbytes}) with", st
            )
        return int(out.value)

    def cuda_free(self, dev_ptr: DevPtr) -> None:
        """
        Free device memory previously allocated by `cuda_malloc`.

        Parameters
        ----------
        dev_ptr : DevPtr
            Device pointer to free.

        Raises
        ------
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_free(c_uint64(int(dev_ptr)))
        if st != 0:
            self._raise_with_native_debug("keydnn_cuda_free failed with", st)

    def cuda_memcpy_h2d(self, dst_dev: DevPtr, src_host: np.ndarray) -> None:
        """
        Copy a host NumPy array into device memory.

        Parameters
        ----------
        dst_dev : DevPtr
            Destination device pointer.
        src_host : np.ndarray
            Source host array. Will be converted to C-contiguous if needed.

        Raises
        ------
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        if not src_host.flags["C_CONTIGUOUS"]:
            src_host = np.ascontiguousarray(src_host)

        st = self.lib.keydnn_cuda_memcpy_h2d(
            c_uint64(int(dst_dev)),
            c_void_p(int(src_host.ctypes.data)),
            c_size_t(int(src_host.nbytes)),
        )
        if st != 0:
            self._raise_with_native_debug("keydnn_cuda_memcpy_h2d failed with", st)

    def cuda_memcpy_d2h(self, dst_host: np.ndarray, src_dev: DevPtr) -> None:
        """
        Copy device memory into a pre-allocated host NumPy array.

        Parameters
        ----------
        dst_host : np.ndarray
            Destination host array. Must be C-contiguous.
        src_dev : DevPtr
            Source device pointer.

        Raises
        ------
        ValueError
            If `dst_host` is not C-contiguous.
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        if not dst_host.flags["C_CONTIGUOUS"]:
            raise ValueError("dst_host must be C-contiguous")

        st = self.lib.keydnn_cuda_memcpy_d2h(
            c_void_p(int(dst_host.ctypes.data)),
            c_uint64(int(src_dev)),
            c_size_t(int(dst_host.nbytes)),
        )
        if st != 0:
            self._raise_with_native_debug("keydnn_cuda_memcpy_d2h failed with", st)

    def cuda_synchronize(self) -> None:
        """
        Block until all previously issued work on the current device is complete.

        Raises
        ------
        RuntimeError
            If the native call returns a non-zero status.
        """
        self._bind_cuda_utils()
        st = self.lib.keydnn_cuda_synchronize()
        if st != 0:
            self._raise_with_native_debug("keydnn_cuda_synchronize failed with", st)

    def cuda_from_host(self, x: np.ndarray) -> DevPtr:
        """
        Convenience helper: allocate device memory and upload a host array.

        Parameters
        ----------
        x : np.ndarray
            Host array to upload. Must be one of float32/float64/int64/uint64.
            Will be converted to C-contiguous if needed.

        Returns
        -------
        DevPtr
            Device pointer to the uploaded buffer.

        Raises
        ------
        TypeError
            If `x.dtype` is unsupported.
        RuntimeError
            If allocation or memcpy fails (device buffer is freed on failure).
        """
        if x.dtype not in (np.float32, np.float64, np.int64, np.uint64):
            raise TypeError(
                f"cuda_from_host only supports float32/float64/int64/uint64, got {x.dtype}"
            )
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        dev = self.cuda_malloc(x.nbytes)
        try:
            self.cuda_memcpy_h2d(dev, x)
        except Exception:
            self.cuda_free(dev)
            raise
        return dev

    def debug_read_u64_array(self, *, u64_array_dev: DevPtr, K: int) -> list[int]:
        """
        Debug helper: copy a device `uint64_t[K]` array back to host.

        Parameters
        ----------
        u64_array_dev : DevPtr
            Device pointer to a uint64 array of length `K`.
        K : int
            Number of uint64 entries.

        Returns
        -------
        list[int]
            Host list of length `K` containing the copied values.

        Notes
        -----
        This is intended for diagnosing pointer-array upload issues and requires
        the `keydnn_cuda_memcpy_d2h` export to function correctly.
        """
        self._bind_cuda_utils()
        K = int(K)
        host = (c_uint64 * K)()
        nbytes = K * ctypes.sizeof(c_uint64)

        st = self.lib.keydnn_cuda_memcpy_d2h(
            c_void_p(int(ctypes.addressof(host))),
            c_uint64(int(u64_array_dev)),
            c_size_t(int(nbytes)),
        )
        if st != 0:
            self._raise_with_native_debug(
                "debug_read_u64_array memcpy_d2h failed with", st
            )

        return [int(host[i]) for i in range(K)]

    # ----------------------------
    # Stack API
    # ----------------------------

    def cuda_upload_u64_array(
        self, *, dst_u64_array_dev: DevPtr, src_ptrs_host: Sequence[int]
    ) -> None:
        """
        Upload a host list of pointers into a device `uint64_t[K]` array.

        Parameters
        ----------
        dst_u64_array_dev : DevPtr
            Device pointer to an allocation of size `K * sizeof(uint64_t)`.
        src_ptrs_host : Sequence[int]
            Host sequence of integer pointer values to upload.

        Raises
        ------
        ValueError
            If `src_ptrs_host` is empty.
        RuntimeError
            If the native upload call fails.
        """
        self._bind_stack()

        K = int(len(src_ptrs_host))
        if K <= 0:
            raise ValueError("src_ptrs_host must be non-empty")

        host_arr = (c_uint64 * K)(*[c_uint64(int(p)) for p in src_ptrs_host])

        st = self.lib.keydnn_cuda_upload_u64_array(
            self._as_dev_ptr(dst_u64_array_dev),  # device pointer (uint64_t*)
            ctypes.cast(host_arr, c_void_p),  # host pointer (uint64_t*)
            c_int64(K),
        )
        if st != 0:
            self._raise_with_native_debug(
                "keydnn_cuda_upload_u64_array failed with", st
            )

    # def stack_forward_cuda(
    #     self,
    #     *,
    #     xs_dev_ptrs: Sequence[DevPtr],
    #     y_dev: DevPtr,
    #     pre: int,
    #     post: int,
    #     dtype: np.dtype,
    #     sync: bool = True,
    #     debug_verify_ptrs: bool = False,
    # ) -> DevPtr:
    #     """
    #     Run the CUDA stack forward kernel.

    #     This method allocates and populates a device `uint64_t[K]` pointer array
    #     for the inputs, dispatches the dtype-specialized native kernel, and
    #     optionally synchronizes.

    #     Parameters
    #     ----------
    #     xs_dev_ptrs : Sequence[DevPtr]
    #         Input device pointers for K tensors/buffers.
    #     y_dev : DevPtr
    #         Output device pointer (pre-allocated) where stacked result is written.
    #     pre : int
    #         Product of input dimensions before the insertion axis.
    #     post : int
    #         Product of input dimensions at/after the insertion axis.
    #     dtype : np.dtype
    #         Either np.float32 or np.float64 to select the native kernel.
    #     sync : bool, optional
    #         If True, calls `cuda_synchronize()` after kernel launch. Defaults to True.
    #     debug_verify_ptrs : bool, optional
    #         If True, reads back the device u64 pointer array and checks it matches
    #         the host inputs. Useful for debugging pointer upload issues.

    #     Returns
    #     -------
    #     DevPtr
    #         Device pointer to the temporary `uint64_t[K]` allocation holding input
    #         pointers. The caller is responsible for freeing it.

    #     Raises
    #     ------
    #     ValueError
    #         If `xs_dev_ptrs` is empty.
    #     TypeError
    #         If `dtype` is unsupported.
    #     RuntimeError
    #         If the native kernel or a CUDA utility call fails.

    #     Resource management
    #     -------------------
    #     - Allocates `xs_u64_dev` (device uint64[K]) internally.
    #     - On failure, frees `xs_u64_dev` before re-raising.
    #     """
    #     self._bind_stack()
    #     self._bind_cuda_utils()

    #     K = int(len(xs_dev_ptrs))
    #     if K <= 0:
    #         raise ValueError("xs_dev_ptrs must be non-empty")

    #     u64_array_bytes = K * ctypes.sizeof(c_uint64)
    #     xs_u64_dev = self.cuda_malloc(u64_array_bytes)

    #     try:
    #         self.cuda_upload_u64_array(
    #             dst_u64_array_dev=xs_u64_dev, src_ptrs_host=xs_dev_ptrs
    #         )

    #         if debug_verify_ptrs:
    #             readback = self.debug_read_u64_array(u64_array_dev=xs_u64_dev, K=K)
    #             if readback != [int(p) for p in xs_dev_ptrs]:
    #                 raise RuntimeError(
    #                     f"xs pointer u64 array mismatch: {readback} vs {list(map(int, xs_dev_ptrs))}"
    #                 )

    #         if dtype == np.float32:
    #             st = self.lib.keydnn_cuda_stack_fwd_u64_f32(
    #                 self._as_dev_ptr(xs_u64_dev),
    #                 c_int64(K),
    #                 c_int64(int(pre)),
    #                 c_int64(int(post)),
    #                 self._as_dev_ptr(y_dev),
    #             )
    #         elif dtype == np.float64:
    #             st = self.lib.keydnn_cuda_stack_fwd_u64_f64(
    #                 self._as_dev_ptr(xs_u64_dev),
    #                 c_int64(K),
    #                 c_int64(int(pre)),
    #                 c_int64(int(post)),
    #                 self._as_dev_ptr(y_dev),
    #             )
    #         else:
    #             raise TypeError(f"Unsupported dtype for stack_forward_cuda: {dtype}")

    #         if st != 0:
    #             self._raise_with_native_debug(
    #                 "keydnn_cuda_stack_fwd_u64 failed with", st
    #             )

    #         if sync:
    #             self.cuda_synchronize()

    #     except Exception:
    #         self.cuda_free(xs_u64_dev)
    #         raise

    #     return xs_u64_dev

    # def stack_backward_cuda(
    #     self,
    #     *,
    #     dy_dev: DevPtr,
    #     dxs_dev_ptrs: Sequence[DevPtr],
    #     pre: int,
    #     post: int,
    #     dtype: np.dtype,
    #     sync: bool = True,
    #     debug_verify_ptrs: bool = True,
    # ) -> DevPtr:
    #     """
    #     Run the CUDA stack backward kernel.

    #     This method allocates and populates a device `uint64_t[K]` pointer array
    #     for the output-gradient buffers (`dxs`), dispatches the dtype-specialized
    #     native backward kernel, and optionally synchronizes.

    #     Parameters
    #     ----------
    #     dy_dev : DevPtr
    #         Device pointer to `grad_out` / `dy` buffer (stacked gradient input).
    #     dxs_dev_ptrs : Sequence[DevPtr]
    #         Sequence of K device pointers, one per per-input `dx` buffer.
    #         These buffers must be pre-allocated by the caller.
    #     pre : int
    #         Product of original input dimensions before the insertion axis.
    #     post : int
    #         Product of original input dimensions at/after the insertion axis.
    #     dtype : np.dtype
    #         Either np.float32 or np.float64 to select the native kernel.
    #     sync : bool, optional
    #         If True, calls `cuda_synchronize()` after kernel launch. Defaults to True.
    #     debug_verify_ptrs : bool, optional
    #         If True, reads back the device u64 pointer array and checks it matches
    #         the host pointers. Defaults to True.

    #     Returns
    #     -------
    #     DevPtr
    #         Device pointer to the temporary `uint64_t[K]` allocation holding `dx`
    #         pointers. The caller is responsible for freeing it.

    #     Raises
    #     ------
    #     ValueError
    #         If `dxs_dev_ptrs` is empty.
    #     TypeError
    #         If `dtype` is unsupported.
    #     RuntimeError
    #         If the native kernel or a CUDA utility call fails.

    #     Resource management
    #     -------------------
    #     - Allocates `dxs_u64_dev` (device uint64[K]) internally.
    #     - On failure, frees `dxs_u64_dev` before re-raising.
    #     """
    #     self._bind_stack()
    #     self._bind_cuda_utils()

    #     K = int(len(dxs_dev_ptrs))
    #     if K <= 0:
    #         raise ValueError("dxs_dev_ptrs must be non-empty")

    #     u64_array_bytes = K * ctypes.sizeof(c_uint64)
    #     dxs_u64_dev = self.cuda_malloc(u64_array_bytes)

    #     try:
    #         self.cuda_upload_u64_array(
    #             dst_u64_array_dev=dxs_u64_dev, src_ptrs_host=dxs_dev_ptrs
    #         )

    #         if debug_verify_ptrs:
    #             readback = self.debug_read_u64_array(u64_array_dev=dxs_u64_dev, K=K)
    #             if readback != [int(p) for p in dxs_dev_ptrs]:
    #                 raise RuntimeError(
    #                     f"dx pointer u64 array mismatch: {readback} vs {list(map(int, dxs_dev_ptrs))}"
    #                 )

    #         if dtype == np.float32:
    #             st = self.lib.keydnn_cuda_stack_bwd_u64_f32(
    #                 self._as_dev_ptr(dy_dev),
    #                 c_int64(K),
    #                 c_int64(int(pre)),
    #                 c_int64(int(post)),
    #                 self._as_dev_ptr(dxs_u64_dev),
    #             )
    #         elif dtype == np.float64:
    #             st = self.lib.keydnn_cuda_stack_bwd_u64_f64(
    #                 self._as_dev_ptr(dy_dev),
    #                 c_int64(K),
    #                 c_int64(int(pre)),
    #                 c_int64(int(post)),
    #                 self._as_dev_ptr(dxs_u64_dev),
    #             )
    #         else:
    #             raise TypeError(f"Unsupported dtype for stack_backward_cuda: {dtype}")

    #         if st != 0:
    #             self._raise_with_native_debug(
    #                 "keydnn_cuda_stack_bwd_u64 failed with", st
    #             )

    #         if sync:
    #             self.cuda_synchronize()

    #     except Exception:
    #         self.cuda_free(dxs_u64_dev)
    #         raise

    #     return dxs_u64_dev

    def _bind_stack(self) -> None:
        if self._stack_bound:
            return

        lib = self.lib

        # (bindings unchanged...)

        # Bind debug exports if present, but DO NOT force enable.
        # Only enable debug if env default requests it.
        try:
            self._bind_debug()
            if self._debug_enabled_default:
                self.cuda_debug_set_enabled(True)
        except AttributeError:
            # debug exports optional in perf build
            pass

        self._stack_bound = True

    def stack_forward_cuda(
        self,
        *,
        xs_dev_ptrs,
        y_dev,
        pre,
        post,
        dtype,
        sync: bool = False,              # <- default False
        debug_verify_ptrs: bool = False,
    ) -> DevPtr:
        self._bind_stack()
        self._bind_cuda_utils()

        K = int(len(xs_dev_ptrs))
        if K <= 0:
            raise ValueError("xs_dev_ptrs must be non-empty")

        xs_u64_dev = self.cuda_malloc(K * ctypes.sizeof(c_uint64))
        try:
            self.cuda_upload_u64_array(dst_u64_array_dev=xs_u64_dev, src_ptrs_host=xs_dev_ptrs)

            if debug_verify_ptrs:
                readback = self.debug_read_u64_array(u64_array_dev=xs_u64_dev, K=K)
                if readback != [int(p) for p in xs_dev_ptrs]:
                    raise RuntimeError(f"xs pointer u64 array mismatch: {readback} vs {list(map(int, xs_dev_ptrs))}")

            if dtype == np.float32:
                st = self.lib.keydnn_cuda_stack_fwd_u64_f32(
                    self._as_dev_ptr(xs_u64_dev), c_int64(K), c_int64(int(pre)), c_int64(int(post)), self._as_dev_ptr(y_dev)
                )
            elif dtype == np.float64:
                st = self.lib.keydnn_cuda_stack_fwd_u64_f64(
                    self._as_dev_ptr(xs_u64_dev), c_int64(K), c_int64(int(pre)), c_int64(int(post)), self._as_dev_ptr(y_dev)
                )
            else:
                raise TypeError(f"Unsupported dtype for stack_forward_cuda: {dtype}")

            if st != 0:
                self._raise_with_native_debug("keydnn_cuda_stack_fwd_u64 failed with", st)

            if sync:
                self.cuda_synchronize()

        except Exception:
            self.cuda_free(xs_u64_dev)
            raise

        return xs_u64_dev

    def stack_backward_cuda(
        self,
        *,
        dy_dev,
        dxs_dev_ptrs,
        pre,
        post,
        dtype,
        sync: bool = False,              # <- default False
        debug_verify_ptrs: bool = False, # <- default False (perf)
    ) -> DevPtr:
        self._bind_stack()
        self._bind_cuda_utils()

        K = int(len(dxs_dev_ptrs))
        if K <= 0:
            raise ValueError("dxs_dev_ptrs must be non-empty")

        dxs_u64_dev = self.cuda_malloc(K * ctypes.sizeof(c_uint64))
        try:
            self.cuda_upload_u64_array(dst_u64_array_dev=dxs_u64_dev, src_ptrs_host=dxs_dev_ptrs)

            if debug_verify_ptrs:
                readback = self.debug_read_u64_array(u64_array_dev=dxs_u64_dev, K=K)
                if readback != [int(p) for p in dxs_dev_ptrs]:
                    raise RuntimeError(f"dx pointer u64 array mismatch: {readback} vs {list(map(int, dxs_dev_ptrs))}")

            if dtype == np.float32:
                st = self.lib.keydnn_cuda_stack_bwd_u64_f32(
                    self._as_dev_ptr(dy_dev), c_int64(K), c_int64(int(pre)), c_int64(int(post)), self._as_dev_ptr(dxs_u64_dev)
                )
            elif dtype == np.float64:
                st = self.lib.keydnn_cuda_stack_bwd_u64_f64(
                    self._as_dev_ptr(dy_dev), c_int64(K), c_int64(int(pre)), c_int64(int(post)), self._as_dev_ptr(dxs_u64_dev)
                )
            else:
                raise TypeError(f"Unsupported dtype for stack_backward_cuda: {dtype}")

            if st != 0:
                self._raise_with_native_debug("keydnn_cuda_stack_bwd_u64 failed with", st)

            if sync:
                self.cuda_synchronize()

        except Exception:
            self.cuda_free(dxs_u64_dev)
            raise

        return dxs_u64_dev


# ---------------------------------------------------------------------
# Functional API (cached singleton like your other wrappers)
# ---------------------------------------------------------------------

_cuda_singleton: CudaLib | None = None


def _get_cuda(lib: ctypes.CDLL) -> CudaLib:
    """
    Get or create a cached `CudaLib` wrapper for a specific `ctypes.CDLL` handle.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.

    Returns
    -------
    CudaLib
        Cached wrapper instance. A new wrapper is created if:
        - no singleton exists yet, or
        - the provided `lib` differs from the cached wrapper's `lib`.
    """
    global _cuda_singleton
    if _cuda_singleton is None or _cuda_singleton.lib is not lib:
        _cuda_singleton = CudaLib(lib)
    return _cuda_singleton


def cuda_set_device(lib: ctypes.CDLL, device: int = 0) -> None:
    """
    Functional wrapper for `CudaLib.cuda_set_device`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    device : int, optional
        CUDA device ordinal. Defaults to 0.
    """
    _get_cuda(lib).cuda_set_device(device)


def cuda_malloc(lib: ctypes.CDLL, nbytes: int) -> DevPtr:
    """
    Functional wrapper for `CudaLib.cuda_malloc`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    nbytes : int
        Number of bytes to allocate.

    Returns
    -------
    DevPtr
        Device pointer to the allocated buffer.
    """
    return _get_cuda(lib).cuda_malloc(nbytes)


def cuda_free(lib: ctypes.CDLL, dev_ptr: DevPtr) -> None:
    """
    Functional wrapper for `CudaLib.cuda_free`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    dev_ptr : DevPtr
        Device pointer to free.
    """
    _get_cuda(lib).cuda_free(dev_ptr)


def cuda_memcpy_h2d(lib: ctypes.CDLL, dst_dev: DevPtr, src_host: np.ndarray) -> None:
    """
    Functional wrapper for `CudaLib.cuda_memcpy_h2d`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    dst_dev : DevPtr
        Destination device pointer.
    src_host : np.ndarray
        Source host array (will be made C-contiguous if needed).
    """
    _get_cuda(lib).cuda_memcpy_h2d(dst_dev, src_host)


def cuda_memcpy_d2h(lib: ctypes.CDLL, dst_host: np.ndarray, src_dev: DevPtr) -> None:
    """
    Functional wrapper for `CudaLib.cuda_memcpy_d2h`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    dst_host : np.ndarray
        Destination host array (must be C-contiguous).
    src_dev : DevPtr
        Source device pointer.
    """
    _get_cuda(lib).cuda_memcpy_d2h(dst_host, src_dev)


def cuda_synchronize(lib: ctypes.CDLL) -> None:
    """
    Functional wrapper for `CudaLib.cuda_synchronize`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    """
    _get_cuda(lib).cuda_synchronize()


def cuda_upload_u64_array(
    lib: ctypes.CDLL, *, dst_u64_array_dev: DevPtr, src_ptrs_host: Sequence[int]
) -> None:
    """
    Functional wrapper for `CudaLib.cuda_upload_u64_array`.

    Parameters
    ----------
    lib : ctypes.CDLL
        Loaded native library handle.
    dst_u64_array_dev : DevPtr
        Device pointer to a `uint64_t[K]` allocation.
    src_ptrs_host : Sequence[int]
        Host pointer values to upload.
    """
    _get_cuda(lib).cuda_upload_u64_array(
        dst_u64_array_dev=dst_u64_array_dev, src_ptrs_host=src_ptrs_host
    )


def stack_forward_cuda(
    lib: ctypes.CDLL,
    *,
    xs_dev_ptrs: Sequence[DevPtr],
    y_dev: DevPtr,
    pre: int,
    post: int,
    dtype: np.dtype,
    sync: bool = True,
    debug_verify_ptrs: bool = False,
) -> DevPtr:
    """
    Functional wrapper for `CudaLib.stack_forward_cuda`.

    Returns
    -------
    DevPtr
        Device pointer to the temporary `uint64_t[K]` pointer-array allocation.
        The caller should free it via `cuda_free(lib, ptr)`.
    """
    return _get_cuda(lib).stack_forward_cuda(
        xs_dev_ptrs=xs_dev_ptrs,
        y_dev=y_dev,
        pre=pre,
        post=post,
        dtype=dtype,
        sync=sync,
        debug_verify_ptrs=debug_verify_ptrs,
    )


def stack_backward_cuda(
    lib: ctypes.CDLL,
    *,
    dy_dev: DevPtr,
    dxs_dev_ptrs: Sequence[DevPtr],
    pre: int,
    post: int,
    dtype: np.dtype,
    sync: bool = True,
    debug_verify_ptrs: bool = True,
) -> DevPtr:
    """
    Functional wrapper for `CudaLib.stack_backward_cuda`.

    Returns
    -------
    DevPtr
        Device pointer to the temporary `uint64_t[K]` pointer-array allocation.
        The caller should free it via `cuda_free(lib, ptr)`.
    """
    return _get_cuda(lib).stack_backward_cuda(
        dy_dev=dy_dev,
        dxs_dev_ptrs=dxs_dev_ptrs,
        pre=pre,
        post=post,
        dtype=dtype,
        sync=sync,
        debug_verify_ptrs=debug_verify_ptrs,
    )
