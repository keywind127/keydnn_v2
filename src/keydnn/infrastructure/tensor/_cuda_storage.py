"""
CUDA storage and lifetime management utilities.

This module defines `_CudaStorage`, a reference-counted wrapper around a single
CUDA device allocation. It provides a unified abstraction for managing CUDA
memory ownership, sharing, and deterministic cleanup across multiple tensors.

Motivation
----------
Historically, CUDA tensors in the codebase were represented directly by raw
device pointers (`dev_ptr`). While simple, this approach makes it difficult to:

- Share device memory safely across multiple tensors (e.g., views, reshapes).
- Define clear ownership and lifetime semantics.
- Avoid memory leaks or double-frees during autograd and graph traversal.
- Gradually migrate legacy code without breaking existing kernels or tests.

`_CudaStorage` addresses these issues by introducing an explicit storage layer
with reference counting and controlled finalization.

Core Concepts
-------------
- **Storage-backed tensors**:
    Tensors created via `_from_storage` or `_ensure_cuda_alloc` own a reference
    to a `_CudaStorage` object. The underlying device memory is freed exactly
    once when the last reference is released.

- **Borrowed storage**:
    Legacy or externally provided device pointers can be wrapped in a
    `_CudaStorage` instance with its finalizer detached. This allows uniform
    handling of storage without taking ownership of the memory.

- **Reference counting**:
    Each `_CudaStorage` maintains an explicit reference count. Operations such
    as view creation, reshape, or detach-view increment the count, while tensor
    destruction or `free_()` decrements it.

- **Finalization safety**:
    A `weakref.finalize` callback acts as a safety net to free device memory
    during garbage collection if explicit cleanup is missed. All finalizers
    are best-effort and never raise exceptions.

Thread Safety
-------------
Reference count updates are protected by an internal lock, allowing safe sharing
of storage objects across threads. CUDA calls themselves are assumed to be
thread-safe at the driver level or externally synchronized.

Migration Strategy
------------------
This module is a foundational part of the gradual migration from raw `dev_ptr`
usage to storage-backed CUDA tensors:

- New code should always allocate via `_ensure_cuda_alloc` and construct tensors
  via `_from_storage`.
- `_from_devptr` and `_cuda_ensure_storage` exist only to support legacy or
  externally owned device pointers.
- Over time, devptr-only tensors can be phased out without breaking ABI or
  kernel interfaces.

Design Notes
------------
- `_CudaStorage` intentionally avoids defining `__del__` to prevent common
  garbage-collection pitfalls and interpreter shutdown issues.
- The storage abstraction is lightweight and does not impose layout, stride,
  or shape semantics; these remain the responsibility of higher-level tensor
  logic.
- This module is infrastructure-layer code and may depend on ctypes-based CUDA
  bindings.

Summary
-------
`_CudaStorage` provides a safe, explicit, and extensible foundation for CUDA
memory management, enabling correct tensor sharing, predictable lifetimes, and
a clean transition away from raw device-pointer-based designs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import threading
import weakref

import numpy as np


@dataclass
class _CudaStorage:
    """
    Reference-counted wrapper around a CUDA device allocation.

    This class represents a single contiguous CUDA memory allocation that may
    be shared by multiple tensors (e.g., via reshape, view, or detach-view
    operations). The underlying device memory is freed **exactly once** when
    the last owning reference is released.

    Design
    ------
    - Storage objects are shared across tensors instead of duplicating device
      memory.
    - Lifetime is managed via explicit reference counting (`incref` / `decref`)
      combined with a `weakref.finalize` fallback for GC-time cleanup.
    - The finalizer ensures that device memory is freed even if explicit
      decref calls are missed, while still allowing deterministic frees when
      reference count reaches zero.

    Ownership semantics
    -------------------
    - A storage instance represents **owned** CUDA memory unless its finalizer
      has been explicitly detached (borrowed storage).
    - Borrowed storage wrappers must detach the finalizer to avoid freeing
      memory owned by external code.
    - When `_refcnt` reaches zero, the device pointer is freed immediately and
      invalidated.

    Thread safety
    -------------
    - Reference count updates are protected by an internal lock to allow safe
      sharing across threads.
    - CUDA calls themselves are assumed to be thread-safe at the driver level
      or externally synchronized.

    Notes
    -----
    - This class intentionally avoids defining `__del__` to prevent common
      garbage-collection pitfalls involving reference cycles and interpreter
      shutdown ordering.
    - All CUDA calls are best-effort and must never raise during finalization.
    """

    lib: object
    device_index: int
    dev_ptr: int
    nbytes: int
    dtype: np.dtype

    _refcnt: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # A finalizer that will free dev_ptr if it is still owned at GC-time.
    _finalizer: weakref.finalize | None = None

    def __post_init__(self) -> None:
        """
        Initialize the CUDA storage finalizer.

        This method installs a `weakref.finalize` callback that frees the underlying
        CUDA device memory when the storage object is garbage-collected, provided
        that it is still owned.

        Behavior
        --------
        - The finalizer captures only primitive values (library handle, device
        index, and device pointer) to avoid reference cycles.
        - The CUDA device is explicitly set before freeing to ensure correctness
        on multi-device systems.
        - The finalizer is only installed if the device pointer is non-zero and
        the allocation size is positive.

        Notes
        -----
        - The finalizer acts as a **safety net**; normal code paths should release
        storage deterministically via `decref()`.
        - Any exceptions raised during finalization are suppressed, as required
        for safe interpreter shutdown behavior.
        """
        # Lazily bind finalizer to avoid __del__ pitfalls.
        # We capture only plain values, not self, to avoid cycles.
        import ctypes

        # Import your cuda_free wrapper (use the same module family you already use)
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_free,
        )

        lib = self.lib
        device_index = int(self.device_index)
        dev_ptr = int(self.dev_ptr)

        def _free_ptr() -> None:
            # Best-effort: at interpreter shutdown modules may be None
            try:
                cuda_set_device(lib, device_index)
                cuda_free(lib, dev_ptr)
            except Exception:
                # Never raise in finalizers
                pass

        if dev_ptr != 0 and int(self.nbytes) > 0:
            self._finalizer = weakref.finalize(self, _free_ptr)

    def incref(self) -> None:
        """
        Increment the storage reference count.

        This method indicates that an additional tensor or view now shares
        ownership of this storage. The underlying device memory will remain
        allocated until a matching number of `decref()` calls are performed.

        Notes
        -----
        - This operation is thread-safe.
        - This method does not allocate or free memory.
        """
        with self._lock:
            self._refcnt += 1

    def decref(self) -> None:
        """
        Decrement the storage reference count and free memory if it reaches zero.

        When the reference count drops to zero, this method triggers immediate,
        deterministic cleanup of the underlying CUDA device memory by invoking
        the registered finalizer (if still alive).

        Behavior
        --------
        - If the reference count becomes zero:
            - The CUDA memory is freed exactly once.
            - The device pointer and size metadata are invalidated.
        - If the storage was created as borrowed and its finalizer was detached,
        no device memory is freed.

        Notes
        -----
        - This operation is thread-safe.
        - This method is idempotent with respect to memory freeing: subsequent
        calls after refcount reaches zero will have no effect.
        - After this call frees memory, the storage object should be considered
        logically invalid and must not be reused.
        """
        with self._lock:
            self._refcnt -= 1
            if self._refcnt == 0:
                # Trigger finalizer now (deterministic), if still alive.
                if self._finalizer is not None and self._finalizer.alive:
                    self._finalizer()
                self.dev_ptr = 0
                self.nbytes = 0
