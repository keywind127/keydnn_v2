"""
CUDA device-memory caching allocator with exact-size reuse semantics.

This module implements a **per-device CUDA memory pool** that reuses device
allocations **only when the requested size matches exactly**. Unlike traditional
bucketed or rounded allocators, this design preserves a strict invariant:

    (dev_ptr, nbytes) always refers to an allocation of exactly `nbytes`.

This invariant is critical for KeyDNN, where raw device pointers may be exposed
to higher layers (e.g., Tensor storage, memcpy boundaries, and unit tests) and
where copy sizes are derived from tensor shape and dtype rather than allocator
metadata.

Design Goals
------------
- Avoid repeated `cudaMalloc` / `cudaFree` churn for hot paths.
- Preserve strict correctness guarantees around pointer size.
- Eliminate size-rounding and bucket aliasing as sources of undefined behavior.
- Keep the allocator simple, debuggable, and predictable.

Non-goals
---------
- Maximum reuse efficiency (intentionally reduced).
- Stream-aware reuse or event-based synchronization.
- Cross-device or cross-context pooling.

Assumptions
-----------
- Callers either use the default CUDA stream or ensure proper synchronization
  before returning memory to the pool.
- Callers provide the **exact allocation size** when freeing a pointer.
- All CUDA calls are best-effort and must not raise during cleanup paths.

This module lives in the infrastructure layer and is used by `_CudaStorage`
to implement deterministic and safe CUDA memory lifetime management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import threading


@dataclass
class _Block:
    """
    A cached CUDA allocation.

    Parameters
    ----------
    ptr : int
        CUDA device pointer (uintptr_t as Python int).
    size : int
        Exact allocation size in bytes.
    """

    ptr: int
    size: int


@dataclass
class _DevicePool:
    """
    Per-device memory pool state.

    Attributes
    ----------
    free : Dict[int, List[_Block]]
        Free lists keyed by **exact allocation size in bytes**.
    cached_bytes : int
        Total number of bytes currently cached for this device.
    allocated_bytes : int
        Cumulative number of bytes ever allocated via cudaMalloc
        (exact sum, not current live bytes).
    """

    free: Dict[int, List[_Block]] = field(default_factory=dict)
    cached_bytes: int = 0
    allocated_bytes: int = 0


class CudaMemoryPool:
    """
    A per-device caching allocator for CUDA devptrs using **exact sizes**.

    Unlike bucketed allocators that round allocation sizes, this pool reuses
    a cached block **only if its size exactly matches the requested size**.
    This eliminates ambiguity around how many bytes are valid for a given
    device pointer and avoids subtle size-mismatch bugs during memcpy or
    kernel execution.

    Design
    ------
    - `malloc(nbytes)`:
        - Reuses a cached block only if `nbytes` matches exactly.
        - Otherwise calls `cudaMalloc(nbytes)`.
    - `free(dev_ptr, nbytes)`:
        - Returns the block to the cache under its exact size.
        - Falls back to `cudaFree` if the cache is full or size is unknown.
    - Thread-safe.
    - Pools are isolated per CUDA device.

    Notes
    -----
    - This allocator is intentionally conservative.
    - It trades reuse efficiency for correctness and debuggability.
    - Suitable for systems where raw dev_ptrs and explicit memcpy sizes
      are part of the public contract (as in KeyDNN).

    Streaming Warning
    -----------------
    This pool assumes either:
    - default CUDA stream usage, or
    - correct external synchronization before reuse.

    If multi-stream execution is introduced, this allocator must be extended
    with event-based deferred reuse.
    """

    def __init__(
        self,
        *,
        max_cached_bytes_per_device: int = 512 * 1024 * 1024,  # 512MB
    ) -> None:
        """
        Initialize a CUDA memory pool.

        Parameters
        ----------
        max_cached_bytes_per_device : int, optional
            Maximum number of bytes to cache per device before falling back
            to `cudaFree`. Defaults to 512 MB.
        """
        self._max_cached_bytes_per_device = int(max_cached_bytes_per_device)
        self._lock = threading.Lock()
        self._dev: Dict[int, _DevicePool] = {}

    def _get_devpool(self, device_index: int) -> _DevicePool:
        """
        Retrieve or create the pool state for a specific CUDA device.

        Parameters
        ----------
        device_index : int
            CUDA device index.

        Returns
        -------
        _DevicePool
            The per-device pool state.
        """
        di = int(device_index)
        dp = self._dev.get(di)
        if dp is None:
            dp = _DevicePool()
            self._dev[di] = dp
        return dp

    def malloc(self, lib: object, device_index: int, nbytes: int) -> int:
        """
        Allocate (or reuse) a CUDA device buffer of **exactly** `nbytes`.

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int
            CUDA device index.
        nbytes : int
            Exact number of bytes to allocate.

        Returns
        -------
        int
            CUDA device pointer (uintptr_t as int).
            Returns 0 on allocation failure.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_malloc,
        )

        req = int(nbytes)
        if req <= 0:
            return 0

        # Exact-size cache lookup
        with self._lock:
            dp = self._get_devpool(device_index)
            freelist = dp.free.get(req)
            if freelist:
                blk = freelist.pop()
                dp.cached_bytes -= blk.size
                if not freelist:
                    dp.free.pop(req, None)
                return int(blk.ptr)

        # Cache miss => real cudaMalloc(EXACT size)
        cuda_set_device(lib, int(device_index))
        ptr = int(cuda_malloc(lib, int(req)))
        if ptr == 0:
            return 0

        with self._lock:
            dp = self._get_devpool(device_index)
            dp.allocated_bytes += req

        return ptr

    def free(self, lib: object, device_index: int, dev_ptr: int, nbytes: int) -> None:
        """
        Return a CUDA device buffer to the cache or free it.

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int
            CUDA device index.
        dev_ptr : int
            CUDA device pointer to release.
        nbytes : int
            **Exact allocation size** used when the pointer was allocated.

        Important
        ---------
        - `nbytes` MUST match the size originally passed to `malloc`.
        - If the size is unknown or invalid, the pointer is freed immediately
          via `cudaFree` to avoid unsafe reuse.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_free,
        )

        ptr = int(dev_ptr)
        if ptr == 0:
            return

        sz = int(nbytes)
        if sz <= 0:
            # Unknown size => do not cache
            cuda_set_device(lib, int(device_index))
            try:
                cuda_free(lib, ptr)
            except Exception:
                pass
            return

        # Attempt to cache
        with self._lock:
            dp = self._get_devpool(device_index)
            if dp.cached_bytes + sz <= self._max_cached_bytes_per_device:
                dp.free.setdefault(sz, []).append(_Block(ptr=ptr, size=sz))
                dp.cached_bytes += sz
                return

        # Cache full => cudaFree
        cuda_set_device(lib, int(device_index))
        try:
            cuda_free(lib, ptr)
        except Exception:
            pass

    def empty_cache(self, lib: object, device_index: Optional[int] = None) -> None:
        """
        Release all cached CUDA allocations back to the driver.

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int or None, optional
            If provided, clears only the specified device.
            If None, clears caches for all devices.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_free,
        )

        to_free: List[Tuple[int, int]] = []

        with self._lock:
            dev_indices = (
                list(self._dev.keys()) if device_index is None else [int(device_index)]
            )
            for di in dev_indices:
                dp = self._dev.get(di)
                if dp is None:
                    continue
                for freelist in dp.free.values():
                    for blk in freelist:
                        to_free.append((di, int(blk.ptr)))
                dp.free.clear()
                dp.cached_bytes = 0

        for di, ptr in to_free:
            try:
                cuda_set_device(lib, int(di))
                cuda_free(lib, int(ptr))
            except Exception:
                pass

    def stats(self) -> dict:
        """
        Return allocator statistics for debugging and diagnostics.

        Returns
        -------
        dict
            Mapping of device index to allocator statistics, including:
            - cached_bytes
            - allocated_bytes
            - num_cached_blocks
            - num_sizes (distinct exact sizes cached)
        """
        with self._lock:
            out = {}
            for di, dp in self._dev.items():
                out[int(di)] = {
                    "cached_bytes": int(dp.cached_bytes),
                    "allocated_bytes": int(dp.allocated_bytes),
                    "num_cached_blocks": int(sum(len(v) for v in dp.free.values())),
                    "num_sizes": int(len(dp.free)),
                }
            return out


# Global singleton pool used throughout the CUDA tensor infrastructure.
GLOBAL_CUDA_MEMORY_POOL = CudaMemoryPool()
