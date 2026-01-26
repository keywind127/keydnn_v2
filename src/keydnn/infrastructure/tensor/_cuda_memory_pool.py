# _cuda_memory_pool.py
"""
CUDA device-memory caching allocator with bucketed allocations and exact-size API.

This module implements a **per-device CUDA memory pool** that reduces the
overhead of repeated `cudaMalloc` / `cudaFree` calls by caching freed device
allocations for later reuse.

Key idea
--------
Callers request memory using an **exact requested size** (`req_nbytes`), while
the allocator may internally round up to a **bucketed allocation size**
(`alloc_nbytes`) to improve reuse and reduce fragmentation.

To keep correctness simple (especially when raw device pointers are exposed
across the system), this pool enforces a critical invariant:

    Every dev_ptr returned by this pool is tracked in a pointer map:
        dev_ptr -> alloc_nbytes

This removes ambiguity during `free()` and prevents unsafe “guessing” of bucket
sizes from requested sizes.

Why track two sizes?
--------------------
- `req_nbytes` (external): the exact size the caller logically needs
  (e.g., `numel * dtype.itemsize`).
- `alloc_nbytes` (internal): the actual size passed to `cudaMalloc`, typically
  a rounded bucket size.

The pool uses `alloc_nbytes` for caching/reuse decisions, while callers keep
using `req_nbytes` for tensor semantics (shape/dtype) and memcpy boundaries.

Design Goals
------------
- Avoid repeated cuda allocation churn on hot paths.
- Provide deterministic, debuggable behavior via explicit metadata tracking.
- Prevent size mismatch bugs by never deriving alloc sizes from req sizes at free time.

Non-goals
---------
- Maximum reuse efficiency (still decent with buckets, but not as aggressive as
  sophisticated allocators).
- Multi-stream safety via CUDA events (not implemented here).
- Cross-context pooling (each process and DLL handle is assumed consistent).

Assumptions / Streaming warning
-------------------------------
This allocator assumes either:
- use the default CUDA stream only, OR
- synchronize appropriately before blocks are freed/reused.

If multi-stream execution is introduced, this allocator should be extended
to record CUDA events on `free()` and only recycle blocks after event completion.

Safety notes
------------
- If a pointer was **not allocated by this pool** (not present in the pointer map),
  `free()` conservatively falls back to `cudaFree` (best-effort).
- Cleanup operations suppress exceptions to remain safe during interpreter shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import threading


def _round_up_power_of_two(n: int) -> int:
    """
    Round `n` up to the smallest power of two >= n.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    int
        Smallest power-of-two >= n. Returns 0 if n <= 0.
    """
    n = int(n)
    if n <= 0:
        return 0
    return 1 << (n - 1).bit_length()


def _bucket_size(req_nbytes: int, *, min_bucket: int = 256) -> int:
    """
    Convert a requested byte size into an allocation bucket size.

    Parameters
    ----------
    req_nbytes : int
        Requested number of bytes.
    min_bucket : int, optional
        Minimum bucket size. Small allocations are rounded up to at least this.

    Returns
    -------
    int
        Bucketed allocation size (power-of-two) >= max(req_nbytes, min_bucket).
        Returns 0 if req_nbytes <= 0.
    """
    req = int(req_nbytes)
    if req <= 0:
        return 0
    if req < int(min_bucket):
        req = int(min_bucket)
    return _round_up_power_of_two(req)


@dataclass
class _Block:
    """
    A cached CUDA allocation.

    Attributes
    ----------
    ptr : int
        CUDA device pointer (uintptr_t represented as Python int).
    alloc_nbytes : int
        Actual allocation size in bytes that was passed to cudaMalloc (bucket size).
    """

    ptr: int
    alloc_nbytes: int


@dataclass
class _DevicePool:
    """
    Per-device allocator state.

    Attributes
    ----------
    free : Dict[int, List[_Block]]
        Free lists keyed by allocation bucket size (alloc_nbytes).
    live_alloc : Dict[int, int]
        Mapping: dev_ptr -> alloc_nbytes for pointers currently handed out by the pool.
        This is the critical metadata that makes free() unambiguous.
    cached_bytes : int
        Total bytes currently cached (sum of alloc_nbytes of free blocks).
    allocated_bytes : int
        Cumulative bytes ever allocated via cudaMalloc (sum of alloc_nbytes).
        This is a diagnostic counter (not “currently live bytes”).

    Instrumentation counters
    ------------------------
    hits : int
        Number of cache hits in malloc().
    misses : int
        Number of cache misses in malloc() (resulting in cudaMalloc attempt).
    cuda_malloc_calls : int
        Number of actual cudaMalloc calls performed by the pool.
    cuda_free_calls : int
        Number of actual cudaFree calls performed by the pool (best-effort counter).
    """

    free: Dict[int, List[_Block]] = field(default_factory=dict)
    live_alloc: Dict[int, int] = field(default_factory=dict)
    cached_bytes: int = 0
    allocated_bytes: int = 0

    # --- Counters (debug / diagnostics) ---
    hits: int = 0
    misses: int = 0
    cuda_malloc_calls: int = 0
    cuda_free_calls: int = 0


class CudaMemoryPool:
    """
    A per-device caching allocator for CUDA devptrs with bucketed allocations.

    Public API uses **exact requested sizes**:
    - `malloc(req_nbytes)` returns a pointer valid for at least `req_nbytes`.
    - `free(dev_ptr, req_nbytes=...)` returns the pointer to the pool or frees it.

    Internally, allocations are rounded up to an `alloc_nbytes` bucket size to
    increase reuse. The pool tracks `dev_ptr -> alloc_nbytes` so that `free()`
    never needs to recompute or guess bucket size.

    Parameters
    ----------
    max_cached_bytes_per_device : int, optional
        Maximum cached bytes per device. When exceeded, freed blocks are returned
        to CUDA via `cudaFree` instead of being cached.
    min_bucket : int, optional
        Minimum bucket size. Requests smaller than this will be rounded up to
        at least `min_bucket`, then to the next power-of-two.

    Notes
    -----
    - Thread-safe (single lock protecting all devices).
    - Per-device isolation: blocks are never reused across devices.
    - Not stream-aware: callers must synchronize appropriately if using non-default streams.
    """

    def __init__(
        self,
        *,
        max_cached_bytes_per_device: int = 512 * 1024 * 1024,  # 512MB
        min_bucket: int = 256,
    ) -> None:
        """
        Initialize the CUDA memory pool.

        Parameters
        ----------
        max_cached_bytes_per_device : int, optional
            Max bytes to cache per device. Defaults to 512MB.
        min_bucket : int, optional
            Minimum bucket size in bytes. Defaults to 256 bytes.
        """
        self._max_cached_bytes_per_device = int(max_cached_bytes_per_device)
        self._min_bucket = int(min_bucket)
        self._lock = threading.RLock()
        self._dev: Dict[int, _DevicePool] = {}

    def _get_devpool(self, device_index: int) -> _DevicePool:
        """
        Get or create the per-device pool state.

        Parameters
        ----------
        device_index : int
            CUDA device index.

        Returns
        -------
        _DevicePool
            Pool state for the specified device.
        """
        di = int(device_index)
        dp = self._dev.get(di)
        if dp is None:
            dp = _DevicePool()
            self._dev[di] = dp
        return dp

    def malloc(self, lib: object, device_index: int, nbytes: int) -> int:
        """
        Allocate (or reuse) a CUDA device buffer for at least `nbytes`.

        This function exposes an exact-size API: caller requests `nbytes`.
        Internally we may allocate a larger bucket (`alloc_nbytes`) for reuse.

        Instrumentation
        --------------
        This function maintains per-device hit/miss counters:
        - A "hit" occurs when an allocation is satisfied from the cache.
        - A "miss" occurs when the cache does not contain a usable block and
          we must fall back to cudaMalloc.

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int
            CUDA device index.
        nbytes : int
            Requested number of bytes.

        Returns
        -------
        int
            CUDA devptr (uintptr_t) as Python int.
            Returns 0 on allocation failure or if nbytes <= 0.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_malloc,
        )

        req = int(nbytes)
        if req <= 0:
            return 0

        alloc = _bucket_size(req, min_bucket=self._min_bucket)

        # Try bucketed cache hit
        with self._lock:
            dp = self._get_devpool(device_index)
            freelist = dp.free.get(alloc)
            if freelist:
                dp.hits += 1

                blk = freelist.pop()
                dp.cached_bytes -= int(blk.alloc_nbytes)
                if not freelist:
                    # keep dict tidy
                    dp.free.pop(alloc, None)

                ptr = int(blk.ptr)
                dp.live_alloc[ptr] = int(blk.alloc_nbytes)
                return ptr

            # cache miss (we will fall back to cudaMalloc)
            dp.misses += 1

        # Cache miss => cudaMalloc(bucket size)
        cuda_set_device(lib, int(device_index))
        ptr = int(cuda_malloc(lib, int(alloc)))
        if ptr == 0:
            return 0

        with self._lock:
            dp = self._get_devpool(device_index)
            dp.cuda_malloc_calls += 1
            dp.allocated_bytes += int(alloc)
            dp.live_alloc[int(ptr)] = int(alloc)

        return int(ptr)

    def free(
        self,
        lib: object,
        device_index: int,
        dev_ptr: int,
        nbytes: Optional[int] = None,
    ) -> None:
        """
        Return a CUDA device buffer to the cache or free it.

        This pool is bucketed internally, so the *actual* allocation size may be
        larger than the caller's requested size. Therefore, `free()` does NOT
        recompute buckets from `nbytes`. Instead, it consults the internal pointer
        map:

            alloc_nbytes = live_alloc.pop(dev_ptr)

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int
            CUDA device index.
        dev_ptr : int
            CUDA device pointer to release.
        nbytes : int or None, optional
            Requested size originally used by the caller (debug-only).
            If provided, we sanity-check `nbytes <= alloc_nbytes`.
            If not provided, the pool can still free safely using pointer metadata.

        Behavior
        --------
        - If dev_ptr was allocated by this pool and caching capacity allows:
            - it is returned to the cache under its alloc bucket.
        - If the cache is full:
            - it is released via cudaFree.
        - If dev_ptr was NOT allocated by this pool:
            - it is conservatively released via cudaFree (best-effort).

        Notes
        -----
        - If `nbytes` is provided and `nbytes > alloc_nbytes`, we treat it as a
          serious bug and fall back to cudaFree (do not cache).

        Instrumentation (best-effort)
        -----------------------------
        - `cuda_free_calls` is incremented when this method chooses to call
          cudaFree. The increment is done without adding any new lock acquisitions
          around CUDA calls to avoid introducing deadlock/hang risks.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_free,
        )

        ptr = int(dev_ptr)
        if ptr == 0:
            return

        # Lookup allocation size from metadata (do NOT guess)
        with self._lock:
            dp = self._get_devpool(device_index)
            alloc = dp.live_alloc.pop(ptr, None)

        if alloc is None:
            # Pointer not owned/tracked by this pool: safest is cudaFree.
            # best-effort counter update WITHOUT acquiring the pool lock
            dp0 = self._dev.get(int(device_index))
            if dp0 is not None:
                dp0.cuda_free_calls += 1

            cuda_set_device(lib, int(device_index))
            try:
                cuda_free(lib, ptr)
            except Exception:
                pass
            return

        # Optional sanity check against caller's requested bytes
        if nbytes is not None:
            req = int(nbytes)
            if req > int(alloc):
                # Do not cache inconsistent pointer; free immediately.
                # best-effort counter update WITHOUT acquiring the pool lock
                with self._lock:
                    dp = self._get_devpool(device_index)
                    dp.cuda_free_calls += 1

                cuda_set_device(lib, int(device_index))
                try:
                    cuda_free(lib, ptr)
                except Exception:
                    pass
                return

        # Try to cache the block
        with self._lock:
            dp = self._get_devpool(device_index)
            if dp.cached_bytes + int(alloc) <= self._max_cached_bytes_per_device:
                dp.free.setdefault(int(alloc), []).append(
                    _Block(ptr=int(ptr), alloc_nbytes=int(alloc))
                )
                dp.cached_bytes += int(alloc)
                return

        # Cache full => cudaFree
        with self._lock:
            dp = self._get_devpool(device_index)
            dp.cuda_free_calls += 1

        cuda_set_device(lib, int(device_index))
        try:
            cuda_free(lib, ptr)
        except Exception:
            pass

    def empty_cache(self, lib: object, device_index: Optional[int] = None) -> None:
        """
        Release cached blocks back to CUDA via cudaFree.

        Parameters
        ----------
        lib : object
            Loaded CUDA shared library handle.
        device_index : int or None, optional
            If provided, clears only that device's cache.
            If None, clears caches for all devices.

        Notes
        -----
        - Only cached blocks are freed. Live blocks (currently checked out) are untouched.
        - Frees are best-effort; exceptions are suppressed.
        """
        from ...infrastructure.native_cuda.python.avgpool2d_ctypes import (
            cuda_set_device,
            cuda_free,
        )

        to_free: List[Tuple[int, int]] = []  # (device_index, ptr)

        with self._lock:
            dev_indices = (
                list(self._dev.keys()) if device_index is None else [int(device_index)]
            )

            for di in dev_indices:
                dp = self._dev.get(int(di))
                if dp is None:
                    continue

                for freelist in dp.free.values():
                    for blk in freelist:
                        to_free.append((int(di), int(blk.ptr)))

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
        Return allocator statistics for debugging/diagnostics.

        Returns
        -------
        dict
            Mapping device_index -> stats dict with keys:
            - cached_bytes
            - allocated_bytes
            - num_cached_blocks
            - num_buckets
            - num_live_blocks
            - hits
            - misses
            - cuda_malloc_calls
            - cuda_free_calls
        """
        with self._lock:
            out = {}
            for di, dp in self._dev.items():
                out[int(di)] = {
                    "cached_bytes": int(dp.cached_bytes),
                    "allocated_bytes": int(dp.allocated_bytes),
                    "num_cached_blocks": int(sum(len(v) for v in dp.free.values())),
                    "num_buckets": int(len(dp.free)),
                    "num_live_blocks": int(len(dp.live_alloc)),
                    # instrumentation
                    "hits": int(dp.hits),
                    "misses": int(dp.misses),
                    "cuda_malloc_calls": int(dp.cuda_malloc_calls),
                    "cuda_free_calls": int(dp.cuda_free_calls),
                }
            return out


# Global singleton pool used throughout the CUDA tensor infrastructure.
GLOBAL_CUDA_MEMORY_POOL = CudaMemoryPool()
