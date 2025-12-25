# src/keydnn/infrastructure/ops/unary_cuda.py
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from ..native_cuda.python.ops.unary_ctypes import exp_cuda as _exp_ctypes


def _is_cdll(obj: object) -> bool:
    return isinstance(obj, ctypes.CDLL)


def exp_cuda(*args: Any, **kwargs: Any) -> None:
    """
    Ops-layer wrapper for CUDA unary exp.

    IMPORTANT
    ---------
    Tests store this function on the test class, then call it via `self.exp_cuda(...)`.
    That turns it into a *bound method* and injects `self` as the first positional arg.

    Therefore this function must accept *args/**kwargs and manually parse arguments,
    otherwise you'll get:
        TypeError: exp_cuda() got multiple values for argument 'x_dev'
    """
    if not args:
        raise TypeError("exp_cuda expected at least a lib argument")

    # Handle bound-method injection from unittest:
    # - If called as self.exp_cuda(lib, ...), args[0] is the TestCase instance.
    # - If called as exp_cuda(lib, ...), args[0] is ctypes.CDLL.
    if _is_cdll(args[0]):
        lib = args[0]
        rest = args[1:]
    else:
        if len(args) < 2 or not _is_cdll(args[1]):
            raise TypeError(
                "exp_cuda expected ctypes.CDLL as first arg (or second arg when bound as a method)"
            )
        lib = args[1]
        rest = args[2:]

    # Support optional positional dev pointers (rare; tests use kwargs)
    x_dev = kwargs.pop("x_dev", None)
    y_dev = kwargs.pop("y_dev", None)

    # tolerate legacy keyword spellings
    if x_dev is None:
        x_dev = kwargs.pop("X_dev", None) or kwargs.pop("x", None) or kwargs.pop("X", None)
    if y_dev is None:
        y_dev = kwargs.pop("Y_dev", None) or kwargs.pop("y", None) or kwargs.pop("Y", None)

    # If still missing, try positional parsing: (x_dev, y_dev)
    if x_dev is None and len(rest) >= 1:
        x_dev = rest[0]
    if y_dev is None and len(rest) >= 2:
        y_dev = rest[1]

    if x_dev is None or y_dev is None:
        raise TypeError("exp_cuda requires x_dev and y_dev device pointers")

    numel = kwargs.pop("numel", None)
    dtype = kwargs.pop("dtype", None)
    _sync = kwargs.pop("sync", True)  # accepted for compatibility; ignored here

    if numel is None or dtype is None:
        raise TypeError("exp_cuda requires numel and dtype")

    if int(numel) <= 0:
        raise ValueError(f"exp_cuda requires numel > 0, got {numel}")

    _exp_ctypes(
        lib,
        x_dev=int(x_dev),
        y_dev=int(y_dev),
        numel=int(numel),
        dtype=np.dtype(dtype),
    )


# Aliases so resolve_func keeps working even if candidates change
cuda_exp = exp_cuda
exp = exp_cuda
unary_exp_cuda = exp_cuda

__all__ = ["exp_cuda", "cuda_exp", "exp", "unary_exp_cuda"]
