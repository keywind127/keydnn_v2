"""
Utility helpers for data preparation and interop.

This module provides small, framework-agnostic helper functions that simplify
common tasks when working with KeyDNN, such as:

- Converting integer class labels into one-hot encoded NumPy arrays.
- Converting NumPy arrays into `Tensor` objects with optional device placement
  and gradient tracking.

These utilities are intentionally lightweight and side-effect free, and are
designed to sit at the boundary between NumPy-based data pipelines and the
KeyDNN tensor system.
"""

from typing import Optional
import numpy as np
import warnings

from ...domain.device._device import Device
from ..tensor._tensor import Tensor


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels to a one-hot encoded NumPy array.

    This function takes a 1D array of integer labels and returns a 2D
    one-hot encoded matrix with `float32` dtype.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer class labels. The input is flattened internally,
        so both shape `(N,)` and `(N, 1)` are supported.
    num_classes : int
        Total number of classes. Each label value must satisfy
        `0 <= label < num_classes`.

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape `(N, num_classes)` with dtype
        `float32`, where `N` is the number of labels.

    Notes
    -----
    - No bounds checking is performed beyond NumPy indexing semantics;
      invalid label values will raise an `IndexError`.
    - This function is intended for dataset preprocessing and loss
      computation (e.g., classification targets).
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def numpy_to_tensor(
    arr: np.ndarray,
    device: Optional[Device] = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Convert a NumPy array into a KeyDNN Tensor.

    This function creates a new `Tensor` instance with the same shape as
    the given NumPy array, copies the data into the tensor's internal
    storage, and optionally enables gradient tracking.

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array. The data is converted to `float32` internally
        before being copied into the tensor.
    device : Device, optional
        Target device on which to allocate the tensor. If `None`, the
        framework default device is used.
    requires_grad : bool, default=False
        Whether the resulting tensor should participate in autograd and
        accumulate gradients during backpropagation.

    Returns
    -------
    Tensor
        A newly allocated tensor containing a copy of the input data.

    Notes
    -----
    - The returned tensor does **not** share memory with the input NumPy
      array; the data is copied explicitly.
    - This function uses only public `Tensor` APIs and is safe to use in
      user-facing code, tests, and examples.
    - Gradient buffers are allocated lazily according to the tensor's
      `requires_grad` setting and the autograd engine behavior.
    """
    if device is None:
        device = Device("cpu")
        warnings.warn(
            "No device specified for numpy_to_tensor; using default CPU device.",
            stacklevel=2,
        )
    a = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=a.shape, device=device, requires_grad=requires_grad)
    t.copy_from_numpy(a)
    return t
