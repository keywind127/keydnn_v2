"""
Domain-level structural typing for NumPy-like n-dimensional arrays.

This module defines :class:`NDArrayLike`, a **backend-agnostic Protocol**
representing objects that behave like NumPy ``ndarray`` instances, without
introducing a dependency on NumPy in the domain layer.

Design intent
-------------
- **PIAD-compliant**: avoids importing infrastructure libraries (e.g., NumPy).
- **Structural typing**: any object satisfying the protocol is accepted,
  regardless of its concrete implementation.
- **Typing precision**: prevents ``Any`` leakage when domain interfaces expose
  array-like data (e.g., via ``ITensor.to_numpy()``).
- **Minimal but practical**: models only the most commonly used ndarray APIs
  required by higher-level tensor logic.

Typical implementers include:
- ``numpy.ndarray``
- Array views or wrappers returned by infrastructure layers
- Backend-specific arrays (e.g., CuPy, JAX) that emulate ndarray semantics

This protocol is intended for **typing and documentation purposes only** and
does not impose runtime behavior beyond optional ``runtime_checkable`` support.
"""

from __future__ import annotations
from typing import Protocol, Tuple, Any, Iterable, overload, runtime_checkable


@runtime_checkable
class NDArrayLike(Protocol):
    """
    NDArrayLike (N-Dimensional Array–Like)

    A structural typing interface for objects that behave like NumPy ndarrays.

    This protocol captures a commonly used subset of ndarray attributes and
    methods, enabling domain-layer code to reason about array-shaped data
    without importing or depending on NumPy or other numerical backends.

    Notes
    -----
    - This is a *Protocol*, not a concrete base class.
    - Implementers are expected to follow NumPy-like semantics, but exact
      behavior (e.g., view vs copy) is backend-defined.
    - The API surface is intentionally limited to avoid re-implementing NumPy
      in the domain layer.
    """

    # -----------------------------
    # Core properties
    # -----------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the array as a tuple of dimension sizes.

        Returns
        -------
        Tuple[int, ...]
            The size of each dimension.
        """
        ...

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the array.

        Returns
        -------
        int
            The dimensionality (rank) of the array.
        """
        ...

    @property
    def size(self) -> int:
        """
        Total number of elements in the array.

        Returns
        -------
        int
            Product of all entries in :pyattr:`shape`.
        """
        ...

    @property
    def dtype(self) -> Any:
        """
        Data type descriptor of the array elements.

        Returns
        -------
        Any
            Backend-defined dtype object (e.g., ``numpy.dtype``).
        """
        ...

    # -----------------------------
    # Reshaping / layout
    # -----------------------------

    def reshape(self, *shape: int) -> NDArrayLike:
        """
        Return an array with a new shape.

        Parameters
        ----------
        *shape : int
            New shape dimensions.

        Returns
        -------
        NDArrayLike
            Reshaped array-like object.
        """
        ...

    def transpose(self, *axes: int) -> NDArrayLike:
        """
        Return a view of the array with permuted axes.

        Parameters
        ----------
        *axes : int
            Permutation of axis indices.

        Returns
        -------
        NDArrayLike
            Transposed array-like object.
        """
        ...

    @property
    def T(self) -> NDArrayLike:
        """
        Transposed view of the array.

        Equivalent to ``transpose()`` with reversed axes.

        Returns
        -------
        NDArrayLike
            Transposed array-like object.
        """
        ...

    def squeeze(self, axis: int | None = None) -> NDArrayLike:
        """
        Remove axes of length one from the array shape.

        Parameters
        ----------
        axis : int or None, optional
            Specific axis to remove. If None, all singleton axes are removed.

        Returns
        -------
        NDArrayLike
            Squeezed array-like object.
        """
        ...

    def ravel(self) -> NDArrayLike:
        """
        Return a flattened 1D view of the array when possible.

        Returns
        -------
        NDArrayLike
            Flattened array-like object.
        """
        ...

    def flatten(self) -> NDArrayLike:
        """
        Return a flattened 1D copy of the array.

        Returns
        -------
        NDArrayLike
            Flattened array-like object.
        """
        ...

    # -----------------------------
    # Type / conversion
    # -----------------------------

    def astype(self, dtype: Any, copy: bool = ...) -> NDArrayLike:
        """
        Cast the array to a specified data type.

        Parameters
        ----------
        dtype : Any
            Target data type.
        copy : bool, optional
            Whether to force a copy even if the dtype is unchanged.

        Returns
        -------
        NDArrayLike
            Array-like object with the requested dtype.
        """
        ...

    def copy(self) -> NDArrayLike:
        """
        Return a (shallow) copy of the array.

        Returns
        -------
        NDArrayLike
            Copied array-like object.
        """
        ...

    def tolist(self) -> list[Any]:
        """
        Convert the array to a (possibly nested) Python list.

        Returns
        -------
        list[Any]
            Python-native representation of the array contents.
        """
        ...

    def __array__(self, dtype: Any = ...) -> Any:
        """
        Return a backend-native array representation.

        This method enables interoperability with NumPy-style APIs such as
        ``np.asarray(obj)`` without importing NumPy in the domain layer.

        Parameters
        ----------
        dtype : Any, optional
            Requested dtype for the returned array.

        Returns
        -------
        Any
            Backend-native array object (e.g., ``numpy.ndarray``).
        """
        ...

    # -----------------------------
    # Indexing / slicing
    # -----------------------------

    @overload
    def __getitem__(self, key: int) -> Any:
        """
        Return a single element selected by integer indexing.
        """
        ...

    @overload
    def __getitem__(self, key: slice | Tuple[Any, ...]) -> NDArrayLike:
        """
        Return a sub-array selected by slicing or tuple-based indexing.
        """
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Assign a value to a location or slice in the array.

        Parameters
        ----------
        key : Any
            Index or slice specification.
        value : Any
            Value to assign.
        """
        ...

    # -----------------------------
    # Reductions
    # -----------------------------

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Any:
        """
        Compute the sum of array elements over a given axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which the sum is computed.
        keepdims : bool, optional
            Whether to retain reduced dimensions.

        Returns
        -------
        Any
            Reduced scalar or array-like result.
        """
        ...

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Any:
        """
        Compute the arithmetic mean over a given axis.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which the mean is computed.
        keepdims : bool, optional
            Whether to retain reduced dimensions.

        Returns
        -------
        Any
            Reduced scalar or array-like result.
        """
        ...

    def max(self, axis: int | None = None, keepdims: bool = False) -> Any:
        """
        Compute the maximum value over a given axis.

        Returns
        -------
        Any
            Reduced scalar or array-like result.
        """
        ...

    def min(self, axis: int | None = None, keepdims: bool = False) -> Any:
        """
        Compute the minimum value over a given axis.

        Returns
        -------
        Any
            Reduced scalar or array-like result.
        """
        ...

    def argmax(self, axis: int | None = None) -> Any:
        """
        Return indices of the maximum values along an axis.

        Returns
        -------
        Any
            Index or array of indices.
        """
        ...

    def argmin(self, axis: int | None = None) -> Any:
        """
        Return indices of the minimum values along an axis.

        Returns
        -------
        Any
            Index or array of indices.
        """
        ...

    # -----------------------------
    # Comparison / boolean
    # -----------------------------

    def all(self, axis: int | None = None) -> bool | NDArrayLike:
        """
        Test whether all elements evaluate to True along an axis.

        Returns
        -------
        bool or NDArrayLike
            Boolean scalar or reduced array.
        """
        ...

    def any(self, axis: int | None = None) -> bool | NDArrayLike:
        """
        Test whether any element evaluates to True along an axis.

        Returns
        -------
        bool or NDArrayLike
            Boolean scalar or reduced array.
        """
        ...

    # -----------------------------
    # Iteration
    # -----------------------------

    def __iter__(self) -> Iterable[Any]:
        """
        Return an iterator over the first axis of the array.

        Returns
        -------
        Iterable[Any]
            Iterator yielding array elements or sub-arrays.
        """
        ...

    def fill(self) -> None: ...
