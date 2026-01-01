"""
Arithmetic mixin defining elementwise Tensor operators.

This module declares :class:`TensorMixinArithmetic`, an abstract mixin that
specifies the public API and mathematical semantics for elementwise arithmetic
operations on tensors.

The mixin itself does not implement numerical kernels. Concrete CPU/CUDA
implementations are provided elsewhere and registered via a control-path
dispatch mechanism. This design keeps the Tensor core lightweight while
supporting device-specific execution strategies behind a unified interface.
"""

from typing import Optional, Union
from abc import ABC

from .....domain._tensor import ITensor

Number = Union[int, float]


class TensorMixinArithmetic(ABC):
    """
    Abstract mixin defining elementwise arithmetic operations for tensors.

    This mixin specifies the signatures, expected behavior, and backward
    semantics for arithmetic operators, including addition, subtraction,
    multiplication, and division.

    Notes
    -----
    - Methods defined here serve as interface declarations and documentation
      of mathematical contracts.
    - No computation is performed in this class.
    - Scalars are conceptually promoted to tensors matching the receiver's
      shape and device before applying the operation.
    - Backward rules described in method docstrings are contractual and must
      be respected by all concrete implementations.
    """

    # ----------------------------
    # True division
    # ----------------------------
    def __truediv__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise true division.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self / other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a / b) / da = 1 / b``
        - ``d(a / b) / db = -a / (b^2)``
        """
        ...

    def __rtruediv__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand true division to support ``scalar / Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other / self``.

        Notes
        -----
        This method promotes the scalar to a tensor compatible with ``self``
        and delegates to :meth:`__truediv__`.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__truediv__(self)

    # ----------------------------
    # Python 2 legacy division alias
    # ----------------------------
    def __div__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise division (legacy alias for true division).

        Notes
        -----
        Python 3 uses :meth:`__truediv__` to implement the ``/`` operator.
        This method exists solely for compatibility with legacy code that
        explicitly calls ``__div__``.
        """
        return self.__truediv__(other)

    def __rdiv__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand division (legacy alias for right true division).

        Notes
        -----
        Python 3 uses :meth:`__rtruediv__`. This method exists solely for
        compatibility with legacy code that explicitly calls ``__rdiv__``.
        """
        return self.__rtruediv__(other)

    # ----------------------------
    # Addition
    # ----------------------------
    def __add__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise addition.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self + other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a + b) / da = 1``
        - ``d(a + b) / db = 1``
        """
        ...

    def __radd__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand addition to support ``scalar + Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other + self``.

        Notes
        -----
        Addition is commutative, so this method simply delegates to
        :meth:`__add__`.
        """
        return self.__add__(other)

    def __sub__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise subtraction.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self - other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a - b) / da = 1``
        - ``d(a - b) / db = -1``
        """
        ...

    def __rsub__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand subtraction to support ``scalar - Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other - self``.

        Notes
        -----
        The scalar is promoted to a tensor compatible with ``self`` before
        applying subtraction.
        """
        other_t = self._as_tensor_like(other, self)
        return other_t.__sub__(self)

    def __imul__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise multiplication: self *= other.

        CUDA behavior
        -------------
        - If self is CUDA AND it is safe to mutate storage (no autograd tracking),
        dispatch to native in-place CUDA kernels to avoid intermediate allocations.
        - Otherwise, fall back to out-of-place multiply + copy_from (same semantics
        pattern as __isub__), which is graph-safer.

        CPU behavior
        ------------
        - Uses out-of-place multiply + copy_from (keeps behavior consistent and avoids
        relying on internal numpy storage layout).
        """
        # -----------------------------
        # Determine whether it's safe to do a true in-place op
        # -----------------------------
        # Only do true in-place when:
        # - CUDA tensor
        # - not tracking gradients
        # - not participating in an autograd graph
        #
        # This avoids subtle autograd bugs from mutating values that are needed by
        # saved tensors in contexts.
        try:
            self_req = bool(getattr(self, "requires_grad", False))
        except Exception:
            self_req = False

        try:
            self_ctx = None
            if hasattr(self, "_get_ctx") and callable(getattr(self, "_get_ctx")):
                self_ctx = self._get_ctx()  # type: ignore[attr-defined]
            elif hasattr(self, "ctx"):
                self_ctx = getattr(self, "ctx")
        except Exception:
            self_ctx = None

        safe_inplace = (not self_req) and (self_ctx is None)

        # -----------------------------
        # CUDA fast-path (true in-place)
        # -----------------------------
        if safe_inplace and hasattr(self, "device") and self.device.is_cuda():
            import numpy as np

            # Handle scalar separately (uses scalar in-place kernel)
            if isinstance(other, (int, float)):
                dt = np.dtype(getattr(self, "dtype", np.float32))
                if dt not in (np.float32, np.float64):
                    raise TypeError(
                        f"mul inplace scalar requires float32/float64, got dtype={dt}"
                    )

                # numel==0 should be a no-op (and must not call cuda_malloc(0))
                try:
                    shape = tuple(int(d) for d in self.shape)
                except Exception:
                    shape = tuple(self.shape)  # type: ignore[arg-type]

                numel = 1
                for d in shape:
                    numel *= int(d)
                if int(numel) <= 0:
                    return self

                device_index = int(getattr(self.device, "index", 0) or 0)

                # Uses your new in-place wrapper at Tensor-boundary level
                from ....ops.mul_cuda_ext import (
                    mul_scalar_inplace as _mul_scalar_inplace,
                )

                _mul_scalar_inplace(self, float(other), device=device_index, sync=False)
                return self

            # Tensor operand path
            other_t = self._as_tensor_like(other, self)

            if not other_t.device.is_cuda():
                self._raise_device_not_supported("imul (mul inplace)")

            # shape + dtype checks (no broadcasting)
            self._binary_op_shape_check(self, other_t)

            if np.dtype(getattr(self, "dtype", np.float32)) != np.dtype(
                getattr(other_t, "dtype", np.float32)
            ):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(getattr(self,'dtype',np.float32))} "
                    f"vs other.dtype={np.dtype(getattr(other_t,'dtype',np.float32))}"
                )

            # numel==0 is a no-op
            try:
                shape = tuple(int(d) for d in self.shape)
            except Exception:
                shape = tuple(self.shape)  # type: ignore[arg-type]

            numel = 1
            for d in shape:
                numel *= int(d)
            if int(numel) <= 0:
                return self

            device_index = int(getattr(self.device, "index", 0) or 0)

            from ....ops.mul_cuda_ext import mul_inplace as _mul_inplace

            _mul_inplace(self, other_t, device=device_index, sync=False)
            return self

        # -----------------------------
        # Fallback: out-of-place compute + write back
        # -----------------------------
        out = self.__mul__(other)
        self.copy_from(out)
        return self

    # ----------------------------
    # Multiplication
    # ----------------------------
    def __mul__(self: ITensor, other: Union["ITensor", Number]) -> "ITensor":
        """
        Elementwise multiplication.

        Parameters
        ----------
        other : Union[ITensor, Number]
            Right-hand operand. Scalars are lifted to tensors matching this
            tensor's shape and device.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``self * other``.

        Notes
        -----
        Backward rule (elementwise, no broadcasting):
        - ``d(a * b) / da = b``
        - ``d(a * b) / db = a``
        """
        ...

    def __rmul__(self: ITensor, other: Number) -> "ITensor":
        """
        Right-hand multiplication to support ``scalar * Tensor``.

        Parameters
        ----------
        other : Number
            Left-hand scalar operand.

        Returns
        -------
        ITensor
            Tensor containing the elementwise result of ``other * self``.

        Notes
        -----
        Multiplication is commutative, so this method simply delegates to
        :meth:`__mul__`.
        """
        return self.__mul__(other)

    def __iadd__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise addition: self += other.

        CUDA fast-path:
        - If safe to mutate (no autograd tracking), dispatch to native CUDA in-place
          kernels to avoid intermediate allocations.

        Fallback:
        - out-of-place add + copy_from.
        """
        # -------- safe inplace gate --------
        try:
            self_req = bool(getattr(self, "requires_grad", False))
        except Exception:
            self_req = False

        try:
            self_ctx = None
            if hasattr(self, "_get_ctx") and callable(getattr(self, "_get_ctx")):
                self_ctx = self._get_ctx()  # type: ignore[attr-defined]
            elif hasattr(self, "ctx"):
                self_ctx = getattr(self, "ctx")
        except Exception:
            self_ctx = None

        safe_inplace = (not self_req) and (self_ctx is None)

        # -------- CUDA true inplace --------
        if safe_inplace and hasattr(self, "device") and self.device.is_cuda():
            import numpy as np

            # numel==0 -> no-op (must not call native with null ptrs)
            try:
                shape = tuple(int(d) for d in self.shape)
            except Exception:
                shape = tuple(self.shape)  # type: ignore[arg-type]

            numel = 1
            for d in shape:
                numel *= int(d)
            if int(numel) <= 0:
                return self

            device_index = int(getattr(self.device, "index", 0) or 0)

            if isinstance(other, (int, float)):
                dt = np.dtype(getattr(self, "dtype", np.float32))
                if dt not in (np.float32, np.float64):
                    raise TypeError(
                        f"add inplace scalar requires float32/float64, got dtype={dt}"
                    )

                from ....ops.tensor_arithmetic_cuda_ext import (
                    add_scalar_inplace as _add_scalar_inplace,
                )

                _add_scalar_inplace(self, float(other), device=device_index)
                return self

            other_t = self._as_tensor_like(other, self)
            if not other_t.device.is_cuda():
                self._raise_device_not_supported("iadd (add inplace)")

            self._binary_op_shape_check(self, other_t)

            if np.dtype(getattr(self, "dtype", np.float32)) != np.dtype(
                getattr(other_t, "dtype", np.float32)
            ):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(getattr(self,'dtype',np.float32))} "
                    f"vs other.dtype={np.dtype(getattr(other_t,'dtype',np.float32))}"
                )

            from ....ops.tensor_arithmetic_cuda_ext import add_inplace as _add_inplace

            _add_inplace(self, other_t, device=device_index)
            return self

        # -------- fallback --------
        out = self.__add__(other)
        self.copy_from(out)
        return self

    def __isub__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise subtraction: self -= other.

        Same in-place safety rules as __iadd__/__imul__.
        """
        # -------- safe inplace gate --------
        try:
            self_req = bool(getattr(self, "requires_grad", False))
        except Exception:
            self_req = False

        try:
            self_ctx = None
            if hasattr(self, "_get_ctx") and callable(getattr(self, "_get_ctx")):
                self_ctx = self._get_ctx()  # type: ignore[attr-defined]
            elif hasattr(self, "ctx"):
                self_ctx = getattr(self, "ctx")
        except Exception:
            self_ctx = None

        safe_inplace = (not self_req) and (self_ctx is None)

        # -------- CUDA true inplace --------
        if safe_inplace and hasattr(self, "device") and self.device.is_cuda():
            import numpy as np

            # numel==0 -> no-op
            try:
                shape = tuple(int(d) for d in self.shape)
            except Exception:
                shape = tuple(self.shape)  # type: ignore[arg-type]

            numel = 1
            for d in shape:
                numel *= int(d)
            if int(numel) <= 0:
                return self

            device_index = int(getattr(self.device, "index", 0) or 0)

            if isinstance(other, (int, float)):
                dt = np.dtype(getattr(self, "dtype", np.float32))
                if dt not in (np.float32, np.float64):
                    raise TypeError(
                        f"sub inplace scalar requires float32/float64, got dtype={dt}"
                    )

                from ....ops.tensor_arithmetic_cuda_ext import (
                    sub_scalar_inplace as _sub_scalar_inplace,
                )

                _sub_scalar_inplace(self, float(other), device=device_index)
                return self

            other_t = self._as_tensor_like(other, self)
            if not other_t.device.is_cuda():
                self._raise_device_not_supported("isub (sub inplace)")

            self._binary_op_shape_check(self, other_t)

            if np.dtype(getattr(self, "dtype", np.float32)) != np.dtype(
                getattr(other_t, "dtype", np.float32)
            ):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(getattr(self,'dtype',np.float32))} "
                    f"vs other.dtype={np.dtype(getattr(other_t,'dtype',np.float32))}"
                )

            from ....ops.tensor_arithmetic_cuda_ext import sub_inplace as _sub_inplace

            _sub_inplace(self, other_t, device=device_index)
            return self

        # -------- fallback --------
        out = self.__sub__(other)
        self.copy_from(out)
        return self

    def __itruediv__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        """
        In-place elementwise true division: self /= other.

        Same in-place safety rules as __iadd__/__imul__.
        """
        # -------- safe inplace gate --------
        try:
            self_req = bool(getattr(self, "requires_grad", False))
        except Exception:
            self_req = False

        try:
            self_ctx = None
            if hasattr(self, "_get_ctx") and callable(getattr(self, "_get_ctx")):
                self_ctx = self._get_ctx()  # type: ignore[attr-defined]
            elif hasattr(self, "ctx"):
                self_ctx = getattr(self, "ctx")
        except Exception:
            self_ctx = None

        safe_inplace = (not self_req) and (self_ctx is None)

        # -------- CUDA true inplace --------
        if safe_inplace and hasattr(self, "device") and self.device.is_cuda():
            import numpy as np

            # numel==0 -> no-op
            try:
                shape = tuple(int(d) for d in self.shape)
            except Exception:
                shape = tuple(self.shape)  # type: ignore[arg-type]

            numel = 1
            for d in shape:
                numel *= int(d)
            if int(numel) <= 0:
                return self

            device_index = int(getattr(self.device, "index", 0) or 0)

            if isinstance(other, (int, float)):
                dt = np.dtype(getattr(self, "dtype", np.float32))
                if dt not in (np.float32, np.float64):
                    raise TypeError(
                        f"div inplace scalar requires float32/float64, got dtype={dt}"
                    )

                from ....ops.tensor_arithmetic_cuda_ext import (
                    div_scalar_inplace as _div_scalar_inplace,
                )

                _div_scalar_inplace(self, float(other), device=device_index)
                return self

            other_t = self._as_tensor_like(other, self)
            if not other_t.device.is_cuda():
                self._raise_device_not_supported("itruediv (div inplace)")

            self._binary_op_shape_check(self, other_t)

            if np.dtype(getattr(self, "dtype", np.float32)) != np.dtype(
                getattr(other_t, "dtype", np.float32)
            ):
                raise TypeError(
                    f"dtype mismatch: self.dtype={np.dtype(getattr(self,'dtype',np.float32))} "
                    f"vs other.dtype={np.dtype(getattr(other_t,'dtype',np.float32))}"
                )

            from ....ops.tensor_arithmetic_cuda_ext import div_inplace as _div_inplace

            _div_inplace(self, other_t, device=device_index)
            return self

        # -------- fallback --------
        out = self.__truediv__(other)
        self.copy_from(out)
        return self

    # Python 2 legacy alias (optional, but nice to keep symmetry with __div__)
    def __idiv__(self: ITensor, other: Union["ITensor", Number]) -> ITensor:
        return self.__itruediv__(other)
