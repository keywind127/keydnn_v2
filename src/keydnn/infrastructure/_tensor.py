from __future__ import annotations
from typing import Any, Union, Callable, Optional, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..domain._tensor import ITensor
from ..domain._device import Device
from ..domain._errors import DeviceNotSupportedError

Number = Union[int, float]


@dataclass
class Context:
    """
    Backward context attached to an output Tensor.

    - parents: input/parameter tensors used to compute this output
    - backward_fn: computes grads for parents given grad_out
    - saved_tensors/meta: anything needed for backward (e.g., indices, shapes)
    """

    parents: Sequence["Tensor"]
    backward_fn: Callable[["Tensor"], Sequence[Optional["Tensor"]]]
    saved_tensors: list["Tensor"] = field(default_factory=list)
    saved_meta: dict[str, Any] = field(default_factory=dict)

    def save_for_backward(self, *tensors: "Tensor") -> None:
        self.saved_tensors.extend(tensors)


class Tensor(ITensor):

    def __initialize_data(self) -> None:
        match self._device:
            case Device() if self._device.is_cpu():
                self._data = np.zeros(self._shape, dtype=np.float32)
            case Device() if self._device.is_cuda():
                self._data = f"CUDA Tensor on device {self._device.index} with shape {self._shape}"
            case _:
                raise ValueError("Unsupported device type")

    def __init__(
        self,
        shape: tuple[int, ...],
        device: Device,
        *,
        requires_grad: bool = False,
        ctx: Optional[Context] = None,
    ) -> None:
        self._shape = shape
        self._device = device
        self.__initialize_data()

        # --- autograd fields (optional) ---
        self._requires_grad: bool = bool(requires_grad)
        self._grad: Optional["Tensor"] = None
        self._ctx: Optional[Context] = ctx

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> Device:
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = bool(value)

    @property
    def grad(self) -> Optional["Tensor"]:
        return self._grad

    def zero_grad(self) -> None:
        self._grad = None

    def _set_ctx(self, ctx: Optional[Context]) -> None:
        """Internal hook: attach/detach backward context."""
        self._ctx = ctx

    def _get_ctx(self) -> Optional[Context]:
        """Internal hook: access backward context (engine will use this)."""
        return self._ctx

    def to_numpy(self) -> np.ndarray:
        if not self._device.is_cpu():
            raise RuntimeError("to_numpy() is only available for CPU tensors.")
        return self._data

    def fill(self, value: float) -> None:
        if not self._device.is_cpu():
            raise RuntimeError("fill() is only available for CPU tensors.")
        self._data.fill(value)

    def debug_storage_repr(self) -> str:
        if self._device.is_cpu():
            arr: np.ndarray = self._data
            return f"CPU ndarray shape={arr.shape} dtype={arr.dtype}"
        return str(self._data)

    def copy_from_numpy(self, arr: np.ndarray) -> None:
        if not self._device.is_cpu():
            raise RuntimeError("copy_from_numpy() is only available for CPU tensors.")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != self._shape:
            raise ValueError(
                f"Shape mismatch: tensor {self._shape} vs array {arr.shape}"
            )
        self._data[...] = arr

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _raise_device_not_supported(self, op: str) -> "None":
        raise DeviceNotSupportedError(op=op, device=str(self._device))

    @staticmethod
    def _result_requires_grad(*parents: "Tensor") -> bool:
        return any(p.requires_grad for p in parents)

    @staticmethod
    def _as_tensor_like(x: Union["Tensor", Number], like: "Tensor") -> "Tensor":
        """
        Convert Python scalar to a Tensor with same shape/device as `like`.
        If already a Tensor, return it.
        """
        if isinstance(x, Tensor):
            return x
        if isinstance(x, (int, float)):
            t = Tensor(shape=like.shape, device=like.device, requires_grad=False)
            t.fill(float(x))
            return t
        raise TypeError(f"Unsupported operand type: {type(x)!r}")

    @staticmethod
    def _binary_op_shape_check(a: "Tensor", b: "Tensor") -> None:
        # Minimal rule for now: exact shape match (no broadcasting yet)
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # ----------------------------
    # Unary ops
    # ----------------------------
    def __neg__(self) -> "Tensor":
        if self._device.is_cpu():
            out = Tensor(
                shape=self.shape, device=self.device, requires_grad=self.requires_grad
            )
            out.copy_from_numpy(-self.to_numpy())

            if self.requires_grad:
                ctx = Context(
                    parents=(self,),
                    backward_fn=lambda grad_out: (-(grad_out),),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("neg")

    # ----------------------------
    # Addition / Subtraction
    # ----------------------------
    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() + other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out, grad_out),
                )
                out._set_ctx(ctx)
            return out

        # pick a consistent error point (left operand op name)
        self._raise_device_not_supported("add")

    def __radd__(self, other: Number) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() - other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out, -grad_out),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("sub")

    def __rsub__(self, other: Number) -> "Tensor":
        other_t = self._as_tensor_like(other, self)
        return other_t.__sub__(self)

    # ----------------------------
    # Multiplication
    # ----------------------------
    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() * other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (grad_out * other_t, grad_out * self),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("mul")

    def __rmul__(self, other: Number) -> "Tensor":
        return self.__mul__(other)

    # ----------------------------
    # True division
    # ----------------------------
    def __truediv__(self, other: Union["Tensor", Number]) -> "Tensor":
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            req = self._result_requires_grad(self, other_t)
            out = Tensor(shape=self.shape, device=self.device, requires_grad=req)
            out.copy_from_numpy(self.to_numpy() / other_t.to_numpy())

            if req:
                ctx = Context(
                    parents=(self, other_t),
                    backward_fn=lambda grad_out: (
                        grad_out / other_t,
                        -(grad_out * self) / (other_t * other_t),
                    ),
                )
                out._set_ctx(ctx)
            return out

        self._raise_device_not_supported("truediv")

    def __rtruediv__(self, other: Number) -> "Tensor":
        other_t = self._as_tensor_like(other, self)
        return other_t.__truediv__(self)

    # ----------------------------
    # Comparisons (no grad)
    # ----------------------------
    def __gt__(self, other: Union["Tensor", Number]) -> "Tensor":
        other_t = self._as_tensor_like(other, self)

        if self._device.is_cpu() and other_t.device.is_cpu():
            self._binary_op_shape_check(self, other_t)

            out = Tensor(shape=self.shape, device=self.device, requires_grad=False)
            out.copy_from_numpy(
                (self.to_numpy() > other_t.to_numpy()).astype(np.float32)
            )
            return out

        self._raise_device_not_supported("gt")
