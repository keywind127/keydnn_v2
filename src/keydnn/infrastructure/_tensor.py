from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..domain._tensor import ITensor
from ..domain._device import Device


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
