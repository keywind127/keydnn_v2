import numpy as np

from ..domain._tensor import _Tensor
from ..domain._device import Device


class Tensor(_Tensor):

    def __initialize_data(self) -> None:
        match self._device:
            case Device() if self._device.is_cpu():
                self._data = np.zeros(self._shape, dtype=np.float32)
            case Device() if self._device.is_cuda():
                # Placeholder for CUDA tensor initialization
                self._data = f"CUDA Tensor on device {self._device.index} with shape {self._shape}"
            case _:
                raise ValueError("Unsupported device type")

    def __init__(self, shape: tuple[int, ...], device: Device) -> None:
        self._shape = shape
        self._device = device
        self.__initialize_data()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def device(self) -> Device:
        return self._device

    def to_numpy(self) -> np.ndarray:
        """
        Return the underlying CPU ndarray.
        CPU-only for now (until CUDA storage is implemented).
        """
        if not self._device.is_cpu():
            raise RuntimeError("to_numpy() is only available for CPU tensors.")
        # If you want stricter encapsulation: return self._data.copy()
        return self._data

    def fill(self, value: float) -> None:
        """
        Fill tensor with a scalar value.
        CPU-only for now.
        """
        if not self._device.is_cpu():
            raise RuntimeError("fill() is only available for CPU tensors.")
        self._data.fill(value)

    def debug_storage_repr(self) -> str:
        """
        A stable string representation of storage for testing/debugging.
        (Useful now because CUDA is a placeholder string.)
        """
        if self._device.is_cpu():
            arr: np.ndarray = self._data
            return f"CPU ndarray shape={arr.shape} dtype={arr.dtype}"
        return str(self._data)
