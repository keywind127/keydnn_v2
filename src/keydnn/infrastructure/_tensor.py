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
