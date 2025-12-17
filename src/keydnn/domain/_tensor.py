from abc import ABC, abstractmethod

from ._device import Device


class _Tensor(ABC):

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def device(self) -> Device:
        pass
