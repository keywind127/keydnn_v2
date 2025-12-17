from enum import Enum
import re


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


class Device:
    __slots__ = ("type", "index")

    _CUDA_PATTERN = re.compile(r"^cuda:(\d+)$")

    def __init__(self, device: str):
        if device == "cpu":
            self.type = DeviceType.CPU
            self.index = None
        else:
            m = self._CUDA_PATTERN.match(device)
            if not m:
                raise ValueError(
                    f"Invalid device '{device}'. " "Expected 'cpu' or 'cuda:<index>'"
                )
            self.type = DeviceType.CUDA
            self.index = int(m.group(1))

    def __str__(self):
        return "cpu" if self.type is DeviceType.CPU else f"cuda:{self.index}"

    def is_cpu(self) -> bool:
        return self.type is DeviceType.CPU

    def is_cuda(self) -> bool:
        return self.type is DeviceType.CUDA
