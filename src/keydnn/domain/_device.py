"""
Device abstraction utilities.

This module defines lightweight abstractions for representing computation
devices (CPU and CUDA GPUs) in a framework-agnostic way. It provides:

- `DeviceType`: an enumeration of supported device categories
- `Device`: a concrete device descriptor that validates and normalizes
  user-facing device strings such as "cpu" or "cuda:0"

The design intentionally avoids backend-specific dependencies and is suitable
for use across domain, application, and infrastructure layers.
"""

from enum import Enum
import re


class DeviceType(Enum):
    """
    Enumeration of supported device categories.

    This enum represents the *type* of a computation device, independent of
    any specific device index or backend implementation.

    Attributes
    ----------
    CPU : DeviceType
        Central Processing Unit.
    CUDA : DeviceType
        NVIDIA CUDA-enabled Graphics Processing Unit.
    """

    CPU = "cpu"
    CUDA = "cuda"


class Device:
    """
    Concrete computation device descriptor.

    This class encapsulates a normalized representation of a computation device,
    including its type (CPU or CUDA) and, for CUDA devices, an optional device
    index (e.g., cuda:0).

    The class performs strict validation of device strings to ensure a small,
    well-defined set of supported device identifiers.

    Parameters
    ----------
    device : str
        Device identifier string. Must be either:
        - "cpu"
        - "cuda:<index>", where <index> is a non-negative integer

    Raises
    ------
    ValueError
        If the provided device string does not match the supported formats.

    Notes
    -----
    - `__slots__` is used to prevent dynamic attribute creation and reduce
      per-instance memory overhead.
    - This class is intentionally lightweight and does not allocate or manage
      any backend resources.
    """

    __slots__ = ("type", "index")

    _CUDA_PATTERN = re.compile(r"^cuda:(\d+)$")

    def __init__(self, device: str):
        """
        Initialize a Device instance from a device string.

        This constructor parses and validates the given device string,
        assigning the appropriate `DeviceType` and device index.

        Parameters
        ----------
        device : str
            Device identifier string ("cpu" or "cuda:<index>").

        Raises
        ------
        ValueError
            If the device string is invalid or unsupported.
        """
        if device == "cpu":
            self.type = DeviceType.CPU
            self.index = None
        else:
            m = self._CUDA_PATTERN.match(device)
            if not m:
                raise ValueError(
                    f"Invalid device '{device}'. Expected 'cpu' or 'cuda:<index>'"
                )
            self.type = DeviceType.CUDA
            self.index = int(m.group(1))

    def __str__(self):
        """
        Return the canonical string representation of the device.

        Returns
        -------
        str
            "cpu" for CPU devices, or "cuda:<index>" for CUDA devices.
        """
        return "cpu" if self.type is DeviceType.CPU else f"cuda:{self.index}"

    def is_cpu(self) -> bool:
        """
        Check whether this device represents a CPU.

        Returns
        -------
        bool
            True if the device type is CPU, False otherwise.
        """
        return self.type is DeviceType.CPU

    def is_cuda(self) -> bool:
        """
        Check whether this device represents a CUDA GPU.

        Returns
        -------
        bool
            True if the device type is CUDA, False otherwise.
        """
        return self.type is DeviceType.CUDA
