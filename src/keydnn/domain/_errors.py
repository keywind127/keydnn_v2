"""
Device- and execution-related exceptions for KeyDNN.

This module defines custom runtime errors used to signal unsupported or
invalid device usage during tensor operations. These exceptions allow
the framework to fail fast and clearly when an operation is invoked on
a device backend that has not yet been implemented (e.g., CUDA), or when
operands reside on incompatible devices.

These errors are intentionally explicit to aid debugging and to prepare
the codebase for future multi-device (CPU/GPU) support.
"""


class DeviceNotSupportedError(RuntimeError):
    """
    Raised when a tensor operation is requested on a device backend
    that is not implemented.

    This error is typically raised by Tensor arithmetic operators
    (e.g., __add__, __mul__) when the current device is not CPU and
    no alternative backend (such as CUDA) has been implemented yet.

    Attributes
    ----------
    op : str
        The name of the operation that was attempted (e.g., "add", "mul").
    device : str
        String representation of the device on which the operation
        was attempted.
    """

    def __init__(self, op: str, device: str) -> None:
        """
        Initialize the DeviceNotSupportedError.

        Parameters
        ----------
        op : str
            The operation name that is not supported on the given device.
        device : str
            The device identifier (e.g., "cuda:0", "mps", etc.).
        """
        super().__init__(f"{op} is not implemented for device '{device}'.")
        self.op = op
        self.device = device


class DeviceMismatchError(RuntimeError):
    """
    Raised when an operation is attempted between tensors on different devices.

    This error is used to prevent undefined behavior when combining tensors
    that reside on incompatible devices (e.g., CPU tensor with CUDA tensor)
    without an explicit device transfer or synchronization step.
    """

    def __init__(self, device_a: str, device_b: str) -> None:
        """
        Initialize the DeviceMismatchError.

        Parameters
        ----------
        device_a : str
            Device identifier of the first operand.
        device_b : str
            Device identifier of the second operand.
        """
        super().__init__(f"Device mismatch: '{device_a}' vs '{device_b}'.")
        self.device_a = device_a
        self.device_b = device_b
