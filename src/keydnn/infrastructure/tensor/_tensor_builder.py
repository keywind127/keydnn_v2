"""
Tensor control-path manager for device-specific dispatch.

This module defines a shared control-path manager used to register and resolve
device-specific implementations of Tensor methods.

The manager is created by specializing the generic `create_path_builder`
utility with the state attribute name ``"device"``. As a result, method
dispatch is performed based on the runtime value of ``self.device`` on
Tensor objects.

Typical usage
-------------
Backend-specific implementations (e.g., CPU vs. CUDA) register themselves
using this manager:

    @tensor_control_path_manager(TensorMixin, TensorMixin.op, Device("cpu"))
    def op_cpu(self, ...): ...

    @tensor_control_path_manager(TensorMixin, TensorMixin.op, Device("cuda:0"))
    def op_cuda(self, ...): ...

At runtime, calling ``Tensor.op(...)`` dispatches to the implementation whose
registered device matches ``self.device``.

Notes
-----
- The dispatch mechanism is state-based and relies on equality comparison of
  the ``device`` attribute.
- All control paths registered via this manager share a single internal
  registry, ensuring consistent dispatch behavior across the Tensor subsystem.
"""

from ...domain.utils._control_path import create_path_builder

# Control-path manager that dispatches Tensor methods based on `self.device`
tensor_control_path_manager = create_path_builder("device")
