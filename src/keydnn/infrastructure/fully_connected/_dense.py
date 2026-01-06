"""
Dense (Keras-style) layer with lazy input-dimension inference.

This module defines a `Dense` layer that combines the ergonomics of Keras'
`Dense` with the lazy parameter initialization behavior of PyTorch's
`LazyLinear`.

Users specify only `out_features` at construction time. The corresponding
`in_features` dimension is inferred from the first input tensor passed to
`forward` (specifically `x.shape[1]`).

Once built, this module delegates all numerical computation, autograd logic,
and device-specific behavior to the existing `Linear` implementation, which
supports both CPU and CUDA backends.

Design Notes
------------
- Inputs must be 2D tensors of shape `(batch, in_features)`, consistent with
  the `Linear` module contract.
- If `device` is not provided at construction time, the layer adopts the device
  of the first input tensor.
- If `device` *is* provided, `forward` enforces that all inputs already reside
  on that device (no implicit device transfers are performed).
- Parameter materialization occurs exactly once, either during the first
  forward pass or eagerly during deserialization if `in_features` is known.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from .._module import Module
from ._linear import Linear
from ..module._serialization_core import register_module
from ...domain.device._device import Device

from ..tensor._tensor import Tensor


@register_module()
class Dense(Module):
    """
    Keras-style Dense layer with lazy input-dimension inference.

    This layer delays parameter creation until the first forward pass, at which
    point the input feature dimension is inferred from the input tensor.
    Internally, it constructs and delegates execution to a fully-initialized
    `Linear` submodule.

    Parameters
    ----------
    out_features : int
        Number of output features per input example.
    bias : bool, optional
        If True, include a learnable bias term. Defaults to True.
    device : Optional[Device], optional
        Desired device placement for parameters and computation. If None, the
        device is inferred from the first input tensor.
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize a lazy Dense layer.

        At construction time, only the output dimensionality is fixed.
        The input dimensionality and parameters are initialized later when
        sufficient runtime information is available.

        Raises
        ------
        ValueError
            If `out_features` is not a positive integer.
        """
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer")

        self.out_features = int(out_features)
        self._use_bias = bool(bias)
        self.device = device  # may be None until first forward

        # Lazy state
        self.in_features: Optional[int] = None
        self._linear: Optional[Linear] = None  # created on first forward

    @property
    def is_built(self) -> bool:
        """
        Return whether this layer has been materialized.

        A Dense layer is considered *built* once its internal `Linear`
        submodule has been constructed and parameters have been allocated.

        Returns
        -------
        bool
            True if parameters have been initialized, False otherwise.
        """
        return self._linear is not None

    def _build(self, in_features: int, *, device: Device) -> None:
        """
        Materialize parameters by constructing the internal `Linear` submodule.

        This method is invoked exactly once during the layer's lifetime,
        either on the first forward pass or during deserialization when
        `in_features` is known ahead of time.

        Parameters
        ----------
        in_features : int
            Number of input features inferred from the input tensor.
        device : Device
            Device on which parameters should be allocated.

        Raises
        ------
        ValueError
            If `in_features` is not a positive integer.
        RuntimeError
            If the layer has already been built with a conflicting
            `in_features` value.
        """
        if in_features <= 0:
            raise ValueError("in_features must be a positive integer")

        if self._linear is not None:
            # already built: enforce consistency
            if self.in_features != int(in_features):
                raise RuntimeError(
                    f"Dense already built with in_features={self.in_features}, "
                    f"but got in_features={int(in_features)}"
                )
            return

        self.in_features = int(in_features)
        self.device = device

        # Delegate all math/autograd/device paths to the Linear implementation
        self._linear = Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self._use_bias,
            device=self.device,
        )
        # Register as submodule so state tracking and serialization can see it
        self.register_module("linear", self._linear)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the Dense transformation to the input tensor.

        On the first invocation, this method infers the input feature dimension
        from `x`, constructs the internal `Linear` submodule, and then delegates
        execution to it.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape `(batch, in_features)`.

        Returns
        -------
        Tensor
            Output tensor of shape `(batch, out_features)`.

        Raises
        ------
        ValueError
            If the input tensor is not 2-dimensional.
        RuntimeError
            If a device mismatch is detected between the input tensor and
            the layer's configured device.
        """
        # Validate 2D input (same contract as Linear)
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Dense expects 2D input (batch, in_features), got {x_shape}"
            )

        inferred_in = int(x_shape[1])

        # Decide device: if self.device is None, adopt x.device on first call
        if self.device is None:
            self._build(inferred_in, device=x.device)
        else:
            # enforce device match (no implicit moves)
            if str(x.device) != str(self.device):
                raise RuntimeError(
                    f"Dense.forward device mismatch: "
                    f"x.device={x.device} vs layer.device={self.device}"
                )
            # build if needed
            self._build(inferred_in, device=self.device)

        assert self._linear is not None
        return self._linear.forward(x)

    # -------------------------------------------------------------------------
    # Serialization (config-only) hooks
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration dictionary.

        For lazy modules, this includes `in_features` if the layer has already
        been built. This enables deterministic reconstruction of the module
        structure during deserialization.

        Returns
        -------
        Dict[str, Any]
            Constructor configuration suitable for JSON serialization.
        """
        return {
            "out_features": int(self.out_features),
            "bias": bool(self._use_bias),
            "device": str(self.device) if self.device is not None else None,
            "in_features": (
                int(self.in_features) if self.in_features is not None else None
            ),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Dense":
        """
        Reconstruct a Dense layer from a serialized configuration.

        If `in_features` is present in the configuration, the layer is eagerly
        built so that its structure matches the saved state. Parameter values
        are typically restored separately from a checkpoint.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config`.

        Returns
        -------
        Dense
            Reconstructed Dense module.
        """
        dev = cfg.get("device", None)
        device = Device(str(dev)) if dev is not None else None

        m = cls(
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=device,
        )

        # If in_features is known (built module was saved), eagerly build
        in_features = cfg.get("in_features", None)
        if in_features is not None:
            # If device is still None here, default to CPU for deterministic rebuild
            # (weights will typically be loaded from checkpoint afterwards anyway).
            build_dev = device if device is not None else Device("cpu")
            m._build(int(in_features), device=build_dev)

        return m


LazyLinear = Dense

__all__ = [
    Dense.__name__,
    "LazyLinear",
]
