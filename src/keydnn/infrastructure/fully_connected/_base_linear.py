# =========================
# file: _base_linear.py
# =========================
"""
Shared core for linear-family layers.

This module defines `_BaseLinear`, a non-registered base class that centralizes
the implementation of an affine projection:

    y = x @ W^T + b

Motivation
----------
A wrapper-style `Dense` that owns an internal `Linear` submodule can introduce
checkpoint/optimizer continuity problems when parameter names or child-module
registration differ across save/load cycles. `_BaseLinear` avoids that by making
both eager (`Linear`) and lazy (`Dense`) layers *own their parameters directly*,
keeping parameter keys stable (`weight`, `bias`) and making state loading
deterministic.

Design
------
- `_BaseLinear` is intentionally NOT decorated with `@register_module()`.
- Parameter allocation is performed by `_materialize(...)` exactly once.
- `forward(...)` implements CPU + CUDA paths and legacy autograd `Context`.

Notes
-----
- Serialization is "config-only": weights are expected to be handled by a
  checkpoint/state mechanism (e.g., payloads from `named_parameters()`).
- Subclasses can expose `from_config` for reconstruction but should rely on
  state loading for parameter values.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np

from ..tensor._tensor_context import Context
from .._module import Module
from .._parameter import Parameter
from ..tensor._tensor import Tensor
from ...domain.device._device import Device


class _BaseLinear(Module):
    """
    Shared implementation for linear-family layers.

    `_BaseLinear` provides a single, stable implementation of an affine layer,
    supporting both eager and lazy construction while keeping parameter naming
    stable across save/load cycles.

    Attributes
    ----------
    in_features : Optional[int]
        Input feature dimension. None until built for lazy layers.
    out_features : int
        Output feature dimension.
    device : Optional[Device]
        Device placement for parameters and outputs. None until built if lazy.
    dtype : Any
        Parameter dtype used for initialization and kernel dispatch.
    weight : Optional[Parameter]
        Weight parameter of shape (out_features, in_features). None until built.
    bias : Optional[Parameter]
        Bias parameter of shape (out_features,) if enabled; otherwise None.
        If the layer is unbuilt, this remains None as well.
    """

    def __init__(
        self,
        *,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        initializer: str = "xavier_uniform",
        in_features: Optional[int] = None,
    ) -> None:
        """
        Initialize shared state for a linear-family layer.

        This constructor stores hyperparameters and (optionally) a preferred
        device, but does not necessarily allocate parameters. Allocation occurs
        in `_materialize()`.

        Parameters
        ----------
        out_features : int
            Output feature dimension.
        bias : bool, optional
            Whether to include a learnable bias term. Defaults to True.
        device : Optional[Device], optional
            Preferred device placement. May be None for lazy layers.
        dtype : Any, optional
            Parameter dtype. Defaults to float32 if not provided.
        initializer : str, optional
            Weight initializer name passed to `Parameter(initializer=...)`.
        in_features : Optional[int], optional
            If provided, stored as the known input dimension. Subclasses may
            materialize eagerly using this value.

        Raises
        ------
        ValueError
            If `out_features` is not positive.
        """
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be a positive integer")

        self.in_features: Optional[int] = (
            int(in_features) if in_features is not None else None
        )
        self.out_features: int = int(out_features)

        self._use_bias: bool = bool(bias)
        self.device: Optional[Device] = device
        self.dtype: Any = dtype if dtype is not None else np.float32
        self.initializer: str = str(initializer)

        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None

    @property
    def is_built(self) -> bool:
        """
        Return whether parameters have been materialized.

        Returns
        -------
        bool
            True if `weight` exists, False otherwise.
        """
        return self.weight is not None

    def _materialize(self, in_features: int, *, device: Device) -> None:
        """
        Allocate and register parameters if not already built.

        This creates and registers:
        - `weight`: (out_features, in_features)
        - `bias`:   (out_features,) if bias is enabled

        Materialization is idempotent: if already built, it validates that
        `in_features` matches.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        device : Device
            Device on which parameters should be allocated.

        Raises
        ------
        ValueError
            If `in_features` is not positive.
        RuntimeError
            If called after build with a mismatched `in_features`.
        """
        if in_features <= 0:
            raise ValueError("in_features must be a positive integer")

        if self.weight is not None:
            if self.in_features != int(in_features):
                raise RuntimeError(
                    f"{type(self).__name__} already built with in_features={self.in_features}, "
                    f"but got in_features={int(in_features)}"
                )
            return

        self.in_features = int(in_features)
        self.device = device

        w = Parameter(
            shape=(int(self.out_features), int(self.in_features)),
            device=device,
            dtype=self.dtype,
            requires_grad=True,
            initializer=self.initializer,
        )
        self.register_parameter("weight", w)
        self.weight = w

        if self._use_bias:
            b = Parameter(
                shape=(int(self.out_features),),
                device=device,
                dtype=self.dtype,
                requires_grad=True,
                initializer="zeros",
            )
            self.register_parameter("bias", b)
            self.bias = b
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the affine transform to a 2D input tensor.

        This method assumes the layer has been materialized (i.e., `weight` exists).
        Lazy subclasses should call `_materialize(...)` before delegating here.

        Computation:
            y = x @ W^T (+ b)

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, out_features).

        Raises
        ------
        RuntimeError
            If the layer is not built, or devices mismatch.
        ValueError
            If input rank/shape is incompatible with the layer.
        """
        if self.weight is None or self.in_features is None or self.device is None:
            raise RuntimeError(
                f"{type(self).__name__} is not built. Call _materialize() first."
            )

        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"{type(self).__name__} expects 2D input (batch, in_features), got {x_shape}"
            )
        if int(x_shape[1]) != int(self.in_features):
            raise ValueError(
                f"{type(self).__name__} expects in_features={self.in_features}, got {x_shape[1]}"
            )

        layer_dev = self.weight.device

        if str(x.device) != str(layer_dev):
            raise RuntimeError(
                f"{type(self).__name__}.forward device mismatch: x.device={x.device} vs weight.device={layer_dev}"
            )

        if self.bias is not None and str(self.bias.device) != str(layer_dev):
            raise RuntimeError(
                f"{type(self).__name__}.forward device mismatch: bias.device={self.bias.device} vs weight.device={layer_dev}"
            )

        x_req = bool(x.requires_grad)
        w_req = bool(self.weight.requires_grad)
        b_req = bool(self.bias is not None and self.bias.requires_grad)
        req = x_req or w_req or b_req

        # -----------------------
        # CUDA path
        # -----------------------
        if layer_dev.is_cuda():
            if __debug__ and int(getattr(x, "data")) == 0:
                raise RuntimeError(
                    "Linear CUDA requires allocated x device buffer (x.data != 0)."
                )
            if __debug__ and int(getattr(self.weight, "data")) == 0:
                raise RuntimeError(
                    "Linear CUDA requires allocated weight device buffer (weight.data != 0)."
                )
            if (
                __debug__
                and self.bias is not None
                and int(getattr(self.bias, "data")) == 0
            ):
                raise RuntimeError(
                    "Linear CUDA requires allocated bias device buffer (bias.data != 0)."
                )

            dt = np.dtype(getattr(x, "dtype", np.float32))
            if __debug__ and np.dtype(getattr(self.weight, "dtype", dt)) != dt:
                raise TypeError(
                    f"Linear CUDA dtype mismatch: x.dtype={np.dtype(getattr(x,'dtype',dt))} "
                    f"vs weight.dtype={np.dtype(getattr(self.weight,'dtype',dt))}"
                )
            if (
                __debug__
                and self.bias is not None
                and np.dtype(getattr(self.bias, "dtype", dt)) != dt
            ):
                raise TypeError(
                    f"Linear CUDA dtype mismatch: x.dtype={dt} vs bias.dtype={np.dtype(getattr(self.bias,'dtype',dt))}"
                )

            y = x @ self.weight.T  # (batch, out_features)

            if self.bias is not None:
                from ..ops.bias_add_cuda_ext import bias_add_inplace

                bias_add_inplace(
                    y, self.bias, device=int(self.device.index or 0), sync=True
                )

            if not req:
                return y

            # Override ctx to exactly what Linear defines
            y.requires_grad = True
            y._set_ctx(None)

            def backward_fn(grad_out: Tensor):
                """
                Backward rule for y = x @ W^T (+ b) on CUDA.

                Returns gradients for parents in the same order as `parents`.
                """
                if not grad_out.device.is_cuda():
                    raise RuntimeError("grad_out must be CUDA for Linear CUDA backward")
                layer_dev = self.weight.device  # authoritative
                if str(grad_out.device) != str(layer_dev):
                    raise RuntimeError(
                        f"grad_out must be on same device as output; got {grad_out.device} vs {layer_dev}"
                    )
                if tuple(grad_out.shape) != (int(x_shape[0]), int(self.out_features)):
                    raise ValueError(
                        f"grad_out shape mismatch: expected {(int(x_shape[0]), int(self.out_features))}, got {tuple(grad_out.shape)}"
                    )
                if int(getattr(grad_out, "data")) == 0:
                    raise RuntimeError(
                        "grad_out CUDA tensor has no allocated devptr (data == 0)"
                    )

                grad_x = None
                grad_w = None
                grad_b = None

                if x.requires_grad:
                    grad_x = grad_out @ self.weight
                    grad_x.requires_grad = False
                    grad_x._set_ctx(None)

                if self.weight.requires_grad:
                    grad_w = grad_out.T @ x
                    grad_w.requires_grad = False
                    grad_w._set_ctx(None)

                if self.bias is not None and self.bias.requires_grad:
                    grad_b = grad_out.sum(axis=0)
                    grad_b.requires_grad = False
                    grad_b._set_ctx(None)

                if self.bias is None:
                    return (grad_x, grad_w)
                return (grad_x, grad_w, grad_b)

            parents = (
                (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
            )
            ctx = Context(parents=parents, backward_fn=backward_fn)
            ctx.save_for_backward(x, self.weight)
            y._set_ctx(ctx)
            return y

        # -----------------------
        # CPU path
        # -----------------------
        y = x @ self.weight.T
        if self.bias is not None:
            batch = int(x_shape[0])
            b2d = Tensor.stack([self.bias] * batch, axis=0)
            y = y + b2d

        if not req:
            return y

        out = Tensor(shape=y.shape, device=self.device, requires_grad=True)
        out.copy_from(y)

        def backward_fn(grad_out: Tensor):
            """
            Backward rule for y = x @ W^T (+ b) on CPU.

            Returns gradients for parents in the same order as `parents`.
            """
            x_saved, w_saved = ctx.saved_tensors

            grad_x = None
            grad_w = None
            grad_b = None

            if x_saved.requires_grad:
                grad_x = grad_out @ w_saved

            if w_saved.requires_grad:
                grad_w = grad_out.T @ x_saved

            if self.bias is not None and self.bias.requires_grad:
                grad_b = grad_out.sum(axis=0)

            if self.bias is None:
                return (grad_x, grad_w)
            return (grad_x, grad_w, grad_b)

        parents = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        ctx = Context(parents=parents, backward_fn=backward_fn)
        ctx.save_for_backward(x, self.weight)
        out._set_ctx(ctx)
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration dict.

        This stores only constructor-level hyperparameters. Parameter values are
        expected to be restored separately via the checkpoint/state mechanism.

        Returns
        -------
        Dict[str, Any]
            Configuration containing `in_features` (if known), `out_features`,
            `bias`, `device`, `dtype`, and `initializer`.
        """
        return {
            "in_features": (
                int(self.in_features) if self.in_features is not None else None
            ),
            "out_features": int(self.out_features),
            "bias": bool(self._use_bias),
            "device": str(self.device) if self.device is not None else None,
            "dtype": str(np.dtype(self.dtype).name),
            "initializer": str(self.initializer),
        }
