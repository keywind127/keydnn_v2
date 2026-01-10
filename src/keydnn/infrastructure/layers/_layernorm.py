"""
Layer Normalization layer for KeyDNN (CPU).

This module implements classic Layer Normalization:

- Normalizes over the *last K dimensions* of the input, where
  K = len(normalized_shape), and normalized_shape must match the trailing
  dimensions of the input tensor.

For an input x of shape:
    (..., *normalized_shape)

LayerNorm computes per-sample statistics over the normalized axes:

    mean = mean(x, axes=last K)
    var  = var(x, axes=last K)
    x_hat = (x - mean) / sqrt(var + eps)

Optionally applies an affine transform:

    y = gamma * x_hat + beta

Where gamma and beta have shape normalized_shape and are broadcast across the
leading dimensions.

Autograd integration
--------------------
Backward propagation is implemented via `Context` closures that capture the
forward-pass intermediates required for the LayerNorm gradient formulas.

Serialization
-------------
Modules are registered via `register_module()` and expose `get_config()` /
`from_config()` for config-based (de)serialization.

Notes
-----
- Current implementation supports **CPU tensors only**.
- Device mismatches are treated as errors to keep module behavior strict and
  consistent with other KeyDNN modules.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from ..tensor._tensor_context import Context

from ...domain.device._device import Device
from .._module import Module
from .._parameter import Parameter
from ..tensor._tensor import Tensor
from ..module._serialization_core import register_module


def _as_tuple_ints(shape: Iterable[int]) -> Tuple[int, ...]:
    t = tuple(int(x) for x in shape)
    if any(d <= 0 for d in t):
        raise ValueError(f"normalized_shape must be positive ints, got {t}")
    return t


@register_module()
class LayerNorm(Module):
    """
    Layer Normalization over the last `len(normalized_shape)` dimensions.

    Parameters
    ----------
    normalized_shape : tuple[int, ...] | list[int]
        The shape of the dimensions to be normalized. Must match the trailing
        dimensions of the input tensor.
    device : Device
        Device on which this module operates. Must match input tensor device.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    affine : bool, default=True
        If True, learnable scale (gamma) and shift (beta) parameters are used.

    Attributes
    ----------
    normalized_shape : tuple[int, ...]
        Normalized trailing dimensions shape.
    gamma : Parameter | None
        Learnable scale parameter of shape normalized_shape if affine=True.
    beta : Parameter | None
        Learnable shift parameter of shape normalized_shape if affine=True.
    """

    def __init__(
        self,
        normalized_shape: Iterable[int],
        *,
        device: Device,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()

        self.normalized_shape = _as_tuple_ints(normalized_shape)
        self.device = device
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.gamma = Parameter(
                self.normalized_shape, self.device, requires_grad=True
            )
            self.beta = Parameter(
                self.normalized_shape, self.device, requires_grad=True
            )

            # init: gamma=1, beta=0 (common default)
            ones = Tensor.ones(
                shape=self.normalized_shape, device=self.device, requires_grad=False
            )
            zeros = Tensor.zeros(
                shape=self.normalized_shape, device=self.device, requires_grad=False
            )
            self.gamma.copy_from(ones)
            self.beta.copy_from(zeros)
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply LayerNorm to an input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., *normalized_shape). Must be a CPU tensor
            and on the same device as the module.

        Returns
        -------
        Tensor
            Output tensor of the same shape as `x`.

        Raises
        ------
        RuntimeError
            If the input tensor is not on CPU.
        ValueError
            If device mismatches, rank is insufficient, or trailing dims do not
            match `normalized_shape`.
        """
        if not x.device.is_cpu():
            raise RuntimeError("LayerNorm is only supported for CPU tensors for now.")

        if x.device != self.device:
            raise ValueError(
                f"Device mismatch: x is {x.device}, module is {self.device}"
            )

        k = len(self.normalized_shape)
        if k == 0:
            raise ValueError("normalized_shape must be non-empty for LayerNorm.")

        if len(x.shape) < k:
            raise ValueError(
                f"LayerNorm expects input rank >= {k}, got shape {x.shape}"
            )

        if tuple(x.shape[-k:]) != self.normalized_shape:
            raise ValueError(
                f"LayerNorm trailing dims must match normalized_shape={self.normalized_shape}, "
                f"got trailing dims {tuple(x.shape[-k:])} from x.shape={x.shape}"
            )

        # prefix dims are the non-normalized leading dims
        prefix_shape = x.shape[:-k]
        out_shape_reduced = prefix_shape + (1,) * k  # keepdims-like reduction target

        # number of elements in normalized part
        m = 1.0
        for d in self.normalized_shape:
            m *= float(d)

        def _sum_norm_axes(t: Tensor) -> Tensor:
            # (..., norm) -> (..., 1,1,...,1) (K ones)
            return t.sum_to_shape(out_shape_reduced)

        def _bc_norm_param_to_x(p: Tensor) -> Tensor:
            # (norm) -> (1,...,1,norm) -> broadcast to x.shape
            view_shape = (1,) * len(prefix_shape) + self.normalized_shape
            return p.reshape(view_shape).broadcast_to(x.shape)

        # mean/var over normalized axes per sample
        sum_x = _sum_norm_axes(x)
        mean = sum_x * (1.0 / m)  # shape = prefix + (1,)*k
        mean_bc = mean.broadcast_to(x.shape)

        x_centered = x - mean_bc
        var = _sum_norm_axes(x_centered * x_centered) * (1.0 / m)  # same reduced shape

        inv_std = 1.0 / (var + self.eps).sqrt()  # reduced shape
        inv_std_bc = inv_std.broadcast_to(x.shape)

        x_hat = x_centered * inv_std_bc

        if self.affine:
            assert self.gamma is not None and self.beta is not None
            gamma_bc = _bc_norm_param_to_x(self.gamma)
            beta_bc = _bc_norm_param_to_x(self.beta)
            y = gamma_bc * x_hat + beta_bc
        else:
            y = x_hat

        req = x.requires_grad or (
            self.affine
            and (
                (self.gamma is not None and self.gamma.requires_grad)
                or (self.beta is not None and self.beta.requires_grad)
            )
        )

        out = Tensor(shape=x.shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from(y)

        if req:
            parents = (x,) if not self.affine else (x, self.gamma, self.beta)

            def backward_fn(grad_out: Tensor):
                """
                Compute gradients for LayerNorm.

                Parameters
                ----------
                grad_out : Tensor
                    Upstream gradient dL/dy of shape x.shape.

                Returns
                -------
                tuple[Tensor, ...]
                    Gradients in the same order as `parents`:
                    - (dx,) if affine=False
                    - (dx, dgamma, dbeta) if affine=True
                """
                g = grad_out  # same shape as x

                if self.affine:
                    assert self.gamma is not None and self.beta is not None
                    dxhat = g * _bc_norm_param_to_x(self.gamma)
                else:
                    dxhat = g

                # reduce over normalized axes per sample
                sum_dxhat = _sum_norm_axes(dxhat)  # reduced shape
                sum_dxhat_xhat = _sum_norm_axes(dxhat * x_hat)  # reduced shape

                sum_dxhat_bc = sum_dxhat.broadcast_to(x.shape)
                sum_dxhat_xhat_bc = sum_dxhat_xhat.broadcast_to(x.shape)

                # dx formula mirrors BN, but per-sample over normalized axes
                dx = (
                    (1.0 / m)
                    * inv_std_bc
                    * (m * dxhat - sum_dxhat_bc - x_hat * sum_dxhat_xhat_bc)
                )

                dx_t = Tensor(
                    shape=x.shape, device=self.device, requires_grad=False, ctx=None
                )
                dx_t.copy_from(dx)

                if not self.affine:
                    return (dx_t,)

                # dgamma/dbeta: reduce over prefix dims (all non-normalized dims)
                # target shape keeps normalized dims, collapses prefix to 1s
                target = (1,) * len(prefix_shape) + self.normalized_shape

                dgamma = (g * x_hat).sum_to_shape(target).reshape(self.normalized_shape)
                dbeta = g.sum_to_shape(target).reshape(self.normalized_shape)

                dgamma_t = Tensor(
                    shape=dgamma.shape,
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                )
                dbeta_t = Tensor(
                    shape=dbeta.shape, device=self.device, requires_grad=False, ctx=None
                )
                dgamma_t.copy_from(dgamma)
                dbeta_t.copy_from(dbeta)

                return (dx_t, dgamma_t, dbeta_t)

            ctx = Context(parents=parents, backward_fn=backward_fn)
            out._set_ctx(ctx)

        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for this module.
        """
        return {
            "normalized_shape": list(self.normalized_shape),
            "eps": self.eps,
            "affine": self.affine,
            "device": str(self.device),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LayerNorm":
        """
        Construct a LayerNorm instance from a configuration dictionary.
        """
        dev_str = str(config.get("device", "cpu"))
        device = Device(dev_str)

        return cls(
            normalized_shape=tuple(int(x) for x in config["normalized_shape"]),
            device=device,
            eps=float(config["eps"]),
            affine=bool(config["affine"]),
        )
