"""
Batch Normalization layers for KeyDNN (CPU).

This module implements classic Batch Normalization for:
- **BatchNorm1d**: 2D inputs of shape (N, C), normalized per feature over N.
- **BatchNorm2d**: 4D inputs of shape (N, C, H, W), normalized per channel over
  (N, H, W).

During **training**, the layer computes batch statistics (mean/variance) and
updates running statistics using an exponential moving average controlled by
`momentum`. During **evaluation**, the layer uses the stored running statistics.

Both layers support an optional affine transform:
    y = gamma * x_hat + beta

Autograd integration
--------------------
Backward propagation is implemented via `Context` closures that capture the
forward-pass intermediates required for the BatchNorm gradient formulas.

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

from typing import Any, Dict

from ..tensor._tensor_context import Context

from ...domain.device._device import Device
from .._module import Module
from .._parameter import Parameter
from ..tensor._tensor import Tensor
from ..module._serialization_core import register_module


@register_module()
class BatchNorm1d(Module):
    """
    Batch Normalization for 2D inputs of shape (N, C).

    This layer normalizes each feature channel independently using batch
    statistics computed over the batch dimension N:

        mean_c = mean(x[:, c])
        var_c  = var(x[:, c])
        x_hat  = (x - mean) / sqrt(var + eps)

    During training, running statistics are updated as:

        running = (1 - momentum) * running + momentum * batch_stat

    During evaluation, the running statistics are used for normalization.

    Parameters
    ----------
    num_features : int
        Number of feature channels C (the second dimension of the input).
    device : Device
        Device on which this module operates. Must match input tensor device.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    momentum : float, default=0.1
        Exponential moving average factor for running statistics.
    affine : bool, default=True
        If True, learnable scale (gamma) and shift (beta) parameters are used.

    Attributes
    ----------
    running_mean : Tensor
        Non-trainable buffer of shape (C,) storing running mean.
    running_var : Tensor
        Non-trainable buffer of shape (C,) storing running variance.
    gamma : Parameter | None
        Learnable scale parameter of shape (C,) if affine=True, else None.
    beta : Parameter | None
        Learnable shift parameter of shape (C,) if affine=True, else None.
    training : bool
        If True, uses batch statistics; if False, uses running statistics.
    """

    def __init__(
        self,
        num_features: int,
        *,
        device: Device,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        """
        Initialize a BatchNorm1d layer.

        Parameters
        ----------
        num_features : int
            Number of feature channels C.
        device : Device
            Device for parameters/buffers and forward computation.
        eps : float, default=1e-5
            Numerical stability constant added to variance.
        momentum : float, default=0.1
            Momentum used to update running statistics.
        affine : bool, default=True
            Whether to include learnable affine parameters (gamma, beta).
        """
        super().__init__()

        self.num_features = int(num_features)
        self.device = device
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.training = True

        # buffers (no NumPy here)
        self.running_mean = Tensor.full(
            (self.num_features,), 0.0, device=self.device, requires_grad=False
        )
        self.running_var = Tensor.full(
            (self.num_features,), 1.0, device=self.device, requires_grad=False
        )

        # affine params
        if self.affine:
            self.gamma = Parameter(
                (self.num_features,), self.device, requires_grad=True
            )
            self.beta = Parameter((self.num_features,), self.device, requires_grad=True)

            self.gamma.copy_from(
                Tensor.full(
                    (self.num_features,), 1.0, device=self.device, requires_grad=False
                )
            )
            self.beta.copy_from(
                Tensor.full(
                    (self.num_features,), 0.0, device=self.device, requires_grad=False
                )
            )
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply BatchNorm1d to an input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C). Must be a CPU tensor and on the same
            device as the module.

        Returns
        -------
        Tensor
            Output tensor of shape (N, C). Requires gradients if the input
            requires gradients and/or (when affine=True) gamma/beta require
            gradients.

        Raises
        ------
        RuntimeError
            If the input tensor is not on CPU.
        ValueError
            If device mismatches, input rank is not 2D, or channel count does
            not match `num_features`.
        """
        if not x.device.is_cpu():
            raise RuntimeError("BatchNorm1d is only supported for CPU tensors for now.")

        if x.device != self.device:
            raise ValueError(
                f"Device mismatch: x is {x.device}, module is {self.device}"
            )

        if len(x.shape) != 2:
            raise ValueError(
                f"BatchNorm1d expects 2D input (N, C), got shape {x.shape}"
            )

        N, C = x.shape
        if C != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got C={C}")

        # --- stats ---
        if self.training:
            mean = x.sum(axis=0) * (1.0 / float(N))  # (C,)
            x_centered = x - mean.broadcast_to(x.shape)
            var = (x_centered * x_centered).sum(axis=0) * (1.0 / float(N))  # (C,)

            # Update running stats WITHOUT graph history
            mean_det = Tensor(
                shape=mean.shape, device=self.device, requires_grad=False, ctx=None
            )
            var_det = Tensor(
                shape=var.shape, device=self.device, requires_grad=False, ctx=None
            )
            mean_det.copy_from(mean)
            var_det.copy_from(var)

            new_rm = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean_det
            new_rv = (1.0 - self.momentum) * self.running_var + self.momentum * var_det
            self.running_mean.copy_from(new_rm)
            self.running_var.copy_from(new_rv)
        else:
            mean = self.running_mean
            var = self.running_var

            x_centered = x - mean.broadcast_to(x.shape)

        inv_std = 1.0 / (var + self.eps).sqrt()  # (C,)
        inv_std_bc = inv_std.broadcast_to(x.shape)  # (N,C)
        x_hat = x_centered * inv_std_bc  # (N,C)

        if self.affine:
            assert self.gamma is not None and self.beta is not None
            gamma_bc = self.gamma.broadcast_to(x.shape)
            beta_bc = self.beta.broadcast_to(x.shape)
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

        # Fresh output tensor: BN defines ctx (avoid inheriting op ctx chain)
        out = Tensor(shape=x.shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from(y)

        if req:
            parents = (x,) if not self.affine else (x, self.gamma, self.beta)

            def backward_fn(grad_out: Tensor):
                """
                Compute gradients for BatchNorm1d.

                Parameters
                ----------
                grad_out : Tensor
                    Upstream gradient dL/dy of shape (N, C).

                Returns
                -------
                tuple[Tensor, ...]
                    Gradients in the same order as `parents`:
                    - (dx,) if affine=False
                    - (dx, dgamma, dbeta) if affine=True
                """
                g = grad_out  # (N,C)

                if self.affine:
                    assert self.gamma is not None and self.beta is not None
                    dxhat = g * self.gamma.broadcast_to(x.shape)
                else:
                    dxhat = g

                sum_dxhat = dxhat.sum(axis=0).broadcast_to(x.shape)  # (N,C)
                sum_dxhat_xhat = (
                    (dxhat * x_hat).sum(axis=0).broadcast_to(x.shape)
                )  # (N,C)

                dx = (
                    (1.0 / float(N))
                    * inv_std_bc
                    * (float(N) * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
                )

                dx_t = Tensor(
                    shape=x.shape, device=self.device, requires_grad=False, ctx=None
                )
                dx_t.copy_from(dx)

                if not self.affine:
                    return (dx_t,)

                dgamma = (g * x_hat).sum(axis=0)  # (C,)
                dbeta = g.sum(axis=0)  # (C,)

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

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary sufficient to reconstruct the module via
            `from_config`.
        """
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "device": str(self.device),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchNorm1d":
        """
        Construct a BatchNorm1d instance from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration as produced by `get_config()`.

        Returns
        -------
        BatchNorm1d
            Reconstructed module instance.
        """
        dev_str = str(config.get("device", "cpu"))
        device = Device(dev_str)

        return cls(
            num_features=int(config["num_features"]),
            device=device,
            eps=float(config["eps"]),
            momentum=float(config["momentum"]),
            affine=bool(config["affine"]),
        )


@register_module()
class BatchNorm2d(Module):
    """
    Batch Normalization for 4D inputs of shape (N, C, H, W).

    This layer normalizes each channel independently using statistics computed
    over the axes (N, H, W):

        mean_c = mean(x[:, c, :, :])
        var_c  = var(x[:, c, :, :])
        x_hat  = (x - mean) / sqrt(var + eps)

    During training, running statistics are updated via exponential moving
    average controlled by `momentum`. During evaluation, running statistics
    are used.

    Parameters
    ----------
    num_features : int
        Number of channels C.
    device : Device
        Device on which this module operates. Must match input tensor device.
    eps : float, default=1e-5
        Small constant added to variance for numerical stability.
    momentum : float, default=0.1
        Exponential moving average factor for running statistics.
    affine : bool, default=True
        If True, learnable scale (gamma) and shift (beta) parameters are used.

    Attributes
    ----------
    running_mean : Tensor
        Non-trainable buffer of shape (C,) storing running mean.
    running_var : Tensor
        Non-trainable buffer of shape (C,) storing running variance.
    gamma : Parameter | None
        Learnable scale parameter of shape (C,) if affine=True, else None.
    beta : Parameter | None
        Learnable shift parameter of shape (C,) if affine=True, else None.
    training : bool
        If True, uses batch statistics; if False, uses running statistics.
    """

    def __init__(
        self,
        num_features: int,
        *,
        device: Device,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        """
        Initialize a BatchNorm2d layer.

        Parameters
        ----------
        num_features : int
            Number of channels C.
        device : Device
            Device for parameters/buffers and forward computation.
        eps : float, default=1e-5
            Numerical stability constant added to variance.
        momentum : float, default=0.1
            Momentum used to update running statistics.
        affine : bool, default=True
            Whether to include learnable affine parameters (gamma, beta).
        """
        super().__init__()

        self.num_features = int(num_features)
        self.device = device
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.training = True

        self.running_mean = Tensor.full(
            (self.num_features,), 0.0, device=self.device, requires_grad=False
        )
        self.running_var = Tensor.full(
            (self.num_features,), 1.0, device=self.device, requires_grad=False
        )

        if self.affine:
            self.gamma = Parameter(
                shape=(self.num_features,), device=self.device, requires_grad=True
            )
            self.beta = Parameter(
                shape=(self.num_features,), device=self.device, requires_grad=True
            )
            self.gamma.copy_from(
                Tensor.full(
                    (self.num_features,), 1.0, device=self.device, requires_grad=False
                )
            )
            self.beta.copy_from(
                Tensor.full(
                    (self.num_features,), 0.0, device=self.device, requires_grad=False
                )
            )
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply BatchNorm2d to an input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, C, H, W). Must be a CPU tensor and on the
            same device as the module.

        Returns
        -------
        Tensor
            Output tensor of shape (N, C, H, W). Requires gradients if the input
            requires gradients and/or (when affine=True) gamma/beta require
            gradients.

        Raises
        ------
        RuntimeError
            If the input tensor is not on CPU.
        ValueError
            If device mismatches, input rank is not 4D, or channel count does
            not match `num_features`.
        """
        if not x.device.is_cpu():
            raise RuntimeError("BatchNorm2d is only supported for CPU tensors for now.")

        if x.device != self.device:
            raise ValueError(
                f"Device mismatch: x is {x.device}, module is {self.device}"
            )

        if len(x.shape) != 4:
            raise ValueError(
                f"BatchNorm2d expects 4D input (N, C, H, W), got shape {x.shape}"
            )

        N, C, H, W = x.shape
        if C != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got C={C}")

        m = float(N * H * W)

        def _bc_c_to_nchw(t_c: Tensor) -> Tensor:
            # (C,) -> (1,C,1,1) -> (N,C,H,W)
            return t_c.reshape((1, C, 1, 1)).broadcast_to(x.shape)

        if self.training:
            # mean over (N,H,W) per channel:
            mean = x.sum(axis=0).sum(axis=1).sum(axis=1) * (1.0 / m)  # (C,)
            mean_bc = _bc_c_to_nchw(mean)
            x_centered = x - mean_bc
            var = (x_centered * x_centered).sum(axis=0).sum(axis=1).sum(axis=1) * (
                1.0 / m
            )  # (C,)

            mean_det = Tensor(
                shape=mean.shape, device=self.device, requires_grad=False, ctx=None
            )
            var_det = Tensor(
                shape=var.shape, device=self.device, requires_grad=False, ctx=None
            )
            mean_det.copy_from(mean)
            var_det.copy_from(var)

            new_rm = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean_det
            new_rv = (1.0 - self.momentum) * self.running_var + self.momentum * var_det
            self.running_mean.copy_from(new_rm)
            self.running_var.copy_from(new_rv)
        else:
            mean = self.running_mean
            var = self.running_var
            mean_bc = _bc_c_to_nchw(mean)
            x_centered = x - mean_bc

        inv_std = 1.0 / (var + self.eps).sqrt()  # (C,)
        inv_std_bc = _bc_c_to_nchw(inv_std)  # (N,C,H,W)
        x_hat = x_centered * inv_std_bc  # (N,C,H,W)

        if self.affine:
            assert self.gamma is not None and self.beta is not None
            gamma_bc = _bc_c_to_nchw(self.gamma)
            beta_bc = _bc_c_to_nchw(self.beta)
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
                Compute gradients for BatchNorm2d.

                Parameters
                ----------
                grad_out : Tensor
                    Upstream gradient dL/dy of shape (N, C, H, W).

                Returns
                -------
                tuple[Tensor, ...]
                    Gradients in the same order as `parents`:
                    - (dx,) if affine=False
                    - (dx, dgamma, dbeta) if affine=True
                """
                g = grad_out  # (N,C,H,W)

                if self.affine:
                    assert self.gamma is not None and self.beta is not None
                    dxhat = g * _bc_c_to_nchw(self.gamma)
                else:
                    dxhat = g

                # sum over (N,H,W): do it via chaining sums
                sum_dxhat = dxhat.sum(axis=0).sum(axis=1).sum(axis=1)  # (C,)
                sum_dxhat_xhat = (
                    (dxhat * x_hat).sum(axis=0).sum(axis=1).sum(axis=1)
                )  # (C,)

                sum_dxhat_bc = _bc_c_to_nchw(sum_dxhat)
                sum_dxhat_xhat_bc = _bc_c_to_nchw(sum_dxhat_xhat)

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

                dgamma = (g * x_hat).sum(axis=0).sum(axis=1).sum(axis=1)  # (C,)
                dbeta = g.sum(axis=0).sum(axis=1).sum(axis=1)  # (C,)

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

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary sufficient to reconstruct the module via
            `from_config`.
        """
        return {
            "num_features": self.num_features,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "device": str(self.device),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BatchNorm2d":
        """
        Construct a BatchNorm2d instance from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration as produced by `get_config()`.

        Returns
        -------
        BatchNorm2d
            Reconstructed module instance.
        """
        dev_str = str(config.get("device", "cpu"))
        device = Device(dev_str)
        return cls(
            num_features=int(config["num_features"]),
            device=device,
            eps=float(config["eps"]),
            momentum=float(config["momentum"]),
            affine=bool(config["affine"]),
        )
