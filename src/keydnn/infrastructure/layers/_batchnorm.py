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

import numpy as np

from ...domain.device._device import Device
from .._module import Module
from .._parameter import Parameter
from .._tensor import Context, Tensor
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
        super().__init__()  # <-- REQUIRED (sets _parameters/_modules)

        self.num_features = int(num_features)
        self.device = device
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.training = True

        # buffers
        self.running_mean = Tensor(
            (self.num_features,), self.device, requires_grad=False, ctx=None
        )
        self.running_var = Tensor(
            (self.num_features,), self.device, requires_grad=False, ctx=None
        )
        self.running_mean.copy_from_numpy(
            np.zeros((self.num_features,), dtype=np.float32)
        )
        self.running_var.copy_from_numpy(
            np.ones((self.num_features,), dtype=np.float32)
        )

        # affine params
        if self.affine:
            self.gamma = Parameter(
                (self.num_features,), self.device, requires_grad=True
            )
            self.beta = Parameter((self.num_features,), self.device, requires_grad=True)

            self.gamma.copy_from_numpy(np.ones((self.num_features,), dtype=np.float32))
            self.beta.copy_from_numpy(np.zeros((self.num_features,), dtype=np.float32))
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
            # Keep it strict for now (consistent with other modules)
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

        x_np = x.to_numpy()

        if self.training:
            mean = x_np.mean(axis=0)
            var = x_np.var(axis=0)

            # Update running stats
            rm = self.running_mean.to_numpy()
            rv = self.running_var.to_numpy()
            self.running_mean.copy_from_numpy(
                (1.0 - self.momentum) * rm + self.momentum * mean
            )
            self.running_var.copy_from_numpy(
                (1.0 - self.momentum) * rv + self.momentum * var
            )
        else:
            mean = self.running_mean.to_numpy()
            var = self.running_var.to_numpy()

        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x_np - mean) * inv_std

        if self.affine:
            gamma_np = self.gamma.to_numpy()
            beta_np = self.beta.to_numpy()
            y_np = gamma_np * x_hat + beta_np
        else:
            y_np = x_hat

        req = x.requires_grad or (
            self.affine and (self.gamma.requires_grad or self.beta.requires_grad)
        )
        out = Tensor(shape=x.shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(y_np)

        if req:
            parents = (x,)
            if self.affine:
                parents = (x, self.gamma, self.beta)

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
                g = grad_out.to_numpy()  # (N, C)

                # If affine: grad wrt x_hat is g * gamma, else just g
                if self.affine:
                    gamma_np_local = self.gamma.to_numpy()
                    dxhat = g * gamma_np_local
                else:
                    dxhat = g

                # BatchNorm backward (for x) using x_hat and inv_std
                # dx = (1/N) * inv_std * (N*dxhat - sum(dxhat) - x_hat*sum(dxhat*x_hat))
                sum_dxhat = dxhat.sum(axis=0)
                sum_dxhat_xhat = (dxhat * x_hat).sum(axis=0)

                dx = (
                    (1.0 / N)
                    * inv_std
                    * (N * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
                )

                dx_t = Tensor(
                    shape=x.shape, device=self.device, requires_grad=False, ctx=None
                )
                dx_t.copy_from_numpy(dx)

                if not self.affine:
                    return (dx_t,)

                dgamma = (g * x_hat).sum(axis=0)
                dbeta = g.sum(axis=0)

                dgamma_t = Tensor(
                    shape=self.gamma.shape,
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                )
                dbeta_t = Tensor(
                    shape=self.beta.shape,
                    device=self.device,
                    requires_grad=False,
                    ctx=None,
                )
                dgamma_t.copy_from_numpy(dgamma.astype(np.float32, copy=False))
                dbeta_t.copy_from_numpy(dbeta.astype(np.float32, copy=False))

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
        super().__init__()  # IMPORTANT: creates _parameters/_modules for auto-registration

        self.num_features = int(num_features)
        self.device = device
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.training = True

        # Running stats (buffers)
        self.running_mean = Tensor(
            shape=(self.num_features,),
            device=self.device,
            requires_grad=False,
            ctx=None,
        )
        self.running_var = Tensor(
            shape=(self.num_features,),
            device=self.device,
            requires_grad=False,
            ctx=None,
        )
        self.running_mean.copy_from_numpy(
            np.zeros((self.num_features,), dtype=np.float32)
        )
        self.running_var.copy_from_numpy(
            np.ones((self.num_features,), dtype=np.float32)
        )

        # Affine parameters
        if self.affine:
            self.gamma = Parameter(
                shape=(self.num_features,),
                device=self.device,
                requires_grad=True,
            )
            self.beta = Parameter(
                shape=(self.num_features,),
                device=self.device,
                requires_grad=True,
            )
            self.gamma.copy_from_numpy(np.ones((self.num_features,), dtype=np.float32))
            self.beta.copy_from_numpy(np.zeros((self.num_features,), dtype=np.float32))
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

        x_np = x.to_numpy()  # (N,C,H,W)

        # Compute stats per-channel over (N,H,W)
        axes = (0, 2, 3)

        if self.training:
            mean = x_np.mean(axis=axes)  # (C,)
            var = x_np.var(axis=axes)  # (C,)

            # Update running stats
            rm = self.running_mean.to_numpy()
            rv = self.running_var.to_numpy()
            self.running_mean.copy_from_numpy(
                (1.0 - self.momentum) * rm + self.momentum * mean
            )
            self.running_var.copy_from_numpy(
                (1.0 - self.momentum) * rv + self.momentum * var
            )
        else:
            mean = self.running_mean.to_numpy()
            var = self.running_var.to_numpy()

        inv_std = 1.0 / np.sqrt(var + self.eps)  # (C,)

        # Broadcast to (N,C,H,W)
        mean_bc = mean.reshape(1, C, 1, 1)
        inv_std_bc = inv_std.reshape(1, C, 1, 1)

        x_hat = (x_np - mean_bc) * inv_std_bc  # (N,C,H,W)

        if self.affine:
            gamma_np = self.gamma.to_numpy().reshape(1, C, 1, 1)
            beta_np = self.beta.to_numpy().reshape(1, C, 1, 1)
            y_np = gamma_np * x_hat + beta_np
        else:
            y_np = x_hat

        req = x.requires_grad or (
            self.affine
            and (
                (self.gamma is not None and self.gamma.requires_grad)
                or (self.beta is not None and self.beta.requires_grad)
            )
        )

        out = Tensor(shape=x.shape, device=self.device, requires_grad=req, ctx=None)
        out.copy_from_numpy(y_np.astype(np.float32, copy=False))

        if req:
            parents = (x,) if not self.affine else (x, self.gamma, self.beta)

            # m = number of elements per channel used in normalization
            m = float(N * H * W)

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
                g = grad_out.to_numpy().astype(np.float32, copy=False)  # (N,C,H,W)

                if self.affine:
                    gamma_np_local = (
                        self.gamma.to_numpy()
                        .astype(np.float32, copy=False)
                        .reshape(1, C, 1, 1)
                    )
                    dxhat = g * gamma_np_local
                else:
                    dxhat = g

                # sums over (N,H,W) per channel => keepdims for broadcast
                sum_dxhat = dxhat.sum(axis=axes, keepdims=True)  # (1,C,1,1)
                sum_dxhat_xhat = (dxhat * x_hat).sum(
                    axis=axes, keepdims=True
                )  # (1,C,1,1)

                dx = (
                    (1.0 / m)
                    * inv_std_bc
                    * (m * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat)
                )
                dx_t = Tensor(
                    shape=x.shape, device=self.device, requires_grad=False, ctx=None
                )
                dx_t.copy_from_numpy(dx.astype(np.float32, copy=False))

                if not self.affine:
                    return (dx_t,)

                dgamma = (g * x_hat).sum(axis=axes)  # (C,)
                dbeta = g.sum(axis=axes)  # (C,)

                dgamma_t = Tensor(
                    shape=(C,), device=self.device, requires_grad=False, ctx=None
                )
                dbeta_t = Tensor(
                    shape=(C,), device=self.device, requires_grad=False, ctx=None
                )
                dgamma_t.copy_from_numpy(dgamma.astype(np.float32, copy=False))
                dbeta_t.copy_from_numpy(dbeta.astype(np.float32, copy=False))

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
