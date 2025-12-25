"""
Optimizer primitives for KeyDNN.

This module provides simple, CPU-only optimizers that update `Parameter`
instances in-place using their accumulated gradients.

Design notes
------------
- Optimizers operate exclusively on `Parameter` / `Tensor` objects and read
  gradients from `p.grad`.
- Parameters with `grad is None` are skipped to support partial graphs and
  frozen weights.
- Updates are applied by writing directly into parameter storage via
  `copy_from`, keeping optimizers independent from the autograd execution
  engine and graph construction details.
- Optimizer math is expressed entirely in terms of Tensor operations;
  backend-specific numerical implementations (e.g., NumPy) are encapsulated
  within Tensor methods only.
- Current implementations are CPU-only. CUDA support may be added once
  device-aware Tensor kernels and gradient storage are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict

from .tensor._tensor import Tensor
from ._parameter import Parameter


@dataclass
class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates parameters in-place using their accumulated
    gradients and a fixed learning rate.

    Update rule
    -----------
    For each parameter ``p`` with gradient ``g``:

    - If ``weight_decay > 0`` (classical L2 regularization):
        ``g <- g + weight_decay * p``
    - Parameter update:
        ``p <- p - lr * g``

    Parameters
    ----------
    params : Sequence[Parameter]
        Parameters to be optimized.
    lr : float, optional
        Learning rate. Must be positive. Defaults to 1e-3.
    weight_decay : float, optional
        Classical L2 weight decay coefficient (coupled). Must be non-negative.
        Defaults to 0.0.

    Notes
    -----
    - Parameters with ``grad is None`` are skipped.
    - Momentum, Nesterov, and other SGD variants are intentionally omitted
      in this minimal implementation.
    - This optimizer is CPU-only in the current KeyDNN implementation.
    """

    params: Sequence[Parameter]
    lr: float = 1e-3
    weight_decay: float = 0.0

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Construct an SGD optimizer.

        Parameters
        ----------
        params : Iterable[Parameter]
            Iterable of parameters to optimize. The iterable is consumed and
            stored internally.
        lr : float, optional
            Learning rate. Must be > 0. Defaults to 1e-3.
        weight_decay : float, optional
            Classical L2 regularization coefficient. Must be >= 0.
            Defaults to 0.0.

        Raises
        ------
        ValueError
            If ``lr <= 0`` or ``weight_decay < 0``.
        """
        self.params = list(params)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

    def zero_grad(self) -> None:
        """
        Clear gradients for all managed parameters.

        Notes
        -----
        This calls ``zero_grad()`` on each parameter, which clears the stored
        gradient tensor (if any). Training loops typically call `zero_grad()`
        before computing a new backward pass to avoid gradient accumulation.
        """
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """
        Apply one SGD update step to all managed parameters.

        Notes
        -----
        - Parameters with ``grad is None`` are skipped.
        - This method is CPU-only and raises a device-not-supported error if
          either a parameter or its gradient resides on a non-CPU device.
        - Weight decay is implemented as classical L2 regularization
          (coupled with the gradient), not as decoupled weight decay.
        """
        for p in self.params:
            g = p.grad
            if g is None:
                continue

            if not p.device.is_cpu() or not g.device.is_cpu():
                p._raise_device_not_supported("sgd_step")

            # Optional L2 weight decay (decoupled is AdamW; this is classical)
            if self.weight_decay != 0.0:
                g = g + (self.weight_decay * p)

            # p <- p - lr * g
            updated = p - (self.lr * g)

            # Write back to parameter storage (CPU-only)
            p.copy_from(updated)


@dataclass
class Adam:
    """
    Adam optimizer.

    Adam maintains exponentially decaying averages of past gradients (first
    moment) and past squared gradients (second moment), and applies bias
    correction to both estimates.

    Update rule
    -----------
    Let ``g_t`` be the gradient at step ``t``:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)

        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)

        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    If ``weight_decay > 0`` (classical L2 regularization):

        g_t <- g_t + weight_decay * p

    Parameters
    ----------
    params : Sequence[Parameter]
        Parameters to be optimized.
    lr : float, optional
        Learning rate. Must be positive. Defaults to 1e-3.
    betas : tuple[float, float], optional
        Exponential decay rates for the first and second moments.
        Each must be in (0, 1). Defaults to (0.9, 0.999).
    eps : float, optional
        Numerical stability epsilon added to the denominator. Must be positive.
        Defaults to 1e-8.
    weight_decay : float, optional
        Classical L2 regularization coefficient (coupled). Must be non-negative.
        Defaults to 0.0.

    Notes
    -----
    - Parameters with ``grad is None`` are skipped.
    - Weight decay here is *classical L2* (not decoupled AdamW).
    - Optimizer state (m, v, t) is stored per-parameter and persists across steps.
    - This optimizer is CPU-only in the current KeyDNN implementation.
    """

    params: Sequence[Parameter]
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        Construct an Adam optimizer.

        Parameters
        ----------
        params : Iterable[Parameter]
            Iterable of parameters to optimize. The iterable is consumed and
            stored internally.
        lr : float, optional
            Learning rate. Must be > 0. Defaults to 1e-3.
        betas : tuple[float, float], optional
            Exponential decay rates (beta1, beta2), each in (0, 1).
            Defaults to (0.9, 0.999).
        eps : float, optional
            Numerical stability epsilon. Must be > 0. Defaults to 1e-8.
        weight_decay : float, optional
            Classical L2 regularization coefficient. Must be >= 0.
            Defaults to 0.0.

        Raises
        ------
        ValueError
            If any hyperparameter is outside its valid range.
        """
        self.params = list(params)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        b1, b2 = self.betas
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0.0 < b1 < 1.0) or not (0.0 < b2 < 1.0):
            raise ValueError(f"betas must be in (0,1), got {self.betas}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

        # Per-parameter state: id(p) -> {t, m, v}
        # Store m, v as NumPy arrays (float32) to avoid depending on Tensor ops.
        self._state: Dict[int, Dict[str, object]] = {}

    def zero_grad(self) -> None:
        """
        Clear gradients for all managed parameters.

        Notes
        -----
        This calls ``zero_grad()`` on each parameter, which clears the stored
        gradient tensor (if any). Training loops typically call `zero_grad()`
        before computing a new backward pass to avoid gradient accumulation.
        """
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """
        Apply one Adam update step to all managed parameters.

        Notes
        -----
        - Parameters with ``grad is None`` are skipped.
        - This method is CPU-only and raises a device-not-supported error if
          either a parameter or its gradient resides on a non-CPU device.
        - Weight decay is implemented as classical L2 regularization
          (coupled with the gradient), not as decoupled AdamW.
        - Optimizer state is created lazily on the first update for each
          parameter.
        """
        b1, b2 = self.betas

        for p in self.params:
            g = p.grad
            if g is None:
                continue

            # Keep legacy behavior: CPU-only for now
            if not p.device.is_cpu() or not g.device.is_cpu():
                p._raise_device_not_supported("adam_step")

            pid = id(p)
            st = self._state.get(pid)
            if st is None:
                # m, v should NOT require grad (optimizer state)
                m = Tensor(shape=p.shape, device=p.device, requires_grad=False)
                v = Tensor(shape=p.shape, device=p.device, requires_grad=False)
                m.fill(0.0)
                v.fill(0.0)
                st = {"t": 0, "m": m, "v": v}
                self._state[pid] = st

            st["t"] = int(st["t"]) + 1
            t = int(st["t"])
            m: Tensor = st["m"]  # type: ignore[assignment]
            v: Tensor = st["v"]  # type: ignore[assignment]

            # Classical L2 weight decay (coupled): g <- g + wd * p
            if self.weight_decay != 0.0:
                g_eff = g + (self.weight_decay * p)
            else:
                g_eff = g

            # m = b1*m + (1-b1)*g
            # v = b2*v + (1-b2)*(g*g)
            m_new = (b1 * m) + ((1.0 - b1) * g_eff)
            v_new = (b2 * v) + ((1.0 - b2) * (g_eff * g_eff))

            # write state in-place
            m.copy_from(m_new)
            v.copy_from(v_new)

            # bias correction
            # NOTE: these are Python scalars; that’s fine.
            m_hat = m / (1.0 - (b1**t))
            v_hat = v / (1.0 - (b2**t))

            # update = lr * m_hat / (sqrt(v_hat) + eps)
            # requires Tensor.sqrt() (or equivalent)
            denom = v_hat.sqrt() + self.eps
            step = self.lr * (m_hat / denom)

            # p <- p - step
            p.copy_from(p - step)
