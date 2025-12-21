"""
Recurrent neural network (RNN) layers for KeyDNN.

This module provides a minimal, CPU-only implementation of a vanilla RNN:

- `RNNCell`: a single-timestep cell computing
    h_t = tanh(x_t W_ih + h_{t-1} W_hh + b_ih + b_hh)
- `RNN`: a simple unidirectional RNN that applies `RNNCell` over a time-major
  sequence input of shape (T, N, D).

Design goals
-----------
- Keep the implementation small and readable (NumPy on CPU).
- Integrate with KeyDNN autograd via `Context` and `Tensor.backward()`.
- Avoid Tensor broadcasting semantics (KeyDNN currently enforces strict shape
  equality for elementwise ops), so biases are expanded explicitly in NumPy.

Notes
-----
- This implementation is intended for learning and correctness first; it is not
  optimized. Python loops over time are used deliberately.
- Device handling is currently CPU-only. Parameters are created on CPU in
  `RNNCell.__post_init__` to match the available backend.
- `RNN` constructs per-timestep `Tensor` objects from NumPy slices; these are
  treated as independent leaf tensors in the current minimal autograd engine
  (i.e., gradients will not automatically flow back into the original `x`
  unless you later implement view/slice tensors).

Public API
----------
- `tensor_from_numpy`
- `RNNCell`
- `RNN`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..domain._device import Device
from ..infrastructure._tensor import Tensor, Context
from ..infrastructure._parameter import Parameter
from ..infrastructure._module import Module


def tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool = False
) -> Tensor:
    """
    Construct a KeyDNN `Tensor` from a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Source array. It will be converted to `np.float32`.
    device : Device
        Target device placement (currently CPU is supported by the Tensor backend).
    requires_grad : bool, optional
        Whether the returned tensor should participate in autograd.
        Defaults to False.

    Returns
    -------
    Tensor
        A new tensor with shape equal to `arr.shape`, whose storage is populated
        via `Tensor.copy_from_numpy`.
    """
    arr = np.asarray(arr, dtype=np.float32)
    out = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad)
    out.copy_from_numpy(arr)
    return out


@dataclass
class RNNCell(Module):
    """
    Vanilla RNN cell for a single timestep.

    This cell computes:

        a_t = x_t @ W_ih + h_{t-1} @ W_hh + b_ih + b_hh
        h_t = tanh(a_t)

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        If True, uses biases `b_ih` and `b_hh` (both shape (H,)).
        Defaults to True.

    Attributes
    ----------
    W_ih : Parameter
        Input-to-hidden weight matrix of shape (D, H).
    W_hh : Parameter
        Hidden-to-hidden weight matrix of shape (H, H).
    b_ih : Optional[Parameter]
        Input bias of shape (H,) if `bias=True`, else None.
    b_hh : Optional[Parameter]
        Hidden bias of shape (H,) if `bias=True`, else None.

    Shapes
    ------
    x_t : (N, D)
    h_prev : (N, H)
    h_t : (N, H)

    Notes
    -----
    - Pure NumPy CPU implementation (no ctypes/kernels).
    - Bias addition is expanded explicitly in NumPy to avoid broadcasting in
      KeyDNN Tensor ops (strict shape equality).
    - Because this is a dataclass, `Module.__init__()` is NOT called
      automatically. We explicitly call it in `__post_init__` so that parameter
      auto-registration in `Module.__setattr__` works.
    """

    input_size: int
    hidden_size: int
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Initialize module bookkeeping and create trainable parameters.

        This method:
        - explicitly calls `Module.__init__` (dataclasses do not automatically),
        - initializes weights (and optional biases) with a uniform distribution
          U(-k, k) where k = 1/sqrt(hidden_size),
        - stores parameters on CPU (current backend support).
        """
        # CRITICAL: dataclass does NOT call Module.__init__ automatically.
        # Without this, __setattr__ will fail when it tries to access _parameters/_modules.
        Module.__init__(self)

        k = 1.0 / np.sqrt(self.hidden_size)
        device = Device("cpu")  # must be a Device instance

        # Weight: input -> hidden (D, H)
        Wih_np = np.random.uniform(
            -k, k, size=(self.input_size, self.hidden_size)
        ).astype(np.float32)
        self.W_ih = Parameter(shape=Wih_np.shape, device=device, requires_grad=True)
        self.W_ih.copy_from_numpy(Wih_np)

        # Weight: hidden -> hidden (H, H)
        Whh_np = np.random.uniform(
            -k, k, size=(self.hidden_size, self.hidden_size)
        ).astype(np.float32)
        self.W_hh = Parameter(shape=Whh_np.shape, device=device, requires_grad=True)
        self.W_hh.copy_from_numpy(Whh_np)

        if self.bias:
            # Biases (H,)
            bih_np = np.random.uniform(-k, k, size=(self.hidden_size,)).astype(
                np.float32
            )
            self.b_ih = Parameter(shape=bih_np.shape, device=device, requires_grad=True)
            self.b_ih.copy_from_numpy(bih_np)

            bhh_np = np.random.uniform(-k, k, size=(self.hidden_size,)).astype(
                np.float32
            )
            self.b_hh = Parameter(shape=bhh_np.shape, device=device, requires_grad=True)
            self.b_hh.copy_from_numpy(bhh_np)
        else:
            # This will also avoid registering bias params in Module.__setattr__
            self.b_ih = None
            self.b_hh = None

    def parameters(self):
        """
        Return parameters owned by this cell.

        Notes
        -----
        `Module.parameters()` already exists and is recursive. This method keeps
        explicit control over ordering and inclusion (e.g., bias toggles).

        Returns
        -------
        list[Parameter]
            A list containing W_ih, W_hh and (optionally) b_ih, b_hh.
        """
        params = [self.W_ih, self.W_hh]
        if self.bias:
            params += [self.b_ih, self.b_hh]
        return params

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute the next hidden state for one timestep.

        Parameters
        ----------
        x_t : Tensor
            Input at timestep t, shape (N, D).
        h_prev : Tensor
            Previous hidden state h_{t-1}, shape (N, H).

        Returns
        -------
        Tensor
            Next hidden state h_t, shape (N, H).

        Autograd
        --------
        If any input/parameter requires gradients, the output tensor receives a
        `Context` that produces gradients for:
        (x_t, h_prev, W_ih, W_hh[, b_ih, b_hh]).

        Notes
        -----
        Biases are expanded explicitly to shape (N, H) using NumPy repeats to
        avoid broadcasting in Tensor ops.
        """
        x = x_t.to_numpy()  # (N, D)
        h = h_prev.to_numpy()  # (N, H)

        Wih = self.W_ih.to_numpy()  # (D, H)
        Whh = self.W_hh.to_numpy()  # (H, H)

        a = x @ Wih + h @ Whh  # (N, H)

        if self.bias:
            # Avoid broadcasting in Tensor ops by expanding explicitly in NumPy
            N = x.shape[0]
            bih = self.b_ih.to_numpy().reshape(1, -1)  # (1, H)
            bhh = self.b_hh.to_numpy().reshape(1, -1)  # (1, H)
            a = a + np.repeat(bih, N, axis=0) + np.repeat(bhh, N, axis=0)

        h_t_np = np.tanh(a).astype(np.float32)  # (N, H)

        req = Tensor._result_requires_grad(
            x_t,
            h_prev,
            self.W_ih,
            self.W_hh,
            *([self.b_ih, self.b_hh] if self.bias else []),
        )

        out = Tensor(shape=h_t_np.shape, device=x_t.device, requires_grad=req)
        out.copy_from_numpy(h_t_np)

        if req:
            parents = (x_t, h_prev, self.W_ih, self.W_hh)
            if self.bias:
                parents = parents + (self.b_ih, self.b_hh)

            ctx = Context(
                parents=parents,
                backward_fn=lambda grad_out: self._backward(ctx, grad_out),
            )

            # Save tensors needed for backward: x, h_prev, and h_t (for tanh').
            ctx.save_for_backward(x_t, h_prev, out)
            out._set_ctx(ctx)

        return out

    def _backward(self, ctx: Context, grad_out: Tensor):
        """
        Backward pass for a single timestep.

        Parameters
        ----------
        ctx : Context
            Context containing saved tensors from forward.
            Expected: (x_t, h_prev, h_t).
        grad_out : Tensor
            Upstream gradient dL/dh_t, shape (N, H).

        Returns
        -------
        tuple[Optional[Tensor], ...]
            Gradients aligned with `ctx.parents` order:
            (dL/dx_t, dL/dh_prev, dL/dW_ih, dL/dW_hh[, dL/db_ih, dL/db_hh]).

        Notes
        -----
        Uses the identity:
            d(tanh(a))/da = 1 - tanh(a)^2 = 1 - h_t^2
        """
        x_t, h_prev, h_t = ctx.saved_tensors

        x = x_t.to_numpy()  # (N, D)
        h = h_prev.to_numpy()  # (N, H)
        ht = h_t.to_numpy()  # (N, H)
        gh = grad_out.to_numpy()  # (N, H)

        # dL/da
        ga = gh * (1.0 - ht * ht)  # (N, H)

        Wih = self.W_ih.to_numpy()  # (D, H)
        Whh = self.W_hh.to_numpy()  # (H, H)

        grad_x_np = ga @ Wih.T  # (N, D)
        grad_h_prev_np = ga @ Whh.T  # (N, H)

        grad_Wih_np = x.T @ ga  # (D, H)
        grad_Whh_np = h.T @ ga  # (H, H)

        gx = (
            tensor_from_numpy(grad_x_np, device=x_t.device, requires_grad=False)
            if x_t.requires_grad
            else None
        )
        ghp = (
            tensor_from_numpy(grad_h_prev_np, device=h_prev.device, requires_grad=False)
            if h_prev.requires_grad
            else None
        )

        gWih = (
            tensor_from_numpy(grad_Wih_np, device=self.W_ih.device, requires_grad=False)
            if self.W_ih.requires_grad
            else None
        )
        gWhh = (
            tensor_from_numpy(grad_Whh_np, device=self.W_hh.device, requires_grad=False)
            if self.W_hh.requires_grad
            else None
        )

        if self.bias:
            grad_bih_np = ga.sum(axis=0).astype(np.float32)  # (H,)
            grad_bhh_np = ga.sum(axis=0).astype(np.float32)  # (H,)

            gbih = (
                tensor_from_numpy(
                    grad_bih_np, device=self.b_ih.device, requires_grad=False
                )
                if self.b_ih.requires_grad
                else None
            )
            gbhh = (
                tensor_from_numpy(
                    grad_bhh_np, device=self.b_hh.device, requires_grad=False
                )
                if self.b_hh.requires_grad
                else None
            )
            return (gx, ghp, gWih, gWhh, gbih, gbhh)

        return (gx, ghp, gWih, gWhh)


class RNN(Module):
    """
    Unidirectional vanilla RNN over a time-major sequence using `RNNCell`.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        Whether to use biases in the underlying cell. Defaults to True.

    Input / Output
    --------------
    Input:
        x : Tensor of shape (T, N, D)  (time-major)
        h0 : Optional[Tensor] of shape (N, H)
    Output:
        h_seq : Tensor of shape (T, N, H)
        h_T : Tensor of shape (N, H)

    Notes
    -----
    - Uses an explicit Python loop over timesteps for clarity.
    - The current implementation constructs per-timestep tensors from NumPy
      slices; without a view/slice Tensor implementation, gradients will not
      automatically accumulate back into the original `x` tensor.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Create an RNN module that owns a single `RNNCell`.

        Parameters
        ----------
        input_size : int
            Input feature dimension D.
        hidden_size : int
            Hidden state dimension H.
        bias : bool, optional
            Whether to use biases in the cell. Defaults to True.
        """
        super().__init__()  # ensure _parameters/_modules exist
        self.cell = RNNCell(input_size, hidden_size, bias=bias)

    def forward(self, x: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Run the RNN over a full sequence (time-major).

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (T, N, D).
        h0 : Optional[Tensor], optional
            Initial hidden state of shape (N, H). If None, initializes to zeros.
            Defaults to None.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - h_seq: all hidden states, shape (T, N, H)
            - h_T: final hidden state, shape (N, H)
        """
        x_np = x.to_numpy()
        T, N, D = x_np.shape

        if h0 is None:
            h_prev = tensor_from_numpy(
                np.zeros((N, self.cell.hidden_size), dtype=np.float32),
                device=x.device,
                requires_grad=False,
            )
        else:
            h_prev = h0

        hs = []
        for t in range(T):
            # if x requires grad, each x_t should require grad too
            x_t = tensor_from_numpy(
                x_np[t], device=x.device, requires_grad=x.requires_grad
            )
            h_prev = self.cell.forward(x_t, h_prev)
            hs.append(h_prev.to_numpy())

        h_seq = tensor_from_numpy(
            np.stack(hs, axis=0), device=x.device, requires_grad=False
        )
        return h_seq, h_prev
