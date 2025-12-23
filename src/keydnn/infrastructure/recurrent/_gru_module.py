"""
Gated Recurrent Unit (GRU) layers for KeyDNN.

This module provides a minimal, CPU-only implementation of a vanilla GRU,
including both a single-timestep cell (`GRUCell`) and a unidirectional
sequence layer (`GRU`) that applies the cell over a time-major input.

Implemented components
----------------------
- `tensor_from_numpy`:
    Utility to construct a KeyDNN `Tensor` from a NumPy array.
- `_sigmoid`:
    Numerically stable sigmoid used for gate activations.
- `GRUCell`:
    Computes one GRU step using packed parameters and a manual backward.
- `GRU`:
    Applies `GRUCell` over a time-major sequence input of shape (T, N, D).

GRU equations (per timestep)
----------------------------
Given x_t (N, D) and h_{t-1} (N, H):

    a_z = x_t W_iz + h_{t-1} W_hz + b_z
    a_r = x_t W_ir + h_{t-1} W_hr + b_r
    z   = sigmoid(a_z)
    r   = sigmoid(a_r)

    a_n = x_t W_in + (r * h_{t-1}) W_hn + b_n
    n   = tanh(a_n)

    h_t = (1 - z) * n + z * h_{t-1}

Parameter packing
-----------------
Weights and biases are stored in packed form, similar to the LSTM module:

    W_ih : (D, 3H)  -> [W_iz | W_ir | W_in]
    W_hh : (H, 3H)  -> [W_hz | W_hr | W_hn]
    b_ih : (3H,) optional
    b_hh : (3H,) optional

Biases are expanded explicitly in NumPy to avoid relying on Tensor broadcasting
semantics (KeyDNN enforces strict shape equality for elementwise operations).

Autograd notes
--------------
- `GRUCell` constructs an explicit `Context` and provides a manual backward.
- `GRU` builds a BPTT graph by indexing x[t] (preserving gradient connectivity)
  and stacking per-timestep hidden states via `Tensor.stack`.

Notes
-----
- This implementation targets correctness and readability (NumPy on CPU),
  not performance.
- Device handling is CPU-only; parameters are created on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ...domain.device._device import Device
from .._module import Module
from .._parameter import Parameter
from .._tensor import Context, Tensor
from ..module._serialization_core import register_module


def tensor_from_numpy(
    arr: np.ndarray, *, device: Device, requires_grad: bool = False
) -> Tensor:
    """
    Construct a KeyDNN Tensor from a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array to copy into tensor storage. The array is converted to
        float32.
    device : Device
        Target device for the constructed tensor.
    requires_grad : bool, optional
        Whether the resulting tensor should participate in autograd.

    Returns
    -------
    Tensor
        A KeyDNN Tensor with the same shape as `arr`, stored on `device`,
        containing a float32 copy of the input data.
    """
    arr = np.asarray(arr, dtype=np.float32)
    out = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad)
    out.copy_from_numpy(arr)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute a numerically stable sigmoid on a NumPy array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output array with sigmoid applied elementwise (float32).

    Notes
    -----
    This implementation avoids overflow by treating positive and negative
    values separately.
    """
    # numerically stable sigmoid
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    ex = np.exp(x[neg], dtype=np.float32)
    out[neg] = ex / (1.0 + ex)
    return out


@register_module()
@dataclass
class GRUCell(Module):
    """
    Vanilla GRU cell operating on a single timestep.

    This cell computes the next hidden state h_t from the current input x_t
    and previous hidden state h_{t-1} using update/reset gates.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        If True, includes packed bias parameters b_ih and b_hh (both shape (3H,)).
        Default is True.

    Attributes
    ----------
    W_ih : Parameter
        Packed input-to-hidden weight matrix of shape (D, 3H).
    W_hh : Parameter
        Packed hidden-to-hidden weight matrix of shape (H, 3H).
    b_ih : Optional[Parameter]
        Packed input bias of shape (3H,) if `bias=True`, else None.
    b_hh : Optional[Parameter]
        Packed hidden bias of shape (3H,) if `bias=True`, else None.

    Input / Output shapes
    ---------------------
    x_t    : (N, D)
    h_prev : (N, H)
    h_t    : (N, H)

    Autograd notes
    -------------
    - A manual backward is implemented in `_backward` and connected through `Context`.
    - Since this is a dataclass, `Module.__init__()` is not invoked automatically;
      it is called explicitly in `__post_init__` to enable parameter auto-registration.
    """

    input_size: int
    hidden_size: int
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Dataclass post-initialization hook.

        Initializes the Module base class (required for parameter/module
        auto-registration) and allocates packed GRU parameters on CPU with
        small uniform random initialization.
        """
        # dataclass does not call Module.__init__
        Module.__init__(self)

        k = 1.0 / np.sqrt(self.hidden_size)
        device = Device("cpu")

        # Packed weights
        Wih = np.random.uniform(
            -k, k, size=(self.input_size, 3 * self.hidden_size)
        ).astype(np.float32)
        Whh = np.random.uniform(
            -k, k, size=(self.hidden_size, 3 * self.hidden_size)
        ).astype(np.float32)

        self.W_ih = Parameter(shape=Wih.shape, device=device, requires_grad=True)
        self.W_hh = Parameter(shape=Whh.shape, device=device, requires_grad=True)
        self.W_ih.copy_from_numpy(Wih)
        self.W_hh.copy_from_numpy(Whh)

        if self.bias:
            bih = np.random.uniform(-k, k, size=(3 * self.hidden_size,)).astype(
                np.float32
            )
            bhh = np.random.uniform(-k, k, size=(3 * self.hidden_size,)).astype(
                np.float32
            )
            self.b_ih = Parameter(shape=bih.shape, device=device, requires_grad=True)
            self.b_hh = Parameter(shape=bhh.shape, device=device, requires_grad=True)
            self.b_ih.copy_from_numpy(bih)
            self.b_hh.copy_from_numpy(bhh)
        else:
            self.b_ih = None
            self.b_hh = None

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute the next hidden state for one GRU timestep.

        Parameters
        ----------
        x_t : Tensor
            Input at the current timestep, shape (N, D).
        h_prev : Tensor
            Previous hidden state, shape (N, H).

        Returns
        -------
        Tensor
            Next hidden state h_t, shape (N, H).

        Notes
        -----
        - Parameters are stored in packed form (3H columns) and split internally.
        - Biases are expanded in NumPy to match batch shape, avoiding reliance on
          Tensor broadcasting.
        - If autograd is required, this method attaches a `Context` that routes
          gradients through `_backward`.
        """
        """
        Forward pass for one timestep.

        x_t : (N, D)
        h_prev : (N, H)
        returns h_t : (N, H)
        """
        x = x_t.to_numpy().astype(np.float32, copy=False)  # (N, D)
        h = h_prev.to_numpy().astype(np.float32, copy=False)  # (N, H)

        Wih = self.W_ih.to_numpy().astype(np.float32, copy=False)  # (D, 3H)
        Whh = self.W_hh.to_numpy().astype(np.float32, copy=False)  # (H, 3H)

        a = x @ Wih + h @ Whh  # (N, 3H)

        if self.bias:
            N = x.shape[0]
            bih = self.b_ih.to_numpy().reshape(1, -1)  # (1, 3H)
            bhh = self.b_hh.to_numpy().reshape(1, -1)  # (1, 3H)
            a = a + np.repeat(bih, N, axis=0) + np.repeat(bhh, N, axis=0)

        H = self.hidden_size
        a_z = a[:, 0:H]
        a_r = a[:, H : 2 * H]
        a_n_base = a[
            :, 2 * H : 3 * H
        ]  # this is xWin + hWhn + biases; we'll adjust with r*h

        # But for GRU, candidate uses (r*h) @ Whn, not h @ Whn.
        # Since our packed 'a' used h @ Whh, we cannot reuse a_n_base directly.
        # So compute gates separately:
        # Split packed weights.
        W_iz = Wih[:, 0:H]
        W_ir = Wih[:, H : 2 * H]
        W_in = Wih[:, 2 * H : 3 * H]

        W_hz = Whh[:, 0:H]
        W_hr = Whh[:, H : 2 * H]
        W_hn = Whh[:, 2 * H : 3 * H]

        # Bias split
        if self.bias:
            b_ih = self.b_ih.to_numpy().astype(np.float32, copy=False)
            b_hh = self.b_hh.to_numpy().astype(np.float32, copy=False)
            b_z = (b_ih[0:H] + b_hh[0:H]).reshape(1, -1)
            b_r = (b_ih[H : 2 * H] + b_hh[H : 2 * H]).reshape(1, -1)
            b_n = (b_ih[2 * H : 3 * H] + b_hh[2 * H : 3 * H]).reshape(1, -1)
        else:
            b_z = None
            b_r = None
            b_n = None

        a_z = x @ W_iz + h @ W_hz
        a_r = x @ W_ir + h @ W_hr
        if b_z is not None:
            a_z = a_z + np.repeat(b_z, x.shape[0], axis=0)
            a_r = a_r + np.repeat(b_r, x.shape[0], axis=0)

        z = _sigmoid(a_z)  # (N, H)
        r = _sigmoid(a_r)  # (N, H)

        rh = r * h  # (N, H)
        a_n = x @ W_in + rh @ W_hn
        if b_n is not None:
            a_n = a_n + np.repeat(b_n, x.shape[0], axis=0)

        n = np.tanh(a_n).astype(np.float32)  # (N, H)

        h_t_np = ((1.0 - z) * n + z * h).astype(np.float32)  # (N, H)

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
            # Save tensors needed for backward:
            # x_t, h_prev, z, r, n, plus raw weights splits
            z_t = tensor_from_numpy(z, device=x_t.device, requires_grad=False)
            r_t = tensor_from_numpy(r, device=x_t.device, requires_grad=False)
            n_t = tensor_from_numpy(n, device=x_t.device, requires_grad=False)
            ctx.save_for_backward(x_t, h_prev, z_t, r_t, n_t)
            out._set_ctx(ctx)

        return out

    def _backward(self, ctx: Context, grad_out: Tensor):
        """
        Backward pass for one GRU timestep.

        Computes gradients with respect to inputs, previous hidden state,
        and packed parameters using the stored forward intermediates.

        Parameters
        ----------
        ctx : Context
            Autograd context containing saved tensors and parent references.
        grad_out : Tensor
            Upstream gradient with respect to h_t, shape (N, H).

        Returns
        -------
        tuple
            Gradients aligned with `ctx.parents`:
            (dx_t, dh_prev, dW_ih, dW_hh[, db_ih, db_hh])
            where entries may be None when the corresponding parent does not
            require gradients.

        Notes
        -----
        - Bias gradients are computed for the combined per-gate bias
          b_gate = b_ih_gate + b_hh_gate and then mirrored to both b_ih and b_hh,
          matching the forward bias composition used in this implementation.
        """
        """
        Returns grads aligned with ctx.parents:
          (dx, dh_prev, dW_ih, dW_hh[, db_ih, db_hh])
        """
        x_t, h_prev, z_t, r_t, n_t = ctx.saved_tensors

        x = x_t.to_numpy().astype(np.float32, copy=False)  # (N, D)
        h = h_prev.to_numpy().astype(np.float32, copy=False)  # (N, H)
        z = z_t.to_numpy().astype(np.float32, copy=False)  # (N, H)
        r = r_t.to_numpy().astype(np.float32, copy=False)  # (N, H)
        n = n_t.to_numpy().astype(np.float32, copy=False)  # (N, H)
        gh = grad_out.to_numpy().astype(np.float32, copy=False)  # (N, H)

        N, D = x.shape
        H = self.hidden_size

        Wih = self.W_ih.to_numpy().astype(np.float32, copy=False)  # (D, 3H)
        Whh = self.W_hh.to_numpy().astype(np.float32, copy=False)  # (H, 3H)

        # Split weights
        W_iz = Wih[:, 0:H]
        W_ir = Wih[:, H : 2 * H]
        W_in = Wih[:, 2 * H : 3 * H]

        W_hz = Whh[:, 0:H]
        W_hr = Whh[:, H : 2 * H]
        W_hn = Whh[:, 2 * H : 3 * H]

        # h_t = n + z*(h - n)
        grad_z = gh * (h - n)  # (N, H)
        grad_n = gh * (1.0 - z)  # (N, H)
        grad_h = gh * z  # (N, H)  (direct path)

        # n = tanh(a_n)
        grad_a_n = grad_n * (1.0 - n * n)  # (N, H)

        # a_n = x @ W_in + (r*h) @ W_hn + b_n
        rh = r * h
        grad_x = grad_a_n @ W_in.T  # (N, D)
        grad_W_in = x.T @ grad_a_n  # (D, H)
        grad_W_hn = rh.T @ grad_a_n  # (H, H)
        grad_rh = grad_a_n @ W_hn.T  # (N, H)

        # rh = r*h
        grad_r = grad_rh * h  # (N, H)
        grad_h += grad_rh * r  # add h path through rh

        # r = sigmoid(a_r)
        grad_a_r = grad_r * (r * (1.0 - r))  # (N, H)

        grad_x += grad_a_r @ W_ir.T
        grad_h += grad_a_r @ W_hr.T
        grad_W_ir = x.T @ grad_a_r
        grad_W_hr = h.T @ grad_a_r

        # z = sigmoid(a_z)
        grad_a_z = grad_z * (z * (1.0 - z))  # (N, H)

        grad_x += grad_a_z @ W_iz.T
        grad_h += grad_a_z @ W_hz.T
        grad_W_iz = x.T @ grad_a_z
        grad_W_hz = h.T @ grad_a_z

        # Pack weight grads to (D, 3H) and (H, 3H)
        grad_Wih = np.concatenate([grad_W_iz, grad_W_ir, grad_W_in], axis=1).astype(
            np.float32
        )
        grad_Whh = np.concatenate([grad_W_hz, grad_W_hr, grad_W_hn], axis=1).astype(
            np.float32
        )

        gx = (
            tensor_from_numpy(grad_x, device=x_t.device, requires_grad=False)
            if x_t.requires_grad
            else None
        )
        gh_prev = (
            tensor_from_numpy(grad_h, device=h_prev.device, requires_grad=False)
            if h_prev.requires_grad
            else None
        )
        gWih = (
            tensor_from_numpy(grad_Wih, device=self.W_ih.device, requires_grad=False)
            if self.W_ih.requires_grad
            else None
        )
        gWhh = (
            tensor_from_numpy(grad_Whh, device=self.W_hh.device, requires_grad=False)
            if self.W_hh.requires_grad
            else None
        )

        if self.bias:
            # For our split-bias design we treat b_ih and b_hh as combined per-gate in forward:
            # b_gate = b_ih_gate + b_hh_gate, so gradient should be split identically to both.
            db_z = grad_a_z.sum(axis=0).astype(np.float32)
            db_r = grad_a_r.sum(axis=0).astype(np.float32)
            db_n = grad_a_n.sum(axis=0).astype(np.float32)
            db = np.concatenate([db_z, db_r, db_n], axis=0).astype(np.float32)

            gbih = (
                tensor_from_numpy(db, device=self.b_ih.device, requires_grad=False)
                if self.b_ih.requires_grad
                else None
            )
            gbhh = (
                tensor_from_numpy(db, device=self.b_hh.device, requires_grad=False)
                if self.b_hh.requires_grad
                else None
            )
            return (gx, gh_prev, gWih, gWhh, gbih, gbhh)

        return (gx, gh_prev, gWih, gWhh)

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this GRUCell.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing constructor arguments.
        """
        return {
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "bias": bool(self.bias),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GRUCell":
        """
        Construct a GRUCell from a configuration dictionary.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        GRUCell
            Reconstructed GRUCell instance.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
        )


@register_module()
class GRU(Module):
    """
    Unidirectional GRU layer over a time-major sequence.

    This module applies `GRUCell` over a sequence input of shape (T, N, D),
    producing either the full sequence of hidden states or only the final
    hidden state depending on configuration.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        Whether the underlying cell uses biases.
    return_sequences : bool, optional
        If True, return the full output sequence (T, N, H).
        If False, return only the final output (N, H).
    return_state : bool, optional
        If True, include the final hidden state in the return structure.
    keras_compat : bool, optional
        If True, return values follow a Keras-like convention:
        - If return_state: (out, h_T)
        - Else: out
        If False, legacy-like behavior is used (matching KeyDNN RNN/LSTM).

    Input
    -----
    x : Tensor
        Time-major input of shape (T, N, D).
    h0 : Optional[Tensor]
        Initial hidden state of shape (N, H). If None, initialized to zeros.

    Notes
    -----
    The BPTT graph is constructed by:
    - indexing `x[t]` to preserve autograd connectivity to the input, and
    - stacking timestep outputs via `Tensor.stack`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        return_sequences: bool = True,
        return_state: bool = True,
        keras_compat: bool = False,
    ):
        """
        Initialize a GRU layer.

        Parameters
        ----------
        input_size : int
            Input feature dimension D.
        hidden_size : int
            Hidden state dimension H.
        bias : bool, optional
            Whether the cell includes bias parameters.
        return_sequences : bool, optional
            Whether to return the full sequence output.
        return_state : bool, optional
            Whether to return the final hidden state.
        keras_compat : bool, optional
            Whether to use Keras-like return conventions.
        """
        super().__init__()
        self.cell = GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)
        self.keras_compat = bool(keras_compat)

    def forward(self, x: Tensor, h0: Optional[Tensor] = None):
        """
        Run the GRU over a time-major input sequence.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (T, N, D).
        h0 : Optional[Tensor], optional
            Initial hidden state of shape (N, H). If None, uses zeros.

        Returns
        -------
        Any
            Return structure depends on `keras_compat`, `return_sequences`,
            and `return_state` as implemented in this method.

        Raises
        ------
        ValueError
            If the input tensor does not have rank 3 (T, N, D).
        """
        if len(x.shape) != 3:
            raise ValueError(f"GRU expects x shape (T,N,D), got {x.shape}")

        x_np = x.to_numpy()
        T, N, _ = x_np.shape

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
            x_t = x[t]
            h_prev = self.cell.forward(x_t, h_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)  # (T,N,H)
        h_T = h_prev  # (N,H)

        # legacy-like behavior (match your RNN/LSTM pattern)
        if not self.keras_compat:
            if self.return_state:
                if self.return_sequences:
                    return h_seq, h_T
                return h_T
            return h_seq if self.return_sequences else h_T

        # Keras-like behavior
        out = h_seq if self.return_sequences else h_T
        if self.return_state:
            return out, h_T
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this GRU layer.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing constructor arguments and
            behavioral flags.
        """
        return {
            "input_size": int(self.cell.input_size),
            "hidden_size": int(self.cell.hidden_size),
            "bias": bool(self.cell.bias),
            "return_sequences": bool(self.return_sequences),
            "return_state": bool(self.return_state),
            "keras_compat": bool(self.keras_compat),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GRU":
        """
        Construct a GRU layer from a configuration dictionary.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        GRU
            Reconstructed GRU instance.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", True)),
            keras_compat=bool(cfg.get("keras_compat", False)),
        )
