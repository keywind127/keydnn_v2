"""
Long Short-Term Memory (LSTM) layers for KeyDNN.

This module provides a minimal, CPU-only implementation of a vanilla LSTM,
including both a single-timestep cell (`LSTMCell`) and a unidirectional
sequence layer (`LSTM`) that applies the cell over a time-major input.

Implemented components
----------------------
- `tensor_from_numpy`:
    Utility to construct a KeyDNN `Tensor` from a NumPy array.
- `LSTMCell`:
    Computes one LSTM step:
        gates = x_t W_ih + h_{t-1} W_hh + b_ih + b_hh
        [i, f, g, o] = [sigmoid, sigmoid, tanh, sigmoid](gates)
        c_t = f * c_{t-1} + i * g
        h_t = o * tanh(c_t)
- `LSTM`:
    Applies `LSTMCell` over a sequence input of shape (T, N, D).

Design notes
------------
- CPU-only reference implementation intended for correctness and clarity.
- Integrates with KeyDNN autograd via `Context` and `Tensor.backward()`.
- KeyDNN elementwise ops enforce strict shape equality; therefore biases are
  explicitly expanded to (N, 4H) without relying on broadcasting.
- Backpropagation through time (BPTT) is obtained naturally through the
  autograd graph: each timestep produces tensors depending on x[t],
  previous states, and parameters.

Serialization
-------------
Both `LSTMCell` and `LSTM` are registered via `register_module` and support
`get_config()` and `from_config()` for persistence.

Public API
----------
- `tensor_from_numpy`
- `LSTMCell`
- `LSTM`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

from ...domain.device._device import Device
from .._tensor import Tensor, Context
from .._parameter import Parameter
from .._module import Module
from ..module._serialization_core import register_module


def tensor_from_numpy(
    arr: Any, *, device: Device, requires_grad: bool = False
) -> Tensor:
    """
    Construct a KeyDNN Tensor from a NumPy array.

    Notes
    -----
    This helper remains for compatibility/testing. Core LSTM math does not
    depend on NumPy.
    """
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    out = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    out.copy_from_numpy(arr)
    return out


def _tanh(x: Tensor) -> Tensor:
    """
    Elementwise tanh using the framework's tanh op / Function.
    """
    return x.tanh()  # Use existing tanh op/function if available


def _sigmoid(x: Tensor) -> Tensor:
    """
    Elementwise sigmoid using the framework's sigmoid op / Function.

    If you don't have SigmoidFn / sigmoid op, implement via exp:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    # Preferred: existing sigmoid op/function
    try:
        return x.sigmoid()
    except Exception:
        # Fallback: exp-based if you have Tensor.exp()
        one = Tensor.full(x.shape, 1.0, device=x.device, requires_grad=False)
        return one / (one + (-x).exp())


def _expand_bias(b: Tensor, N: int) -> Tensor:
    """
    Expand a (4H,) bias vector into shape (N, 4H) without broadcasting.

    Uses: ones(N,1) @ b(1,4H)
    """
    if len(b.shape) != 1:
        raise ValueError(f"Bias must be 1D (4H,), got shape {b.shape}")

    ones = Tensor.full((N, 1), 1.0, device=b.device, requires_grad=False)
    b_row = b.reshape((1, b.shape[0]))  # (1, 4H)
    return ones @ b_row  # (N, 4H)


@register_module()
@dataclass
class LSTMCell(Module):
    """
    Vanilla LSTM cell operating on a single timestep.

    (Docstring unchanged except "NumPy-based gate activation/backward" is no longer true;
    the cell now uses Tensor ops to build an autograd graph directly.)
    """

    input_size: int
    hidden_size: int
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Dataclass post-initialization hook.

        Initializes Module base class and allocates parameters on CPU.
        """
        Module.__init__(self)

        H = int(self.hidden_size)
        D = int(self.input_size)
        device = Device("cpu")

        # Initialization:
        # Use Tensor.rand/full if you want to avoid NumPy here too.
        # If Parameter.copy_from(other_tensor) exists, this is fully NumPy-free.
        k = 1.0 / (H**0.5)

        W_ih = Tensor.rand((D, 4 * H), device=device, requires_grad=False)
        W_hh = Tensor.rand((H, 4 * H), device=device, requires_grad=False)
        W_ih = (W_ih * (2.0 * k)) - k
        W_hh = (W_hh * (2.0 * k)) - k

        self.W_ih = Parameter(shape=(D, 4 * H), device=device, requires_grad=True)
        self.W_hh = Parameter(shape=(H, 4 * H), device=device, requires_grad=True)

        # NOTE: this assumes you have copy_from(other_tensor). If not, see note below.
        self.W_ih.copy_from(W_ih)
        self.W_hh.copy_from(W_hh)

        if self.bias:
            b_ih = Tensor.rand((4 * H,), device=device, requires_grad=False)
            b_hh = Tensor.rand((4 * H,), device=device, requires_grad=False)
            b_ih = (b_ih * (2.0 * k)) - k
            b_hh = (b_hh * (2.0 * k)) - k

            self.b_ih = Parameter(shape=(4 * H,), device=device, requires_grad=True)
            self.b_hh = Parameter(shape=(4 * H,), device=device, requires_grad=True)
            self.b_ih.copy_from(b_ih)
            self.b_hh.copy_from(b_hh)
        else:
            self.b_ih = None
            self.b_hh = None

    def parameters(self):
        params = [self.W_ih, self.W_hh]
        if self.bias:
            params += [self.b_ih, self.b_hh]
        return params

    def forward(
        self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the next hidden and cell state for one timestep.

        (Behavior unchanged; implementation is now Tensor-op based.)
        """
        if not x_t.device.is_cpu():
            raise RuntimeError("LSTMCell is CPU-only for now.")
        if x_t.device != h_prev.device or x_t.device != c_prev.device:
            raise ValueError("x_t, h_prev, c_prev must be on the same device.")

        if len(x_t.shape) != 2:
            raise ValueError(f"x_t must be 2D (N,D), got {x_t.shape}")
        if len(h_prev.shape) != 2 or len(c_prev.shape) != 2:
            raise ValueError("h_prev and c_prev must be 2D (N,H)")

        N, D = x_t.shape
        _, H = h_prev.shape
        if H != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got H={H}")

        # gates = xW_ih + hW_hh + b_ih + b_hh
        gates = (x_t @ self.W_ih) + (h_prev @ self.W_hh)  # (N, 4H)

        if self.bias:
            gates = gates + _expand_bias(self.b_ih, N) + _expand_bias(self.b_hh, N)

        # Split gates (no concat needed)
        # ai, af, ag, ao: each (N, H)
        ai = gates[:, 0:H]
        af = gates[:, H : 2 * H]
        ag = gates[:, 2 * H : 3 * H]
        ao = gates[:, 3 * H : 4 * H]

        i = _sigmoid(ai)
        f = _sigmoid(af)
        g = _tanh(ag)
        o = _sigmoid(ao)

        c_t = (f * c_prev) + (i * g)  # (N, H)
        h_t = o * _tanh(c_t)  # (N, H)

        return h_t, c_t

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "bias": bool(self.bias),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LSTMCell":
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
        )


@register_module()
class LSTM(Module):
    """
    Unidirectional vanilla LSTM layer over a time-major sequence.

    (Docstring unchanged; implementation no longer uses NumPy for sequence shape.)
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
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size, bias=bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)
        self.keras_compat = bool(keras_compat)

    def forward(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
        c0: Optional[Tensor] = None,
    ):
        """
        Run the LSTM over a time-major input sequence.

        Notes
        -----
        - Avoids x.to_numpy(); uses x.shape.
        """
        if len(x.shape) != 3:
            raise ValueError(f"LSTM expects 3D input (T,N,D), got {x.shape}")

        T, N, _D = x.shape
        H = self.cell.hidden_size

        if h0 is None:
            h_prev = Tensor.full((N, H), 0.0, device=x.device, requires_grad=False)
        else:
            h_prev = h0

        if c0 is None:
            c_prev = Tensor.full((N, H), 0.0, device=x.device, requires_grad=False)
        else:
            c_prev = c0

        hs = []
        for t in range(T):
            x_t = x[t]  # keep autograd path through __getitem__
            h_prev, c_prev = self.cell.forward(x_t, h_prev, c_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)  # (T,N,H)
        h_T = h_prev
        c_T = c_prev

        if not self.keras_compat:
            if self.return_state:
                if self.return_sequences:
                    return h_seq, (h_T, c_T)
                return h_T, (h_T, c_T)
            return h_seq if self.return_sequences else h_T

        out = h_seq if self.return_sequences else h_T
        if self.return_state:
            return out, (h_T, c_T)
        return out

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_size": int(self.cell.input_size),
            "hidden_size": int(self.cell.hidden_size),
            "bias": bool(self.cell.bias),
            "return_sequences": bool(self.return_sequences),
            "return_state": bool(self.return_state),
            "keras_compat": bool(self.keras_compat),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LSTM":
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", True)),
            keras_compat=bool(cfg.get("keras_compat", False)),
        )
