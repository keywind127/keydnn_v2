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
from typing import Optional, Any, Dict

import numpy as np

from ...domain.device._device import Device
from .._tensor import Tensor, Context
from .._parameter import Parameter
from .._module import Module
from ..module._serialization_core import register_module


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


@register_module()
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

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this cell.

        Notes
        -----
        This captures constructor hyperparameters only.
        Trainable parameters (W_ih, W_hh, b_ih, b_hh) are serialized separately
        by the checkpoint/state_dict mechanism.
        """
        return {
            "input_size": int(self.input_size),
            "hidden_size": int(self.hidden_size),
            "bias": bool(self.bias),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RNNCell":
        """
        Reconstruct a cell from config.

        Notes
        -----
        We intentionally do not load weights here; weight loading is handled by
        `load_state_payload_` after the module graph is instantiated.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
        )


@register_module()
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
    return_sequences : bool, optional
        If True, include the full hidden-state sequence `h_seq` (T, N, H) in the output.
        Defaults to True for backward compatibility (historically `forward()` returned h_seq).
    return_state : bool, optional
        If True, also include the final hidden state `h_T` (N, H) in the output.
        Defaults to True for backward compatibility (historically `forward()` returned (h_seq, h_T)).
    keras_compat : bool, optional
        If True, use Keras-like return behavior:
          - return_sequences=False, return_state=False -> h_T
          - return_sequences=True,  return_state=False -> h_seq
          - return_sequences=False, return_state=True  -> (h_T, h_T)
          - return_sequences=True,  return_state=True  -> (h_seq, h_T)
        If False (default), preserve legacy behavior where `forward()` returns:
          - (h_seq, h_T) when return_state=True (default)
          - h_seq when return_state=False
        Defaults to False.

    Input / Output
    --------------
    Input:
        x : Tensor of shape (T, N, D)  (time-major)
        h0 : Optional[Tensor] of shape (N, H)
    Output (default legacy behavior):
        h_seq : Tensor of shape (T, N, H)
        h_T : Tensor of shape (N, H)

    Notes
    -----
    - Uses an explicit Python loop over timesteps for clarity.
    - The current implementation uses Tensor slicing (`x[t]`) and `Tensor.stack`,
      so BPTT emerges naturally from the autograd graph (no fused RNN backward).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        return_sequences: bool = True,  # <-- legacy default
        return_state: bool = True,  # <-- legacy default
        keras_compat: bool = False,  # <-- opt-in
    ):
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
        return_sequences : bool, optional
            If True, return the full hidden-state sequence `h_seq` of shape (T, N, H).
            If False, return only the final hidden state `h_T` of shape (N, H).
            Defaults to True.
        return_state : bool, optional
            If True, also return the final hidden state `h_T` as an additional output.
            Defaults to True.
        """
        super().__init__()  # ensure _parameters/_modules exist
        self.cell = RNNCell(input_size, hidden_size, bias=bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)
        self.keras_compat = bool(keras_compat)

    def forward(self, x: Tensor, h0: Optional[Tensor] = None):
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
        Tensor | Tuple[Tensor, Tensor]
            Output determined by `return_sequences` and `return_state`:

            - return_sequences=False, return_state=False:
                out : Tensor, final hidden state `h_T` of shape (N, H)

            - return_sequences=True, return_state=False:
                out : Tensor, full sequence `h_seq` of shape (T, N, H)

            - return_sequences=False, return_state=True:
                (out, h_T) : Tuple[Tensor, Tensor], both are `h_T` of shape (N, H)

            - return_sequences=True, return_state=True:
                (out, h_T) : Tuple[Tensor, Tensor], where out is `h_seq` of shape (T, N, H)
                and h_T is final hidden state of shape (N, H)
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
            x_t = x[t]  # uses __getitem__ (keeps grad path to x)
            h_prev = self.cell.forward(x_t, h_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)  # (T, N, H)
        h_T = h_prev  # (N, H)

        # --- Legacy behavior (default): keep tests working ---
        if not self.keras_compat:
            if self.return_state:
                # default legacy: (h_seq, h_T)
                if self.return_sequences:
                    return h_seq, h_T
                # rarely used: (h_T,) isn't great; return h_T to keep sane behavior
                return h_T
            # return_state=False: historically some code might expect h_seq only
            return h_seq if self.return_sequences else h_T

        # --- Keras-like behavior (opt-in) ---
        out = h_seq if self.return_sequences else h_T
        if self.return_state:
            return out, h_T
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this RNN.

        Notes
        -----
        This captures constructor hyperparameters only. The child `cell`
        is a submodule and will be serialized structurally by `module_to_config`.
        Trainable parameters are serialized separately by the weights payload.
        """
        return {
            # architectural hyperparameters
            "input_size": int(self.cell.input_size),
            "hidden_size": int(self.cell.hidden_size),
            "bias": bool(self.cell.bias),
            # output behavior flags
            "return_sequences": bool(self.return_sequences),
            "return_state": bool(self.return_state),
            "keras_compat": bool(self.keras_compat),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RNN":
        """
        Reconstruct an RNN from config.

        Notes
        -----
        - We construct RNN with the same hyperparameters.
        - The deserializer will attach child modules into `_modules` afterward.
          Since `RNN.__init__` creates `self.cell` already, the serializer should
          remain consistent: either rely on this default cell or overwrite it
          via `_modules["cell"]` depending on your core deserialization design.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", True)),
            keras_compat=bool(cfg.get("keras_compat", False)),
        )
