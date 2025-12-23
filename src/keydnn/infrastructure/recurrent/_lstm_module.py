"""
Long Short-Term Memory (LSTM) layers for KeyDNN.

This module provides a minimal, CPU-only implementation of a vanilla LSTM,
including both a single-timestep cell (`LSTMCell`) and a unidirectional
sequence layer (`LSTM`) that applies the cell over a time-major input.

Implemented components
----------------------
- `tensor_from_numpy`:
    Utility to construct a KeyDNN `Tensor` from a NumPy array.
- `_sigmoid`:
    Numerically stable sigmoid used for gate activations.
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
- Bias broadcasting is performed explicitly using NumPy repeats because
  KeyDNN elementwise operations enforce strict shape equality.
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

    This implementation avoids overflow by treating positive and negative
    values separately.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Output array with sigmoid applied elementwise (float32).
    """
    # Stable sigmoid
    x = x.astype(np.float32, copy=False)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    expx = np.exp(x[neg], dtype=np.float32)
    out[neg] = expx / (1.0 + expx)
    return out


@register_module()
@dataclass
class LSTMCell(Module):
    """
    Vanilla LSTM cell operating on a single timestep.

    This cell computes the next hidden and cell states (h_t, c_t) from an
    input x_t and previous states (h_{t-1}, c_{t-1}) using the standard
    LSTM gate formulation.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        If True, includes bias parameters b_ih and b_hh (both shape (4H,)).
        Default is True.

    Attributes
    ----------
    W_ih : Parameter
        Input-to-hidden weight matrix of shape (D, 4H).
    W_hh : Parameter
        Hidden-to-hidden weight matrix of shape (H, 4H).
    b_ih : Optional[Parameter]
        Input bias vector of shape (4H,) if `bias=True`, else None.
    b_hh : Optional[Parameter]
        Hidden bias vector of shape (4H,) if `bias=True`, else None.

    Input / Output shapes
    ---------------------
    x_t    : (N, D)
    h_prev : (N, H)
    c_prev : (N, H)
    h_t    : (N, H)
    c_t    : (N, H)

    Autograd notes
    -------------
    - This module builds explicit `Context` objects for both outputs
      `h_t` and `c_t`. If both are used downstream, gradient contributions
      accumulate naturally.
    - Since this is a dataclass, `Module.__init__()` is not invoked
      automatically; it is called explicitly in `__post_init__`.
    """

    input_size: int
    hidden_size: int
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Dataclass post-initialization hook.

        Initializes the Module base class (required for parameter/module
        auto-registration) and allocates parameters on CPU with small
        uniform random initialization.
        """
        # CRITICAL: dataclass does NOT call Module.__init__ automatically.
        Module.__init__(self)

        H = int(self.hidden_size)
        D = int(self.input_size)
        k = 1.0 / np.sqrt(H)
        device = Device("cpu")

        Wih_np = np.random.uniform(-k, k, size=(D, 4 * H)).astype(np.float32)
        Whh_np = np.random.uniform(-k, k, size=(H, 4 * H)).astype(np.float32)

        self.W_ih = Parameter(shape=Wih_np.shape, device=device, requires_grad=True)
        self.W_ih.copy_from_numpy(Wih_np)

        self.W_hh = Parameter(shape=Whh_np.shape, device=device, requires_grad=True)
        self.W_hh.copy_from_numpy(Whh_np)

        if self.bias:
            bih_np = np.random.uniform(-k, k, size=(4 * H,)).astype(np.float32)
            bhh_np = np.random.uniform(-k, k, size=(4 * H,)).astype(np.float32)

            self.b_ih = Parameter(shape=bih_np.shape, device=device, requires_grad=True)
            self.b_ih.copy_from_numpy(bih_np)

            self.b_hh = Parameter(shape=bhh_np.shape, device=device, requires_grad=True)
            self.b_hh.copy_from_numpy(bhh_np)
        else:
            self.b_ih = None
            self.b_hh = None

    def parameters(self):
        """
        Return the parameters of this LSTM cell.

        Returns
        -------
        list
            List of Parameter objects including weights and (optionally) biases.
        """
        params = [self.W_ih, self.W_hh]
        if self.bias:
            params += [self.b_ih, self.b_hh]
        return params

    def forward(
        self, x_t: Tensor, h_prev: Tensor, c_prev: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute the next hidden and cell state for one timestep.

        Parameters
        ----------
        x_t : Tensor
            Input at the current timestep, shape (N, D).
        h_prev : Tensor
            Previous hidden state, shape (N, H).
        c_prev : Tensor
            Previous cell state, shape (N, H).

        Returns
        -------
        Tuple[Tensor, Tensor]
            (h_t, c_t) where both tensors have shape (N, H).

        Raises
        ------
        RuntimeError
            If the input tensor is not on CPU.
        ValueError
            If x_t, h_prev, and c_prev are not on the same device.
        """
        """
        Compute (h_t, c_t) for one timestep.
        """
        if not x_t.device.is_cpu():
            raise RuntimeError("LSTMCell is CPU-only for now.")
        if x_t.device != h_prev.device or x_t.device != c_prev.device:
            raise ValueError("x_t, h_prev, c_prev must be on the same device.")

        x = x_t.to_numpy()  # (N, D)
        h = h_prev.to_numpy()  # (N, H)
        c = c_prev.to_numpy()  # (N, H)

        Wih = self.W_ih.to_numpy()  # (D, 4H)
        Whh = self.W_hh.to_numpy()  # (H, 4H)

        gates = x @ Wih + h @ Whh  # (N, 4H)

        if self.bias:
            N = x.shape[0]
            bih = self.b_ih.to_numpy().reshape(1, -1)  # (1, 4H)
            bhh = self.b_hh.to_numpy().reshape(1, -1)  # (1, 4H)
            gates = gates + np.repeat(bih, N, axis=0) + np.repeat(bhh, N, axis=0)

        H = self.hidden_size
        ai = gates[:, 0:H]
        af = gates[:, H : 2 * H]
        ag = gates[:, 2 * H : 3 * H]
        ao = gates[:, 3 * H : 4 * H]

        i = _sigmoid(ai)
        f = _sigmoid(af)
        g = np.tanh(ag).astype(np.float32)
        o = _sigmoid(ao)

        c_t_np = (f * c + i * g).astype(np.float32)  # (N, H)
        tanh_c = np.tanh(c_t_np).astype(np.float32)  # (N, H)
        h_t_np = (o * tanh_c).astype(np.float32)  # (N, H)

        req = Tensor._result_requires_grad(
            x_t,
            h_prev,
            c_prev,
            self.W_ih,
            self.W_hh,
            *([self.b_ih, self.b_hh] if self.bias else []),
        )

        h_t = Tensor(shape=h_t_np.shape, device=x_t.device, requires_grad=req, ctx=None)
        h_t.copy_from_numpy(h_t_np)

        c_t = Tensor(shape=c_t_np.shape, device=x_t.device, requires_grad=req, ctx=None)
        c_t.copy_from_numpy(c_t_np)

        if req:
            parents = (x_t, h_prev, c_prev, self.W_ih, self.W_hh)
            if self.bias:
                parents = parents + (self.b_ih, self.b_hh)

            # Save intermediates needed for backward.
            saved = {
                "i": i,
                "f": f,
                "g": g,
                "o": o,
                "c_t": c_t_np,
                "tanh_c": tanh_c,
            }

            # ctx for h_t: grad_out is dL/dh_t
            ctx_h = Context(
                parents=parents,
                backward_fn=lambda grad_out: self._backward_h(ctx_h, grad_out, saved),
            )
            ctx_h.saved_meta["kind"] = "h"
            ctx_h.save_for_backward(x_t, h_prev, c_prev)
            h_t._set_ctx(ctx_h)

            # ctx for c_t: grad_out is dL/dc_t
            ctx_c = Context(
                parents=parents,
                backward_fn=lambda grad_out: self._backward_c(ctx_c, grad_out, saved),
            )
            ctx_c.saved_meta["kind"] = "c"
            ctx_c.save_for_backward(x_t, h_prev, c_prev)
            c_t._set_ctx(ctx_c)

        return h_t, c_t

    def _common_grads_from_dc(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c_prev: np.ndarray,
        dc: np.ndarray,
        saved: dict[str, np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        """
        Compute gate-related gradients given dL/dc_t.

        This helper handles the portion of the LSTM backward pass that depends
        only on the gradient w.r.t. the cell state c_t (dc). It computes the
        pre-activation gradients for the input, forget, and candidate gates,
        as well as the gradient flowing back to the previous cell state.

        Parameters
        ----------
        x : np.ndarray
            Current input batch x_t as a NumPy array of shape (N, D).
            Included for signature consistency/debugging; not used directly.
        h : np.ndarray
            Previous hidden state h_{t-1} as a NumPy array of shape (N, H).
            Included for signature consistency/debugging; not used directly.
        c_prev : np.ndarray
            Previous cell state c_{t-1} as a NumPy array of shape (N, H).
        dc : np.ndarray
            Gradient w.r.t. current cell state c_t, shape (N, H).
        saved : dict[str, np.ndarray]
            Saved intermediates from forward pass containing i, f, g.

        Returns
        -------
        tuple
            (dai, daf, dag, dc_prev) where:
            - dai : dL/da_i, shape (N, H)
            - daf : dL/da_f, shape (N, H)
            - dag : dL/da_g, shape (N, H)
            - dc_prev : dL/dc_{t-1}, shape (N, H)

        Notes
        -----
        The output gate gradient is handled separately because it depends on
        dL/dh_t rather than directly on dL/dc_t.
        """
        """
        Given dL/dc_t (dc), compute gate preactivation grads and parameter grads.

        Returns:
            grad_x_np, grad_h_np, grad_c_prev_np,
            grad_Wih_np, grad_Whh_np,
            grad_bih_np, grad_bhh_np,
            da (N,4H) (for debugging)
        """
        i = saved["i"]
        f = saved["f"]
        g = saved["g"]

        # c_t = f * c_prev + i * g
        di = dc * g
        dg = dc * i
        df = dc * c_prev
        dc_prev = dc * f

        # derivatives wrt preactivations
        dai = di * (i * (1.0 - i))
        daf = df * (f * (1.0 - f))
        dag = dg * (1.0 - g * g)
        # o gate handled outside (depends on dh)
        return dai, daf, dag, dc_prev

    def _pack_grads(
        self,
        x_t: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
        grad_x_np: np.ndarray,
        grad_h_np: np.ndarray,
        grad_c_np: np.ndarray,
        grad_Wih_np: np.ndarray,
        grad_Whh_np: np.ndarray,
        grad_b_np: Optional[np.ndarray],
    ):
        """
        Pack NumPy gradients into KeyDNN Tensors matching parent ordering.

        This helper converts computed NumPy gradient arrays into KeyDNN
        tensors (with requires_grad=False) and returns them in the same
        positional order as the corresponding parents used in Context.

        Parameters
        ----------
        x_t, h_prev, c_prev : Tensor
            Input and previous state tensors for this timestep.
        grad_x_np : np.ndarray
            Gradient w.r.t. x_t, shape (N, D).
        grad_h_np : np.ndarray
            Gradient w.r.t. h_prev, shape (N, H).
        grad_c_np : np.ndarray
            Gradient w.r.t. c_prev, shape (N, H).
        grad_Wih_np : np.ndarray
            Gradient w.r.t. W_ih, shape (D, 4H).
        grad_Whh_np : np.ndarray
            Gradient w.r.t. W_hh, shape (H, 4H).
        grad_b_np : Optional[np.ndarray]
            Gradient w.r.t. biases (shared for b_ih and b_hh here), shape (4H,).

        Returns
        -------
        tuple
            Tuple of gradients aligned with the Context parents, containing
            Tensor objects or None for inputs/parameters that do not require
            gradients.
        """
        gx = (
            tensor_from_numpy(grad_x_np, device=x_t.device, requires_grad=False)
            if x_t.requires_grad
            else None
        )
        gh = (
            tensor_from_numpy(grad_h_np, device=h_prev.device, requires_grad=False)
            if h_prev.requires_grad
            else None
        )
        gc = (
            tensor_from_numpy(grad_c_np, device=c_prev.device, requires_grad=False)
            if c_prev.requires_grad
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
            assert grad_b_np is not None
            gbih = (
                tensor_from_numpy(
                    grad_b_np, device=self.b_ih.device, requires_grad=False
                )
                if self.b_ih.requires_grad
                else None
            )
            gbhh = (
                tensor_from_numpy(
                    grad_b_np, device=self.b_hh.device, requires_grad=False
                )
                if self.b_hh.requires_grad
                else None
            )
            return (gx, gh, gc, gWih, gWhh, gbih, gbhh)

        return (gx, gh, gc, gWih, gWhh)

    def _backward_h(self, ctx: Context, grad_out: Tensor, saved: dict[str, np.ndarray]):
        """
        Backward pass for the hidden output h_t.

        Computes gradients for all parents given an upstream gradient
        dL/dh_t. This path includes the contribution of h_t through
        tanh(c_t), which induces an additional dL/dc_t term.

        Parameters
        ----------
        ctx : Context
            Autograd context containing saved tensors and parent references.
        grad_out : Tensor
            Upstream gradient with respect to h_t, shape (N, H).
        saved : dict[str, np.ndarray]
            Saved forward intermediates (gate activations, tanh(c_t), etc.).

        Returns
        -------
        tuple
            Gradients aligned with Context parents:
            (dx_t, dh_prev, dc_prev, dW_ih, dW_hh[, db_ih, db_hh])
            with entries possibly being None if a parent does not require grad.

        Notes
        -----
        - If downstream uses both h_t and c_t, KeyDNN will accumulate the
          resulting gradients for shared parents across both contexts.
        """
        """
        Backward for h_t output only (grad_out = dL/dh_t).
        This produces gradients for (x_t, h_prev, c_prev, W_ih, W_hh[, b_ih, b_hh]).
        """
        x_t, h_prev, c_prev = ctx.saved_tensors
        x = x_t.to_numpy()
        h = h_prev.to_numpy()
        c = c_prev.to_numpy()

        gh = grad_out.to_numpy()  # dL/dh_t (N,H)

        o = saved["o"]
        tanh_c = saved["tanh_c"]
        c_t_np = saved["c_t"]

        # h_t = o * tanh(c_t)
        do = gh * tanh_c
        dtanh_c = gh * o
        dc = dtanh_c * (1.0 - tanh_c * tanh_c)  # dL/dc_t from h path

        # o = sigmoid(a_o)
        dao = do * (o * (1.0 - o))  # dL/da_o

        # c_t path into i,f,g and c_prev
        dai, daf, dag, dc_prev = self._common_grads_from_dc(x, h, c, dc, saved)

        # Pack gate preact grads
        da = np.concatenate([dai, daf, dag, dao], axis=1).astype(np.float32)  # (N,4H)

        Wih = self.W_ih.to_numpy()
        Whh = self.W_hh.to_numpy()

        grad_x_np = da @ Wih.T
        grad_h_np = da @ Whh.T

        grad_Wih_np = x.T @ da
        grad_Whh_np = h.T @ da

        grad_b_np = da.sum(axis=0).astype(np.float32) if self.bias else None

        return self._pack_grads(
            x_t,
            h_prev,
            c_prev,
            grad_x_np.astype(np.float32),
            grad_h_np.astype(np.float32),
            dc_prev.astype(np.float32),
            grad_Wih_np.astype(np.float32),
            grad_Whh_np.astype(np.float32),
            grad_b_np,
        )

    def _backward_c(self, ctx: Context, grad_out: Tensor, saved: dict[str, np.ndarray]):
        """
        Backward pass for the cell output c_t.

        Computes gradients for all parents given an upstream gradient
        dL/dc_t that arrives directly from downstream consumers of c_t.

        Parameters
        ----------
        ctx : Context
            Autograd context containing saved tensors and parent references.
        grad_out : Tensor
            Upstream gradient with respect to c_t, shape (N, H).
        saved : dict[str, np.ndarray]
            Saved forward intermediates (gate activations, etc.).

        Returns
        -------
        tuple
            Gradients aligned with Context parents:
            (dx_t, dh_prev, dc_prev, dW_ih, dW_hh[, db_ih, db_hh])
            with entries possibly being None if a parent does not require grad.

        Notes
        -----
        - This path does not include the contribution from h_t -> c_t.
          That contribution is handled by `_backward_h`.
        - If both c_t and h_t are used downstream, KeyDNN will naturally
          accumulate gradients from both contexts.
        """
        """
        Backward for c_t output only (grad_out = dL/dc_t).
        This does NOT include the h_t -> c_t contribution (that is handled by _backward_h),
        so if both outputs are used downstream, the engine will accumulate them.
        """
        x_t, h_prev, c_prev = ctx.saved_tensors
        x = x_t.to_numpy()
        h = h_prev.to_numpy()
        c = c_prev.to_numpy()

        dc = grad_out.to_numpy()  # external dL/dc_t

        # c_t = f*c_prev + i*g
        o = saved[
            "o"
        ]  # still affects parameter grads via gates, but not via tanh(c_t) path here
        # o does not influence c_t directly, so dao is zero for this output-only backward.
        dao = np.zeros_like(saved["o"], dtype=np.float32)

        dai, daf, dag, dc_prev = self._common_grads_from_dc(x, h, c, dc, saved)

        da = np.concatenate([dai, daf, dag, dao], axis=1).astype(np.float32)

        Wih = self.W_ih.to_numpy()
        Whh = self.W_hh.to_numpy()

        grad_x_np = da @ Wih.T
        grad_h_np = da @ Whh.T

        grad_Wih_np = x.T @ da
        grad_Whh_np = h.T @ da

        grad_b_np = da.sum(axis=0).astype(np.float32) if self.bias else None

        return self._pack_grads(
            x_t,
            h_prev,
            c_prev,
            grad_x_np.astype(np.float32),
            grad_h_np.astype(np.float32),
            dc_prev.astype(np.float32),
            grad_Wih_np.astype(np.float32),
            grad_Whh_np.astype(np.float32),
            grad_b_np,
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this LSTMCell.

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
    def from_config(cls, cfg: Dict[str, Any]) -> "LSTMCell":
        """
        Construct an LSTMCell from a configuration dictionary.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        LSTMCell
            Reconstructed LSTMCell instance.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
        )


@register_module()
class LSTM(Module):
    """
    Unidirectional vanilla LSTM layer over a time-major sequence.

    This module applies `LSTMCell` over a sequence input of shape (T, N, D),
    producing either a full sequence of hidden states or the final hidden
    state depending on configuration.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        Whether the underlying cell uses biases.
    return_sequences : bool, optional
        If True, return a time-major sequence output (T, N, H).
        If False, return only the final output (N, H).
    return_state : bool, optional
        If True, also return the final states (h_T, c_T).
    keras_compat : bool, optional
        If True, return values follow a Keras-like convention:
        - If return_state: (out, (h_T, c_T))
        - Else: out
        If False, legacy-like behavior is used.

    Input
    -----
    x : Tensor
        Time-major input of shape (T, N, D).
    h0 : Optional[Tensor]
        Initial hidden state of shape (N, H). If None, initialized to zeros.
    c0 : Optional[Tensor]
        Initial cell state of shape (N, H). If None, initialized to zeros.

    Outputs
    -------
    Output structure depends on `keras_compat`, `return_sequences`, and
    `return_state` as implemented in `forward`.
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
        Initialize an LSTM layer.

        Parameters
        ----------
        input_size : int
            Input feature dimension D.
        hidden_size : int
            Hidden state dimension H.
        bias : bool, optional
            Whether to include biases in the cell.
        return_sequences : bool, optional
            Whether to return the full output sequence.
        return_state : bool, optional
            Whether to return the final states.
        keras_compat : bool, optional
            Whether to use Keras-like return conventions.
        """
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

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (T, N, D).
        h0 : Optional[Tensor], optional
            Initial hidden state of shape (N, H). If None, uses zeros.
        c0 : Optional[Tensor], optional
            Initial cell state of shape (N, H). If None, uses zeros.

        Returns
        -------
        Any
            Return structure depends on configuration:
            - If keras_compat is False: legacy-like return patterns.
            - If keras_compat is True: Keras-like return patterns.

        Notes
        -----
        - The per-timestep tensor `x[t]` is obtained via `__getitem__` to
          preserve the autograd path to the input.
        - The output sequence is constructed via `Tensor.stack` to keep
          gradients connected to per-timestep states.
        """
        x_np = x.to_numpy()
        T, N, D = x_np.shape
        H = self.cell.hidden_size

        if h0 is None:
            h_prev = tensor_from_numpy(
                np.zeros((N, H), dtype=np.float32),
                device=x.device,
                requires_grad=False,
            )
        else:
            h_prev = h0

        if c0 is None:
            c_prev = tensor_from_numpy(
                np.zeros((N, H), dtype=np.float32),
                device=x.device,
                requires_grad=False,
            )
        else:
            c_prev = c0

        hs = []
        for t in range(T):
            x_t = x[t]  # keeps grad path to x via __getitem__
            h_prev, c_prev = self.cell.forward(x_t, h_prev, c_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)  # (T,N,H)
        h_T = h_prev
        c_T = c_prev

        # --- Legacy-like behavior (default) ---
        if not self.keras_compat:
            if self.return_state:
                if self.return_sequences:
                    return h_seq, (h_T, c_T)
                return h_T, (h_T, c_T)
            return h_seq if self.return_sequences else h_T

        # --- Keras-like behavior (opt-in) ---
        out = h_seq if self.return_sequences else h_T
        if self.return_state:
            return out, (h_T, c_T)
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this LSTM layer.

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
    def from_config(cls, cfg: Dict[str, Any]) -> "LSTM":
        """
        Construct an LSTM layer from a configuration dictionary.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        LSTM
            Reconstructed LSTM instance.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", True)),
            keras_compat=bool(cfg.get("keras_compat", False)),
        )
