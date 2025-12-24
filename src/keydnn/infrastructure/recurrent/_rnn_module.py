"""
Recurrent neural network (RNN) layers for KeyDNN.

This module provides a minimal, CPU-only implementation of a vanilla RNN:

- `RNNCell`: a single-timestep cell computing
    h_t = tanh(x_t W_ih + h_{t-1} W_hh + b_ih + b_hh)
- `RNN`: a simple unidirectional RNN that applies `RNNCell` over a time-major
  sequence input of shape (T, N, D).

Design goals
-----------
- Keep the implementation small and readable (CPU-first).
- Integrate with KeyDNN autograd via `Context` and `Tensor.backward()`.
- Avoid Tensor broadcasting semantics (KeyDNN currently enforces strict shape
  equality for elementwise ops), so biases are expanded explicitly.

Notes
-----
- This implementation is intended for learning and correctness first; it is not
  optimized. Python loops over time are used deliberately.
- Device handling is currently CPU-only.
- `RNN` uses Tensor slicing (`x[t]`) and `Tensor.stack`, so BPTT emerges naturally
  from the autograd graph (no fused RNN backward).

Public API
----------
- `RNNCell`
- `RNN`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

from ..tensor._tensor_context import Context

from ...domain.device._device import Device
from .._tensor import Tensor
from .._parameter import Parameter
from .._module import Module
from ..module._serialization_core import register_module


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
    - CPU-first implementation.
    - Bias addition is expanded explicitly to match strict elementwise semantics
      (no implicit broadcasting assumed).
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
        Module.__init__(self)

        device = Device("cpu")
        k = 1.0 / float(self.hidden_size) ** 0.5

        # W_ih: (D, H), U(-k, k)
        self.W_ih = Parameter(
            shape=(self.input_size, self.hidden_size),
            device=device,
            requires_grad=True,
        )
        self.W_ih.copy_from(
            (Tensor.rand(self.W_ih.shape, device=device) * 2.0 - 1.0) * k
        )

        # W_hh: (H, H), U(-k, k)
        self.W_hh = Parameter(
            shape=(self.hidden_size, self.hidden_size),
            device=device,
            requires_grad=True,
        )
        self.W_hh.copy_from(
            (Tensor.rand(self.W_hh.shape, device=device) * 2.0 - 1.0) * k
        )

        if self.bias:
            self.b_ih = Parameter(
                shape=(self.hidden_size,),
                device=device,
                requires_grad=True,
            )
            self.b_hh = Parameter(
                shape=(self.hidden_size,),
                device=device,
                requires_grad=True,
            )
            self.b_ih.copy_from(
                (Tensor.rand(self.b_ih.shape, device=device) * 2.0 - 1.0) * k
            )
            self.b_hh.copy_from(
                (Tensor.rand(self.b_hh.shape, device=device) * 2.0 - 1.0) * k
            )
        else:
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
        Biases are expanded explicitly to shape (N, H) to match strict elementwise
        semantics (no implicit broadcasting assumed).
        """
        if len(x_t.shape) != 2 or len(h_prev.shape) != 2:
            raise ValueError(
                f"RNNCell expects x_t,h_prev as 2D tensors, got {x_t.shape}, {h_prev.shape}"
            )
        N, D = x_t.shape
        if D != self.input_size:
            raise ValueError(f"RNNCell expects input_size={self.input_size}, got {D}")
        if h_prev.shape[0] != N or h_prev.shape[1] != self.hidden_size:
            raise ValueError(
                f"RNNCell expects h_prev shape (N,H)=({N},{self.hidden_size}), got {h_prev.shape}"
            )

        a = (x_t @ self.W_ih) + (h_prev @ self.W_hh)  # (N, H)

        if self.bias:
            # Explicit expansion to (N, H) without relying on implicit broadcasting
            b_ih_2d = self.b_ih.broadcast_to((N, self.hidden_size))
            b_hh_2d = self.b_hh.broadcast_to((N, self.hidden_size))
            a = a + b_ih_2d + b_hh_2d

        h_t = a.tanh()  # (N, H)

        req = Tensor._result_requires_grad(
            x_t,
            h_prev,
            self.W_ih,
            self.W_hh,
            *([self.b_ih, self.b_hh] if self.bias else []),
        )

        # Create a "fresh" output tensor with ctx owned by RNNCell (legacy style)
        out = Tensor(shape=h_t.shape, device=x_t.device, requires_grad=req, ctx=None)
        out.copy_from(h_t)

        if req:
            parents = (x_t, h_prev, self.W_ih, self.W_hh)
            if self.bias:
                parents = parents + (self.b_ih, self.b_hh)

            def backward_fn(grad_out: Tensor):
                return self._backward(ctx, grad_out)

            ctx = Context(parents=parents, backward_fn=backward_fn)
            # Save x_t, h_prev, h_t for tanh'(a)=1-h_t^2
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

        # grad_out should already be a non-requires-grad tensor from the engine;
        # Tensor ops here should remain no-grad by construction.
        ga = grad_out * (1.0 - (h_t * h_t))  # (N, H)

        grad_x = None
        grad_h_prev = None
        grad_Wih = None
        grad_Whh = None
        grad_bih = None
        grad_bhh = None

        if x_t.requires_grad:
            grad_x = ga @ self.W_ih.T  # (N, D)

        if h_prev.requires_grad:
            grad_h_prev = ga @ self.W_hh.T  # (N, H)

        if self.W_ih.requires_grad:
            grad_Wih = x_t.T @ ga  # (D, H)

        if self.W_hh.requires_grad:
            grad_Whh = h_prev.T @ ga  # (H, H)

        if self.bias:
            if self.b_ih.requires_grad:
                grad_bih = ga.sum(axis=0)  # (H,)
            if self.b_hh.requires_grad:
                grad_bhh = ga.sum(axis=0)  # (H,)

            return (grad_x, grad_h_prev, grad_Wih, grad_Whh, grad_bih, grad_bhh)

        return (grad_x, grad_h_prev, grad_Wih, grad_Whh)

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
        super().__init__()
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
        if len(x.shape) != 3:
            raise ValueError(f"RNN expects 3D input (T,N,D), got {x.shape}")

        T, N, D = x.shape
        if D != self.cell.input_size:
            raise ValueError(
                f"RNN expects input_size={self.cell.input_size}, got D={D}"
            )

        if h0 is None:
            h_prev = Tensor.full(
                (N, self.cell.hidden_size), 0.0, device=x.device, requires_grad=False
            )
        else:
            h_prev = h0

        hs = []
        for t in range(T):
            x_t = x[t]  # keeps autograd path through slicing (as implemented)
            h_prev = self.cell.forward(x_t, h_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)  # (T, N, H)
        h_T = h_prev

        if not self.keras_compat:
            if self.return_state:
                if self.return_sequences:
                    return h_seq, h_T
                return h_T
            return h_seq if self.return_sequences else h_T

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
            "input_size": int(self.cell.input_size),
            "hidden_size": int(self.cell.hidden_size),
            "bias": bool(self.cell.bias),
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
