"""
Gated Recurrent Unit (GRU) layers for KeyDNN (CPU-only).

This module implements a minimal GRU stack on top of KeyDNN's `Tensor` and
autograd `Context`. It includes:

- `tensor_from_numpy`
    A small helper that copies a NumPy array into a KeyDNN `Tensor`. This helper
    intentionally keeps the NumPy boundary local to this module.
- `_sigmoid`
    A numerically-stable NumPy sigmoid used by the (manual) GRUCell backward.
- `_expand_bias_1d_to_batch`
    Expands a per-hidden-unit bias of shape (H,) to an explicit (N, H) tensor,
    because elementwise broadcasting is intentionally not supported in KeyDNN.
- `GRUCell`
    A single-timestep GRU cell. The current `forward` builds its computation
    using `Tensor` ops. A separate `_backward` method exists for a manual
    backward path, but **it is not wired into autograd by `forward` in the
    current code** (no `Context` is attached inside `forward`).
- `GRU`
    A unidirectional GRU layer that applies `GRUCell` over a time-major input
    sequence of shape (T, N, D), returning either the full sequence, the final
    output, and/or the final hidden state depending on flags.

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
The cell stores parameters in packed form (PyTorch-like packing):

    W_ih : (D, 3H)  -> [W_iz | W_ir | W_in]
    W_hh : (H, 3H)  -> [W_hz | W_hr | W_hn]
    b_ih : (3H,) optional
    b_hh : (3H,) optional

Bias handling explicitly expands to (N, H) without relying on broadcasting.

Autograd notes
--------------
- `GRU` builds a differentiable BPTT graph by indexing `x[t]` (which preserves
  gradient connectivity via `Tensor.__getitem__`) and stacking per-timestep
  hidden states via `Tensor.stack`.
- `GRUCell.forward` is expressed in terms of `Tensor` ops (matmul, add, mul,
  sigmoid/tanh, etc.), so gradients flow through those ops if they are
  autograd-enabled.
- Although `GRUCell._backward` exists and documents a manual backward, it is
  currently **not** connected to a `Context` in `forward` as written.

Notes
-----
- CPU-only: parameters are allocated on CPU, and the implementation targets
  correctness and readability over performance.
- Broadcasting is intentionally avoided; helper routines expand biases and
  construct "ones" explicitly.

Implementation notes
--------------------
- The forward path and parameter initialization are fully Tensor-based and do
  not depend on NumPy.
- NumPy is used only in legacy or auxiliary routines (`tensor_from_numpy`,
  `_sigmoid`, and `GRUCell._backward`) that are not invoked by the default
  autograd execution path.
- These routines are retained for reference, testing, or future refactors and
  may be removed once Tensor-native equivalents are complete.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

from ...domain.device._device import Device
from .._module import Module
from .._parameter import Parameter
from .._tensor import Context, Tensor
from ..module._serialization_core import register_module


def tensor_from_numpy(
    arr: "Any", *, device: Device, requires_grad: bool = False
) -> Tensor:
    """
    Construct a KeyDNN `Tensor` by copying data from a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Source array. It is converted to `float32` before copying.
    device : Device
        Target device for the resulting tensor (CPU-only in current backend).
    requires_grad : bool, optional
        If True, marks the created tensor as participating in gradient tracking.
        Defaults to False.

    Returns
    -------
    Tensor
        A tensor with `shape == arr.shape` containing a float32 copy of `arr`.

    Notes
    -----
    This helper exists to isolate NumPy usage for testing, debugging, or legacy
    manual backward routines. Core GRU execution does not depend on NumPy.

    """
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    out = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad)
    out.copy_from_numpy(arr)
    return out


def _expand_bias_1d_to_batch(b: Tensor, N: int) -> Tensor:
    """
    Expand a bias vector from shape (H,) to an explicit batch tensor (N, H).

    Parameters
    ----------
    b : Tensor
        Per-hidden-unit bias of shape (H,).
    N : int
        Batch size.

    Returns
    -------
    Tensor
        Expanded bias tensor of shape (N, H).

    Notes
    -----
    KeyDNN currently enforces strict shape equality for elementwise operations.
    This helper avoids relying on broadcasting by explicitly materializing the
    expanded bias via `Tensor.stack` and `reshape`.
    """
    b2 = b.reshape((1, -1))
    return Tensor.stack([b2] * N, axis=0).reshape((N, b.shape[0]))


@register_module()
@dataclass
class GRUCell(Module):
    """
    Vanilla GRU cell operating on a single timestep (CPU-only).

    The cell computes the next hidden state `h_t` given:
    - current input `x_t` of shape (N, D)
    - previous hidden state `h_prev` of shape (N, H)

    Parameters are stored in packed form (3H columns) and are sliced into
    per-gate matrices inside `forward`.

    Parameters
    ----------
    input_size : int
        Input feature dimension D.
    hidden_size : int
        Hidden state dimension H.
    bias : bool, optional
        If True, includes packed bias parameters `b_ih` and `b_hh`, each of
        shape (3H,). Defaults to True.

    Attributes
    ----------
    W_ih : Parameter
        Packed input-to-hidden weights of shape (D, 3H).
    W_hh : Parameter
        Packed hidden-to-hidden weights of shape (H, 3H).
    b_ih : Optional[Parameter]
        Packed input bias of shape (3H,), if enabled.
    b_hh : Optional[Parameter]
        Packed hidden bias of shape (3H,), if enabled.

    Autograd behavior
    -----------------
    - `forward` is implemented using `Tensor` operations, so gradients flow
      through those ops if they are autograd-enabled.
    - `_backward` implements a NumPy-based manual backward routine, but the
      current `forward` does **not** attach a `Context` that calls `_backward`.
      (It remains available for future refactors or alternative execution paths.)

    Notes
    -----
    This class is a dataclass, so `Module.__init__()` is called explicitly in
    `__post_init__` to enable KeyDNN's parameter registration behavior.
    """

    input_size: int
    hidden_size: int
    bias: bool = True

    def __post_init__(self) -> None:
        """
        Initialize module state and allocate packed parameters.

        This hook:
        - explicitly calls `Module.__init__()` (required for dataclasses),
        - allocates parameters on CPU,
        - initializes packed weights (and biases, if enabled) using `Tensor.rand`
        and pure `Tensor` arithmetic,
        - copies initialized tensors directly into `Parameter` storage.

        Notes
        -----
        - This initialization path is **NumPy-free** by design.
        - It relies on the existence of a `Parameter.copy_from(Tensor)` API.
        - No fallbacks are provided: missing Tensor or Parameter functionality
        is expected to raise errors so that incomplete framework APIs can be
        identified and fixed explicitly.
        - Initialization is part of module construction and is not tracked by
        autograd.
        """

        Module.__init__(self)

        device = Device("cpu")
        H = int(self.hidden_size)
        D = int(self.input_size)

        k = 1.0 / math.sqrt(H)

        # NumPy-free init: Tensor.rand in [0,1) then affine to [-k, k]
        Wih_t = Tensor.rand((D, 3 * H), device=device, requires_grad=False)
        Whh_t = Tensor.rand((H, 3 * H), device=device, requires_grad=False)
        Wih_t = (Wih_t * (2.0 * k)) - k
        Whh_t = (Whh_t * (2.0 * k)) - k

        self.W_ih = Parameter(shape=(D, 3 * H), device=device, requires_grad=True)
        self.W_hh = Parameter(shape=(H, 3 * H), device=device, requires_grad=True)

        # Require Tensor-to-Parameter copy. If missing, let it raise.
        self.W_ih.copy_from(Wih_t)
        self.W_hh.copy_from(Whh_t)

        if self.bias:
            bih_t = Tensor.rand((3 * H,), device=device, requires_grad=False)
            bhh_t = Tensor.rand((3 * H,), device=device, requires_grad=False)
            bih_t = (bih_t * (2.0 * k)) - k
            bhh_t = (bhh_t * (2.0 * k)) - k

            self.b_ih = Parameter(shape=(3 * H,), device=device, requires_grad=True)
            self.b_hh = Parameter(shape=(3 * H,), device=device, requires_grad=True)

            # Require Tensor-to-Parameter copy. If missing, let it raise.
            self.b_ih.copy_from(bih_t)
            self.b_hh.copy_from(bhh_t)
        else:
            self.b_ih = None
            self.b_hh = None

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Compute the next hidden state for one GRU timestep.

        Parameters
        ----------
        x_t : Tensor
            Current timestep input of shape (N, D).
        h_prev : Tensor
            Previous hidden state of shape (N, H).

        Returns
        -------
        Tensor
            Next hidden state `h_t` of shape (N, H).

        What this implementation does
        -----------------------------
        - Slices packed parameters (`W_ih`, `W_hh`, and optional biases) into
          per-gate components.
        - Computes update/reset pre-activations via matmul/add.
        - Expands bias vectors to (N, H) explicitly (no broadcasting).
        - Applies `sigmoid` to z/r gates and `tanh` to candidate activation.
        - Combines candidate and previous hidden state:
              h_t = (1 - z) * n + z * h_prev

        Notes
        -----
        This method currently relies on `Tensor`-level ops for autograd.
        It does not attach a custom `Context` for a manual backward routine.
        """
        H = self.hidden_size

        W_iz = self.W_ih[:, 0:H]
        W_ir = self.W_ih[:, H : 2 * H]
        W_in = self.W_ih[:, 2 * H : 3 * H]

        W_hz = self.W_hh[:, 0:H]
        W_hr = self.W_hh[:, H : 2 * H]
        W_hn = self.W_hh[:, 2 * H : 3 * H]

        a_z = x_t @ W_iz + h_prev @ W_hz
        a_r = x_t @ W_ir + h_prev @ W_hr

        if self.bias:
            b_ih = self.b_ih
            b_hh = self.b_hh

            b_z = b_ih[0:H] + b_hh[0:H]
            b_r = b_ih[H : 2 * H] + b_hh[H : 2 * H]
            b_n = b_ih[2 * H : 3 * H] + b_hh[2 * H : 3 * H]

            N = x_t.shape[0]
            b_z = _expand_bias_1d_to_batch(b_z, N)
            b_r = _expand_bias_1d_to_batch(b_r, N)
            b_n = _expand_bias_1d_to_batch(b_n, N)

            a_z = a_z + b_z
            a_r = a_r + b_r
        else:
            b_n = None

        z = a_z.sigmoid()
        r = a_r.sigmoid()

        rh = r * h_prev
        a_n = x_t @ W_in + rh @ W_hn
        if self.bias:
            a_n = a_n + b_n

        n = a_n.tanh()

        one = Tensor.ones(shape=z.shape, device=z.device, requires_grad=False)
        h_t = (one - z) * n + z * h_prev
        return h_t

    def _backward(self, ctx: Context, grad_out: Tensor):
        """
        Manual backward for one GRU timestep (NumPy implementation).

        Parameters
        ----------
        ctx : Context
            Context expected to contain saved forward intermediates in
            `ctx.saved_tensors` in the following order:
            (x_t, h_prev, z_t, r_t, n_t).
        grad_out : Tensor
            Upstream gradient w.r.t. `h_t`, shape (N, H).

        Returns
        -------
        tuple
            Gradients aligned with the expected `ctx.parents` ordering:
            (dx_t, dh_prev, dW_ih, dW_hh[, db_ih, db_hh]).
            Entries may be None if the corresponding parent does not require
            gradients.

        Notes
        -----
        - This routine computes gradients in NumPy for clarity.
        - Bias gradients are computed for the combined per-gate bias
          `b_gate = b_ih_gate + b_hh_gate` and then returned for both `b_ih`
          and `b_hh`, matching the forward composition used here.
        - This method is not currently invoked by `forward` unless you attach a
          `Context` that calls it.

        Important
        ---------
        This method is a legacy NumPy-based reference implementation and is not invoked
        by the default Tensor-based forward/autograd path. It is expected to be removed
        or rewritten once a Tensor-native custom backward is implemented.

        - All differentiable execution is performed via Tensor operations in `forward`.

        """
        import numpy as np

        x_t, h_prev, z_t, r_t, n_t = ctx.saved_tensors

        x = x_t.to_numpy().astype(np.float32, copy=False)
        h = h_prev.to_numpy().astype(np.float32, copy=False)
        z = z_t.to_numpy().astype(np.float32, copy=False)
        r = r_t.to_numpy().astype(np.float32, copy=False)
        n = n_t.to_numpy().astype(np.float32, copy=False)
        gh = grad_out.to_numpy().astype(np.float32, copy=False)

        N, D = x.shape
        H = self.hidden_size

        Wih = self.W_ih.to_numpy().astype(np.float32, copy=False)
        Whh = self.W_hh.to_numpy().astype(np.float32, copy=False)

        W_iz = Wih[:, 0:H]
        W_ir = Wih[:, H : 2 * H]
        W_in = Wih[:, 2 * H : 3 * H]

        W_hz = Whh[:, 0:H]
        W_hr = Whh[:, H : 2 * H]
        W_hn = Whh[:, 2 * H : 3 * H]

        grad_z = gh * (h - n)
        grad_n = gh * (1.0 - z)
        grad_h = gh * z

        grad_a_n = grad_n * (1.0 - n * n)

        rh = r * h
        grad_x = grad_a_n @ W_in.T
        grad_W_in = x.T @ grad_a_n
        grad_W_hn = rh.T @ grad_a_n
        grad_rh = grad_a_n @ W_hn.T

        grad_r = grad_rh * h
        grad_h += grad_rh * r

        grad_a_r = grad_r * (r * (1.0 - r))

        grad_x += grad_a_r @ W_ir.T
        grad_h += grad_a_r @ W_hr.T
        grad_W_ir = x.T @ grad_a_r
        grad_W_hr = h.T @ grad_a_r

        grad_a_z = grad_z * (z * (1.0 - z))

        grad_x += grad_a_z @ W_iz.T
        grad_h += grad_a_z @ W_hz.T
        grad_W_iz = x.T @ grad_a_z
        grad_W_hz = h.T @ grad_a_z

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
            A JSON-serializable dict containing the constructor arguments needed
            to reconstruct the cell.
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
            A reconstructed `GRUCell` instance.
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

    This module applies `GRUCell.forward` over a time-major input `x` of shape
    (T, N, D). It supports multiple return conventions via flags:

    - `return_sequences`
        If True, returns the full hidden sequence stacked into shape (T, N, H).
        If False, uses only the final output (N, H).
    - `return_state`
        If True, includes the final hidden state `h_T` in the return structure.
    - `keras_compat`
        If True, uses Keras-style returns:
            - if return_state: (out, h_T)
            - else: out
        If False, uses the current KeyDNN legacy-like branching implemented in
        `forward`.

    Autograd behavior
    -----------------
    - The loop indexes `x[t]` via `Tensor.__getitem__`, preserving gradient
      connectivity to the input.
    - Hidden states are stacked via `Tensor.stack`, which is autograd-enabled.

    Notes
    -----
    - Initial hidden state `h0` defaults to an explicit zeros tensor when not
      provided.
    - This implementation is CPU-oriented and prioritizes clarity.
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
            Whether the underlying `GRUCell` includes biases. Defaults to True.
        return_sequences : bool, optional
            If True, return the full output sequence (T, N, H). Defaults to True.
        return_state : bool, optional
            If True, include the final hidden state in the return value(s).
            Defaults to True.
        keras_compat : bool, optional
            If True, return values follow a Keras-like convention. Defaults to False.
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
            Initial hidden state of shape (N, H). If None, initializes to zeros
            with `requires_grad=False`.

        Returns
        -------
        Any
            A return structure determined by `return_sequences`, `return_state`,
            and `keras_compat`:

            - keras_compat=False:
                - return_state=True,  return_sequences=True  -> (h_seq, h_T)
                - return_state=True,  return_sequences=False -> h_T
                - return_state=False, return_sequences=True  -> h_seq
                - return_state=False, return_sequences=False -> h_T
            - keras_compat=True:
                - out = h_seq if return_sequences else h_T
                - return_state=True  -> (out, h_T)
                - return_state=False -> out

        Raises
        ------
        ValueError
            If `x` does not have rank 3 (T, N, D).

        Notes
        -----
        The BPTT graph is built by:
        - indexing `x[t]` to keep input connectivity, and
        - stacking hidden states via `Tensor.stack`.
        """
        if len(x.shape) != 3:
            raise ValueError(f"GRU expects x shape (T,N,D), got {x.shape}")

        T, N, _D = x.shape
        H = self.cell.hidden_size

        if h0 is None:
            h_prev = Tensor.zeros(shape=(N, H), device=x.device, requires_grad=False)
        else:
            h_prev = h0

        hs = []
        for t in range(T):
            x_t = x[t]
            h_prev = self.cell.forward(x_t, h_prev)
            hs.append(h_prev)

        h_seq = Tensor.stack(hs, axis=0)
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
        Return a serializable configuration for this GRU layer.

        Returns
        -------
        Dict[str, Any]
            A JSON-serializable dict containing constructor arguments and behavior
            flags needed to reconstruct the layer.
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
            A reconstructed `GRU` instance with matching configuration.
        """
        return cls(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", True)),
            keras_compat=bool(cfg.get("keras_compat", False)),
        )
