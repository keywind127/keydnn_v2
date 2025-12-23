"""
Bidirectional wrapper for recurrent modules (Keras-like).

This module implements a minimal Keras-style Bidirectional wrapper for KeyDNN's
time-major vanilla RNN:

    x: (T, N, D)

It runs a forward RNN over t=0..T-1 and a backward RNN over t=T-1..0, then
merges their outputs.

Default behavior matches Keras merge_mode="concat":
- return_sequences=True  -> y_seq: (T, N, 2H)
- return_sequences=False -> y_T:   (N, 2H)

If return_state=True, also returns the final states of both directions:
- h_f_T: (N, H)  (forward final)
- h_b_T: (N, H)  (backward final; corresponds to original t=0)

Notes
-----
- CPU-only, consistent with the current RNN backend.
- Merge modes supported: "concat" only (for now).
- Internally forces child RNNs to return both (out, h_T) via keras_compat=True,
  then applies wrapper-level return flags.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from .._module import Module
from .._tensor import Tensor
from ..module._serialization_core import register_module
from ._rnn_module import RNN


InitialStateLike = Union[None, Tensor, Tuple[Tensor, Tensor]]


@register_module()
class Bidirectional(Module):
    """
    Keras-like Bidirectional wrapper for KeyDNN RNN.

    Parameters
    ----------
    layer : RNN
        A unidirectional RNN instance used as a template for hyperparameters.
        Two independent copies are created internally (forward/backward).
    merge_mode : str, optional
        Merge mode for forward/backward outputs. Only "concat" is supported.
        Default is "concat".
    return_sequences : bool, optional
        If True, return the full merged sequence (T, N, 2H).
        If False, return only the merged final output (N, 2H).
        Default is True.
    return_state : bool, optional
        If True, also return (h_f_T, h_b_T), both shape (N, H).
        Default is False.

    Examples
    --------
    birnn = Bidirectional(RNN(D, H), return_sequences=True, return_state=True)
    y, h_f, h_b = birnn(x)   # y: (T,N,2H), h_f/h_b: (N,H)
    """

    def __init__(
        self,
        layer: RNN,
        *,
        merge_mode: str = "concat",
        return_sequences: bool = True,
        return_state: bool = False,
    ) -> None:
        super().__init__()

        if merge_mode != "concat":
            raise ValueError(
                f"Unsupported merge_mode={merge_mode!r}. Only 'concat' is supported."
            )

        self.merge_mode = str(merge_mode)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

        # Clone two independent RNNs with the same hyperparameters.
        # Force keras_compat=True and return_state=True internally so we always get (out, h_T).
        cfg = layer.get_config()

        self.forward_rnn = RNN(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=True,  # always produce sequence internally
            return_state=True,  # always produce h_T internally
            keras_compat=True,  # always return (out, h_T)
        )

        self.backward_rnn = RNN(
            input_size=int(cfg["input_size"]),
            hidden_size=int(cfg["hidden_size"]),
            bias=bool(cfg.get("bias", True)),
            return_sequences=True,
            return_state=True,
            keras_compat=True,
        )

    def forward(self, x: Tensor, h0: InitialStateLike = None):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input sequence, shape (T, N, D).
        h0 : None | Tensor | (Tensor, Tensor), optional
            Optional initial states.
            - None: both directions start from zeros.
            - Tensor: used as initial state for BOTH directions.
            - (h0_f, h0_b): separate initial states for forward/backward.

        Returns
        -------
        Tensor | Tuple[Tensor, Tensor, Tensor]
            If return_state=False:
                out
            If return_state=True:
                (out, h_f_T, h_b_T)
        """
        if len(x.shape) != 3:
            raise ValueError(f"Bidirectional expects x shape (T,N,D), got {x.shape}")

        # Unpack h0
        h0_f: Optional[Tensor]
        h0_b: Optional[Tensor]
        if h0 is None:
            h0_f, h0_b = None, None
        elif isinstance(h0, Tensor):
            h0_f, h0_b = h0, h0
        elif (
            isinstance(h0, tuple)
            and len(h0) == 2
            and isinstance(h0[0], Tensor)
            and isinstance(h0[1], Tensor)
        ):
            h0_f, h0_b = h0
        else:
            raise TypeError("h0 must be None, a Tensor, or a (Tensor, Tensor) tuple")

        # Forward direction: (h_seq_f, h_f_T)
        h_seq_f, h_f_T = self.forward_rnn.forward(
            x, h0=h0_f
        )  # keras_compat=True => (out, h_T)

        # Backward direction:
        # iterate from T-1 to 0 using x[t] to preserve grad path to original x via __getitem__
        T = x.shape[0]
        hs_b = []
        # We can't call backward_rnn.forward on a reversed Tensor without a reverse op,
        # so we do an explicit reversed loop using the cell.
        if h0_b is None:
            # Let backward_rnn initialize zeros internally by passing None
            h_prev = None
        else:
            h_prev = h0_b

        # Use the backward_rnn's cell directly to build a BPTT graph across reversed time.
        # This mirrors RNN.forward but with reversed indices.
        if h_prev is None:
            # backward_rnn.forward would allocate zeros internally, but we are doing manual loop:
            # reuse forward_rnn behavior: call backward_rnn.forward on a stacked reversed sequence
            # would require reverse op. Instead, we emulate init by calling backward_rnn.forward
            # on the first timestep to get correct zero init.
            # So: create an explicit zero h_prev with correct shape on x.device.
            import numpy as np
            from ._rnn_module import tensor_from_numpy  # same helper used in RNN

            N = x.shape[1]
            H = self.backward_rnn.cell.hidden_size
            h_prev = tensor_from_numpy(
                np.zeros((N, H), dtype=np.float32), device=x.device, requires_grad=False
            )

        for t in range(T - 1, -1, -1):
            x_t = x[t]
            h_prev = self.backward_rnn.cell.forward(x_t, h_prev)
            hs_b.append(h_prev)

        # hs_b is in reversed-time order: [h_{T-1}^b, ..., h_0^b]
        # Align to original time order by reversing the list before stacking
        hs_b_aligned = list(reversed(hs_b))
        h_seq_b = Tensor.stack(hs_b_aligned, axis=0)  # (T,N,H)
        h_b_T = hs_b[
            -1
        ]  # last computed in reversed loop == state after processing original t=0

        # Merge (concat) like Keras default
        if self.return_sequences:
            out = Tensor.concat([h_seq_f, h_seq_b], axis=2)  # (T,N,2H)
        else:
            out = Tensor.concat([h_f_T, h_b_T], axis=1)  # (N,2H)

        if self.return_state:
            return out, h_f_T, h_b_T
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Serializable configuration for this wrapper.

        Notes
        -----
        We store the wrapped RNN hyperparameters (via forward_rnn.get_config()) plus
        wrapper flags. We do not serialize weights here.
        """
        return {
            "merge_mode": self.merge_mode,
            "return_sequences": bool(self.return_sequences),
            "return_state": bool(self.return_state),
            "layer": self.forward_rnn.get_config(),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Bidirectional":
        """
        Reconstruct Bidirectional wrapper from config.
        """
        layer_cfg = dict(cfg["layer"])
        layer = RNN.from_config(layer_cfg)

        return cls(
            layer=layer,
            merge_mode=str(cfg.get("merge_mode", "concat")),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", False)),
        )
