"""
Bidirectional wrapper for recurrent modules (Keras-like).

This module implements a generic Bidirectional wrapper that can be applied
to KeyDNN recurrent layers such as RNN or LSTM. The wrapper closely follows
Keras semantics and supports concatenation of forward and backward outputs.

The wrapped recurrent module must provide:
- get_config() -> Dict[str, Any]
- from_config(cfg) -> Module
- forward(x: Tensor, h0: Optional[state]) -> output

The wrapped forward method must follow Keras-compatible return conventions:
- If return_state=True: returns (out, state)
- Otherwise: returns out
- out is either a full sequence (T, N, H) or a final output (N, H),
  depending on return_sequences.

This wrapper implements merge_mode="concat" only (Keras default) and exposes:
- return_sequences: sequence output (T, N, 2H) vs final output (N, 2H)
- return_state: optionally returns final states for both directions

Notes
-----
- CPU-only behavior is inherited from the wrapped recurrent modules.
- Time reversal is implemented via slicing and Tensor.stack to preserve
  autograd connectivity.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from .._module import Module
from .._tensor import Tensor
from ..module._serialization_core import register_module


# State can be:
# - None
# - Tensor (RNN: h0)
# - Tuple[Tensor, Tensor] (LSTM: (h0, c0))
InitialStateLike = Union[None, Tensor, Tuple[Tensor, Tensor]]


def _reverse_time(x: Tensor) -> Tensor:
    """
    Reverse a time-major tensor along the time dimension.

    This function reverses a tensor of shape (T, ...) along its first
    (time) axis while preserving the autograd path by using Tensor
    indexing and Tensor.stack instead of NumPy operations.

    Parameters
    ----------
    x : Tensor
        Input tensor with time as the first dimension.

    Returns
    -------
    Tensor
        Time-reversed tensor with the same shape as the input.
    """
    T = x.shape[0]
    xs = [x[t] for t in range(T - 1, -1, -1)]
    return Tensor.stack(xs, axis=0)


def _is_state_tuple(s: object) -> bool:
    """
    Check whether an object represents an LSTM-like state tuple.

    A valid state tuple is defined as a pair of Tensors, typically
    corresponding to (h, c) in LSTM modules.

    Parameters
    ----------
    s : object
        Object to test.

    Returns
    -------
    bool
        True if the object is a tuple of two Tensor instances.
    """
    return (
        isinstance(s, tuple)
        and len(s) == 2
        and isinstance(s[0], Tensor)
        and isinstance(s[1], Tensor)
    )


@register_module()
class Bidirectional(Module):
    """
    Keras-like Bidirectional wrapper for recurrent modules.

    This wrapper runs two independent copies of a recurrent layer:
    one processing the sequence forward in time and the other processing
    the reversed sequence. Their outputs are merged by concatenation.

    Parameters
    ----------
    layer : Module
        A unidirectional recurrent module instance (e.g., RNN or LSTM)
        used as a template. Two independent clones are created internally
        via get_config() and from_config().
    merge_mode : str, optional
        Merge strategy for combining forward and backward outputs.
        Only "concat" is supported. Default is "concat".
    return_sequences : bool, optional
        If True, return the full output sequence of shape (T, N, 2H).
        If False, return only the final output of shape (N, 2H).
        Default is True.
    return_state : bool, optional
        If True, also return the final states from both directions.
        Default is False.

    Returns
    -------
    If return_state is False:
        Tensor
            The merged output tensor.
    If return_state is True:
        Tuple
            (out, state_f, state_b), where state_f and state_b are the
            final states from the forward and backward directions.

    Notes
    -----
    - Forward and backward layers are fully independent.
    - Internal layers are forced to return sequences and states so that
      the wrapper can control the exposed outputs.
    """

    def __init__(
        self,
        layer: Module,
        *,
        merge_mode: str = "concat",
        return_sequences: bool = True,
        return_state: bool = False,
    ) -> None:
        """
        Initialize a Bidirectional wrapper.

        Parameters
        ----------
        layer : Module
            Base recurrent layer to wrap.
        merge_mode : str, optional
            Merge strategy. Only "concat" is supported.
        return_sequences : bool, optional
            Whether to return full sequences.
        return_state : bool, optional
            Whether to return final states.
        """
        super().__init__()

        if merge_mode != "concat":
            raise ValueError(
                f"Unsupported merge_mode={merge_mode!r}. Only 'concat' is supported."
            )

        self.merge_mode = str(merge_mode)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)

        if not hasattr(layer, "get_config") or not callable(
            getattr(layer, "get_config")
        ):
            raise TypeError("Wrapped layer must implement get_config().")
        if not hasattr(layer.__class__, "from_config") or not callable(
            getattr(layer.__class__, "from_config")
        ):
            raise TypeError("Wrapped layer class must implement from_config().")

        # Clone two independent layers
        base_cfg = layer.get_config()
        self.forward_layer = layer.__class__.from_config(dict(base_cfg))
        self.backward_layer = layer.__class__.from_config(dict(base_cfg))

        # Force Keras-like internal behavior
        for m in (self.forward_layer, self.backward_layer):
            cfg = m.get_config()
            cfg["return_sequences"] = True
            cfg["return_state"] = True
            cfg["keras_compat"] = True
            rebuilt = m.__class__.from_config(cfg)

            if m is self.forward_layer:
                self.forward_layer = rebuilt
            else:
                self.backward_layer = rebuilt

        # Backward-compat aliases
        self.forward_rnn = self.forward_layer
        self.backward_rnn = self.backward_layer

    def forward(self, x: Tensor, h0: InitialStateLike = None):
        """
        Apply bidirectional recurrent processing to an input sequence.

        Parameters
        ----------
        x : Tensor
            Input sequence tensor of shape (T, N, D).
        h0 : InitialStateLike, optional
            Initial state(s) for the recurrent layers. Can be:
            - None
            - A Tensor (shared initial state)
            - An LSTM-style state tuple (h0, c0)
            - A pair of states (state_f, state_b)

        Returns
        -------
        Tensor or Tuple
            If return_state is False, returns the merged output tensor.
            If return_state is True, returns (out, state_f, state_b).

        Raises
        ------
        ValueError
            If input shape is invalid.
        TypeError
            If h0 is of an unsupported type.
        RuntimeError
            If wrapped layers do not return expected outputs.
        """
        if len(x.shape) != 3:
            raise ValueError(f"Bidirectional expects x shape (T,N,D), got {x.shape}")

        # Unpack initial states
        if h0 is None:
            h0_f, h0_b = None, None
        elif isinstance(h0, Tensor):
            h0_f, h0_b = h0, h0
        elif _is_state_tuple(h0):
            h0_f, h0_b = h0, h0
        elif (
            isinstance(h0, tuple)
            and len(h0) == 2
            and (_is_state_tuple(h0[0]) or isinstance(h0[0], Tensor))
            and (_is_state_tuple(h0[1]) or isinstance(h0[1], Tensor))
        ):
            h0_f, h0_b = h0  # type: ignore[assignment]
        else:
            raise TypeError(
                "h0 must be None, a Tensor, a (Tensor,Tensor) LSTM-state, "
                "or a pair of states (state_f, state_b)."
            )

        # Forward direction
        y_f_seq, state_f = self.forward_layer.forward(x, h0=h0_f)

        # Backward direction
        x_rev = _reverse_time(x)
        y_b_seq_rev, state_b = self.backward_layer.forward(x_rev, h0=h0_b)
        y_b_seq = _reverse_time(y_b_seq_rev)

        if self.return_sequences:
            out = Tensor.concat([y_f_seq, y_b_seq], axis=2)
        else:
            out = Tensor.concat([y_f_seq[-1], y_b_seq[-1]], axis=1)

        if self.return_state:
            return out, state_f, state_b
        return out

    def get_config(self) -> Dict[str, Any]:
        """
        Return a serializable configuration for this wrapper.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary describing the Bidirectional wrapper.
        """
        return {
            "merge_mode": self.merge_mode,
            "return_sequences": bool(self.return_sequences),
            "return_state": bool(self.return_state),
            "layer_type": self.forward_layer.__class__.__name__,
            "layer": self.forward_layer.get_config(),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Bidirectional":
        """
        Reconstruct a Bidirectional wrapper from a configuration dictionary.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Serialized configuration.

        Returns
        -------
        Bidirectional
            Reconstructed Bidirectional module.
        """
        layer_cfg = dict(cfg["layer"])
        layer_type = cfg.get("layer_type", None)

        if layer_type == "RNN":
            from ._rnn_module import RNN

            layer = RNN.from_config(layer_cfg)
        elif layer_type == "LSTM":
            from ._lstm_module import LSTM

            layer = LSTM.from_config(layer_cfg)
        else:
            try:
                from ._rnn_module import RNN

                layer = RNN.from_config(layer_cfg)
            except Exception:
                from ._lstm_module import LSTM

                layer = LSTM.from_config(layer_cfg)

        return cls(
            layer=layer,
            merge_mode=str(cfg.get("merge_mode", "concat")),
            return_sequences=bool(cfg.get("return_sequences", True)),
            return_state=bool(cfg.get("return_state", False)),
        )
