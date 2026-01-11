"""
Recurrent neural network public exports.

This module defines the public API surface for recurrent neural network (RNN)
layers and cells. It re-exports multiple recurrent architectures and their
cell-level building blocks, along with a bidirectional wrapper, under a single
stable namespace.

Exports
-------
Simple RNN
~~~~~~~~~~
- `RNN`: Recurrent neural network layer.
- `RNNCell`: Cell-level implementation of a simple RNN.

Gated RNNs
~~~~~~~~~~
- `GRU`: Gated Recurrent Unit layer.
- `GRUCell`: Cell-level implementation of a GRU.

LSTM
~~~~
- `LSTM`: Long Short-Term Memory layer.
- `LSTMCell`: Cell-level implementation of an LSTM.

Wrappers
~~~~~~~~
- `Bidirectional`: Wrapper for constructing bidirectional recurrent layers.

Design intent
-------------
- Provide a unified import location for all recurrent modules.
- Allow both layer-level and cell-level composition.
- Align with conventions from PyTorch and Keras.
"""

from ._rnn_module import RNN, RNNCell
from ._gru_module import GRU, GRUCell
from ._lstm_module import LSTM, LSTMCell
from ._bidirectional import Bidirectional


__all__ = [
    RNN.__name__,
    RNNCell.__name__,
    GRU.__name__,
    GRUCell.__name__,
    LSTM.__name__,
    LSTMCell.__name__,
    Bidirectional.__name__,
]
