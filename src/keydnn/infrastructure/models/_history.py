"""
Training history utilities.

This module defines lightweight data structures used to record and expose
training metrics over time, in a manner similar to Keras' `History` object.

The primary purpose of this module is to provide a simple, serializable,
and framework-agnostic container for per-epoch metrics produced by
high-level training APIs such as `Model.fit()`.

Design goals
------------
- Minimal surface area: no dependency on tensors, devices, or autograd
- Deterministic ordering and explicit epoch indexing
- Human-readable and debugger-friendly representation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Union


Number = Union[int, float]


@dataclass
class History:
    """
    Container for per-epoch training metrics.

    `History` is a lightweight, Keras-inspired object returned by high-level
    training routines (e.g., `Model.fit`). It stores aggregated metric values
    for each completed epoch and provides convenience accessors for inspection.

    Attributes
    ----------
    history : Dict[str, List[float]]
        Mapping from metric name to a list of per-epoch values.
        Each list is ordered by epoch index.
    epoch : List[int]
        List of epoch indices (0-based) corresponding to entries in `history`.

    Notes
    -----
    - All metric values are stored as Python `float` for portability.
    - This object is intentionally passive: it performs no aggregation logic
      beyond appending values supplied by the training loop.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    epoch: List[int] = field(default_factory=list)

    def _ensure_key(self, k: str) -> None:
        """
        Ensure a metric key exists in the history mapping.

        Parameters
        ----------
        k : str
            Metric name to initialize if not already present.

        Notes
        -----
        This is an internal helper used to lazily create metric lists
        when new metrics are first encountered.
        """
        if k not in self.history:
            self.history[k] = []

    def append_epoch(self, epoch_idx: int, logs: Mapping[str, Number]) -> None:
        """
        Append metrics for a completed epoch.

        Parameters
        ----------
        epoch_idx : int
            Zero-based index of the completed epoch.
        logs : Mapping[str, Number]
            Mapping from metric name to aggregated epoch value
            (e.g., mean loss, accuracy).

        Notes
        -----
        - Metric values are coerced to `float` before storage.
        - The caller (typically `Model.fit`) is responsible for ensuring
          that `logs` contains already-aggregated values.
        """
        self.epoch.append(int(epoch_idx))
        for k, v in logs.items():
            self._ensure_key(k)
            self.history[k].append(float(v))

    def last(self) -> Dict[str, float]:
        """
        Return metrics from the most recent epoch.

        Returns
        -------
        Dict[str, float]
            Mapping from metric name to its latest recorded value.
            Metrics with no recorded values are omitted.

        Notes
        -----
        This is a convenience accessor commonly used after training
        to retrieve final loss/metric values without manual indexing.
        """
        out: Dict[str, float] = {}
        for k, vs in self.history.items():
            if vs:
                out[k] = float(vs[-1])
        return out
