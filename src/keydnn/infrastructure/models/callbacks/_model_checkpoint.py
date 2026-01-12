"""
ModelCheckpoint callback.

This module provides a Keras-style `ModelCheckpoint` callback for training loops
such as `Model.fit()`. It saves model checkpoints to disk using the model's
`save_json(path)` method, either at the end of every epoch or only when a
monitored metric improves.

Filename formatting
-------------------
The `filepath` argument may be either a fixed path or a Python format string.
If formatting fields are present, they are filled using:

- `epoch`: the 1-based epoch index (Keras-like convention)
- all scalar entries from `logs` (e.g., `loss`, `val_loss`, etc.)

Example:
    "checkpoints/ckpt_epoch{epoch:03d}_valloss{val_loss:.6f}.json"

Contracts
---------
- The model is expected (but not required) to implement `save_json(path)`.
- If `save_json` is not available, the callback degrades to a no-op.
- Early-stopping or checkpointing decisions are deterministic and based solely
  on explicit epoch indices and `logs`.

Design goals
------------
- Minimal coupling to the rest of the framework.
- Filesystem-based persistence only (no in-memory snapshots).
- Behavior consistent with Keras-style checkpoint callbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ._base import Callback
from ._base import _as_float, _is_better


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    `ModelCheckpoint` writes checkpoints via the model's `save_json(path)` method.
    It supports saving at the end of every epoch or only when a monitored metric
    improves (best-only), similar to Keras.

    Formatting
    ----------
    `filepath` may be a plain path or a format string. If it contains Python
    format fields, they are formatted with:
    - `epoch`: the 1-based epoch number (Keras-like)
    - all keys present in `logs` (e.g., `loss`, `val_loss`, etc.)

    Parameters
    ----------
    filepath : str | Path
        Output path or a path template.
    monitor : str, optional
        Name of the metric to monitor for best-only saving. Default is "val_loss".
    mode : str, optional
        One of {"min", "max", "auto"} determining optimization direction.
    save_best_only : bool, optional
        If True, only save checkpoints when `monitor` improves.
    verbose : int, optional
        If non-zero, prints save decisions.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        monitor: str = "val_loss",
        mode: str = "auto",
        save_best_only: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        Create a new ModelCheckpoint callback.

        Parameters
        ----------
        filepath : str | Path
            Output path or a path template.
        monitor : str, optional
            Metric name to monitor for best-only saving.
        mode : str, optional
            One of {"min", "max", "auto"} determining optimization direction.
        save_best_only : bool, optional
            Whether to save checkpoints only when the monitored metric improves.
        verbose : int, optional
            Verbosity level; if non-zero, save decisions are printed.
        """
        self.filepath = Path(filepath)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.save_best_only = bool(save_best_only)
        self.verbose = int(verbose)

        self.best = float("inf")

    def _resolve_mode(self) -> str:
        """
        Resolve the effective optimization mode.

        Returns
        -------
        str
            Either "min" or "max", inferred from the configured `mode`
            and `monitor` name if necessary.

        Raises
        ------
        ValueError
            If the configured mode is unsupported.
        """
        if self.mode in ("min", "max"):
            return self.mode
        if self.mode == "auto":
            m = self.monitor.lower()
            if "acc" in m or "accuracy" in m:
                return "max"
            return "min"
        raise ValueError(f"Unsupported mode: {self.mode!r}")

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize internal state at the start of training.

        This sets the initial best metric value according to the resolved
        optimization direction.

        Parameters
        ----------
        logs : dict[str, float], optional
            Initial training logs (unused).
        """
        mode = self._resolve_mode()
        self.best = float("inf") if mode == "min" else -float("inf")

    def _format_path(self, epoch: int, logs: Mapping[str, float]) -> Path:
        """
        Format the checkpoint output path for a given epoch.

        The formatting dictionary includes:
        - `epoch`: the 1-based epoch index
        - all key-value pairs from `logs`

        If formatting fails for any reason, the raw `filepath` is used.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : Mapping[str, float]
            Metrics logged for the completed epoch.

        Returns
        -------
        Path
            The resolved checkpoint output path.
        """
        # Keras typically formats epoch as 1-based in filenames.
        fmt_dict: Dict[str, Any] = {"epoch": epoch + 1}
        fmt_dict.update(logs)

        s = str(self.filepath)
        try:
            s = s.format(**fmt_dict)
        except Exception:
            # If formatting fails, just use the raw path.
            pass
        return Path(s)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Save a model checkpoint at epoch end if criteria are met.

        Depending on configuration, this method either saves a checkpoint
        unconditionally or only when the monitored metric improves.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict[str, float], optional
            Metrics logged for the completed epoch.
        """
        logs = logs or {}
        model = getattr(self, "model", None)
        if model is None:
            return

        out_path = self._format_path(epoch, logs)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mode = self._resolve_mode()
        current = _as_float(logs.get(self.monitor, float("nan")))

        should_save = True
        if self.save_best_only:
            should_save = _is_better(current, self.best, mode, min_delta=0.0)

        if should_save:
            if self.save_best_only:
                self.best = current
            save_json = getattr(model, "save_json", None)
            if callable(save_json):
                save_json(out_path)
                if self.verbose:
                    print(f"ModelCheckpoint: saved to {out_path}")
        else:
            if self.verbose:
                print(
                    f"ModelCheckpoint: did not improve {self.monitor} "
                    f"(current={current}, best={self.best}); skipping"
                )
