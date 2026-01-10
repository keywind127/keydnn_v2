"""
High-level model utilities.

This module defines the infrastructure-level `Model` base class, which extends
the core `Module` abstraction with conveniences typically expected at the
"top-level network" boundary:

- Inference helpers (`predict`)
- Checkpoint serialization (`save_json`, `load_json`)
- Lightweight, Keras-like training helpers (`train_on_batch`, `fit`)

Important project note
----------------------
Historically, this file also hosted container and training-support utilities.
Those responsibilities have since been split into dedicated modules:

- `History` has been extracted into `._history` to keep metric tracking
  independent of `Model` and training loops.
- `Sequential` has been extracted into its own module to keep container
  composition separate from the core `Model` API surface.

As a result, this module now focuses on the `Model` abstraction and small
training-loop helpers only, while remaining agnostic to specific optimizers,
loss functions, tensor implementations, and execution engines.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Iterable,
    Iterator,
    Optional,
    Tuple,
)
from pathlib import Path
import json

from ._history import History

from .._module import Module

from ..module._serialization_core import (
    module_to_config,
    module_from_config,
)
from ..module._serialization_weights import (
    extract_state_payload,
    load_state_payload_,
)


def _to_float_scalar(x: Any) -> float:
    """
    Convert a scalar-like value to a Python `float`.

    This helper exists to normalize values returned by losses/metrics into a
    portable representation suitable for logging and for `History` storage.

    Supported inputs
    ----------------
    - Python numbers (`int`, `float`)
    - Objects exposing `.item()` (NumPy scalars, some tensor APIs)
    - KeyDNN tensors exposing `.to_numpy()`
    - NumPy scalars/arrays (best-effort)

    Parameters
    ----------
    x : Any
        Scalar-like object to convert.

    Returns
    -------
    float
        Converted scalar value.

    Notes
    -----
    This function is intentionally permissive to reduce friction across CPU/CUDA
    backends and different tensor implementations.
    """
    if isinstance(x, (int, float)):
        return float(x)

    # Common tensor conventions
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            pass

    # KeyDNN Tensor convention: to_numpy()
    to_numpy = getattr(x, "to_numpy", None)
    if callable(to_numpy):
        try:
            import numpy as np

            v = np.asarray(to_numpy())
            return float(v.reshape(-1)[0])
        except Exception:
            pass

    # NumPy scalar / array(1,)
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        if isinstance(x, np.generic):
            return float(x)
    except Exception:
        pass

    return float(x)


def _batch_size_of(xb: Any) -> int:
    """
    Infer batch size from a batch-like object.

    Parameters
    ----------
    xb : Any
        Batch input object. If it exposes a tuple `shape` with at least one
        dimension, `shape[0]` is interpreted as batch size.

    Returns
    -------
    int
        Best-effort batch size, defaulting to 1 if unknown.

    Notes
    -----
    This is used by `fit()` to compute weighted epoch averages when batches may
    have different sizes.
    """
    shape = getattr(xb, "shape", None)
    if shape is not None and isinstance(shape, tuple) and len(shape) >= 1:
        n0 = shape[0]
        if isinstance(n0, int) and n0 >= 1:
            return n0
    return 1


def _iter_minibatches_xy(
    x: Any,
    y: Any,
    *,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[Any, Any]]:
    """
    Yield mini-batches from array-like `(x, y)` datasets.

    This is a small utility used by `Model.fit()` for dataset inputs that support
    `__len__` and `__getitem__` (e.g., NumPy arrays, Python lists, many tensor
    wrappers).

    Parameters
    ----------
    x : Any
        Dataset inputs. Must support `len(x)` and indexing (`x[i]` or
        `x[list_of_indices]`).
    y : Any
        Dataset targets. Must support `len(y)` and indexing.
    batch_size : int
        Desired mini-batch size.
    shuffle : bool, optional
        If True, shuffle indices each epoch. Default is True.

    Yields
    ------
    Iterator[Tuple[Any, Any]]
        `(x_batch, y_batch)` pairs.

    Raises
    ------
    TypeError
        If `x` or `y` do not support `len()` / indexing.
    ValueError
        If `x` and `y` have different lengths.

    Notes
    -----
    - When `shuffle=True`, we shuffle indices, not the underlying storage.
    - If vectorized advanced indexing fails (e.g., `x[batch_ids]`), this
      function falls back to Python list-gathering.
    """
    n = len(x)
    if len(y) != n:
        raise ValueError(
            f"x and y must have same length, got len(x)={n}, len(y)={len(y)}"
        )

    idxs = list(range(n))
    if shuffle:
        import random

        random.shuffle(idxs)

    for start in range(0, n, batch_size):
        batch_ids = idxs[start : start + batch_size]

        # Try to slice efficiently if supported
        try:
            xb = x[batch_ids]  # type: ignore[index]
            yb = y[batch_ids]  # type: ignore[index]
            yield xb, yb
            continue
        except Exception:
            pass

        # Fallback: gather
        xb = [x[i] for i in batch_ids]
        yb = [y[i] for i in batch_ids]
        yield xb, yb


def _call_metric(metric: Callable[..., Any], y_true: Any, y_pred: Any) -> Any:
    """
    Invoke a metric callable using a forgiving argument convention.

    Conventions supported
    ---------------------
    - `metric(y_true, y_pred)` (Keras style)
    - `metric(y_pred, y_true)` (alternate style)

    Parameters
    ----------
    metric : Callable[..., Any]
        Metric function/callable.
    y_true : Any
        Ground-truth batch targets.
    y_pred : Any
        Model predictions for the batch.

    Returns
    -------
    Any
        Metric output, typically a scalar tensor/number.

    Notes
    -----
    This helper exists to reduce integration friction with user-defined metric
    callables that may swap argument order.
    """
    try:
        return metric(y_true, y_pred)
    except TypeError:
        return metric(y_pred, y_true)


class Model(Module):
    """
    Base class for top-level neural network models.

    `Model` is a semantic specialization of `Module` intended to represent
    complete networks rather than individual layers. It preserves all core
    `Module` behavior (parameter registration, recursion, callable semantics)
    while providing higher-level convenience methods:

    - `predict` for inference-style forward passes
    - `save_json` / `load_json` for architecture + weights checkpointing
    - `train_on_batch` and `fit` for lightweight training loops and metric logging

    Notes
    -----
    - This class remains optimizer-agnostic: optimizers are duck-typed and only
      expected to expose `zero_grad()` and `step()` when used by the training APIs.
    - Mode management (`train` / `eval`) is optional and only used when present.
    - `History` is imported from `._history` (extracted into a separate module).
    """

    def __init__(self) -> None:
        """
        Initialize a model with no additional state beyond `Module`.

        Notes
        -----
        Subclasses typically define parameters/modules during their own
        initialization; `Model` itself does not add extra fields.
        """
        super().__init__()

    def predict(self, x, *, requires_grad: bool = False):
        """
        Perform an inference-style forward pass.

        This method invokes `forward()` directly and optionally switches the model
        into evaluation mode if such a mode is supported.

        Parameters
        ----------
        x : ITensor
            Input tensor (or tensor-like object) for inference.
        requires_grad : bool, optional
            Placeholder for future gradient-control semantics. Currently unused.

        Returns
        -------
        ITensor
            Output produced by the model's forward computation.

        Notes
        -----
        - If the model implements `eval()` and `train()`, they may be used here.
        - Gradient suppression (`no_grad`) is not yet implemented.
        """
        eval_fn = getattr(self, "eval", None)
        train_fn = getattr(self, "train", None)
        was_training = getattr(self, "training", None)

        if callable(eval_fn):
            eval_fn()

        out = self.forward(x)

        if callable(train_fn) and was_training is True:
            train_fn()

        return out

    def save_json(self, path: str | Path) -> None:
        """
        Save model architecture and weights into a single JSON file.

        Parameters
        ----------
        path : str | Path
            Output JSON file path, e.g. "checkpoint.json".

        Format
        ------
        {
          "format": "keydnn.json.ckpt.v1",
          "arch": {...},
          "state": {
            "layer1.weight": {"b64": "...", "dtype": "<f4", "shape": [...], "order": "C"},
            ...
          }
        }

        Notes
        -----
        - Avoids pickle and HDF5 dependencies.
        - JSON file can get large; base64 adds ~33% size overhead.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "keydnn.json.ckpt.v1",
            "arch": module_to_config(self),
            "state": extract_state_payload(self),
        }

        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Model":
        """
        Load a model from a single JSON checkpoint created by `save_json()`.

        Parameters
        ----------
        path : str | Path
            Checkpoint JSON path.

        Returns
        -------
        Model
            Reconstructed model with weights loaded.

        Raises
        ------
        ValueError
            If the checkpoint format is unsupported.
        TypeError
            If the reconstructed object is not an instance of `cls`.
        """
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))

        fmt = payload.get("format")
        if fmt != "keydnn.json.ckpt.v1":
            raise ValueError(f"Unsupported checkpoint format: {fmt!r}")

        model = module_from_config(payload["arch"])
        load_state_payload_(model, payload["state"])

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected {cls.__name__}."
            )

        return model

    def train_on_batch(
        self,
        x_batch: Any,
        y_batch: Any,
        *,
        loss: Callable[[Any, Any], Any],
        optimizer: Any,
        metrics: Optional[Sequence[Callable[..., Any]]] = None,
        metric_names: Optional[Sequence[str]] = None,
        zero_grad: bool = True,
        backward: bool = True,
        step: bool = True,
    ) -> Dict[str, float]:
        """
        Run a single training step on one mini-batch.

        This method provides a Keras-like primitive that:
        - switches the model to training mode when available (`train()`)
        - computes predictions and loss
        - runs backpropagation (`loss.backward()`) when enabled
        - steps the optimizer (`optimizer.step()`) when enabled
        - returns a dictionary of scalar logs (loss + metrics)

        Parameters
        ----------
        x_batch, y_batch : Any
            One mini-batch of inputs and targets.
        loss : Callable[[Any, Any], Any]
            Callable producing a scalar loss: `loss(y_pred, y_true)`.
        optimizer : Any
            Optimizer-like object. Expected (duck-typed) methods:
            - `zero_grad()` (optional)
            - `step()` (optional)
        metrics : Optional[Sequence[Callable[..., Any]]], optional
            Optional metric callables. Expected signature:
            `metric(y_true, y_pred) -> scalar tensor/number`.
            A reversed argument order is also supported as a fallback.
        metric_names : Optional[Sequence[str]], optional
            Optional names matching `metrics`. If omitted, names are inferred from
            `metric.__name__` when available, else `"metric_{i}"`.
        zero_grad : bool, optional
            If True, clears gradients before the backward pass. Default is True.
        backward : bool, optional
            If True, calls `loss.backward()`. Default is True.
        step : bool, optional
            If True, calls `optimizer.step()`. Default is True.

        Returns
        -------
        Dict[str, float]
            Batch logs, e.g. `{"loss": 0.123, "accuracy": 0.98}`.

        Raises
        ------
        TypeError
            If the loss return value does not support `.backward()` while
            `backward=True`.
        ValueError
            If `metric_names` is provided but its length does not match `metrics`.

        Notes
        -----
        The returned logs are converted to Python floats using `_to_float_scalar`
        for consistent reporting across CPU/CUDA tensor backends.
        """
        train_fn = getattr(self, "train", None)
        if callable(train_fn):
            train_fn()

        if zero_grad:
            opt_zero = getattr(optimizer, "zero_grad", None)
            if callable(opt_zero):
                opt_zero()
            else:
                mdl_zero = getattr(self, "zero_grad", None)
                if callable(mdl_zero):
                    mdl_zero()

        y_pred = self(x_batch)
        loss_tensor = loss(y_pred, y_batch)
        loss_value = _to_float_scalar(loss_tensor)

        if backward:
            bw = getattr(loss_tensor, "backward", None)
            if not callable(bw):
                raise TypeError(
                    "Loss return value does not support backward(). "
                    "Expected a scalar Tensor-like object."
                )
            bw()

        if step:
            opt_step = getattr(optimizer, "step", None)
            if callable(opt_step):
                opt_step()

        logs: Dict[str, float] = {"loss": float(loss_value)}

        if metrics:
            if metric_names is not None and len(metric_names) != len(metrics):
                raise ValueError("metric_names must have the same length as metrics")

            for i, m in enumerate(metrics):
                name = (
                    metric_names[i]
                    if metric_names is not None
                    else (getattr(m, "__name__", None) or f"metric_{i}")
                )
                mv = _call_metric(m, y_batch, y_pred)
                logs[str(name)] = _to_float_scalar(mv)

        return logs

    def fit(
        self,
        x: Any,
        y: Optional[Any] = None,
        *,
        loss: Callable[[Any, Any], Any],
        optimizer: Any,
        metrics: Optional[Sequence[Callable[..., Any]]] = None,
        metric_names: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        epochs: int = 1,
        shuffle: bool = True,
        verbose: int = 1,
    ) -> History:
        """
        Train the model for a fixed number of epochs.

        This method provides a Keras-like training loop built on top of
        `train_on_batch()`. It aggregates batch logs into per-epoch metrics and
        records them in a `History` object.

        Supported input forms
        ---------------------
        1) `(x, y)` dataset:
           - `x` and `y` are array-like and support `len()` and indexing
           - batching/shuffling is handled internally via `_iter_minibatches_xy`
        2) Iterable-of-batches:
           - `y` is `None`
           - `x` is an iterable yielding `(x_batch, y_batch)` tuples

        Parameters
        ----------
        x : Any
            Dataset inputs, or an iterable yielding `(x_batch, y_batch)` tuples.
        y : Optional[Any], optional
            Dataset targets. Must be provided for dataset inputs; must be `None`
            for iterable-of-batches inputs.
        loss : Callable[[Any, Any], Any]
            Callable producing a scalar loss: `loss(y_pred, y_true)`.
        optimizer : Any
            Optimizer-like object, expected to expose `zero_grad()` and `step()`.
        metrics : Optional[Sequence[Callable[..., Any]]], optional
            Metric callables to compute per batch and aggregate per epoch.
        metric_names : Optional[Sequence[str]], optional
            Optional names matching `metrics`. If omitted, names are inferred.
        batch_size : int, optional
            Mini-batch size for dataset inputs. Default is 32.
        epochs : int, optional
            Number of epochs to train for. Default is 1.
        shuffle : bool, optional
            Whether to shuffle dataset inputs each epoch. Default is True.
        verbose : int, optional
            If non-zero, prints a simple epoch summary. Default is 1.

        Returns
        -------
        History
            A `History` instance (from `._history`) containing per-epoch metrics.

        Raises
        ------
        ValueError
            If `epochs < 1` or `batch_size < 1`.
        TypeError
            If `y is None` but `x` is not an iterable of `(x_batch, y_batch)`.

        Notes
        -----
        Per-epoch metric values are computed as weighted means over batches,
        using `_batch_size_of()` to determine the weight for each batch.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        hist = History()

        def _epoch_batches() -> Iterator[Tuple[Any, Any]]:
            """
            Yield `(x_batch, y_batch)` pairs for one epoch.

            Returns
            -------
            Iterator[Tuple[Any, Any]]
                Batch iterator for the current epoch.

            Notes
            -----
            - For `y is None`, the caller supplies the batch iterator directly.
            - For dataset inputs, batching is handled via `_iter_minibatches_xy`.
            """
            if y is None:
                if not isinstance(x, Iterable):
                    raise TypeError(
                        "If y is None, x must be an iterable of (x_batch, y_batch)"
                    )
                for b in x:  # type: ignore[assignment]
                    if not (isinstance(b, tuple) and len(b) == 2):
                        raise TypeError(
                            "Iterable x must yield (x_batch, y_batch) tuples"
                        )
                    yield b[0], b[1]
            else:
                yield from _iter_minibatches_xy(
                    x, y, batch_size=batch_size, shuffle=shuffle
                )

        for epoch_idx in range(epochs):
            sums: Dict[str, float] = {}
            count: Dict[str, int] = {}
            seen = 0

            for xb, yb in _epoch_batches():
                logs = self.train_on_batch(
                    xb,
                    yb,
                    loss=loss,
                    optimizer=optimizer,
                    metrics=metrics,
                    metric_names=metric_names,
                )

                bs = _batch_size_of(xb)
                seen += bs

                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs
                    count[k] = count.get(k, 0) + bs

            epoch_logs: Dict[str, float] = {}
            for k, s in sums.items():
                denom = float(count.get(k, 0) or 1)
                epoch_logs[k] = s / denom

            hist.append_epoch(epoch_idx, epoch_logs)

            if verbose:
                parts = [f"Epoch {epoch_idx + 1}/{epochs}"]
                for k, v in epoch_logs.items():
                    parts.append(f"{k}: {v:.6f}")
                parts.append(f"seen: {seen}")
                print(" - ".join(parts))

        return hist
