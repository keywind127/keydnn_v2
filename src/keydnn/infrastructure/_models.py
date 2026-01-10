"""
Model and container implementations.

This module defines infrastructure-level model abstractions built on top of
`Module`, including:

- `Model`: a semantic alias for top-level neural networks
- `Sequential`: a container module that applies child modules in sequence

These classes provide a foundation for composing neural networks while
remaining agnostic to training loops, optimizers, and execution engines.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
from dataclasses import dataclass, field
from pathlib import Path
import json

from ._module import Module

from .module._serialization_core import (
    module_to_config,
    module_from_config,
    register_module,
)
from .module._serialization_weights import (
    extract_state_payload,
    load_state_payload_,
)


Number = Union[int, float]


def _to_float_scalar(x: Any) -> float:
    """
    Best-effort conversion of a scalar-like tensor/ndarray/python number into float.
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

    # Numpy scalar / array(1,)
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
    Best-effort batch size inference (defaults to 1 if unknown).
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
    Generic mini-batcher for array-like datasets supporting __len__ and __getitem__.
    Works for numpy arrays, python lists, and many tensor wrappers.

    Notes
    -----
    - If `shuffle=True`, we shuffle indices, not the underlying storage.
    - If x/y don't support len/getitem, this will raise TypeError.
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
    Call a metric callable in a forgiving way.

    Conventions supported
    ---------------------
    - metric(y_true, y_pred)   (Keras style)
    - metric(y_pred, y_true)   (some libs do this)
    """
    try:
        return metric(y_true, y_pred)
    except TypeError:
        return metric(y_pred, y_true)


@dataclass
class History:
    """
    Keras-like history object.

    Attributes
    ----------
    history : dict[str, list[float]]
        Mapping from metric name -> per-epoch values.
    epoch : list[int]
        Epoch indices (0-based) appended as training progresses.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)
    epoch: List[int] = field(default_factory=list)

    def _ensure_key(self, k: str) -> None:
        if k not in self.history:
            self.history[k] = []

    def append_epoch(self, epoch_idx: int, logs: Mapping[str, Number]) -> None:
        """
        Append aggregated epoch logs.
        """
        self.epoch.append(int(epoch_idx))
        for k, v in logs.items():
            self._ensure_key(k)
            self.history[k].append(float(v))

    def last(self) -> Dict[str, float]:
        """
        Return the latest epoch's logs.
        """
        out: Dict[str, float] = {}
        for k, vs in self.history.items():
            if vs:
                out[k] = float(vs[-1])
        return out


class Model(Module):
    """
    Base class for top-level neural network models.

    `Model` is a semantic specialization of `Module` intended to represent
    complete networks rather than individual layers. It preserves all core
    `Module` behavior (parameter registration, recursion, callable semantics)
    while serving as an extension point for higher-level training and inference
    utilities (e.g., `fit`, `evaluate`, `predict`, checkpointing).

    Notes
    -----
    - This class currently adds minimal functionality beyond `Module`.
    - Mode management (`train` / `eval`) is optional and only applied when
      supported by the underlying implementation.
    """

    def __init__(self) -> None:
        """
        Initialize a model with no additional state beyond `Module`.
        """
        super().__init__()

    def predict(self, x, *, requires_grad: bool = False):
        """
        Perform an inference-style forward pass.

        This method invokes `forward` directly and optionally switches the model
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
        Train for a single batch (Keras-like).

        Parameters
        ----------
        x_batch, y_batch
            One mini-batch of inputs and targets.
        loss
            Callable: loss(y_pred, y_true) -> scalar tensor/number.
        optimizer
            Optimizer-like object. Expected (duck-typed) methods:
                - zero_grad() (optional)
                - step() (optional)
        metrics
            Optional list of metric callables. Expected signature:
                metric(y_true, y_pred) -> scalar tensor/number
            (Also supports metric(y_pred, y_true) as a fallback.)
        metric_names
            Optional names matching `metrics`. If omitted, names are inferred from
            `metric.__name__` when available, else "metric_{i}".
        zero_grad, backward, step
            Control hooks for training steps.

        Returns
        -------
        dict[str, float]
            Batch logs, e.g. {"loss": 0.123, "accuracy": 0.98}
        """
        train_fn = getattr(self, "train", None)
        if callable(train_fn):
            train_fn()

        if zero_grad:
            # Prefer optimizer.zero_grad if available; fallback to model.zero_grad if present.
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
            # Loss object expected to have backward() (autograd convention)
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
                name = None
                if metric_names is not None:
                    name = metric_names[i]
                else:
                    name = getattr(m, "__name__", None) or f"metric_{i}"

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
        Keras-like training loop.

        Supported input forms
        ---------------------
        1) (x, y) array-like datasets (must support len() and __getitem__)
        2) x as an iterable yielding (x_batch, y_batch), with y=None

        Returns
        -------
        History
            Per-epoch aggregated metrics.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        hist = History()

        # Build epoch iterator
        def _epoch_batches() -> Iterator[Tuple[Any, Any]]:
            if y is None:
                # Expect x yields (xb, yb)
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
                # x/y dataset
                yield from _iter_minibatches_xy(
                    x, y, batch_size=batch_size, shuffle=shuffle
                )

        for epoch_idx in range(epochs):
            # Running sums for weighted average
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

            # Aggregate epoch means
            epoch_logs: Dict[str, float] = {}
            for k, s in sums.items():
                denom = float(count.get(k, 0) or 1)
                epoch_logs[k] = s / denom

            hist.append_epoch(epoch_idx, epoch_logs)

            if verbose:
                # Simple Keras-ish line: "Epoch 1/5 - loss: ... - acc: ..."
                parts = [f"Epoch {epoch_idx + 1}/{epochs}"]
                for k, v in epoch_logs.items():
                    parts.append(f"{k}: {v:.6f}")
                parts.append(f"seen: {seen}")
                print(" - ".join(parts))

        return hist


@register_module()
class Sequential(Model):
    """
    Sequential container module.

    Applies a sequence of child modules in order:

        y = L_n(...L_2(L_1(x)))

    This container is useful for constructing simple feedforward networks
    without explicitly defining a custom `forward` method.

    Key Features
    ------------
    - Deterministic layer ordering
    - Automatic submodule registration
    - Supports indexing, iteration, and dynamic extension via `add`
    """

    def __init__(self, *layers: Module) -> None:
        """
        Initialize a Sequential container.

        Parameters
        ----------
        *layers : Module
            Zero or more modules to be added to the container in order.
        """
        super().__init__()
        self._layers: List[Module] = []
        for layer in layers:
            self.add(layer)

    def get_config(self) -> dict[str, Any]:
        return {}

    def add(self, layer: Module, name: Optional[str] = None) -> None:
        """
        Append a module to the container.

        The module is registered as a child submodule, enabling recursive
        parameter discovery.

        Parameters
        ----------
        layer : Module
            The module to append.
        name : Optional[str], optional
            Explicit name for the submodule. If omitted, a numeric name
            ("0", "1", ...) is assigned automatically.

        Raises
        ------
        TypeError
            If `layer` is not an instance of `Module`.
        ValueError
            If the provided name conflicts with an existing submodule.
        """
        if not isinstance(layer, Module):
            raise TypeError(f"Sequential.add expects a Module, got: {type(layer)}")

        idx = len(self._layers)
        layer_name = name if name is not None else str(idx)

        if hasattr(self, "_modules") and layer_name in self._modules:
            raise ValueError(f"Duplicate layer name '{layer_name}' in Sequential.")

        self._layers.append(layer)

        # Register as child module for parameter recursion
        if hasattr(self, "_modules"):
            self._modules[layer_name] = layer
        else:
            # Fallback for minimal Module implementations
            setattr(self, layer_name, layer)

    def forward(self, x):
        """
        Apply all layers sequentially.

        Parameters
        ----------
        x : ITensor
            Input tensor to the first layer.

        Returns
        -------
        ITensor
            Output of the final layer.
        """
        out = x
        for layer in self._layers:
            out = layer(out)
        return out

    def __len__(self) -> int:
        """
        Return the number of layers in the container.
        """
        return len(self._layers)

    def __iter__(self) -> Iterator[Module]:
        """
        Iterate over contained layers in order.
        """
        return iter(self._layers)

    def __getitem__(self, idx: int) -> Module:
        """
        Retrieve a layer by index.

        Parameters
        ----------
        idx : int
            Index of the layer.

        Returns
        -------
        Module
            The requested layer.
        """
        return self._layers[idx]

    def layers(self) -> Tuple[Module, ...]:
        """
        Return all layers as an immutable tuple.

        Returns
        -------
        Tuple[Module, ...]
            Tuple of contained modules.
        """
        return tuple(self._layers)

    def summary(self) -> str:
        """
        Generate a lightweight textual summary of the container.

        Returns
        -------
        str
            Human-readable representation listing layer indices and types.

        Notes
        -----
        - No shape inference or parameter counting is performed.
        """
        lines = [f"{self.__class__.__name__}("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}): {layer.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)

    def predict(self, x, *, requires_grad: bool = False):
        """
        Perform an inference-style forward pass for the Sequential model.

        Parameters
        ----------
        x : ITensor
            Input tensor.
        requires_grad : bool, optional
            Placeholder for future gradient-control semantics.

        Returns
        -------
        ITensor
            Output of the sequential computation.
        """
        eval_fn = getattr(self, "eval", None)
        if callable(eval_fn):
            eval_fn()

        return self.forward(x)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Sequential":
        """
        Construct a Sequential container from config.

        Notes
        -----
        Children are attached later by the deserializer into `self._modules`.
        This method restores the ordered `_layers` view from `_modules` once present.
        """
        return cls()

    def _post_load(self) -> None:
        """
        Internal hook invoked by JSON deserialization after children are attached.

        Restores `_layers` ordering from `_modules` so that indexing, iteration,
        and forward() work correctly after load.
        """
        if not hasattr(self, "_modules") or not isinstance(self._modules, dict):
            return

        # Deterministic order: numeric names first ("0", "1", ...)
        def _key_order(k: str) -> tuple[int, int | str]:
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)

        self._layers = [
            self._modules[k] for k in sorted(self._modules.keys(), key=_key_order)
        ]
