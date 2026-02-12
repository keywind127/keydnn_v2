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
    Mapping,
    Union,
)
from pathlib import Path
import json
import math
import sys

import numpy as np

from ..tensor._tensor import Tensor

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
from .callbacks import (
    CallbackList,
    Callback,
)


LossLike = Union[str, Callable[[Any, Any], Any]]
OptimizerLike = Union[str, Any]
MetricLike = Union[str, Callable[..., Any]]

ShapeLike = Union[Sequence[int], Tuple[int, ...]]


def _normalize_key(s: str) -> str:
    return s.strip().lower().replace("-", "").replace("_", "")


def _num_batches_for_epoch(
    x: Any,
    y: Optional[Any],
    *,
    batch_size: int,
) -> Optional[int]:
    """
    Best-effort compute number of batches in the epoch.

    Returns None if unknown (e.g., generator with no __len__).
    """
    if y is not None:
        # Dataset input (x, y) assumed array-like with shape[0]
        n = getattr(x, "shape", None)
        if isinstance(n, tuple) and len(n) >= 1 and isinstance(n[0], int):
            total = n[0]
        else:
            # fallback to len(x) if available
            try:
                total = len(x)
            except Exception:
                return None
        return int(math.ceil(total / float(batch_size)))

    # Iterable-of-batches input
    try:
        return int(len(x))  # type: ignore[arg-type]
    except Exception:
        return None


def _render_progress_bar(
    *,
    epoch_idx: int,
    epochs: int,
    batch_idx: int,
    total_batches: Optional[int],
    logs_left: Optional[Dict[str, float]],
    logs_right: Optional[Dict[str, float]] = None,
    width: int = 30,
) -> str:
    """
    Render a single-line progress bar string for the current epoch/batch.
    """
    prefix = f"Epoch {epoch_idx + 1}/{epochs}"

    parts: list[str] = []

    if total_batches is None or total_batches <= 0:
        parts.append(prefix)
        parts.append(f"{batch_idx} batches")
    else:
        done = min(batch_idx, total_batches)
        frac = done / float(total_batches)
        pct = int(round(frac * 100.0))

        filled = int(round(frac * width))
        filled = max(0, min(width, filled))
        bar = "=" * filled + "." * (width - filled)

        parts.append(f"{prefix} [{bar}] {done}/{total_batches} ({pct:3d}%)")

    # left logs: training metrics
    if logs_left:
        for k, v in logs_left.items():
            parts.append(f"{k}: {v:.6f}")

    # right logs: validation metrics (val_*)
    if logs_right:
        for k, v in logs_right.items():
            parts.append(f"{k}: {v:.6f}")

    return " - ".join(parts)


def _resolve_loss(loss: Any) -> Callable[[Any, Any], Any]:
    """
    Resolve loss argument into a callable loss(y_pred, y_true) -> scalar Tensor.

    Supported string aliases:
    - "mse": mean((pred - target)^2)
    - "sse": sum((pred - target)^2)
    - "bce": mean(-[y*log(p) + (1-y)*log(1-p)])
    - "cce": -(sum(y*log(p))) / N   (N=batch size)
    """
    if callable(loss):
        return loss

    if not isinstance(loss, str):
        raise TypeError(
            "loss must be a callable loss(y_pred, y_true) or a string like "
            "'mse'/'sse'/'bce'/'cce'."
        )

    key = loss.strip().lower()

    if key == "mse":

        from ..losses._functions import MSEFn

        return MSEFn.as_loss()

    if key == "sse":

        from ..losses._functions import SSEFn

        return SSEFn.as_loss()

    if key in ("bce", "binary_crossentropy", "binarycrossentropy"):

        from ..losses._functions import BinaryCrossEntropyFn

        return BinaryCrossEntropyFn.as_loss()

    if key in ("cce", "categorical_crossentropy", "categoricalcrossentropy"):

        from ..losses._functions import CategoricalCrossEntropyFn

        return CategoricalCrossEntropyFn.as_loss()

    raise ValueError(
        f"Unknown loss {loss!r}. Supported: 'mse', 'sse', 'bce', 'cce' "
        "or a callable loss(y_pred, y_true)."
    )


def _resolve_optimizer(
    model: "Model",
    optimizer: OptimizerLike,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Resolve optimizer spec into an optimizer instance.
    Supports:
      - instance (returned as-is)
      - string names: "sgd", "adam"  -> instantiated with model.parameters()
    """
    if not isinstance(optimizer, str):
        return optimizer

    key = _normalize_key(optimizer)
    kwargs = dict(optimizer_kwargs or {})

    # defaults (so user can just pass "sgd" without kwargs)
    if "lr" not in kwargs and "learning_rate" not in kwargs:
        kwargs["lr"] = 1.0

    if key == "sgd":
        from ...infrastructure.optimizers._sgd import SGD  # type: ignore

        return SGD(model.parameters(), **kwargs)

    if key == "adam":
        from ...infrastructure.optimizers._adam import Adam  # type: ignore

        return Adam(model.parameters(), **kwargs)

    raise ValueError(
        f"Unknown optimizer {optimizer!r}. Supported: 'sgd', 'adam' or an optimizer instance."
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
    n = x.shape[0]
    if y.shape[0] != n:
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


def _to_numpy_any(x: Any):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        raise

    if hasattr(x, "to_numpy") and callable(getattr(x, "to_numpy")):
        return np.asarray(x.to_numpy())
    return np.asarray(x)


def _accuracy_metric(y_true: Any, y_pred: Any) -> float:
    """
    Compute accuracy for common KeyDNN training setups.

    Rules (best-effort):
    - If y_pred last dim > 1: multiclass -> argmax over last dim
      - If y_true has same shape as y_pred: assume one-hot -> argmax
      - Else assume integer labels
    - Else: binary -> threshold y_pred >= 0.5, compare against y_true in {0,1}
    """
    import numpy as np

    yt = _to_numpy_any(y_true).astype(np.float32, copy=False)
    yp = _to_numpy_any(y_pred).astype(np.float32, copy=False)

    # Flatten common (N,1) -> (N,)
    if yt.ndim >= 2 and yt.shape[-1] == 1:
        yt2 = yt.reshape(-1)
    else:
        yt2 = yt

    if yp.ndim >= 2 and yp.shape[-1] == 1:
        yp2 = yp.reshape(-1)
    else:
        yp2 = yp

    # Multiclass: (N, C)
    if yp.ndim >= 2 and yp.shape[-1] > 1:
        pred_cls = np.argmax(yp, axis=-1)

        if yt.shape == yp.shape:
            true_cls = np.argmax(yt, axis=-1)
        else:
            true_cls = yt.astype(np.int64, copy=False)
            if true_cls.ndim > 1:
                true_cls = true_cls.reshape(-1)

        pred_cls = pred_cls.reshape(-1)
        true_cls = true_cls.reshape(-1)

        n = min(pred_cls.size, true_cls.size)
        if n == 0:
            return 0.0
        return float((pred_cls[:n] == true_cls[:n]).mean())

    # Binary
    pred_bin = (yp2 >= 0.5).astype(np.float32, copy=False)
    true_bin = (yt2 >= 0.5).astype(np.float32, copy=False)

    pred_bin = pred_bin.reshape(-1)
    true_bin = true_bin.reshape(-1)

    n = min(pred_bin.size, true_bin.size)
    if n == 0:
        return 0.0
    return float((pred_bin[:n] == true_bin[:n]).mean())


def _resolve_metrics_and_names(
    metrics: Optional[Sequence[MetricLike]],
    metric_names: Optional[Sequence[str]],
) -> Tuple[Optional[Sequence[Callable[..., Any]]], Optional[Sequence[str]]]:
    """
    Resolve metrics spec to callables + names.

    Supports:
    - callable metrics
    - string metrics: "acc" / "accuracy"

    If `metric_names` is None, names are inferred:
    - string metric -> the string itself (normalized display kept original)
    - callable -> callable.__name__ or "metric_{i}"
    """
    if metrics is None:
        return None, metric_names

    resolved: list[Callable[..., Any]] = []
    names: list[str] = []

    if metric_names is not None and len(metric_names) != len(metrics):
        raise ValueError("metric_names must have the same length as metrics")

    for i, m in enumerate(metrics):
        # name selection
        if metric_names is not None:
            name = str(metric_names[i])
        else:
            name = (
                str(m)
                if isinstance(m, str)
                else (getattr(m, "__name__", None) or f"metric_{i}")
            )

        # metric resolution
        if isinstance(m, str):
            key = _normalize_key(m)
            if key in ("acc", "accuracy"):
                resolved.append(_accuracy_metric)
                names.append(name)
                continue
            raise ValueError(
                f"Unknown metric {m!r}. Supported: 'acc'/'accuracy' or a callable metric(y_true, y_pred)."
            )

        resolved.append(m)
        names.append(name)

    return resolved, names


def _resolve_metric(metric: Any) -> Callable[[Any, Any], Any]:
    """
    Resolve a metric specification into a callable.

    Supported forms
    ---------------
    - callable: returned as-is
    - string shortcuts:
        - "acc", "accuracy" -> binary accuracy (threshold=0.5)

    Returns
    -------
    Callable[[y_true, y_pred], scalar]
    """
    if callable(metric):
        return metric

    if not isinstance(metric, str):
        raise TypeError(
            "Metric must be a callable or string identifier "
            f"(got {type(metric).__name__})"
        )

    key = metric.lower()

    if key in {"acc", "accuracy"}:

        def _accuracy(y_true, y_pred):
            # Tensor-friendly, CPU/CUDA safe
            yp = y_pred.to_numpy()
            yt = y_true.to_numpy()

            import numpy as np

            yp = np.asarray(yp, dtype=np.float32)
            yt = np.asarray(yt, dtype=np.float32)

            y_hat = (yp >= 0.5).astype(np.float32)
            return float((y_hat == yt).mean())

        _accuracy.__name__ = "acc"
        return _accuracy

    raise ValueError(
        f"Unknown metric {metric!r}. Supported: 'acc' or a callable metric."
    )


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
        # "Built" means: all lazy layers have materialized parameters and
        # the optimizer can safely be created from model.parameters().
        self._is_built: bool = False
        # Internal flag used only to allow build() to run forward once.
        self._building: bool = False

    @property
    def is_built(self) -> bool:
        """
        Return whether the model has been built (lazy layers materialized).

        A model is considered built once `build()` has successfully executed
        a forward pass on a representative input and all lazy layers have
        created their parameters.

        Returns
        -------
        bool
            True if built, else False.
        """
        return bool(getattr(self, "_is_built", False))

    def _make_dummy_input(
        self, shape: Tuple[int, ...], *, device=None, dtype=np.float32
    ) -> Tensor:
        """
        Create a dummy input tensor for model building.

        This helper is used by `Model.build()` when the user provides an input
        *shape* instead of a real Tensor. It constructs a zero-filled Tensor with
        the requested shape, device, and dtype, sufficient to trigger a forward
        pass and materialize all lazy modules.

        Device resolution follows this order:
        1. Explicit `device` argument (if provided)
        2. `self.device` attribute on the model (if present)
        3. CPU fallback (`Device("cpu")`)

        Parameters
        ----------
        shape : Tuple[int, ...]
            Full input shape, including batch dimension
            (e.g., `(1, 3, 224, 224)`).
        device : optional
            Device on which to create the dummy Tensor. If None, inferred from the
            model or defaulted to CPU.
        dtype : Any, optional
            Data type of the dummy Tensor. Defaults to `np.float32`.

        Returns
        -------
        Tensor
            A zero-initialized Tensor suitable for triggering model build.

        Notes
        -----
        - The contents of the tensor are not semantically meaningful.
        - This method exists purely to support lazy initialization paths.
        """

        # Prefer: infer device from model parameters/modules if possible
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            # or fallback to CPU explicitly
            from ...domain.device._device import Device

            device = Device("cpu")

        # Create Tensor without needing numpy data
        t = Tensor(
            shape=tuple(shape),
            device=device,
            dtype=dtype,
            init_zeros=True,
        )
        return t

    def build(
        self,
        x: Union[Tensor, ShapeLike],
        *,
        device: Optional[Any] = None,
        dtype: Any = np.float32,
    ) -> None:
        """
        Build (materialize) the model using a representative input.

        This method performs a single forward pass to force initialization of all
        lazy modules (e.g., `Dense` inferring `in_features` and allocating
        parameters). After `build()`, calls to `model.parameters()` will return a
        complete and stable parameter set suitable for optimizer construction.

        The build input may be provided either as:
        - a real `Tensor`, or
        - a shape-like object (tuple/list), in which case a dummy Tensor is
        internally constructed.

        Parameters
        ----------
        x : Tensor or ShapeLike
            Representative model input.
            - If a `Tensor`, it is used directly.
            - If a shape (e.g., `(1, 784)`), a zero-filled Tensor is created
            internally to trigger the forward pass.
        device : optional
            Device to use when constructing a dummy Tensor from a shape.
            Ignored if `x` is already a Tensor.
        dtype : Any, optional
            Data type to use for a dummy Tensor created from a shape.
            Ignored if `x` is already a Tensor.

        Raises
        ------
        TypeError
            If `x` is neither a Tensor nor a valid shape-like object.
        RuntimeError
            If the forward pass fails during model building.

        Notes
        -----
        - Calling `build()` multiple times is safe; subsequent calls are no-ops.
        - This method must be called before inference, training, or optimizer
        creation when the model contains lazy layers.
        """

        if isinstance(x, (list, tuple)):
            x = self._make_dummy_input(
                shape=tuple(x),
                device=device,
                dtype=dtype,
            )

        if not isinstance(x, Tensor):
            raise TypeError("Model.build(x) expects x to be a Tensor")

        # If already built, no-op.
        if self.is_built:
            return

        self._building = True
        try:
            # Run one forward pass. We call `self(x)` so we follow the normal
            # call path, but __call__ will allow execution while _building=True.
            _ = self.forward(x, skip_norm=True)
        except Exception as e:
            # Keep state as unbuilt on failure.
            self._is_built = False
            raise RuntimeError(f"Model.build() failed during forward: {e!r}") from e
        finally:
            self._building = False

        self._is_built = True

    def _assert_built(self, where: str) -> None:
        """
        Raise a clear error if the model has not been built.

        Parameters
        ----------
        where : str
            Name of the public API that requires a built model.
        """
        if not self.is_built:
            raise RuntimeError(
                f"{where} requires a built model, but this model is not built yet. "
                "Call `model.build(x_sample)` (e.g., `model.build(x[:1])`) before "
                "calling training/inference APIs or creating an optimizer."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Invoke the model.

        This override enforces that the model must be built before any normal
        forward usage, except while inside `build()` itself.
        """
        if not getattr(self, "_building", False) and not self.is_built:
            raise RuntimeError(
                "Model is not built yet. Call `model.build(x_sample)` before calling the model."
            )
        return super().__call__(*args, **kwargs)

    def predict(self, x: Tensor, *, requires_grad: bool = False):
        """
        Perform an inference-style forward pass.

        This method invokes `forward()` directly and optionally switches the model
        into evaluation mode if such a mode is supported.

        Parameters
        ----------
        x : Tensor
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

        self._assert_built("Model.predict")

        if not isinstance(x, Tensor):
            raise TypeError("Input x must be a Tensor")

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

        payload = self.to_json_payload()
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
        payload: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

        fmt = payload.get("format")
        if fmt != "keydnn.json.ckpt.v1":
            raise ValueError(f"Unsupported checkpoint format: {fmt!r}")

        model = module_from_config(payload["arch"])
        load_state_payload_(model, payload["state"])

        setattr(model, "_is_built", True)

        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, expected {cls.__name__}."
            )

        return model

    def train_on_batch(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        *,
        loss: Any,
        optimizer: Any,
        metrics: Optional[Sequence[Any]] = None,
        metric_names: Optional[Sequence[str]] = None,
        zero_grad: bool = True,
        backward: bool = True,
        step: bool = True,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Run a single training step on one mini-batch.

        Supports string shortcuts for:
        - loss: "mse", "sse", "bce", "cce"
        - optimizer: "sgd", "adam"
        - metrics: "acc"

        See Model.fit() for full semantics.
        """

        self._assert_built("Model.train_on_batch")

        if isinstance(x_batch, np.ndarray):
            raise TypeError("x_batch must be a Tensor")
        if isinstance(y_batch, np.ndarray):
            raise TypeError("y_batch must be a Tensor")

        # ------------------------------------------------------------------
        # Resolve loss / optimizer (shared with fit)
        # ------------------------------------------------------------------
        loss_fn = _resolve_loss(loss)  # type: ignore[name-defined]
        opt = _resolve_optimizer(
            self, optimizer, optimizer_kwargs=optimizer_kwargs
        )  # type: ignore[name-defined]

        # ------------------------------------------------------------------
        # Resolve metrics (NEW)
        # ------------------------------------------------------------------
        resolved_metrics: Optional[Sequence[Callable[..., Any]]] = None
        resolved_names: Optional[Sequence[str]] = None

        if metrics:
            resolved_metrics = []
            resolved_names = []

            if metric_names is not None and len(metric_names) != len(metrics):
                raise ValueError("metric_names must have the same length as metrics")

            for i, m in enumerate(metrics):
                fn = _resolve_metric(m)
                resolved_metrics.append(fn)

                if metric_names is not None:
                    resolved_names.append(metric_names[i])
                else:
                    resolved_names.append(getattr(fn, "__name__", f"metric_{i}"))

        # ------------------------------------------------------------------
        # Training mode
        # ------------------------------------------------------------------
        train_fn = getattr(self, "train", None)
        if callable(train_fn):
            train_fn()

        if zero_grad:
            opt_zero = getattr(opt, "zero_grad", None)
            if callable(opt_zero):
                opt_zero()
            else:
                mdl_zero = getattr(self, "zero_grad", None)
                if callable(mdl_zero):
                    mdl_zero()

        # ------------------------------------------------------------------
        # Forward + loss
        # ------------------------------------------------------------------
        y_pred = self(x_batch)
        loss_tensor = loss_fn(y_pred, y_batch)
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
            opt_step = getattr(opt, "step", None)
            if callable(opt_step):
                opt_step()

        # ------------------------------------------------------------------
        # Logs
        # ------------------------------------------------------------------
        logs: Dict[str, float] = {"loss": float(loss_value)}

        if resolved_metrics:
            for name, fn in zip(resolved_names, resolved_metrics):
                mv = _call_metric(fn, y_batch, y_pred)
                logs[name] = _to_float_scalar(mv)

        return logs

    def fit(
        self,
        x: Union[Tensor, Iterable[Tuple[Tensor, Tensor]]],
        y: Optional[Tensor] = None,
        *,
        loss: LossLike,
        optimizer: OptimizerLike,
        metrics: Optional[Sequence[MetricLike]] = None,
        metric_names: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        epochs: int = 1,
        shuffle: bool = True,
        verbose: int = 1,
        validation_data: Optional[Tuple[Any, Any]] = None,
        callbacks: Optional[Sequence["Callback"]] = None,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
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
        x : Union[Tensor, Iterable[Tuple[Tensor, Tensor]]]
            Dataset inputs, or an iterable yielding `(x_batch, y_batch)` tuples.
        y : Optional[Tensor], optional
            Dataset targets. Must be provided for dataset inputs; must be `None`
            for iterable-of-batches inputs.
        loss : LossLike
            Callable producing a scalar loss: `loss(y_pred, y_true)`.
        optimizer : OptimizerLike
            Optimizer-like object, expected to expose `zero_grad()` and `step()`.
        metrics : MetricLike, optional
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

        self._assert_built("Model.fit")

        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if y is not None:
            if isinstance(x, np.ndarray):
                raise TypeError("x must be a Tensor when y is provided")
            if isinstance(y, np.ndarray):
                raise TypeError("y must be a Tensor when provided")

        loss_fn = _resolve_loss(loss)
        opt_obj = _resolve_optimizer(self, optimizer, optimizer_kwargs)

        metrics_fn, metric_names2 = _resolve_metrics_and_names(metrics, metric_names)

        cb_list = CallbackList(callbacks)
        cb_list.set_model(self)
        cb_list.on_train_begin(logs={})

        hist = History()

        def _eval_epoch(xv: Any, yv: Any) -> Dict[str, float]:
            eval_fn = getattr(self, "eval", None)
            train_fn = getattr(self, "train", None)
            was_training = getattr(self, "training", None)

            if callable(eval_fn):
                eval_fn()

            sums: Dict[str, float] = {}
            count: Dict[str, int] = {}

            for xb, yb in _iter_minibatches_xy(
                xv, yv, batch_size=batch_size, shuffle=False
            ):
                y_pred = self(xb)
                loss_tensor = loss_fn(y_pred, yb)
                logs_b: Dict[str, float] = {"loss": _to_float_scalar(loss_tensor)}

                if metrics_fn:
                    for i, m in enumerate(metrics_fn):
                        name = (
                            metric_names2[i]
                            if metric_names2 is not None
                            else (getattr(m, "__name__", None) or f"metric_{i}")
                        )
                        mv = _call_metric(m, yb, y_pred)
                        logs_b[str(name)] = _to_float_scalar(mv)

                bs = _batch_size_of(xb)
                for k, v in logs_b.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs
                    count[k] = count.get(k, 0) + bs

            out: Dict[str, float] = {}
            for k, s in sums.items():
                denom = float(count.get(k, 0) or 1)
                out[f"val_{k}"] = s / denom

            if callable(train_fn) and was_training is True:
                train_fn()
            return out

        def _epoch_batches() -> Iterator[Tuple[Any, Any]]:
            if y is None:
                if not isinstance(x, Iterable):
                    raise TypeError(
                        "If y is None, x must be an iterable of (x_batch, y_batch)"
                    )
                for b in x:
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
            cb_list.on_epoch_begin(epoch_idx, logs={})

            sums: Dict[str, float] = {}
            count: Dict[str, int] = {}
            seen = 0

            total_batches = _num_batches_for_epoch(x, y, batch_size=batch_size)

            batch_idx = 0
            last_logs: Optional[Dict[str, float]] = None

            for xb, yb in _epoch_batches():
                cb_list.on_batch_begin(batch_idx, logs={})

                logs = self.train_on_batch(
                    xb,
                    yb,
                    loss=loss_fn,
                    optimizer=opt_obj,
                    metrics=metrics_fn,
                    metric_names=metric_names2,
                )

                cb_list.on_batch_end(batch_idx, logs=logs)

                bs = _batch_size_of(xb)
                seen += bs
                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * bs
                    count[k] = count.get(k, 0) + bs

                batch_idx += 1
                # last_logs = logs

                # ------------------------------
                # Progress bar (running averages like Keras)
                # ------------------------------
                if verbose:
                    # weighted running average over seen samples
                    avg_logs: Dict[str, float] = {}
                    for k in sums.keys():
                        denom = float(count.get(k, 0) or 1)
                        avg_logs[k] = float(sums[k]) / denom

                    line = _render_progress_bar(
                        epoch_idx=epoch_idx,
                        epochs=epochs,
                        batch_idx=batch_idx,
                        total_batches=total_batches,
                        logs_left=avg_logs,  # <- cumulative avg (smooth)
                        logs_right=None,
                    )
                    sys.stdout.write("\r" + line)
                    sys.stdout.flush()

            # Build epoch (train) averages
            epoch_logs: Dict[str, float] = {}
            for k, s in sums.items():
                denom = float(count.get(k, 0) or 1)
                epoch_logs[k] = s / denom

            # Validation averages (val_*)
            if validation_data is not None:
                xv, yv = validation_data
                epoch_logs.update(_eval_epoch(xv, yv))

            # Keras-like: re-render final progress line at 100% including val_* metrics
            if verbose:
                train_side = {
                    k: v for k, v in epoch_logs.items() if not str(k).startswith("val_")
                }
                val_side = {
                    k: v for k, v in epoch_logs.items() if str(k).startswith("val_")
                }

                final_line = _render_progress_bar(
                    epoch_idx=epoch_idx,
                    epochs=epochs,
                    batch_idx=total_batches if total_batches is not None else batch_idx,
                    total_batches=total_batches,
                    logs_left=train_side,
                    logs_right=val_side,
                )
                sys.stdout.write("\r" + final_line + " " * 10 + "\n")
                sys.stdout.flush()

            # History + callbacks
            hist.append_epoch(epoch_idx, epoch_logs)
            cb_list.on_epoch_end(epoch_idx, logs=epoch_logs)

            if verbose >= 2:
                parts = [f"Epoch {epoch_idx + 1}/{epochs}"]
                for k, v in epoch_logs.items():
                    parts.append(f"{k}: {v:.6f}")
                parts.append(f"seen: {seen}")
                print(" - ".join(parts))

            if cb_list.stop_training:
                break

        cb_list.on_train_end(logs=hist.last())
        return hist

    def to_json_payload(self) -> Dict[str, Any]:
        """
        Serialize model architecture and weights into a JSON-serializable payload.

        This is the in-memory counterpart to `save_json(path)` and is useful for:
        - callback-based checkpointing without filesystem writes
        - snapshotting best weights for early stopping restore
        - programmatic checkpoint transport (e.g., RPC, DB, etc.)

        Returns
        -------
        Dict[str, Any]
            JSON-serializable checkpoint payload with keys:
            - "format": str
            - "arch": dict
            - "state": dict

        Notes
        -----
        The payload is compatible with the on-disk format produced by `save_json()`.
        """
        return {
            "format": "keydnn.json.ckpt.v1",
            "arch": module_to_config(self),
            "state": extract_state_payload(self),
        }

    def from_json_payload_(self, payload: Dict[str, Any]) -> None:
        """
        Load weights in-place from a JSON checkpoint payload.

        This method restores only the parameter state into the current model
        instance. It is intended for in-memory restores (e.g., EarlyStopping).

        Parameters
        ----------
        payload : Dict[str, Any]
            A checkpoint payload produced by `to_json_payload()` or loaded from
            disk via `json.loads(...)`.

        Raises
        ------
        ValueError
            If the checkpoint format is unsupported.

        Notes
        -----
        - This method does not reconstruct the module graph.
        - It assumes the current model architecture matches the payload.
        - Shape mismatches are detected by `load_state_payload_()`.
        """
        fmt = payload.get("format")
        if fmt != "keydnn.json.ckpt.v1":
            raise ValueError(f"Unsupported checkpoint format: {fmt!r}")

        # Only restore weights into this instance.
        load_state_payload_(self, payload["state"])

        self._is_built = True
