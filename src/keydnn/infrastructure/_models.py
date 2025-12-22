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

from typing import Iterator, List, Optional, Tuple, Any
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
