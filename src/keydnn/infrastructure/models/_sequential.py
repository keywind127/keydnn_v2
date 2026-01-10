"""
Sequential container module.

This module defines `Sequential`, a simple container that composes a list of
child `Module` objects into a single `Model` by applying them in order:

    y = L_n(...L_2(L_1(x)))

`Sequential` is intended for straightforward feedforward architectures where a
custom `forward()` is unnecessary. It supports:

- deterministic layer ordering
- automatic submodule registration for parameter recursion/serialization
- indexing/iteration and dynamic extension via `add`
- JSON serialization hooks via `register_module` and `_post_load`

Notes
-----
- This container is infrastructure-level: it does not implement training logic
  itself; training utilities live on `Model` (e.g., `fit`, `train_on_batch`).
- The `_layers` list is the authoritative ordered view used by `forward()`.
  During deserialization, `_post_load()` rebuilds `_layers` from `_modules`.
"""

from typing import Any, Iterator, List, Tuple, Optional

from ..module._serialization_core import register_module
from .._module import Module
from ._models import Model


@register_module()
class Sequential(Model):
    """
    Sequential container model.

    `Sequential` composes multiple `Module` instances and applies them
    sequentially during `forward()`:

        out = layer_n(...layer_2(layer_1(x)))

    This container is useful for building simple pipelines such as
    MLPs and CNN feature stacks without writing a custom `forward`.

    Key Features
    ------------
    - Deterministic layer ordering via an internal `_layers` list
    - Automatic submodule registration into `_modules` for recursion and save/load
    - Supports indexing, iteration, and dynamic extension via `add`
    - Provides a lightweight `summary()` for quick inspection
    """

    def __init__(self, *layers: Module) -> None:
        """
        Initialize a `Sequential` container.

        Parameters
        ----------
        *layers : Module
            Zero or more child modules to be appended in order.

        Notes
        -----
        Layers are registered immediately via `add()` so they participate in
        parameter recursion and serialization.
        """
        super().__init__()
        self._layers: List[Module] = []
        for layer in layers:
            self.add(layer)

    def get_config(self) -> dict[str, Any]:
        """
        Return the (de)serialization config for this container.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary used by the JSON serializer.

        Notes
        -----
        `Sequential` stores its children separately via the module tree, so
        no additional config is required here.
        """
        return {}

    def add(self, layer: Module, name: Optional[str] = None) -> None:
        """
        Append a module to the container and register it as a submodule.

        Parameters
        ----------
        layer : Module
            The module to append.
        name : Optional[str], optional
            Explicit registration name. If omitted, a numeric name ("0", "1", ...)
            is assigned based on insertion order.

        Raises
        ------
        TypeError
            If `layer` is not an instance of `Module`.
        ValueError
            If the provided `name` conflicts with an existing submodule.

        Notes
        -----
        - `_layers` preserves the execution order used by `forward()`.
        - `_modules` enables parameter discovery, recursion, and serialization.
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

        Notes
        -----
        This method uses the ordered `_layers` list (not `_modules`) to ensure
        deterministic execution order.
        """
        out = x
        for layer in self._layers:
            out = layer(out)
        return out

    def __len__(self) -> int:
        """
        Return the number of layers in the container.

        Returns
        -------
        int
            Number of child modules tracked in `_layers`.
        """
        return len(self._layers)

    def __iter__(self) -> Iterator[Module]:
        """
        Iterate over contained layers in order.

        Returns
        -------
        Iterator[Module]
            Iterator over child modules.
        """
        return iter(self._layers)

    def __getitem__(self, idx: int) -> Module:
        """
        Retrieve a layer by index.

        Parameters
        ----------
        idx : int
            Index of the desired layer.

        Returns
        -------
        Module
            The requested layer.

        Raises
        ------
        IndexError
            If `idx` is out of range.
        """
        return self._layers[idx]

    def layers(self) -> Tuple[Module, ...]:
        """
        Return all layers as an immutable tuple.

        Returns
        -------
        Tuple[Module, ...]
            Tuple of contained modules in execution order.
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
        - This is intended for quick debugging/inspection only.
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
            Placeholder for future gradient-control semantics. Currently unused.

        Returns
        -------
        ITensor
            Output of the sequential computation.

        Notes
        -----
        If the model exposes `eval()`, this method switches to eval mode before
        running the forward pass.
        """
        eval_fn = getattr(self, "eval", None)
        if callable(eval_fn):
            eval_fn()

        return self.forward(x)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Sequential":
        """
        Construct a `Sequential` container from configuration.

        Parameters
        ----------
        cfg : dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        Sequential
            A newly constructed `Sequential` instance.

        Notes
        -----
        Child modules are attached later by the deserializer into `self._modules`.
        After attachment, `_post_load()` rebuilds the ordered `_layers` view.
        """
        _ = cfg  # config is currently empty; kept for API stability
        return cls()

    def _post_load(self) -> None:
        """
        Post-deserialization hook to restore execution order.

        This hook is invoked by JSON deserialization after children have been
        attached into `self._modules`. It rebuilds `_layers` from `_modules`
        so that indexing, iteration, and `forward()` work correctly after load.

        Notes
        -----
        Ordering rules:
        - Numeric names ("0", "1", ...) are sorted numerically first
        - Non-numeric names are sorted lexicographically afterward
        """
        if not hasattr(self, "_modules") or not isinstance(self._modules, dict):
            return

        def _key_order(k: str) -> tuple[int, int | str]:
            """
            Sort key that prioritizes numeric submodule names.

            Parameters
            ----------
            k : str
                Submodule key from `_modules`.

            Returns
            -------
            tuple[int, int | str]
                Sort key where numeric keys come first and are ordered numerically.
            """
            try:
                return (0, int(k))
            except ValueError:
                return (1, k)

        self._layers = [
            self._modules[k] for k in sorted(self._modules.keys(), key=_key_order)
        ]
