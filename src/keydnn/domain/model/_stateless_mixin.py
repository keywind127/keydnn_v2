"""
Stateless configuration mixin.

This module defines `StatelessConfigMixin`, a helper mixin for modules whose
behavior does not depend on any configurable hyperparameters.

It provides no-op JSON serialization and deserialization hooks, allowing
stateless layers to participate uniformly in model configuration export
and reconstruction workflows without introducing special cases.
"""

from typing import Any, Dict
from typing_extensions import Self


class StatelessConfigMixin:
    """
    Mixin providing configuration hooks for stateless modules.

    This mixin is intended for layers or components whose behavior is fully
    determined by their class definition and does not require any runtime
    configuration parameters.

    Examples include identity layers, fixed activations, or structural
    components that carry no tunable state.
    """

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration dictionary.

        For stateless modules, this method returns an empty dictionary,
        indicating that no parameters are required to reconstruct the object.

        Returns
        -------
        Dict[str, Any]
            An empty configuration dictionary.
        """
        return {}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> Self:
        """
        Reconstruct the module from a configuration dictionary.

        Since stateless modules do not require any configuration parameters,
        the provided configuration is ignored and a default instance of the
        class is returned.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary (unused).

        Returns
        -------
        StatelessConfigMixin
            A newly constructed instance of the module.
        """
        return cls()
