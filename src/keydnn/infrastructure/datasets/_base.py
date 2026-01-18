"""
This module defines a mixin class for verbosity control in dataset classes.
"""

from __future__ import annotations
from abc import ABC


class _VerboseMixin(ABC):
    """
    Mixin class to add verbosity control to dataset classes.
    """

    _verbose: bool = True

    @classmethod
    def set_verbose(cls, verbose: bool) -> None:
        """
        Set the verbosity level for the dataset class.

        Parameters
        ----------
        verbose : bool
            If True, enable verbose output; otherwise, disable it.
        """
        cls._verbose = bool(verbose)
