"""
2D convolution public exports.

This module defines the public API surface for 2D convolution operations.
It re-exports both the standard convolution and transposed convolution
(Conv2D transpose / deconvolution) interfaces so they can be imported from
a single, consistent namespace.

Exports
-------
Standard convolution
~~~~~~~~~~~~~~~~~~~~
- `Conv2dFn`: Low-level functional convolution operator.
- `Conv2d`: High-level convolution module with parameters.

Transposed convolution
~~~~~~~~~~~~~~~~~~~~~~
- `Conv2dTransposeFn`: Functional transposed convolution operator.
- `Conv2dTranspose`: Module form of transposed convolution.

Design intent
-------------
- Provide a unified import location for all 2D convolution variants.
- Keep internal implementation modules private.
- Align with PyTorch/Keras-style API organization.
"""

from ._conv2d_function import Conv2dFn
from ._conv2d_module import Conv2d
from .transpose import *

__all__ = [
    Conv2d.__name__,
    Conv2dFn.__name__,
    Conv2dTranspose.__name__,
    Conv2dTransposeFn.__name__,
]
