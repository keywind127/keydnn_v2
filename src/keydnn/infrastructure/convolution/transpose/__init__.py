"""
Conv2D transpose (deconvolution) public exports.

This module exposes the public API for transposed 2D convolution
(also known as deconvolution or fractionally strided convolution).
It re-exports both the functional form and the module/layer form so
they can be imported from a single, stable location.

Exports
-------
- `Conv2dTransposeFn`: Low-level functional interface used by autograd.
- `Conv2dTranspose`: High-level module wrapping the transpose convolution
  operation with parameters and forward logic.

Design intent
-------------
- Keep internal implementation details private.
- Provide a clean, explicit public surface for convolution transpose APIs.
- Align naming and structure with other convolution modules.
"""

from ._conv2d_transpose_function import Conv2dTransposeFn
from ._conv2d_transpose_module import Conv2dTranspose

__all__ = [
    Conv2dTransposeFn.__name__,
    Conv2dTranspose.__name__,
]
