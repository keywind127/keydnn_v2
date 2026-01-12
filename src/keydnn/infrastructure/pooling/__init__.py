"""
2D pooling layer public exports.

This module defines the public API surface for 2D pooling operations commonly
used in convolutional neural networks. It re-exports both functional operators
and high-level module wrappers for average pooling, max pooling, and global
average pooling.

Exports
-------
Functional operators
~~~~~~~~~~~~~~~~~~~~
- `AvgPool2dFn`: Functional 2D average pooling operator.
- `MaxPool2dFn`: Functional 2D max pooling operator.
- `GlobalAvgPool2dFn`: Functional global average pooling operator.

Module wrappers
~~~~~~~~~~~~~~~
- `AvgPool2d`: Module form of 2D average pooling.
- `MaxPool2d`: Module form of 2D max pooling.
- `GlobalAvgPool2d`: Module form of global average pooling.

Design intent
-------------
- Provide a unified import location for all 2D pooling variants.
- Separate low-level functional ops from high-level modules.
- Align API structure with other convolutional and pooling layers.
"""

from ._pooling_function import AvgPool2dFn, MaxPool2dFn, GlobalAvgPool2dFn
from ._pooling_module import AvgPool2d, MaxPool2d, GlobalAvgPool2d


__all__ = [
    AvgPool2dFn.__name__,
    MaxPool2dFn.__name__,
    GlobalAvgPool2dFn.__name__,
    AvgPool2d.__name__,
    MaxPool2d.__name__,
    GlobalAvgPool2d.__name__,
]
