"""
Optimizer implementations public exports.

This module exposes concrete optimizer implementations that conform to the
framework's optimizer interface. These optimizers are responsible for updating
model parameters during training based on accumulated gradients.

Exports
-------
- `Adam`: Adaptive Moment Estimation optimizer.
- `SGD`: Stochastic Gradient Descent optimizer.

Design intent
-------------
- Provide a stable public import path for optimizer implementations.
- Keep individual optimizer logic encapsulated in private modules.
- Align optimizer naming and behavior with common deep learning frameworks.
"""

from ._adam import Adam
from ._sgd import SGD


__all__ = [
    Adam.__name__,
    SGD.__name__,
]
