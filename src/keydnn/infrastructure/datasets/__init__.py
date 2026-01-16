# src/keydnn/infrastructure/datasets/__init__.py
from ._mnist import MNIST
from ._cifar import (
    CIFAR100,
    CIFAR10,
)

__all__ = [
    MNIST.__name__,
    CIFAR10.__name__,
    CIFAR100.__name__,
]
