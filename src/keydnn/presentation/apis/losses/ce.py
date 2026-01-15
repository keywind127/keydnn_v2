from ....infrastructure.losses import (
    BinaryCrossEntropyFn,
    CategoricalCrossEntropyFn,
)
from ....infrastructure.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)

BCEFn = BinaryCrossEntropyFn
CCEFn = CategoricalCrossEntropyFn

BCE = BinaryCrossEntropy
CCE = CategoricalCrossEntropy

__all__ = [
    "BCE",
    "CCE",
    "BCEFn",
    "CCEFn",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "BinaryCrossEntropyFn",
    "CategoricalCrossEntropyFn",
]
