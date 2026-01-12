from ....infrastructure._losses import BinaryCrossEntropyFn, CategoricalCrossEntropyFn

BinaryCrossEntropy = BinaryCrossEntropyFn
CategoricalCrossEntropy = CategoricalCrossEntropyFn
BCE = BinaryCrossEntropyFn
CCE = CategoricalCrossEntropyFn

__all__ = [
    "BCE",
    "CCE",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "BinaryCrossEntropyFn",
    "CategoricalCrossEntropyFn",
]
