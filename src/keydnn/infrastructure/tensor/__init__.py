# from ._core import TensorCoreMixin
# from ._factories import TensorFactoriesMixin
# from ._autograd_core import TensorAutogradCoreMixin
# from ._io_and_copy import TensorIOAndCopyMixin
# from ._op_helpers import TensorOpHelpersMixin
# from ._element_wise_ops import TensorElementwiseOpsMixin
# from ._reductions import TensorReductionsMixin
from ._shape_and_indexing import TensorShapeAndIndexingMixin


class _TensorAllMixin(
    # TensorCoreMixin,
    # TensorIOAndCopyMixin,
    # TensorFactoriesMixin,
    # TensorAutogradCoreMixin,
    # TensorOpHelpersMixin,
    # TensorElementwiseOpsMixin,
    # TensorReductionsMixin,
    TensorShapeAndIndexingMixin,
):
    pass


__all__ = [_TensorAllMixin.__name__]
