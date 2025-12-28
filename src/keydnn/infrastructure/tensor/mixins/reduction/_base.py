"""
Reduction mixin defining the public Tensor reduction API.

This module declares :class:`TensorMixinReduction`, an abstract mixin that
specifies the *interface and semantics* of common reduction operations
(`max`, `mean`, `sum`) on tensors.

The mixin itself does not implement any numerical logic. Instead, concrete
implementations are provided elsewhere and registered via a control-path
dispatch mechanism (e.g., CPU vs. CUDA backends). This separation allows the
Tensor class to expose a single, stable API while supporting multiple
device-specific execution strategies.
"""

from typing import Optional
from abc import ABC

from .....domain._tensor import ITensor


class TensorMixinReduction(ABC):
    """
    Abstract mixin defining reduction operations for tensors.

    This mixin specifies the method signatures, expected behavior, and
    mathematical semantics of reduction operations. Actual implementations
    are provided by backend-specific control paths (e.g., CPU, CUDA) and
    selected at runtime based on the tensor's device.

    Notes
    -----
    - Methods defined here should be treated as *pure interface declarations*.
    - No computation is performed in this class.
    - Backward rules described in the docstrings are contractual and must be
      respected by all concrete implementations.
    """

    def max(self: ITensor, axis: int = -1, keepdims: bool = False) -> ITensor:
        """
        Compute the maximum values along a given axis.

        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the maximum. Defaults to -1.
        keepdims : bool, optional
            Whether to retain reduced dimensions with size 1. Defaults to False.

        Returns
        -------
        ITensor
            Tensor containing the maximum values along the specified axis.

        CUDA support
        ------------
        - Only supports 2D input tensors.
        - ``axis`` must reduce exactly one dimension:
          ``axis in {0, 1, -1, -2}``.
        - Backward propagation routes the gradient to a single argmax index per
          slice (ties are not split).

        CPU notes
        ---------
        Backward rule:
            Gradients are routed to all positions equal to the maximum value
            using a mask, i.e.:

                ``dx = grad_out * 1[x == max(x)]``

        Notes
        -----
        The exact behavior (including tie handling and shape semantics) is
        backend-dependent but must conform to the rules documented here.
        """

    def mean(self: ITensor) -> "ITensor":
        """
        Compute the arithmetic mean of all elements in the tensor.

        This operation always reduces the tensor to a scalar value.

        Returns
        -------
        ITensor
            A scalar tensor (shape=()) containing the mean value.

        Notes
        -----
        Backward rule:
            The gradient is distributed uniformly to all input elements:

                ``d(mean(x)) / dx = 1 / numel(x)``

        No axis argument is currently supported; the reduction is always
        performed over all elements.
        """

    def sum(
        self: ITensor, axis: Optional[int] = None, keepdims: bool = False
    ) -> "ITensor":
        """
        Compute the sum of tensor elements.

        Parameters
        ----------
        axis : Optional[int], optional
            Axis along which to compute the sum. If None, all elements are
            reduced into a scalar.
        keepdims : bool, optional
            If True, retains reduced dimensions with size 1. Defaults to False.

        Returns
        -------
        ITensor
            Tensor containing the summed values. The shape depends on the
            ``axis`` and ``keepdims`` arguments.

        Notes
        -----
        Backward rule:
            The upstream gradient is broadcast back to the input tensor's
            shape, i.e., each input element receives the gradient of the
            corresponding reduced output.

        Backend-specific implementations may impose additional constraints
        (e.g., limited axis support on CUDA).
        """
