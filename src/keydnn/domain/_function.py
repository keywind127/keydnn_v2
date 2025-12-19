"""
Autograd function interface definitions.

This module defines the abstract base class for differentiable operations
used in the automatic differentiation system. Concrete subclasses of
`Function` implement both the forward computation and its corresponding
backward gradient computation.

This design is inspired by function-level autograd systems (e.g., PyTorch's
`autograd.Function`) while remaining lightweight and framework-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Union, Any
from ._tensor import ITensor


class Function(ABC):
    """
    Abstract base class for differentiable operations.

    A `Function` represents a single node in the computation graph and
    encapsulates both:
    - the forward computation
    - the backward (gradient) computation

    Subclasses must implement both `forward` and `backward` as static methods.
    Any intermediate values required for gradient computation should be stored
    on the provided `ctx` object during the forward pass.

    Notes
    -----
    - Methods are declared as `@staticmethod` to avoid implicit state on the
      function object itself.
    - The `ctx` argument acts as a per-invocation context, allowing safe reuse
      of `Function` classes across multiple computation graphs.
    """

    @staticmethod
    @abstractmethod
    def forward(ctx, *inputs: Union[ITensor, Any]) -> ITensor:
        """
        Perform the forward computation.

        This method computes the output tensor(s) from the given input
        tensor(s). Any values needed for the backward pass should be saved
        to the provided context object.

        Parameters
        ----------
        ctx : Context
            A mutable context object used to store intermediate values
            required for gradient computation.
        *inputs : Tensor
            Input tensor(s) to the operation.

        Returns
        -------
        Tensor
            The output tensor resulting from the forward computation.
        """
        ...

    @staticmethod
    @abstractmethod
    def backward(ctx, grad_out: ITensor) -> ITensor:
        """
        Compute gradients with respect to the input tensors.

        This method receives the gradient of the loss with respect to the
        output of this operation and must return the gradient(s) with respect
        to each input provided to `forward`.

        Parameters
        ----------
        ctx : Context
            The context object populated during the forward pass.
        grad_out : Tensor
            Gradient of the loss with respect to the output tensor.

        Returns
        -------
        tuple[Tensor | None, ...]
            Gradients with respect to each input tensor. Entries may be None
            for inputs that do not require gradients.
        """
        ...
