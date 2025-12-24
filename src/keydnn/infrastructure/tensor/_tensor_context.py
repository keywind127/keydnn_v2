from typing import Any, Callable, Sequence, Optional
from dataclasses import dataclass, field

from ...domain._tensor import ITensor


@dataclass
class Context:
    """
    Backward context attached to a Tensor produced by an operation.

    A `Context` records the information required to compute gradients for an
    operation during backpropagation.

    Attributes
    ----------
    parents : Sequence[Tensor]
        The input tensors (and/or parameters) used to compute the output tensor.
        Gradients will be produced for these parents during the backward pass.
    backward_fn : Callable[[Tensor], Sequence[Optional[Tensor]]]
        A function that takes the gradient w.r.t. the output (`grad_out`) and
        returns gradients w.r.t. each `parents` entry, in the same order.
        Entries may be None for parents that do not require gradients.
    saved_tensors : list[ITensor]
        Tensors explicitly saved during the forward pass for use in backward.
        These are distinct from `parents`: they may include transformed values
        (e.g., outputs, masks, indices, cached intermediates).
    saved_meta : dict[str, Any]
        Non-tensor metadata required for backward (e.g., shapes, axes, indices).

    Notes
    -----
    `saved_tensors` and `saved_meta` are intentionally generic to support a wide
    range of operations without coupling the Context type to specific kernels.
    """

    parents: Sequence["ITensor"]
    backward_fn: Callable[["ITensor"], Sequence[Optional["ITensor"]]]
    saved_tensors: list["ITensor"] = field(default_factory=list)
    saved_meta: dict[str, Any] = field(default_factory=dict)

    def save_for_backward(self, *tensors: "ITensor") -> None:
        """
        Save tensors for use during the backward computation.

        Parameters
        ----------
        *tensors : Tensor
            Any number of tensors to be stored in `saved_tensors`.

        Notes
        -----
        Saved tensors are often intermediate results that are cheaper to reuse
        than to recompute during the backward pass (e.g., activation masks).
        """
        self.saved_tensors.extend(tensors)
