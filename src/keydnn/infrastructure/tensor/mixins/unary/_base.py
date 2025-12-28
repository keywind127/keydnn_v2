"""
Unary operation mixin defining elementwise Tensor unary APIs.

This module declares :class:`TensorMixinUnary`, an abstract mixin that specifies
the public interface and mathematical semantics of common unary tensor
operations such as ``exp``, ``sqrt``, ``log``, negation, ``tanh``, and
``sigmoid``.

The mixin itself does not implement numerical kernels. Concrete CPU and CUDA
implementations (where available) are provided elsewhere and selected at
runtime, or delegated to existing autograd Functions. This separation keeps
the core Tensor class small while allowing backend-specific optimization.
"""

from abc import ABC

from .....domain._tensor import ITensor


class TensorMixinUnary(ABC):
    """
    Abstract mixin defining unary tensor operations.

    This mixin specifies the method signatures, expected behavior, and
    differentiation rules for elementwise unary operations. Implementations
    may differ by backend (CPU vs. CUDA), but must conform to the contracts
    documented here.

    Notes
    -----
    - Methods in this class are interface declarations and documentation of
      semantics; no computation is performed here unless explicitly shown.
    - Backward rules described in method docstrings are contractual and must
      be respected by all concrete implementations.
    """

    def exp(self: ITensor) -> "ITensor":
        """
        Compute the elementwise exponential of the tensor.

        Returns
        -------
        ITensor
            A tensor of the same shape as ``self`` with ``exp`` applied
            elementwise.

        Notes
        -----
        Backward rule:
            ``d(exp(x)) / dx = exp(x)``

        CUDA behavior
        -------------
        - Uses the native CUDA unary exponential kernel via
          ``unary_cuda_ext.exp_forward``.
        - Operates directly on device pointers without a NumPy round-trip.
        """
        ...

    def sqrt(self: ITensor) -> "ITensor":
        """
        Compute the elementwise square root of the tensor.

        Returns
        -------
        ITensor
            A tensor with the same shape as ``self``, containing
            ``sqrt(self)`` applied elementwise.

        Notes
        -----
        CPU behavior
        ------------
        - Uses NumPy to compute the forward pass for CPU tensors.

        CUDA behavior (workaround)
        --------------------------
        - For CUDA tensors, this method currently performs a CPU round-trip:
          device → host (``to_numpy``) → NumPy ``sqrt`` → device
          (``copy_from_numpy``).
        - This preserves correctness and autograd semantics but is not
          performance-optimal.

        Autograd
        --------
        If ``self.requires_grad`` is True, the returned tensor participates in
        autograd with parent ``self``.

        TODO
        ----
        Implement a native CUDA kernel for ``sqrt`` (and optionally a fused
        backward) to avoid device↔host transfers.
        """
        ...

    def log(self: ITensor) -> "ITensor":
        """
        Compute the elementwise natural logarithm of the tensor.

        Returns
        -------
        ITensor
            A tensor of the same shape as ``self``, where each element is
            replaced by its natural logarithm.

        Notes
        -----
        CPU behavior
        ------------
        - Uses NumPy to compute the forward pass for CPU tensors.

        CUDA behavior (workaround)
        --------------------------
        - For CUDA tensors, this method currently performs a CPU round-trip:
          device → host (``to_numpy``) → NumPy ``log`` → device
          (``copy_from_numpy``).
        - This preserves correctness and autograd semantics but is not
          performance-optimal.

        Autograd
        --------
        If ``self.requires_grad`` is True, the backward rule is:

            ``d(log(x)) / dx = 1 / x``

        The behavior for non-positive input values follows NumPy semantics
        (e.g., ``-inf`` or ``nan``).

        TODO
        ----
        Implement a native CUDA kernel for ``log`` (and a fused backward) to
        avoid device↔host transfers.
        """
        ...

    # ----------------------------
    # Unary operators
    # ----------------------------
    def __neg__(self: ITensor) -> "ITensor":
        """
        Compute the elementwise negation of the tensor.

        Returns
        -------
        ITensor
            A tensor containing ``-self`` applied elementwise.

        Notes
        -----
        If ``self.requires_grad`` is True, the returned tensor participates in
        autograd with the backward rule:

            ``d(-x) / dx = -1``
        """
        ...

    def tanh(self) -> "ITensor":
        """
        Compute the elementwise hyperbolic tangent of the tensor.

        Returns
        -------
        ITensor
            A tensor with the same shape as ``self``, with ``tanh`` applied
            elementwise.

        Notes
        -----
        - This method delegates to the autograd ``TanhFn`` Function.
        - NumPy is not used directly here; numerical kernels remain encapsulated
          inside Tensor operations or autograd Functions.
        """
        from ...._function import TanhFn
        from ..._tensor_context import Context

        ctx = Context(parents=(self,), backward_fn=None)
        out = TanhFn.forward(ctx, self)
        ctx.backward_fn = lambda grad_out: (TanhFn.backward(ctx, grad_out),)
        out._set_ctx(ctx)
        return out

    def sigmoid(self) -> "ITensor":
        """
        Compute the elementwise logistic sigmoid of the tensor.

        The sigmoid function is defined as:

            ``sigmoid(x) = 1 / (1 + exp(-x))``

        Returns
        -------
        ITensor
            A tensor with the same shape as ``self``, with ``sigmoid`` applied
            elementwise.

        Notes
        -----
        This method is a thin convenience wrapper around ``SigmoidFn`` defined
        in ``._function`` and integrates with the autograd system.
        """
        from ..._tensor_context import Context
        from ...._function import SigmoidFn

        # Build context with parents and a callable backward_fn
        ctx = Context(parents=(self,), backward_fn=None)

        out = SigmoidFn.forward(ctx, self)

        # IMPORTANT: Tensor.backward() expects ctx.backward_fn to be callable
        ctx.backward_fn = lambda grad_out: (SigmoidFn.backward(ctx, grad_out),)

        out._set_ctx(ctx)
        return out
