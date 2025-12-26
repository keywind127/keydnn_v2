"""
Linear (fully-connected) layer implementation.

This module provides an infrastructure-level `Linear` layer for KeyDNN. It is a
trainable `Module` (and is registered for serialization via `register_module`)
that performs an affine projection of 2D, batch-major inputs:

    y = x @ W^T + b

Shape conventions
-----------------
- x : (batch, in_features)
- W : (out_features, in_features)
- b : (out_features,)  (optional; omitted if bias=False)
- y : (batch, out_features)

Computation and backend constraints
-----------------------------------
- CPU path: NumPy-backed tensors (unchanged behavior).
- CUDA path: supported when `x`, `weight`, and `bias` (if present) are CUDA tensors
  on the same CUDA device. The forward/backward use CUDA-capable Tensor ops
  (matmul/transpose/stack) and may fall back to a host reduction for `db` until a
  dedicated CUDA reduce-by-axis kernel is available.
- Input must be 2D. Higher-rank inputs are not implicitly flattened.
- Bias addition avoids implicit broadcasting by explicitly expanding `b` to
  (batch, out_features) using `Tensor.stack`.

Autograd integration
--------------------
`forward()` computes results using Tensor operations (e.g., `@`, `.T`, `+`) and,
when gradients are required, attaches a *legacy* `Context` to the returned output
tensor. The context uses parents ordered as:
- (x, weight) if bias is disabled
- (x, weight, bias) if bias is enabled

The backward rule for out = x @ W^T (+ b) is:
- dL/dx = dL/dout @ W
- dL/dW = (dL/dout)^T @ x
- dL/db = sum(dL/dout, axis=0)

Design note
-----------
Parameter initialization is isolated in `_reset_parameters()` and currently uses
NumPy (Xavier/Glorot uniform for weights, zeros for bias). This allows future
replacement with a device-aware RNG/initializer without changing constructor
logic.
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from .tensor._tensor_context import Context

from .module._serialization_core import register_module
from ._module import Module
from ._parameter import Parameter
from .tensor._tensor import Tensor
from ..domain.device._device import Device


import numpy as np


def _load_param_tensor_from_numpy(t, arr: np.ndarray) -> None:
    """
    Load a NumPy array into an existing Tensor `t` using public, device-aware APIs.

    - CPU tensors: use from_numpy / copy_from_numpy.
    - CUDA tensors: allocate device buffer (if needed) then HtoD memcpy into `t.data`.
      This avoids relying on Tensor.copy_from(cpu_tensor), since your copy_from enforces
      same-device copies only.
    """
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # -----------------------------
    # CPU path
    # -----------------------------
    if hasattr(t, "device") and getattr(t.device, "is_cpu", lambda: False)():
        if hasattr(t, "from_numpy") and callable(getattr(t, "from_numpy")):
            t.from_numpy(arr)
            return
        if hasattr(t, "copy_from_numpy") and callable(getattr(t, "copy_from_numpy")):
            t.copy_from_numpy(arr)
            return
        raise AssertionError(
            "CPU tensor cannot be loaded from NumPy via public APIs. "
            "Implement Tensor.from_numpy()/copy_from_numpy()."
        )

    # -----------------------------
    # CUDA path
    # -----------------------------
    if not (hasattr(t, "device") and getattr(t.device, "is_cuda", lambda: False)()):
        raise AssertionError(
            f"Unsupported device for parameter init: {getattr(t, 'device', None)}"
        )

    # Ensure device allocation exists
    if hasattr(t, "_ensure_cuda_alloc") and callable(getattr(t, "_ensure_cuda_alloc")):
        t._ensure_cuda_alloc(dtype=np.dtype(arr.dtype))

    dst = int(getattr(t, "data", 0))
    if dst == 0:
        raise RuntimeError(
            "CUDA parameter tensor has no allocated device buffer (t.data == 0) "
            "after _ensure_cuda_alloc()."
        )

    # Locate a cudaMemcpyHtoD wrapper (be resilient to module moves)
    def _import_cudaMemcpyHtoD():
        try:
            from .native_cuda.python.maxpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        try:
            from .native_cuda.python.global_avgpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        try:
            from .native_cuda.python.avgpool2d_ctypes import cudaMemcpyHtoD  # type: ignore

            return cudaMemcpyHtoD
        except Exception:
            pass
        raise ImportError(
            "Could not import cudaMemcpyHtoD from known native_cuda ctypes modules. "
            "Expose a cudaMemcpyHtoD wrapper (recommended), or add its module path here."
        )

    cudaMemcpyHtoD = _import_cudaMemcpyHtoD()

    # Get CUDA native lib handle (Tensor caches it via _get_cuda_lib)
    if hasattr(t, "_get_cuda_lib") and callable(getattr(t, "_get_cuda_lib")):
        lib = t._get_cuda_lib()
    else:
        from .tensor._tensor import Tensor as _Tensor

        lib = _Tensor._get_cuda_lib()

    cudaMemcpyHtoD(lib, dst, arr, int(arr.nbytes))


@register_module()
class Linear(Module):
    """
    Fully-connected (dense) layer performing an affine transform: y = x @ W^T + b.

    This layer projects 2D batch-major inputs from `in_features` to `out_features`
    using a learnable weight matrix and an optional bias vector.

    Parameters
    ----------
    in_features : int
        Number of input features per example.
    out_features : int
        Number of output features per example.
    bias : bool, optional
        If True, include a learnable bias term. Defaults to True.
    device : Optional[Device], optional
        Device placement for parameters and outputs. Defaults to CPU if not provided.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    device : Device
        Device placement for parameters and outputs.
    weight : Parameter
        Trainable weight matrix of shape (out_features, in_features).
    bias : Optional[Parameter]
        Trainable bias vector of shape (out_features,), or None if disabled.

    Raises
    ------
    ValueError
        If `in_features` or `out_features` is not a positive integer.

    Notes
    -----
    - Weight initialization uses Xavier/Glorot uniform.
    - Bias (when enabled) is initialized to zeros.
    - Expects 2D inputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        """
        Initialize a Linear layer and register its parameters.

        This constructor validates sizes, allocates `Parameter` storage for weights
        (and optional bias), registers parameters for state management, and then
        initializes parameter values via `_reset_parameters()`.

        Parameters
        ----------
        in_features : int
            Number of input features per example.
        out_features : int
            Number of output features per example.
        bias : bool, optional
            If True, include a learnable bias term. Defaults to True.
        device : Optional[Device], optional
            Device placement. Defaults to CPU if not provided.

        Raises
        ------
        ValueError
            If `in_features` or `out_features` is not positive.
        """
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.device = device if device is not None else Device("cpu")

        # Allocate parameters (storage + metadata)
        self.weight = Parameter(
            shape=(self.out_features, self.in_features),
            device=self.device,
            requires_grad=True,
        )
        self.register_parameter("weight", self.weight)

        if bias:
            self.bias = Parameter(
                shape=(self.out_features,),
                device=self.device,
                requires_grad=True,
            )
            self.register_parameter("bias", self.bias)
        else:
            self.bias = None

        # Initialize values
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        import math
        import numpy as np

        k = 1.0 / math.sqrt(float(self.in_features))
        rng = np.random.default_rng()

        w_np = rng.uniform(-k, k, size=(self.out_features, self.in_features)).astype(
            np.float32, copy=False
        )
        _load_param_tensor_from_numpy(self.weight, w_np)

        if self.bias is not None:
            b_np = rng.uniform(-k, k, size=(self.out_features,)).astype(
                np.float32, copy=False
            )
            _load_param_tensor_from_numpy(self.bias, b_np)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the affine transform to a 2D input tensor.

        The forward pass computes:
            y = x @ W^T (+ b)

        where `x` must have shape (batch, in_features). Bias addition is performed
        without implicit broadcasting by explicitly expanding `b` across the batch
        dimension using `Tensor.stack`.

        Autograd behavior
        -----------------
        If gradients are required for any of (x, weight, bias), this method attaches
        a legacy `Context` to a fresh output tensor whose parents are exactly:
        - (x, weight) if bias is disabled
        - (x, weight, bias) if bias is enabled

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, out_features).

        Raises
        ------
        ValueError
            If `x` is not 2D or if its second dimension does not match `in_features`.
        """
        # --- validate input shape ---
        x_shape = x.shape
        if len(x_shape) != 2:
            raise ValueError(
                f"Linear expects 2D input (batch, in_features), got {x_shape}"
            )
        if x_shape[1] != self.in_features:
            raise ValueError(
                f"Linear expects in_features={self.in_features}, got {x_shape[1]}"
            )

        # --- enforce device consistency (no implicit device moves) ---
        if str(x.device) != str(self.device):
            raise RuntimeError(
                f"Linear.forward device mismatch: x.device={x.device} vs layer.device={self.device}"
            )
        if str(self.weight.device) != str(self.device):
            raise RuntimeError(
                f"Linear.forward device mismatch: weight.device={self.weight.device} vs layer.device={self.device}"
            )
        if self.bias is not None and str(self.bias.device) != str(self.device):
            raise RuntimeError(
                f"Linear.forward device mismatch: bias.device={self.bias.device} vs layer.device={self.device}"
            )

        # --- decide whether output should track gradients ---
        x_req = bool(x.requires_grad)
        w_req = bool(self.weight.requires_grad)
        b_req = bool(self.bias is not None and self.bias.requires_grad)
        req = x_req or w_req or b_req

        # ------------------------------------------------------------------
        # CUDA path
        # ------------------------------------------------------------------
        if self.device.is_cuda():
            import numpy as np

            # Sanity: require allocated device buffers for inputs/params.
            # (We do not implicitly allocate or H2D-copy parameters here.)
            if int(getattr(x, "data")) == 0:
                raise RuntimeError(
                    "Linear CUDA requires allocated x device buffer (x.data != 0)."
                )
            if int(getattr(self.weight, "data")) == 0:
                raise RuntimeError(
                    "Linear CUDA requires allocated weight device buffer (weight.data != 0)."
                )
            if self.bias is not None and int(getattr(self.bias, "data")) == 0:
                raise RuntimeError(
                    "Linear CUDA requires allocated bias device buffer (bias.data != 0)."
                )

            # Enforce dtype compatibility for CUDA matmul kernels (your Tensor.matmul checks too).
            dt = np.dtype(getattr(x, "dtype", np.float32))
            if np.dtype(getattr(self.weight, "dtype", dt)) != dt:
                raise TypeError(
                    f"Linear CUDA dtype mismatch: x.dtype={np.dtype(getattr(x,'dtype',dt))} vs weight.dtype={np.dtype(getattr(self.weight,'dtype',dt))}"
                )
            if (
                self.bias is not None
                and np.dtype(getattr(self.bias, "dtype", dt)) != dt
            ):
                raise TypeError(
                    f"Linear CUDA dtype mismatch: x.dtype={dt} vs bias.dtype={np.dtype(getattr(self.bias,'dtype',dt))}"
                )

            # --- forward compute using CUDA-capable Tensor ops ---
            y = x @ self.weight.T  # (batch, out_features)

            if self.bias is not None:
                batch = x_shape[0]
                b2d = Tensor.stack([self.bias] * int(batch), axis=0)  # (batch, out)
                y = y + b2d

            if not req:
                return y

            # Attach legacy ctx (override any ctx produced by internal ops)
            y.requires_grad = True
            y._set_ctx(None)

            def backward_fn(grad_out: Tensor):
                """
                Compute gradients for Linear's parents given gradient at the output.

                Parameters
                ----------
                grad_out : Tensor
                    Gradient w.r.t. the output of this layer, shape (batch, out_features).

                Returns
                -------
                tuple[Optional[Tensor], ...]
                    Gradients for (x, weight) or (x, weight, bias) depending on whether
                    bias is enabled. Non-required gradients may be returned as None.
                """
                if not grad_out.device.is_cuda():
                    raise RuntimeError("grad_out must be CUDA for Linear CUDA backward")
                if str(grad_out.device) != str(self.device):
                    raise RuntimeError(
                        f"grad_out must be on the same CUDA device as output; got {grad_out.device} vs {self.device}"
                    )
                if tuple(grad_out.shape) != (int(x_shape[0]), int(self.out_features)):
                    raise ValueError(
                        f"grad_out shape mismatch: expected {(int(x_shape[0]), int(self.out_features))}, got {tuple(grad_out.shape)}"
                    )
                if int(getattr(grad_out, "data")) == 0:
                    raise RuntimeError(
                        "grad_out CUDA tensor has no allocated devptr (data == 0)"
                    )

                grad_x = None
                grad_w = None
                grad_b = None

                # dX = dY @ W
                if x.requires_grad:
                    grad_x = grad_out @ self.weight  # (batch, in_features)
                    grad_x.requires_grad = False
                    grad_x._set_ctx(None)

                # dW = dY^T @ X
                if self.weight.requires_grad:
                    grad_w = grad_out.T @ x  # (out_features, in_features)
                    grad_w.requires_grad = False
                    grad_w._set_ctx(None)

                # dB = sum(dY, axis=0)  (fallback: do reduction on host then H2D copy)
                if self.bias is not None and self.bias.requires_grad:
                    # NOTE:
                    # Your current Tensor.sum CUDA only supports axis=None.
                    # Until a sum-axis kernel exists, we do a correctness-first fallback:
                    #   1) DtoH grad_out
                    #   2) host reduction
                    #   3) HtoD into a CUDA tensor grad_b
                    go_np = grad_out.to_numpy()  # (batch, out_features) on host
                    gb_np = go_np.sum(axis=0).astype(np.float32, copy=False)  # (out,)

                    grad_b = Tensor(
                        shape=(int(self.out_features),),
                        device=self.device,
                        requires_grad=False,
                        ctx=None,
                        dtype=np.dtype(gb_np.dtype),
                    )
                    grad_b._ensure_cuda_alloc(dtype=np.dtype(gb_np.dtype))

                    # HtoD copy
                    from .native_cuda.python import maxpool2d_ctypes as m

                    lib = grad_b._get_cuda_lib()
                    m.cudaMemcpyHtoD(lib, int(grad_b.data), gb_np, int(gb_np.nbytes))

                if self.bias is None:
                    return (grad_x, grad_w)
                return (grad_x, grad_w, grad_b)

            parents = (
                (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
            )
            ctx = Context(parents=parents, backward_fn=backward_fn)
            # Keep the same saved tensors as CPU path (x, weight)
            ctx.save_for_backward(x, self.weight)
            y._set_ctx(ctx)

            return y

        # ------------------------------------------------------------------
        # CPU path (unchanged behavior)
        # ------------------------------------------------------------------
        # --- compute forward using Tensor ops only ---
        y = x @ self.weight.T  # (batch, out_features)

        if self.bias is not None:
            # No broadcasting: expand bias to 2D by stacking along batch axis.
            batch = x_shape[0]
            b2d = Tensor.stack([self.bias] * batch, axis=0)  # (batch, out_features)
            y = y + b2d

        # --- return early if no autograd needed ---
        if not req:
            return y

        # --- attach legacy Context with parents (x, weight, bias?) ---
        # Create a fresh output tensor so its ctx is exactly what Linear defines.
        out = Tensor(shape=y.shape, device=self.device, requires_grad=True)
        out.copy_from(y)

        def backward_fn(grad_out: Tensor):
            """
            Compute gradients for Linear's parents given gradient at the output.

            Parameters
            ----------
            grad_out : Tensor
                Gradient w.r.t. the output of this layer, shape (batch, out_features).

            Returns
            -------
            tuple[Optional[Tensor], ...]
                Gradients for (x, weight) or (x, weight, bias) depending on whether
                bias is enabled. Non-required gradients may be returned as None.
            """
            x_saved, w_saved = ctx.saved_tensors

            grad_x = None
            grad_w = None
            grad_b = None

            if x_saved.requires_grad:
                grad_x = grad_out @ w_saved  # (batch, in_features)

            if w_saved.requires_grad:
                grad_w = grad_out.T @ x_saved  # (out_features, in_features)

            if self.bias is not None and self.bias.requires_grad:
                grad_b = grad_out.sum(axis=0)  # (out_features,)

            if self.bias is None:
                return (grad_x, grad_w)
            return (grad_x, grad_w, grad_b)

        parents = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        ctx = Context(parents=parents, backward_fn=backward_fn)
        ctx.save_for_backward(x, self.weight)
        out._set_ctx(ctx)

        return out

    # -------------------------------------------------------------------------
    # ADD-ON ONLY: JSON serialization hooks (no change to existing logic above)
    # -------------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration for reconstructing this layer.

        The returned configuration contains constructor-level hyperparameters only.
        Trainable parameter values (weights/bias) are expected to be handled by the
        checkpoint/state mechanism.

        Returns
        -------
        Dict[str, Any]
            A JSON-serializable dict containing `in_features`, `out_features`, `bias`,
            and `device` (as a string).
        """
        return {
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "bias": bool(self.bias is not None),
            # Store device as a string to keep JSON stable.
            # Assumes Device can be reconstructed from its string form (e.g., "cpu").
            "device": str(self.device),
        }

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Linear":
        """
        Construct a Linear layer from a configuration dict.

        This reconstructs the module structure (hyperparameters). Weights are
        expected to be loaded afterward from the checkpoint/state.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Configuration dictionary produced by `get_config()`.

        Returns
        -------
        Linear
            A newly constructed `Linear` instance with matching hyperparameters.
        """
        dev = cfg.get("device", "cpu")
        return cls(
            in_features=int(cfg["in_features"]),
            out_features=int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            device=Device(str(dev)),
        )
