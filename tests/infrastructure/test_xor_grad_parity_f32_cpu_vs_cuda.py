# from __future__ import annotations

# import os
# import unittest
# import numpy as np


# def _cuda_available() -> bool:
#     try:
#         from src.keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
#             load_keydnn_cuda_native,  # type: ignore
#         )

#         _ = load_keydnn_cuda_native()
#         return True
#     except Exception:
#         return False


# def _to_numpy(t) -> np.ndarray:
#     if hasattr(t, "to_numpy"):
#         return np.asarray(t.to_numpy())
#     raise AttributeError(f"Object has no to_numpy(): type={type(t)}")


# def _copy_param_cpu_to_cuda(p_cpu, p_cuda) -> None:
#     """
#     Best-effort: copy parameter values CPU -> CUDA via numpy.
#     Assumes p_cpu and p_cuda are Tensors with copy_from_numpy.
#     """
#     arr = _to_numpy(p_cpu).astype(np.float32, copy=False)
#     p_cuda.copy_from_numpy(arr)


# @unittest.skipUnless(_cuda_available(), "CUDA native DLL/wrappers not available")
# class TestXORGradParityF32CPUvsCUDA(unittest.TestCase):
#     def test_xor_grad_parity_f32(self) -> None:
#         from src.keydnn.infrastructure._models import Sequential
#         from src.keydnn.infrastructure._linear import Linear
#         from src.keydnn.infrastructure._activations import Sigmoid
#         from src.keydnn.infrastructure.tensor._tensor import Tensor
#         from src.keydnn.domain.device._device import Device

#         # -----------------------
#         # Config / tolerances
#         # -----------------------
#         # Default tolerances are pragmatic for early CUDA backends.
#         # Tighten later as correctness improves.
#         loss_atol = float(os.environ.get("KEYDNN_TEST_LOSS_ATOL", "1e-4"))
#         grad_max_abs_atol = float(os.environ.get("KEYDNN_TEST_GRAD_ATOL", "2e-3"))

#         # Optional: print per-param diffs always (useful while debugging)
#         verbose = os.environ.get("KEYDNN_TEST_VERBOSE", "1") != "0"

#         # -----------------------
#         # XOR data (float32)
#         # -----------------------
#         x_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
#         y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)

#         # -----------------------
#         # Model factory
#         # -----------------------
#         def make_model(device: Device) -> Sequential:
#             return Sequential(
#                 Linear(2, 8, device=device),
#                 Sigmoid(),
#                 Linear(8, 1, device=device),
#                 Sigmoid(),
#             )

#         def mse(pred, target):
#             diff = pred - target
#             sq = diff * diff
#             if hasattr(sq, "mean"):
#                 return sq.mean()
#             return sq.sum() * (1.0 / target.shape[0])

#         # -----------------------
#         # CPU graph
#         # -----------------------
#         dev_cpu = Device("cpu")

#         x_cpu = Tensor(
#             shape=x_np.shape, device=dev_cpu, requires_grad=False, dtype=np.float32
#         )
#         x_cpu.copy_from_numpy(x_np)

#         y_cpu = Tensor(
#             shape=y_np.shape, device=dev_cpu, requires_grad=False, dtype=np.float32
#         )
#         y_cpu.copy_from_numpy(y_np)

#         model_cpu = make_model(dev_cpu)

#         pred_cpu = model_cpu(x_cpu)
#         loss_cpu = mse(pred_cpu, y_cpu)
#         loss_cpu.backward()

#         loss_cpu_val = float(_to_numpy(loss_cpu))

#         # -----------------------
#         # CUDA graph
#         #   IMPORTANT: we copy CPU params -> CUDA params
#         #   so we compare the same weights.
#         # -----------------------
#         dev_cuda = Device("cuda:0")

#         x_cuda = Tensor(
#             shape=x_np.shape, device=dev_cuda, requires_grad=False, dtype=np.float32
#         )
#         x_cuda.copy_from_numpy(x_np)

#         y_cuda = Tensor(
#             shape=y_np.shape, device=dev_cuda, requires_grad=False, dtype=np.float32
#         )
#         y_cuda.copy_from_numpy(y_np)

#         model_cuda = make_model(dev_cuda)

#         ps_cpu = list(model_cpu.parameters())
#         ps_cuda = list(model_cuda.parameters())
#         self.assertEqual(len(ps_cpu), len(ps_cuda), "param count mismatch CPU vs CUDA")

#         # Copy values
#         for pc, pg in zip(ps_cpu, ps_cuda):
#             _copy_param_cpu_to_cuda(pc, pg)

#         pred_cuda = model_cuda(x_cuda)
#         loss_cuda = mse(pred_cuda, y_cuda)
#         loss_cuda.backward()

#         loss_cuda_val = float(_to_numpy(loss_cuda))

#         # -----------------------
#         # Loss parity
#         # -----------------------
#         loss_diff = abs(loss_cpu_val - loss_cuda_val)
#         if verbose:
#             print(
#                 f"[xor-parity f32] loss cpu={loss_cpu_val:.8f} cuda={loss_cuda_val:.8f} diff={loss_diff:.3e}"
#             )
#         self.assertLessEqual(
#             loss_diff,
#             loss_atol,
#             f"loss mismatch too large: diff={loss_diff:.3e} > atol={loss_atol:.3e}",
#         )

#         # -----------------------
#         # Gradient parity (max abs per parameter)
#         # -----------------------
#         max_abs = 0.0
#         worst_i = -1
#         worst_shape = None
#         worst_val = None

#         for i, (pc, pg) in enumerate(zip(ps_cpu, ps_cuda)):
#             gc_t = getattr(pc, "grad", None)
#             gg_t = getattr(pg, "grad", None)

#             self.assertIsNotNone(gc_t, f"CPU param[{i}] grad is None")
#             self.assertIsNotNone(gg_t, f"CUDA param[{i}] grad is None")

#             gc = _to_numpy(gc_t).astype(np.float32, copy=False)
#             gg = _to_numpy(gg_t).astype(np.float32, copy=False)

#             self.assertEqual(gc.shape, gg.shape, f"grad shape mismatch param[{i}]")

#             d = float(np.max(np.abs(gc - gg)))
#             if verbose:
#                 print(
#                     f"[xor-parity f32] param[{i}] grad max|diff|={d:.6e} shape={gc.shape}"
#                 )

#             if d > max_abs:
#                 max_abs = d
#                 worst_i = i
#                 worst_shape = gc.shape
#                 worst_val = d

#         if verbose:
#             print(f"[xor-parity f32] MAX grad max|diff| across params = {max_abs:.6e}")

#         self.assertLessEqual(
#             max_abs,
#             grad_max_abs_atol,
#             (
#                 f"grad mismatch too large: max_abs={max_abs:.6e} > atol={grad_max_abs_atol:.3e}. "
#                 f"worst param index={worst_i}, shape={worst_shape}, value={worst_val}"
#             ),
#         )


# if __name__ == "__main__":
#     unittest.main()
