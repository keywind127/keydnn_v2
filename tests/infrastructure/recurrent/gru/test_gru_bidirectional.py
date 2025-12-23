from __future__ import annotations

import unittest
import numpy as np

from src.keydnn.domain.device._device import Device
from src.keydnn.infrastructure._tensor import Tensor
from src.keydnn.infrastructure.recurrent._gru_module import GRU
from src.keydnn.infrastructure.recurrent._bidirectional import Bidirectional


def _unwrap_output(ret):
    """
    Bidirectional's internal layers may run with keras_compat+return_state=True,
    so forward() can return (out, state) or (out, state_f, state_b).
    This helper extracts just the `out` tensor.
    """
    if isinstance(ret, Tensor):
        return ret
    if isinstance(ret, tuple) and len(ret) >= 1 and isinstance(ret[0], Tensor):
        return ret[0]
    raise TypeError(f"Unexpected forward() return type: {type(ret)!r}")


def _tensor_from_numpy(arr: np.ndarray, device: Device, requires_grad: bool) -> Tensor:
    arr = np.asarray(arr, dtype=np.float32)
    t = Tensor(shape=arr.shape, device=device, requires_grad=requires_grad, ctx=None)
    t.copy_from_numpy(arr)
    return t


def _unwrap_param_tensor(p):
    if hasattr(p, "to_numpy") and hasattr(p, "copy_from_numpy"):
        return p
    if hasattr(p, "data"):
        return p.data
    if hasattr(p, "tensor"):
        return p.tensor
    raise TypeError(f"Unsupported Parameter structure: {type(p)!r}")


def _reverse_time(x: Tensor) -> Tensor:
    T = x.shape[0]
    xs = [x[t] for t in range(T - 1, -1, -1)]
    return Tensor.stack(xs, axis=0)


def _set_gru_deterministic_params(gru: GRU, *, D: int, H: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cell = gru.cell

    Wih = (rng.standard_normal((D, 3 * H)).astype(np.float32)) * 0.05
    Whh = (rng.standard_normal((H, 3 * H)).astype(np.float32)) * 0.05
    _unwrap_param_tensor(cell.W_ih).copy_from_numpy(Wih)
    _unwrap_param_tensor(cell.W_hh).copy_from_numpy(Whh)

    if getattr(cell, "bias", True):
        bih = (rng.standard_normal((3 * H,)).astype(np.float32)) * 0.02
        bhh = (rng.standard_normal((3 * H,)).astype(np.float32)) * 0.02
        _unwrap_param_tensor(cell.b_ih).copy_from_numpy(bih)
        _unwrap_param_tensor(cell.b_hh).copy_from_numpy(bhh)


class TestBidirectionalWithGRU(unittest.TestCase):
    def setUp(self) -> None:
        self.device = Device("cpu")

    def test_bidirectional_gru_forward_shapes_return_sequences_true(self):
        np.random.seed(0)
        T, N, D, H = 5, 2, 3, 4

        layer = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        _set_gru_deterministic_params(bi.forward_layer, D=D, H=H, seed=10)
        _set_gru_deterministic_params(bi.backward_layer, D=D, H=H, seed=20)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        y = bi.forward(x)

        self.assertIsInstance(y, Tensor)
        self.assertEqual(y.shape, (T, N, 2 * H))

    def test_bidirectional_gru_return_state_structure(self):
        """
        For GRU: state is a single Tensor h_T per direction.
        Expect: (out, h_f_T, h_b_T) when return_state=True.
        """
        np.random.seed(1)
        T, N, D, H = 4, 2, 3, 5

        layer = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=False,
            return_state=True,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=False,
            return_state=True,
        )

        _set_gru_deterministic_params(bi.forward_layer, D=D, H=H, seed=111)
        _set_gru_deterministic_params(bi.backward_layer, D=D, H=H, seed=222)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=False
        )
        ret = bi.forward(x)

        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 3)

        out, h_f, h_b = ret
        self.assertEqual(out.shape, (N, 2 * H))
        self.assertEqual(h_f.shape, (N, H))
        self.assertEqual(h_b.shape, (N, H))

    def test_bidirectional_gru_forward_matches_manual_concat_reference(self):
        """
        Deterministic numerical correctness:

          y = concat( y_fwd(x),
                      reverse_time( y_bwd(reverse_time(x)) ),
                      axis=-1 )
        """
        np.random.seed(2)
        T, N, D, H = 6, 2, 3, 4

        layer = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        # set deterministic params on cloned layers
        _set_gru_deterministic_params(bi.forward_layer, D=D, H=H, seed=123)
        _set_gru_deterministic_params(bi.backward_layer, D=D, H=H, seed=456)

        x_np = np.random.randn(T, N, D).astype(np.float32) * 0.1
        x = _tensor_from_numpy(x_np, self.device, requires_grad=False)

        y = bi.forward(x)
        y_np = y.to_numpy()

        y_f = _unwrap_output(bi.forward_layer.forward(x))  # Tensor (T,N,H)
        x_rev = _reverse_time(x)  # Tensor (T,N,D)
        y_b_rev = _unwrap_output(bi.backward_layer.forward(x_rev))  # Tensor (T,N,H)
        y_b = _reverse_time(y_b_rev)  # align to original time
        y_ref = Tensor.concat([y_f, y_b], axis=-1).to_numpy()

        self.assertEqual(y.shape, (T, N, 2 * H))
        np.testing.assert_allclose(y_np, y_ref, rtol=1e-6, atol=1e-6)

    def test_bidirectional_gru_backward_runs_and_populates_x_grad(self):
        """
        Smoke test: x -> BiGRU -> sum -> backward should run and populate x.grad.
        """
        np.random.seed(3)
        T, N, D, H = 7, 2, 3, 4

        layer = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )

        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        _set_gru_deterministic_params(bi.forward_layer, D=D, H=H, seed=1)
        _set_gru_deterministic_params(bi.backward_layer, D=D, H=H, seed=2)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        y = bi.forward(x)

        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(np.isfinite(x.grad.to_numpy()).all())

    def test_backward_populates_parameter_grads_both_directions(self):
        """
        Sanity: backward should populate at least one parameter grad
        in BOTH forward and backward layers.
        """
        np.random.seed(4)
        T, N, D, H = 5, 2, 3, 4

        layer = GRU(
            input_size=D,
            hidden_size=H,
            bias=True,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )
        bi = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=True,
            return_state=False,
        )

        _set_gru_deterministic_params(bi.forward_layer, D=D, H=H, seed=10)
        _set_gru_deterministic_params(bi.backward_layer, D=D, H=H, seed=20)

        x = _tensor_from_numpy(
            np.random.randn(T, N, D), self.device, requires_grad=True
        )
        y = bi.forward(x)
        y.sum().backward()

        def _has_any_grad(mod) -> bool:
            for p in mod.parameters():
                pt = _unwrap_param_tensor(p)
                if getattr(pt, "grad", None) is not None:
                    return True
            return False

        self.assertTrue(_has_any_grad(bi.forward_layer))
        self.assertTrue(_has_any_grad(bi.backward_layer))

    def test_get_config_and_from_config_roundtrip(self):
        """
        Requires your Bidirectional.from_config to reconstruct the wrapped layer
        (which you already do in other tests for RNN/LSTM).
        """
        layer = GRU(
            input_size=3,
            hidden_size=4,
            bias=False,
            return_sequences=True,
            return_state=False,
            keras_compat=True,
        )
        bi1 = Bidirectional(
            layer=layer,
            merge_mode="concat",
            return_sequences=False,
            return_state=True,
        )

        cfg = bi1.get_config()
        bi2 = Bidirectional.from_config(cfg)

        self.assertEqual(bi2.merge_mode, bi1.merge_mode)
        self.assertEqual(bi2.return_sequences, bi1.return_sequences)
        self.assertEqual(bi2.return_state, bi1.return_state)

        # forward/backward layers should exist (new API)
        self.assertTrue(hasattr(bi2, "forward_layer"))
        self.assertTrue(hasattr(bi2, "backward_layer"))


if __name__ == "__main__":
    unittest.main()
