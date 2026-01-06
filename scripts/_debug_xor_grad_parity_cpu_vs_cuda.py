from __future__ import annotations

import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _cuda_available() -> bool:
    try:
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def main() -> None:
    from keydnn.infrastructure._models import Sequential
    from keydnn.infrastructure.fully_connected._linear import Linear
    from keydnn.infrastructure._activations import Sigmoid
    from keydnn.infrastructure.tensor._tensor import Tensor
    from keydnn.domain.device._device import Device

    # XOR data
    x_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)

    # model factory (same init seed => same weights if your init uses numpy RNG)
    def make_model(device: Device):
        return Sequential(
            Linear(2, 8, device=device),
            Sigmoid(),
            Linear(8, 1, device=device),
            Sigmoid(),
        )

    def mse(pred, target):
        diff = pred - target
        sq = diff * diff
        if hasattr(sq, "mean"):
            return sq.mean()
        return sq.sum() * (1.0 / target.shape[0])

    # CPU
    dev_cpu = Device("cpu")
    x_cpu = Tensor(shape=x_np.shape, device=dev_cpu)
    x_cpu.copy_from_numpy(x_np)
    y_cpu = Tensor(shape=y_np.shape, device=dev_cpu)
    y_cpu.copy_from_numpy(y_np)
    model_cpu = make_model(dev_cpu)

    pred_cpu = model_cpu(x_cpu)
    loss_cpu = mse(pred_cpu, y_cpu)
    loss_cpu.backward()

    cpu_loss = float(loss_cpu.to_numpy())

    # CUDA
    if not _cuda_available():
        raise SystemExit("CUDA not available")

    dev_cuda = Device("cuda:0")
    x_cuda = Tensor(shape=x_np.shape, device=dev_cuda)
    x_cuda.copy_from_numpy(x_np)
    y_cuda = Tensor(shape=y_np.shape, device=dev_cuda)
    y_cuda.copy_from_numpy(y_np)
    model_cuda = make_model(dev_cuda)

    pred_cuda = model_cuda(x_cuda)
    loss_cuda = mse(pred_cuda, y_cuda)
    loss_cuda.backward()

    cuda_loss = float(loss_cuda.to_numpy())

    print(
        f"loss cpu={cpu_loss:.8f}  cuda={cuda_loss:.8f}  diff={abs(cpu_loss-cuda_loss):.3e}"
    )

    # Compare grads
    ps_cpu = list(model_cpu.parameters())
    ps_cuda = list(model_cuda.parameters())

    if len(ps_cpu) != len(ps_cuda):
        print(f"param count mismatch: cpu={len(ps_cpu)} cuda={len(ps_cuda)}")
        return

    max_abs = 0.0
    for i, (pc, pg) in enumerate(zip(ps_cpu, ps_cuda)):
        gc = pc.grad.to_numpy() if getattr(pc, "grad", None) is not None else None
        gg = pg.grad.to_numpy() if getattr(pg, "grad", None) is not None else None

        if gc is None or gg is None:
            print(
                f"param[{i}] missing grad: cpu_grad={gc is not None} cuda_grad={gg is not None}"
            )
            continue

        gc = np.asarray(gc)
        gg = np.asarray(gg)
        if gc.shape != gg.shape:
            print(f"param[{i}] grad shape mismatch: cpu={gc.shape} cuda={gg.shape}")
            continue

        diff = np.max(np.abs(gc - gg))
        max_abs = max(max_abs, float(diff))
        print(f"param[{i}] grad max|diff| = {diff:.6e}  shape={gc.shape}")

    print(f"MAX grad max|diff| across params = {max_abs:.6e}")


if __name__ == "__main__":
    main()
