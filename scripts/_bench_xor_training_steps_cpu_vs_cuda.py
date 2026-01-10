"""
scripts/bench_xor_training_steps_cpu_vs_cuda.py

Step-level microbenchmark for XOR training on CPU vs CUDA for KeyDNN.

What this measures (per training iteration)
------------------------------------------
- forward:     pred = model(x)
- loss:        mse(pred, y)
- backward:    loss.backward()
- step:        opt.step()
- zero_grad:   model.zero_grad() (or per-parameter)

Timing policy
-------------
- Uses warmup iterations (not recorded).
- Then repeats iterations and records per-step durations.
- For CUDA, optionally synchronizes after each measured step so timings are accurate.

Why CUDA can be slower on XOR
-----------------------------
For tiny models, overhead dominates:
- Python dispatch per op
- many tiny CUDA kernel launches
- sync points (explicit or implicit)
- host<->device transfers if any hidden fallbacks occur
- autograd graph traversal overhead

Usage
-----
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cpu
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cuda:0 --sync_each_step
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cuda:0 --epochs 200 --warmup 50 --repeats 200
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np

# -------------------------
# Make repo_root/src importable
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# tracer utils (your local helper module)
from _funct_call_debug_utils import trace_calls, timed_trace


def _cuda_available() -> bool:
    try:
        from keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes import (
            load_keydnn_cuda_native,  # type: ignore
        )

        _ = load_keydnn_cuda_native()
        return True
    except Exception:
        return False


def _fmt_seconds(x: float) -> str:
    if x < 1e-6:
        return f"{x*1e9:.2f} ns"
    if x < 1e-3:
        return f"{x*1e6:.2f} µs"
    if x < 1:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _p95(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = int(0.95 * (len(xs2) - 1))
    return xs2[k]


def _maybe_cuda_sync(tensor_or_lib, *, device_str: str) -> None:
    """
    Best-effort synchronize for accurate CUDA timing.
    Works even if different modules expose sync in different places.
    """
    if not device_str.startswith("cuda"):
        return

    # Try: Tensor._get_cuda_lib().keydnn_cuda_synchronize
    try:
        lib = None
        if hasattr(tensor_or_lib, "_get_cuda_lib") and callable(
            getattr(tensor_or_lib, "_get_cuda_lib")
        ):
            lib = tensor_or_lib._get_cuda_lib()  # type: ignore[attr-defined]
        else:
            lib = tensor_or_lib

        if lib is not None and hasattr(lib, "keydnn_cuda_synchronize"):
            fn = lib.keydnn_cuda_synchronize
            fn.argtypes = []
            fn.restype = int
            st = int(fn())
            if st != 0:
                raise RuntimeError(f"keydnn_cuda_synchronize failed: status={st}")
            return
    except Exception:
        pass

    # Try known wrapper modules
    for mod_path in (
        "keydnn.infrastructure.native_cuda.python.global_avgpool2d_ctypes",
        "keydnn.infrastructure.native_cuda.python.avgpool2d_ctypes",
        "keydnn.infrastructure.native_cuda.python.maxpool2d_ctypes",
    ):
        try:
            m = __import__(mod_path, fromlist=["cuda_synchronize"])
            if hasattr(m, "cuda_synchronize"):
                m.cuda_synchronize(tensor_or_lib)  # type: ignore[misc]
                return
        except Exception:
            continue

    # If no sync available, do nothing (timings may under-report CUDA work).


@dataclass
class StepStats:
    forward: list[float]
    loss: list[float]
    backward: list[float]
    step: list[float]
    zero_grad: list[float]
    total: list[float]
    eval_to_numpy: list[float]


def _time_step(fn: Callable[[], None], *, sync_after: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    sync_after()
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    ap.add_argument(
        "--epochs", type=int, default=200, help="training iterations per trial"
    )
    ap.add_argument(
        "--warmup", type=int, default=50, help="warmup iterations (not recorded)"
    )
    ap.add_argument("--repeats", type=int, default=200, help="recorded iterations")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sync_each_step",
        action="store_true",
        help="CUDA: synchronize after each measured step",
    )
    ap.add_argument(
        "--eval_each_iter",
        action="store_true",
        help="also time pred.to_numpy() each iter (can force sync)",
    )
    ap.add_argument(
        "--batch_repeat",
        type=int,
        default=1,
        help="Repeat the 4 XOR samples this many times to increase batch size (amortize CUDA overhead).",
    )

    # tracing (counts-only via trace_calls OR timed via timed_trace)
    ap.add_argument(
        "--trace_iter",
        type=int,
        default=-1,
        help="If >=0, enable tracing on this recorded iteration index (0..repeats-1).",
    )
    ap.add_argument(
        "--trace_stage",
        type=str,
        default="step",
        choices=["forward", "loss", "backward", "step", "zero_grad", "total"],
        help="Which stage to trace when trace_iter is enabled.",
    )
    ap.add_argument(
        "--trace_limit",
        type=int,
        default=200,
        help="Max number of lines to print from tracer summary.",
    )
    ap.add_argument(
        "--trace_modules",
        type=str,
        default="keydnn,src.keydnn,ctypes,numpy",
        help="Comma-separated module prefixes to include in trace.",
    )
    ap.add_argument(
        "--trace_timed",
        action="store_true",
        help="Use timed_trace (per-function inclusive/self time). If not set, uses trace_calls (counts only).",
    )
    ap.add_argument(
        "--trace_sort",
        type=str,
        default="inclusive",
        choices=["inclusive", "self", "calls"],
        help="Timed trace sort key (only relevant with --trace_timed).",
    )

    args = ap.parse_args()

    dev_str = args.device.strip().lower()

    if dev_str.startswith("cuda") and not _cuda_available():
        raise SystemExit("CUDA requested but CUDA native DLL/wrappers not available.")

    # Imports (after sys.path fix)
    from keydnn.infrastructure.models._sequential import Sequential
    from keydnn.infrastructure.fully_connected._linear import Linear
    from keydnn.infrastructure._activations import Sigmoid
    from keydnn.infrastructure.tensor._tensor import Tensor
    from keydnn.domain.device._device import Device
    from keydnn.infrastructure.optimizers._sgd import SGD

    # IMPORTANT: ensure deterministic (and non-degenerate) init across runs.
    # Many initializers still use np.random.* global RNG.
    np.random.seed(int(args.seed))
    _ = np.random.default_rng(int(args.seed))  # kept for potential future use

    # ---------------- Dataset ----------------
    x_small = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y_small = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    rep = int(args.batch_repeat)
    if rep < 1:
        raise ValueError("--batch_repeat must be >= 1")

    x_np = np.repeat(x_small, rep, axis=0)
    y_np = np.repeat(y_small, rep, axis=0)

    device = Device(dev_str)

    x = Tensor(shape=x_np.shape, device=device)
    x.copy_from_numpy(x_np)

    y = Tensor(shape=y_np.shape, device=device)
    y.copy_from_numpy(y_np)

    # ---------------- Model ----------------
    hidden_dim = int(args.hidden)

    model = Sequential(
        Linear(2, hidden_dim, device=device),
        Sigmoid(),
        Linear(hidden_dim, 1, device=device),
        Sigmoid(),
    )

    # ---------------- Loss (MSE) ----------------
    def mse(pred, target):
        diff = pred - target
        sq = diff * diff
        if hasattr(sq, "mean"):
            return sq.mean()
        if hasattr(sq, "sum"):
            return sq.sum() * (1.0 / target.shape[0])
        raise AttributeError("Tensor must implement mean() or sum()")

    # ---------------- Optimizer ----------------
    opt = SGD(model.parameters(), lr=float(args.lr))

    # ---------------- Sanity: ensure params actually change after one step ----------------
    try:
        params = list(model.parameters()) if hasattr(model, "parameters") else []
        if params and hasattr(params[0], "to_numpy"):
            p0 = params[0]
            w0 = p0.to_numpy().copy()

            pred0 = model(x)
            loss0 = mse(pred0, y)
            loss0.backward()
            opt.step()

            if hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                for p in params:
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()

            w1 = p0.to_numpy()
            if np.allclose(w0, w1):
                raise RuntimeError(
                    "Sanity check failed: parameters did not change after one optimization step. "
                    "This usually means grads are not flowing, opt.step() is a no-op, or init is degenerate."
                )
    except Exception as e:
        raise RuntimeError(f"Sanity check failed before benchmarking: {e}") from e

    # ---------------- Sync function ----------------
    if hasattr(Tensor, "_get_cuda_lib") and callable(getattr(Tensor, "_get_cuda_lib")):
        lib_for_sync = Tensor._get_cuda_lib()
    else:
        lib_for_sync = None

    def sync_after() -> None:
        if dev_str.startswith("cuda") and args.sync_each_step:
            _maybe_cuda_sync(lib_for_sync or x, device_str=dev_str)

    trace_only_modules = [s.strip() for s in args.trace_modules.split(",") if s.strip()]
    trace_prefixes = tuple(trace_only_modules)

    @contextmanager
    def _maybe_trace(stage_name: str, recorded_iter_index: int):
        """
        Enable tracing only on the chosen iteration + stage to avoid slowing the whole benchmark.
        recorded_iter_index is 0..repeats-1 (excludes warmup). Negative during warmup.
        """
        do_trace = (args.trace_iter >= 0) and (
            recorded_iter_index == int(args.trace_iter)
        )
        want_stage = args.trace_stage == stage_name or args.trace_stage == "total"
        if not do_trace or not want_stage:
            yield
            return

        if args.trace_timed:
            # timed_trace: shows calls + inclusive/self time per function
            with timed_trace(
                include_stdlib=True,  # allow ctypes/numpy etc if you include them
                include_modules_prefix=trace_prefixes,
                limit=int(args.trace_limit),
                sort_by=str(args.trace_sort),
                print_files=True,
            ):
                yield
        else:
            # trace_calls: your existing counts-only tracer
            with trace_calls(
                only_modules=list(trace_only_modules),
                limit=int(args.trace_limit),
            ):
                yield

    # ---------------- Bench loop ----------------
    stats = StepStats(
        forward=[],
        loss=[],
        backward=[],
        step=[],
        zero_grad=[],
        total=[],
        eval_to_numpy=[],
    )

    total_iters = args.warmup + args.repeats
    for it in range(total_iters):
        rec_i = (
            it - args.warmup
        )  # 0..repeats-1 for recorded iters, negative during warmup

        t_iter0 = time.perf_counter()

        # forward
        pred_holder = {"pred": None}

        def _fwd():
            pred_holder["pred"] = model(x)

        with _maybe_trace("forward", rec_i):
            dt_fwd = _time_step(_fwd, sync_after=sync_after)

        pred = pred_holder["pred"]

        # loss
        loss_holder = {"loss": None}

        def _loss():
            loss_holder["loss"] = mse(pred, y)

        with _maybe_trace("loss", rec_i):
            dt_loss = _time_step(_loss, sync_after=sync_after)

        loss_t = loss_holder["loss"]

        # backward
        def _bwd():
            loss_t.backward()

        with _maybe_trace("backward", rec_i):
            dt_bwd = _time_step(_bwd, sync_after=sync_after)

        # step
        def _step():
            opt.step()

        with _maybe_trace("step", rec_i):
            dt_step = _time_step(_step, sync_after=sync_after)

        # zero_grad
        def _zg():
            if hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                for p in model.parameters():
                    if hasattr(p, "zero_grad"):
                        p.zero_grad()

        with _maybe_trace("zero_grad", rec_i):
            dt_zg = _time_step(_zg, sync_after=sync_after)

        # optional eval (forces potential sync / D2H)
        dt_eval = 0.0
        if args.eval_each_iter:

            def _eval():
                _ = pred.to_numpy()

            # If you want to trace eval too, add "eval" as a stage and wrap it.
            dt_eval = _time_step(_eval, sync_after=sync_after)

        t_iter1 = time.perf_counter()
        dt_total = t_iter1 - t_iter0

        # record (skip warmup)
        if it >= args.warmup:
            stats.forward.append(dt_fwd)
            stats.loss.append(dt_loss)
            stats.backward.append(dt_bwd)
            stats.step.append(dt_step)
            stats.zero_grad.append(dt_zg)
            stats.eval_to_numpy.append(dt_eval)
            stats.total.append(dt_total)

    # ---------------- Report ----------------
    def row(name: str, xs: list[float]) -> str:
        return (
            f"{name:<14}  "
            f"median={_fmt_seconds(_median(xs)):>10}  "
            f"p95={_fmt_seconds(_p95(xs)):>10}"
        )

    print("\n" + "=" * 88)
    print(
        f"XOR training step benchmark | device={dev_str} hidden={hidden_dim} "
        f"warmup={args.warmup} repeats={args.repeats} "
        f"sync_each_step={args.sync_each_step} eval_each_iter={args.eval_each_iter}"
    )
    print("=" * 88)
    print(row("forward", stats.forward))
    print(row("loss", stats.loss))
    print(row("backward", stats.backward))
    print(row("opt.step", stats.step))
    print(row("zero_grad", stats.zero_grad))
    if args.eval_each_iter:
        print(row("to_numpy", stats.eval_to_numpy))
    print("-" * 88)
    print(row("TOTAL", stats.total))
    print("=" * 88)

    # Quick “share” of total composition (median-based)
    med_total = _median(stats.total)
    if med_total > 0:
        med_f = _median(stats.forward)
        med_l = _median(stats.loss)
        med_b = _median(stats.backward)
        med_s = _median(stats.step)
        med_z = _median(stats.zero_grad)
        extra = _median(stats.eval_to_numpy) if args.eval_each_iter else 0.0

        print("\nMedian composition (% of total):")

        def pct(x: float) -> str:
            return f"{(100.0 * x / med_total):6.2f}%"

        print(f"  forward    {pct(med_f)}")
        print(f"  loss       {pct(med_l)}")
        print(f"  backward   {pct(med_b)}")
        print(f"  opt.step   {pct(med_s)}")
        print(f"  zero_grad  {pct(med_z)}")
        if args.eval_each_iter:
            print(f"  to_numpy   {pct(extra)}")

    # Sanity: print final accuracy (not timed)
    pred_final = model(x)
    pred_np = pred_final.to_numpy()
    y_hat = (pred_np >= 0.5).astype(np.float32)
    acc = float((y_hat == y_np).mean())
    print(f"\nFinal XOR acc: {acc:.3f} | preds={pred_np.reshape(-1).tolist()[:10]}")


if __name__ == "__main__":
    main()
