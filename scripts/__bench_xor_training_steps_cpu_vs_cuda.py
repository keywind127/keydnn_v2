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

Backward profiling (new)
-----------------------
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cpu --profile_backward
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cuda:0 --sync_each_step --profile_backward
python scripts/bench_xor_training_steps_cpu_vs_cuda.py --device cuda:0 --profile_backward --profile_backward_iter 0
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Any, Optional

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
    bwd_profiles: list[Any]  # BackwardProfile objects (or None)


def _time_step(fn: Callable[[], None], *, sync_after: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    sync_after()
    t1 = time.perf_counter()
    return t1 - t0


def _profile_stat_ms(stat_obj: Any, field: str) -> Optional[float]:
    """
    Extract average ms value for a profile stat object.

    Supports:
    - stat.cpu_ms / stat.gpu_ms (+ stat.count)
    - dict-like {'cpu_ms': ..., 'gpu_ms': ..., 'count': ...}

    Returns average-per-call ms: field_ms / count
    """
    if stat_obj is None:
        return None

    # attribute style
    if hasattr(stat_obj, field):
        v = getattr(stat_obj, field, None)
        if v is None:
            return None
        cnt = getattr(stat_obj, "count", 1)
        try:
            return float(v) / max(int(cnt), 1)
        except Exception:
            return None

    # dict style
    if isinstance(stat_obj, dict) and field in stat_obj:
        v = stat_obj.get(field, None)
        if v is None:
            return None
        cnt = stat_obj.get("count", 1)
        try:
            return float(v) / max(int(cnt), 1)
        except Exception:
            return None

    return None


def _extract_sections(profile_obj: Any) -> dict[str, Any]:
    if profile_obj is None:
        return {}
    if hasattr(profile_obj, "sections"):
        sec = getattr(profile_obj, "sections")
        if isinstance(sec, dict):
            return sec
    if isinstance(profile_obj, dict) and "sections" in profile_obj:
        sec = profile_obj.get("sections")
        if isinstance(sec, dict):
            return sec
    return {}


def _extract_per_op(profile_obj: Any) -> dict[str, Any]:
    if profile_obj is None:
        return {}
    if hasattr(profile_obj, "per_op"):
        d = getattr(profile_obj, "per_op")
        if isinstance(d, dict):
            return d
    if isinstance(profile_obj, dict) and "per_op" in profile_obj:
        d = profile_obj.get("per_op")
        if isinstance(d, dict):
            return d
    return {}


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

    # backward() internal profiling
    ap.add_argument(
        "--profile_backward",
        action="store_true",
        help="Collect and report Tensor.backward() internal profiling (CPU or CUDA).",
    )
    ap.add_argument(
        "--profile_backward_iter",
        type=int,
        default=-1,
        help="If >=0, only collect backward profile on this recorded iteration (0..repeats-1). Default: all recorded iters.",
    )
    ap.add_argument(
        "--profile_backward_topk",
        type=int,
        default=10,
        help="Top-k ops to print from backward profiling (by total GPU time if present, else CPU time).",
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
        Linear(hidden_dim, hidden_dim, device=device),
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

    # ---------------- Sanity ----------------
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
        do_trace = (args.trace_iter >= 0) and (
            recorded_iter_index == int(args.trace_iter)
        )
        want_stage = args.trace_stage == stage_name or args.trace_stage == "total"
        if not do_trace or not want_stage:
            yield
            return

        if args.trace_timed:
            with timed_trace(
                include_stdlib=True,
                include_modules_prefix=trace_prefixes,
                limit=int(args.trace_limit),
                sort_by=str(args.trace_sort),
                print_files=True,
            ):
                yield
        else:
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
        bwd_profiles=[],
    )

    total_iters = args.warmup + args.repeats
    for it in range(total_iters):
        rec_i = it - args.warmup  # 0..repeats-1 recorded, negative warmup

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

        # backward (collect internal profiling on CPU or CUDA)
        bwd_prof_holder: dict[str, Any] = {"prof": None}

        def _bwd():
            want_prof = bool(args.profile_backward) and rec_i >= 0
            if want_prof:
                only_i = int(args.profile_backward_iter)
                if only_i >= 0 and rec_i != only_i:
                    loss_t.backward()
                    bwd_prof_holder["prof"] = None
                else:
                    # Expect Tensor.backward(profile=True) to return a profile object (CPU or CUDA).
                    bwd_prof_holder["prof"] = loss_t.backward(profile=True)
            else:
                loss_t.backward()
                bwd_prof_holder["prof"] = None

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

        # optional eval
        dt_eval = 0.0
        if args.eval_each_iter:

            def _eval():
                _ = pred.to_numpy()

            dt_eval = _time_step(_eval, sync_after=sync_after)

        t_iter1 = time.perf_counter()
        dt_total = t_iter1 - t_iter0

        if it >= args.warmup:
            stats.forward.append(dt_fwd)
            stats.loss.append(dt_loss)
            stats.backward.append(dt_bwd)
            stats.step.append(dt_step)
            stats.zero_grad.append(dt_zg)
            stats.eval_to_numpy.append(dt_eval)
            stats.total.append(dt_total)
            stats.bwd_profiles.append(bwd_prof_holder["prof"])

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
        f"sync_each_step={args.sync_each_step} eval_each_iter={args.eval_each_iter} "
        f"profile_backward={args.profile_backward}"
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

    # Composition
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

    # ---------------- Backward profiling report (CPU or CUDA) ----------------
    if args.profile_backward:
        profs = [p for p in stats.bwd_profiles if p is not None]
        print("\n" + "=" * 88)
        print(
            "Tensor.backward() internal profiling (aggregated over recorded iterations)"
        )
        print("=" * 88)

        if not profs:
            only_i = int(args.profile_backward_iter)
            if only_i >= 0:
                print(
                    "No backward profiles collected. "
                    f"(You requested --profile_backward_iter {only_i}, but it may be outside 0..repeats-1, "
                    "or backward(profile=True) returned None.)"
                )
            else:
                print(
                    "No backward profiles collected. "
                    "Make sure Tensor.backward(profile=True) returns a profile object on this device."
                )
        else:
            # Sections
            section_names: set[str] = set()
            for p in profs:
                section_names.update(_extract_sections(p).keys())

            preferred = [
                "seed_grad",
                "build_topo_dfs",
                "build_nodes_dict",
                "reverse_traverse_total",
                "backward_fn_total",
                "grad_accum_add",
                "grad_accum_set",
                "writeback_leaf_accum",
            ]
            ordered = [n for n in preferred if n in section_names] + sorted(
                [n for n in section_names if n not in preferred]
            )

            # Determine whether gpu_ms exists on any stat
            any_gpu = False
            for p in profs:
                for s in _extract_sections(p).values():
                    if _profile_stat_ms(s, "gpu_ms") is not None:
                        any_gpu = True
                        break
                if any_gpu:
                    break

            if any_gpu:
                print("\n[Sections] avg ms per section-call (CPU vs GPU if available)")
                header = f"{'section':<24}  {'cpu_med':>9}  {'cpu_p95':>9}  {'gpu_med':>9}  {'gpu_p95':>9}"
                print(header)
                print("-" * len(header))
            else:
                print("\n[Sections] avg CPU ms per section-call")
                header = f"{'section':<24}  {'cpu_med':>9}  {'cpu_p95':>9}"
                print(header)
                print("-" * len(header))

            printed_any = False
            for name in ordered:
                cpu_xs: list[float] = []
                gpu_xs: list[float] = []
                for p in profs:
                    sec = _extract_sections(p).get(name)
                    c = _profile_stat_ms(sec, "cpu_ms")
                    g = _profile_stat_ms(sec, "gpu_ms")
                    if c is not None:
                        cpu_xs.append(float(c))
                    if g is not None:
                        gpu_xs.append(float(g))

                if not cpu_xs and not gpu_xs:
                    continue
                printed_any = True

                if any_gpu:
                    cpu_med = statistics.median(cpu_xs) if cpu_xs else 0.0
                    cpu_p95 = _p95(cpu_xs) if cpu_xs else 0.0
                    gpu_med = statistics.median(gpu_xs) if gpu_xs else 0.0
                    gpu_p95 = _p95(gpu_xs) if gpu_xs else 0.0
                    print(
                        f"{name:<24}  {cpu_med:9.3f}  {cpu_p95:9.3f}  {gpu_med:9.3f}  {gpu_p95:9.3f}"
                    )
                else:
                    cpu_med = statistics.median(cpu_xs) if cpu_xs else 0.0
                    cpu_p95 = _p95(cpu_xs) if cpu_xs else 0.0
                    print(f"{name:<24}  {cpu_med:9.3f}  {cpu_p95:9.3f}")

            if not printed_any:
                print(
                    "No recognizable section timings found on profile objects. "
                    "Expected profile.sections[name].cpu_ms/count (and optionally gpu_ms)."
                )

            # Per-op (optional)
            per_op_totals_cpu: dict[str, float] = {}
            per_op_totals_gpu: dict[str, float] = {}
            per_op_counts: dict[str, int] = {}
            saw_ops = False
            saw_op_gpu = False

            for p in profs:
                per_op = _extract_per_op(p)
                if per_op:
                    saw_ops = True
                for op, stat in per_op.items():
                    c = None
                    g = None
                    if hasattr(stat, "cpu_ms"):
                        c = getattr(stat, "cpu_ms", None)
                    elif isinstance(stat, dict):
                        c = stat.get("cpu_ms", None)

                    if hasattr(stat, "gpu_ms"):
                        g = getattr(stat, "gpu_ms", None)
                    elif isinstance(stat, dict):
                        g = stat.get("gpu_ms", None)

                    cnt = 0
                    if hasattr(stat, "count"):
                        cnt = int(getattr(stat, "count", 0))
                    elif isinstance(stat, dict):
                        cnt = int(stat.get("count", 0))

                    if c is not None:
                        per_op_totals_cpu[op] = per_op_totals_cpu.get(op, 0.0) + float(
                            c
                        )
                    if g is not None:
                        saw_op_gpu = True
                        per_op_totals_gpu[op] = per_op_totals_gpu.get(op, 0.0) + float(
                            g
                        )
                    per_op_counts[op] = per_op_counts.get(op, 0) + int(cnt)

            if saw_ops and per_op_totals_cpu:
                topk = int(args.profile_backward_topk)
                # sort key: prefer gpu totals if they exist, else cpu totals
                if saw_op_gpu and per_op_totals_gpu:
                    sort_items = sorted(
                        per_op_totals_gpu.items(), key=lambda kv: kv[1], reverse=True
                    )
                    print(
                        f"\n[Per-op top {topk}] total ms over all profiled backward() calls (CPU/GPU)"
                    )
                    for op, tot_gpu in sort_items[:topk]:
                        tot_cpu = per_op_totals_cpu.get(op, 0.0)
                        calls = per_op_counts.get(op, 1)
                        print(
                            f"{op:<24} cpu_total={tot_cpu:9.3f}  gpu_total={tot_gpu:9.3f}  "
                            f"calls={calls:6d}  gpu_avg={tot_gpu/max(calls,1):7.4f}"
                        )
                else:
                    sort_items = sorted(
                        per_op_totals_cpu.items(), key=lambda kv: kv[1], reverse=True
                    )
                    print(
                        f"\n[Per-op top {topk}] total CPU ms over all profiled backward() calls"
                    )
                    for op, tot_cpu in sort_items[:topk]:
                        calls = per_op_counts.get(op, 1)
                        print(
                            f"{op:<24} cpu_total={tot_cpu:9.3f}  calls={calls:6d}  avg={tot_cpu/max(calls,1):7.4f}"
                        )
            elif saw_ops:
                print(
                    "\n[Per-op] per_op dict exists but contained no recognizable cpu_ms/gpu_ms."
                )
            else:
                print(
                    "\n[Per-op] No per-op stats found on profile objects (expected profile.per_op)."
                )

        print("=" * 88)

    # Final accuracy (not timed)
    pred_final = model(x)
    pred_np = pred_final.to_numpy()
    y_hat = (pred_np >= 0.5).astype(np.float32)
    acc = float((y_hat == y_np).mean())
    print(f"\nFinal XOR acc: {acc:.3f} | preds={pred_np.reshape(-1).tolist()[:10]}")


if __name__ == "__main__":
    main()
