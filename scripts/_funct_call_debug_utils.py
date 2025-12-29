# scripts/funct_call_debug_utils.py
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class FuncStat:
    calls: int = 0
    inclusive_ns: int = 0  # total time including children
    self_ns: int = 0  # exclusive time (minus children)


@dataclass
class FrameRec:
    key: Tuple[str, str, int]  # (module, qualname, firstlineno)
    t0_ns: int
    child_ns: int = 0


def _frame_key(frame) -> Tuple[str, str, int]:
    code = frame.f_code
    mod = frame.f_globals.get("__name__", "<unknown>")
    qual = code.co_name
    firstlineno = code.co_firstlineno
    return (mod, qual, int(firstlineno))


def _frame_file(frame) -> str:
    return frame.f_code.co_filename


def _format_ns(ns: int) -> str:
    # human-ish formatting
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms"
    return f"{ns/1_000_000_000:.3f} s"


@contextmanager
def trace_calls_timed(
    *,
    include_stdlib: bool = False,
    include_modules_prefix: Optional[Tuple[str, ...]] = None,
    exclude_modules_prefix: Tuple[str, ...] = (),
    limit: int = 50,
    sort_by: str = "inclusive",  # "inclusive" | "self" | "calls"
    print_files: bool = True,
) -> Iterator[Dict[Tuple[str, str, int], FuncStat]]:
    """
    Profile function calls with per-function time.

    Returns a dict keyed by (module, qualname, firstlineno) -> FuncStat.

    Timing notes
    ------------
    - Uses perf_counter_ns() for high-resolution timing.
    - inclusive_ns: time from call->return including child calls.
    - self_ns: inclusive_ns - time spent in child calls.

    Filtering
    ---------
    - include_modules_prefix: if provided, only record modules whose __name__ starts with any prefix.
    - exclude_modules_prefix: exclude those module prefixes.
    - include_stdlib: if False, attempts to drop stdlib modules by heuristic (not perfect).
    """
    stats: Dict[Tuple[str, str, int], FuncStat] = {}
    stack: List[FrameRec] = []

    # crude stdlib filter heuristic (optional)
    def _is_probably_stdlib(mod_name: str, filename: str) -> bool:
        if mod_name.startswith("ctypes") or mod_name.startswith("typing"):
            return True
        # often stdlib path contains "...\\lib\\pythonX.Y\\"
        f = filename.replace("/", "\\").lower()
        return "\\lib\\python" in f and "\\site-packages\\" not in f

    def _allowed(frame) -> bool:
        mod = frame.f_globals.get("__name__", "<unknown>")
        fn = _frame_file(frame)
        if include_modules_prefix is not None:
            ok = any(mod.startswith(p) for p in include_modules_prefix)
            if not ok:
                return False
        if exclude_modules_prefix and any(
            mod.startswith(p) for p in exclude_modules_prefix
        ):
            return False
        if not include_stdlib and _is_probably_stdlib(mod, fn):
            return False
        return True

    def profiler(frame, event, arg):
        nonlocal stats, stack
        if event == "call":
            if not _allowed(frame):
                stack.append(FrameRec(key=("<filtered>", "<filtered>", 0), t0_ns=0))
                return profiler

            key = _frame_key(frame)
            stack.append(FrameRec(key=key, t0_ns=time.perf_counter_ns(), child_ns=0))
            return profiler

        if event == "return":
            if not stack:
                return profiler
            rec = stack.pop()
            if rec.key == ("<filtered>", "<filtered>", 0):
                # filtered frame placeholder
                # still counts as a child for parent? we do nothing
                return profiler

            dt = time.perf_counter_ns() - rec.t0_ns
            self_dt = dt - rec.child_ns
            st = stats.get(rec.key)
            if st is None:
                st = FuncStat()
                stats[rec.key] = st
            st.calls += 1
            st.inclusive_ns += int(dt)
            st.self_ns += int(self_dt)

            # add to parent's child time
            if stack:
                # only attribute child time to parent if parent isn't filtered placeholder
                if stack[-1].key != ("<filtered>", "<filtered>", 0):
                    stack[-1].child_ns += int(dt)
            return profiler

        return profiler

    old = sys.getprofile()
    sys.setprofile(profiler)
    t0 = time.perf_counter_ns()
    try:
        yield stats
    finally:
        sys.setprofile(old)
        t1 = time.perf_counter_ns()

        # Pretty print summary here (optional). If you prefer printing outside, delete below.
        total_ms = (t1 - t0) / 1_000_000
        items = list(stats.items())

        def sort_key(item):
            k, st = item
            if sort_by == "calls":
                return st.calls
            if sort_by == "self":
                return st.self_ns
            return st.inclusive_ns

        items.sort(key=sort_key, reverse=True)
        items = items[: int(limit)]

        print(
            f"\n[trace_calls_timed] captured in {total_ms:.3f} ms | sort_by={sort_by}"
        )
        for (mod, qual, firstlineno), st in items:
            if print_files:
                # cannot always recover file reliably from just key; we print module/line
                loc = f"{mod} (line {firstlineno})"
            else:
                loc = f"{mod}:{firstlineno}"

            print(
                f"  {st.calls:5d}  inc={_format_ns(st.inclusive_ns):>10}  "
                f"self={_format_ns(st.self_ns):>10}  {qual:<30}  {loc}"
            )


@contextmanager
def timed_trace(
    *,
    include_stdlib: bool = False,
    include_modules_prefix: Optional[Tuple[str, ...]] = None,
    exclude_modules_prefix: Tuple[str, ...] = (),
    limit: int = 50,
    sort_by: str = "inclusive",
    print_files: bool = True,
):
    """
    Convenience contextmanager wrapper around trace_calls_timed().

    Usage:
        with timed_trace(include_modules_prefix=("keydnn.",), sort_by="self"):
            opt.step()
    """
    with trace_calls_timed(
        include_stdlib=include_stdlib,
        include_modules_prefix=include_modules_prefix,
        exclude_modules_prefix=exclude_modules_prefix,
        limit=limit,
        sort_by=sort_by,
        print_files=print_files,
    ) as stats:
        yield stats


import sys
import time
from collections import Counter
from contextlib import contextmanager


@contextmanager
def trace_calls(only_modules=None, limit=200):
    """
    only_modules: list[str] | None
      e.g. ["src.keydnn", "keydnn", "numpy", "ctypes"]
    """
    counts = Counter()
    stack = []

    def ok_module(modname: str) -> bool:
        if not only_modules:
            return True
        return any(modname.startswith(p) for p in only_modules)

    def profiler(frame, event, arg):
        if event == "call":
            mod = frame.f_globals.get("__name__", "")
            if ok_module(mod):
                func = frame.f_code.co_name
                filename = frame.f_code.co_filename
                counts[(mod, func, filename)] += 1
                stack.append((mod, func))
        elif event == "c_call":
            # C-extension function call
            cfunc = getattr(arg, "__name__", repr(arg))
            mod = getattr(arg, "__module__", "") or ""
            if ok_module(mod):
                counts[(mod, f"[C]{cfunc}", "<built-in>")] += 1
        elif event in ("return", "c_return"):
            if stack:
                stack.pop()
        return profiler

    sys.setprofile(profiler)
    t0 = time.perf_counter()
    try:
        yield counts
    finally:
        dt = (time.perf_counter() - t0) * 1e3
        sys.setprofile(None)

        print(f"\n[trace_calls] captured in {dt:.3f} ms")
        for (mod, func, filename), n in counts.most_common(limit):
            print(f"{n:6d}  {mod}.{func}   ({filename})")


# ---- usage ----
# with trace_calls(only_modules=["src.keydnn", "keydnn", "numpy", "ctypes"], limit=120):
#     optimizer.step()
