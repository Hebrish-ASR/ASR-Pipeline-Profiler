"""The resource sampler: a separate process that watches the profiled one.

    python -m asrprof.sampler PID INTERVAL_MS OUT.jsonl

It runs outside the profiled process so it never competes for that process's GIL and its own
CPU time is not counted against any stage. Each tick it reads, for PID:

- `/proc/PID/task/*/schedstat`: nanoseconds each thread has spent on a CPU (ns precision,
  unlike the 10 ms ticks of `/proc/PID/stat`). The sum is the process's CPU time; the per-thread
  deltas show how many threads a stage really kept busy (ONNX Runtime's intra-op pool, for one).
- `/proc/PID/statm`: resident memory.
- the GPU backend from `asrprof.gpu`, if any.

Line 1 is a header (`{"header": ...}`); then one JSON object per tick:
`{"t": monotonic_ns, "cpu": ns, "thr": {tid: delta_ns}, "rss": bytes, "gpu": {...}}`.
Timestamps are CLOCK_MONOTONIC, the clock the tracer uses, so they line up with the spans.
It stops on SIGTERM or when PID exits.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

from . import gpu as gpumod

PAGE = os.sysconf("SC_PAGE_SIZE")
stop = False


def _thread_ns(pid: int) -> dict[int, int]:
    out = {}
    base = f"/proc/{pid}/task"
    for tid in os.listdir(base):
        try:
            with open(f"{base}/{tid}/schedstat") as f:
                out[int(tid)] = int(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            pass
    return out


def _comm(pid: int, tid: int) -> str:
    try:
        with open(f"/proc/{pid}/task/{tid}/comm") as f:
            return f.read().strip()
    except OSError:
        return "?"


def _rss(pid: int) -> int:
    with open(f"/proc/{pid}/statm") as f:
        return int(f.read().split()[1]) * PAGE


def main(argv: list[str]) -> int:
    global stop
    pid, interval, out = int(argv[0]), float(argv[1]) / 1000, argv[2]

    def _term(*_):
        global stop
        stop = True
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    g = gpumod.probe()
    names: dict[int, str] = {}
    with open(out, "w", buffering=1 << 16) as f:
        f.write(json.dumps({"header": {"pid": pid, "interval_ms": interval * 1000, "page": PAGE,
                                       "gpu": g.info() if g else None,
                                       "clk": "CLOCK_MONOTONIC"}}) + "\n")
        prev = _thread_ns(pid)
        nxt = time.monotonic()
        while not stop:
            nxt += interval
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()          # fell behind: don't burst to catch up
            try:
                t = time.monotonic_ns()
                cur = _thread_ns(pid)
                rss = _rss(pid)
            except (FileNotFoundError, ProcessLookupError):
                break
            thr = {}
            for tid, ns in cur.items():
                d = ns - prev.get(tid, 0)
                if d > 0:
                    thr[tid] = d
                    if tid not in names:
                        names[tid] = _comm(pid, tid)
            rec = {"t": t, "cpu": sum(cur.values()), "thr": thr, "rss": rss}
            if g:
                try:
                    rec["gpu"] = g.read(pid)
                except Exception:
                    pass
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            prev = cur
        f.write(json.dumps({"names": names, "self_cpu_ns": time.process_time_ns()}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
