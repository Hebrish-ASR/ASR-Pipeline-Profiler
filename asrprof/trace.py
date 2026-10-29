"""Spans: what ran when, in the profiled process.

A span is one timed piece of work: `stage` (the pipeline stage it belongs to), `name`, `kind`
and `phase`. Kinds:

- `infer`: a model is computing (a sherpa-onnx decode, an ONNX Runtime `run`, a VAD window batch).
- `glue`: work between model calls that is not a model: resampling, normalising, cutting,
  tokenising, padding, copying arrays. Glue is what fills a junction.
- `stage`: the whole of one stage's turn (wraps its infer and glue spans).
- `load`: making a model ready (file read, session creation, warm-up).
- `run`: one pass of the whole pipeline over the input.

Timestamps are `time.monotonic_ns()` (CLOCK_MONOTONIC, shared with the sampler process).
Each span also records the process CPU time (`time.process_time_ns()`) at both ends, which gives
an exact average CPU for spans too short for the sampler to see.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Span:
    id: int
    parent: int | None
    stage: str
    name: str
    kind: str
    phase: str
    t0: int
    t1: int = 0
    cpu0: int = 0
    cpu1: int = 0
    thread: int = 0
    run: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return (self.t1 - self.t0) / 1e6


class Tracer:
    def __init__(self, interval_ms: float = 20.0, sample: bool = True):
        self.spans: list[Span] = []
        self.events: list[dict] = []
        self.interval_ms = interval_ms
        self._stack = threading.local()
        self._lock = threading.Lock()
        self._next = 0
        self.run: int | None = None
        self.phase = "load"
        self._sampler: subprocess.Popen | None = None
        self._sample_file: Path | None = None
        self.sample = sample
        self.t_start = time.monotonic_ns()

    # sampler ---------------------------------------------------------------------------------
    def start(self):
        if self.sample:
            fd, path = tempfile.mkstemp(prefix="asrprof-", suffix=".jsonl")
            os.close(fd)
            self._sample_file = Path(path)
            pkg_root = str(Path(__file__).resolve().parent.parent)
            env = dict(os.environ, PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
            self._sampler = subprocess.Popen(
                [sys.executable, "-m", "asrprof.sampler", str(os.getpid()), str(self.interval_ms), path],
                env=env, stdin=subprocess.DEVNULL)
            # wait for the header so the first span is covered
            for _ in range(200):
                if self._sample_file.stat().st_size:
                    break
                time.sleep(0.01)
        self.t_start = time.monotonic_ns()

    def stop(self) -> tuple[dict | None, list[dict], dict]:
        """Stop the sampler and return (header, samples, trailer)."""
        if not self._sampler:
            return None, [], {}
        time.sleep(self.interval_ms / 1000 * 2)       # one more tick after the last span
        self._sampler.terminate()
        self._sampler.wait(timeout=10)
        import json
        header, samples, trailer = None, [], {}
        for line in self._sample_file.read_text().splitlines():
            rec = json.loads(line)
            if "header" in rec:
                header = rec["header"]
            elif "t" in rec:
                samples.append(rec)
            else:
                trailer = rec
        self._sample_file.unlink(missing_ok=True)
        return header, samples, trailer

    # spans -----------------------------------------------------------------------------------
    @contextmanager
    def span(self, stage: str, name: str, kind: str, **meta):
        stack = getattr(self._stack, "s", None)
        if stack is None:
            stack = self._stack.s = []
        with self._lock:
            sid = self._next
            self._next += 1
        sp = Span(sid, stack[-1].id if stack else None, stage, name, kind, self.phase, 0,
                  thread=threading.get_native_id(), run=self.run, meta=meta)
        stack.append(sp)
        sp.cpu0 = time.process_time_ns()
        sp.t0 = time.monotonic_ns()
        try:
            yield sp
        finally:
            sp.t1 = time.monotonic_ns()
            sp.cpu1 = time.process_time_ns()
            stack.pop()
            with self._lock:
                self.spans.append(sp)

    def event(self, name: str, **data):
        """A point in time (an audio chunk arriving, a segment closing, text out)."""
        with self._lock:
            self.events.append({"t": time.monotonic_ns(), "name": name, "run": self.run, **data})

    def dump(self) -> list[dict]:
        return [asdict(s) for s in sorted(self.spans, key=lambda s: (s.t0, s.id))]


class NullTracer(Tracer):
    """For code paths used outside a profile (tests)."""

    def __init__(self):
        super().__init__(sample=False)
