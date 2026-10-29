"""Stage adapters: the seam between the profiler and a real pipeline's code.

A pipeline has up to three roles, run in this order:

- `vad` (optional in batch mode, required in replay): finds speech in the audio.
- `asr` (required): audio to text.
- `text` (any number, in file order): text to text, e.g. paragraph breaks, punctuation,
  an LLM post-processing pass.

An adapter wraps one stage. It does its own model calls inside `self.infer(...)` spans and its
own non-model work inside `self.glue(...)` spans; that split is what makes junctions measurable.
See docs/adapters.md for writing one.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class VadResult:
    """Speech segments as (start, end) sample indices, plus anything the next stage may reuse
    (`extra`, e.g. a per-frame speech mask)."""
    segments: list[tuple[int, int]]
    extra: dict = field(default_factory=dict)


@dataclass
class Segment:
    """A speech segment closed by a streaming VAD: audio positions in samples."""
    start: int
    end: int
    samples: np.ndarray


class Adapter:
    role = ""

    def __init__(self, name: str, cfg: dict, tracer):
        self.name, self.cfg, self.tracer = name, cfg, tracer

    def infer(self, name: str, **meta):
        return self.tracer.span(self.name, name, "infer", **meta)

    def glue(self, name: str, **meta):
        return self.tracer.span(self.name, name, "glue", **meta)

    @contextmanager
    def loading(self, name: str, **meta):
        with self.tracer.span(self.name, name, "load", **meta) as sp:
            yield sp

    def load(self):
        pass

    def info(self) -> dict:
        """What the report should say about this stage: model path and size, execution provider
        (`provider`: where the compute runs; anything but "cpu" is GPU/NPU offload), threads."""
        return {}


class VadAdapter(Adapter):
    role = "vad"

    def detect(self, audio: np.ndarray) -> VadResult:
        raise NotImplementedError

    def stream_reset(self):
        raise NotImplementedError(f"{type(self).__name__} has no streaming mode")

    def stream_feed(self, chunk: np.ndarray) -> list[Segment]:
        raise NotImplementedError

    def stream_flush(self) -> list[Segment]:
        return []


class AsrAdapter(Adapter):
    role = "asr"

    def transcribe(self, audio: np.ndarray, vad: VadResult | None) -> list[str]:
        """The whole input; returns its text pieces in order."""
        raise NotImplementedError

    def transcribe_segment(self, samples: np.ndarray) -> str:
        """One closed segment (replay mode)."""
        raise NotImplementedError


class TextAdapter(Adapter):
    role = "text"

    def process(self, text: str) -> str:
        raise NotImplementedError


BUILTIN = {
    "hebrish.silero": "asrprof.adapters.hebrish:SileroVad",
    "hebrish.energy": "asrprof.adapters.hebrish:EnergyVad",
    "hebrish.parakeet": "asrprof.adapters.hebrish:Parakeet",
    "hebrish.paragraphs": "asrprof.adapters.hebrish:Paragraphs",
    "python": "asrprof.adapters.python_callable:PythonText",
}


def load_adapter(spec: str):
    """`spec` is a BUILTIN key or `package.module:Class`."""
    target = BUILTIN.get(spec, spec)
    if ":" not in target:
        raise ValueError(f"unknown adapter {spec!r} (builtins: {', '.join(BUILTIN)})")
    mod, cls = target.split(":", 1)
    return getattr(importlib.import_module(mod), cls)
