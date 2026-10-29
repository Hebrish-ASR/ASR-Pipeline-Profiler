"""Pipeline files, audio input, and the two run modes.

A pipeline file is TOML:

    name = "hebrish-desktop"
    [vars]                       # ${name} anywhere below; ${env:NAME} reads the environment
    models = "~/.local/share/hebrish/models"
    [[stages]]
    name = "vad"                 # label in the report
    role = "vad"                 # vad | asr | text
    adapter = "hebrish.silero"   # a builtin (asrprof.adapters.BUILTIN) or "module:Class"
    threads = 1                  # every other key goes to the adapter; [vars] are defaults

`--var key=value` overrides a [vars] entry, and `--var stage.key=value` one stage key.

Modes:

- `batch`: the whole file at once, as a "transcribe this recording" flow: vad.detect, then
  asr.transcribe, then each text stage.
- `replay`: the file fed at real time (1x) in `chunk_ms` chunks, as if from a microphone; the
  VAD closes segments as it hears them and each is transcribed straight away; the text stages
  run on the joined text after the last chunk. Measures end-of-speech-to-text latency.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tomllib
import wave
from pathlib import Path

import numpy as np

from .adapters import SAMPLE_RATE, load_adapter
from .trace import Tracer

_VAR = re.compile(r"\$\{([^}]+)\}")


def _expand(value, vars_: dict):
    if isinstance(value, str):
        def sub(m):
            key = m.group(1)
            if key.startswith("env:"):
                name, _, default = key[4:].partition(":-")
                return os.environ.get(name, default)
            if key not in vars_:
                raise KeyError(f"undefined variable ${{{key}}}")
            return str(vars_[key])
        for _ in range(10):                   # vars may refer to vars
            new = _VAR.sub(sub, value)
            if new == value:
                break
            value = new
        return value
    if isinstance(value, list):
        return [_expand(v, vars_) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, vars_) for k, v in value.items()}
    return value


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def load_pipeline(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    doc = tomllib.loads(path.read_text())
    vars_ = dict(doc.get("vars", {}))
    stage_over: dict[str, dict] = {}
    for item in overrides or []:
        k, _, v = item.partition("=")
        if "." in k:
            st, _, key = k.partition(".")
            stage_over.setdefault(st, {})[key] = _coerce(v)
        else:
            vars_[k] = v
    vars_ = {k: _expand(v, vars_) for k, v in vars_.items()}
    stages = []
    for st in doc.get("stages", []):
        st = _expand(st, vars_)
        st.update(stage_over.get(st["name"], {}))
        stages.append(st)
    roles = [s["role"] for s in stages]
    if roles.count("asr") != 1 or roles.count("vad") > 1:
        raise ValueError(f"{path}: need exactly one asr stage and at most one vad stage (got {roles})")
    order = {"vad": 0, "asr": 1, "text": 2}
    if roles != sorted(roles, key=order.__getitem__):
        raise ValueError(f"{path}: stages must be in the order vad, asr, text...")
    return {"name": doc.get("name", path.stem), "description": doc.get("description", ""),
            "file": str(path.resolve()), "vars": vars_, "stages": stages,
            "replay": doc.get("replay", {})}


def load_audio(path: str | Path) -> np.ndarray:
    """16 kHz mono float32. 16 kHz mono 16-bit WAV is read directly; anything else goes through
    ffmpeg."""
    path = Path(path)
    try:
        with wave.open(str(path)) as w:
            if w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getsampwidth() == 2:
                return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    except (wave.Error, EOFError):
        pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path} is not 16 kHz mono 16-bit WAV and ffmpeg is not installed")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                          "-ar", str(SAMPLE_RATE), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


class Pipeline:
    def __init__(self, spec: dict, tracer: Tracer):
        self.spec, self.tracer = spec, tracer
        self.stages = []
        for st in spec["stages"]:
            cls = load_adapter(st["adapter"])
            # [vars] are defaults for every stage key (hebrish_src, say); the stage's own win
            cfg = {**spec["vars"], **{k: v for k, v in st.items() if k not in ("name", "role", "adapter")}}
            a = cls(st["name"], cfg, tracer)
            if a.role != st["role"]:
                raise ValueError(f"stage {st['name']}: adapter {st['adapter']} is a {a.role} adapter, not {st['role']}")
            self.stages.append(a)
        self.vad = next((s for s in self.stages if s.role == "vad"), None)
        self.asr = next(s for s in self.stages if s.role == "asr")
        self.text = [s for s in self.stages if s.role == "text"]

    def load(self):
        self.tracer.phase = "load"
        for s in self.stages:
            with self.tracer.span(s.name, "load", "stage"):
                s.load()
        self.infos = {s.name: s.info() for s in self.stages}

    def _text_stages(self, text: str) -> str:
        for s in self.text:
            with self.tracer.span(s.name, "process", "stage", chars_in=len(text)) as sp:
                text = s.process(text)
                sp.meta["chars_out"] = len(text)
        return text

    def run_batch(self, audio: np.ndarray) -> dict:
        tr = self.tracer
        vr = None
        if self.vad:
            with tr.span(self.vad.name, "detect", "stage") as sp:
                vr = self.vad.detect(audio)
                sp.meta["segments"] = len(vr.segments)
        with tr.span(self.asr.name, "transcribe", "stage") as sp:
            pieces = self.asr.transcribe(audio, vr)
            sp.meta["pieces"] = len(pieces)
        with tr.span("pipeline", "join", "glue"):
            text = " ".join(pieces)
        raw = text
        text = self._text_stages(text)
        return {"pieces": pieces, "asr_text": raw, "text": text,
                "vad_segments": [[a / SAMPLE_RATE, b / SAMPLE_RATE] for a, b in vr.segments] if vr else None}

    def run_replay(self, audio: np.ndarray, chunk_ms: float = 100.0, speed: float = 1.0) -> dict:
        """Feed `audio` in real time from a feeder thread; this thread plays the app's
        processing thread. `speed` > 1 replays faster than real time."""
        if not self.vad:
            raise ValueError("replay mode needs a vad stage")
        tr = self.tracer
        q: queue.Queue = queue.Queue()
        n = max(1, int(SAMPLE_RATE * chunk_ms / 1000))
        self.vad.stream_reset()
        t0 = time.monotonic_ns() + 50_000_000           # first chunk 50 ms from now

        def wall(pos: int) -> int:                        # when audio up to `pos` has "arrived"
            return t0 + int(pos / SAMPLE_RATE / speed * 1e9)

        def feeder():
            for a in range(0, len(audio), n):
                b = min(a + n, len(audio))
                due = wall(b)
                delay = (due - time.monotonic_ns()) / 1e9
                if delay > 0:
                    time.sleep(delay)
                q.put((a, b))
            q.put(None)

        th = threading.Thread(target=feeder, name="asrprof-feeder", daemon=True)
        th.start()
        pieces, segs, lags = [], [], []

        def handle(seg_list):
            for sg in seg_list:
                closed = time.monotonic_ns()
                with tr.span(self.asr.name, "segment", "stage", segment=len(segs)) as sp:
                    text = self.asr.transcribe_segment(sg.samples)
                done = time.monotonic_ns()
                if text:
                    pieces.append(text)
                tr.event("text_out", segment=len(segs), words=len(text.split()))
                segs.append({"segment": len(segs), "start_s": sg.start / SAMPLE_RATE, "end_s": sg.end / SAMPLE_RATE,
                             "speech_end_wall": wall(sg.end), "closed": closed, "asr_t0": sp.t0,
                             "asr_t1": sp.t1, "text_out": done, "words": len(text.split())})

        while (item := q.get()) is not None:
            a, b = item
            got = time.monotonic_ns()
            lags.append((got - wall(b)) / 1e6)
            with tr.span(self.vad.name, "feed", "stage", audio_end_s=b / SAMPLE_RATE):
                closed = self.vad.stream_feed(audio[a:b])
            handle(closed)
        with tr.span(self.vad.name, "flush", "stage"):
            closed = self.vad.stream_flush()
        handle(closed)
        end_of_audio = wall(len(audio))
        with tr.span("pipeline", "join", "glue"):
            text = " ".join(pieces)
        raw = text
        text = self._text_stages(text)
        return {"pieces": pieces, "asr_text": raw, "text": text, "segments": segs,
                "chunk_ms": chunk_ms, "speed": speed, "audio_end_wall": end_of_audio,
                "final_text_wall": time.monotonic_ns(),
                "chunk_lag_ms": {"max": round(max(lags), 2) if lags else 0,
                                 "mean": round(sum(lags) / len(lags), 2) if lags else 0, "chunks": len(lags)}}
