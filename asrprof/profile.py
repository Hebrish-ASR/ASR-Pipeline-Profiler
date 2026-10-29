"""One profile: load the pipeline, run it over the audio `repeat` times, write the report."""

from __future__ import annotations

import datetime as dt
import os
import platform
import time
from importlib import metadata
from pathlib import Path

from . import SCHEMA, __version__
from . import quality, report, samples as samplesmod
from .pipeline import Pipeline, load_audio, load_pipeline
from .trace import Tracer


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "?"


def _versions() -> dict:
    out = {}
    for pkg in ("numpy", "sherpa-onnx", "onnxruntime", "onnxruntime-gpu", "psutil", "PySide6"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def host_info(gpu: dict | None) -> dict:
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return {"os": platform.platform(), "python": platform.python_version(),
            "cpu": _cpu_name(), "cores": os.cpu_count(), "ram_gb": mem, "gpu": gpu,
            "packages": _versions()}


def run_profile(pipeline_file: str, audio_file: str, mode: str = "batch", overrides: list[str] | None = None,
                sample: str | None = None, repeat: int = 1, interval_ms: float = 20.0,
                chunk_ms: float | None = None, speed: float = 1.0, log=print) -> dict:
    spec = load_pipeline(pipeline_file, overrides)
    audio = load_audio(audio_file)
    duration = len(audio) / 16000
    tr = Tracer(interval_ms=interval_ms)
    tr.start()
    pipe = Pipeline(spec, tr)
    log(f"loading {len(pipe.stages)} stages")
    pipe.load()
    outputs = []
    for r in range(repeat):
        tr.run, tr.phase = r, "run"
        log(f"run {r + 1}/{repeat} ({mode}, {duration:.1f} s of audio)")
        with tr.span("pipeline", mode, "run"):
            if mode == "batch":
                out = pipe.run_batch(audio)
            elif mode == "replay":
                out = pipe.run_replay(audio, chunk_ms or spec["replay"].get("chunk_ms", 100.0), speed)
            else:
                raise ValueError(f"unknown mode {mode!r}")
        outputs.append(out)
        time.sleep(0.05)
    header, raw, trailer = tr.stop()
    spans = tr.dump()
    smp = report.Samples(raw, tr.t_start) if raw else None
    names = {str(k): v for k, v in (trailer.get("names") or {}).items()}
    order = [s.name for s in pipe.stages]

    runs = []
    for r, out in enumerate(outputs):
        sm = report.summarise_run(spans, r, order, smp, names, tr.t_start)
        sm["rtf"] = round(sm["wall_ms"] / 1000 / duration, 4) if duration else None
        if mode == "replay":
            sm["replay"] = report.summarise_replay(out, tr.t_start)
        runs.append(sm)

    ref = None
    if sample:
        ref = samplesmod.get(sample)
    final = outputs[-1]
    rep = {
        "schema": SCHEMA, "tool": f"asrprof {__version__}",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(header.get("gpu") if header else None),
        "pipeline": {"name": spec["name"], "file": spec["file"], "description": spec["description"],
                     "vars": spec["vars"],
                     "stages": [{"name": s.name, "role": s.role, "adapter": c["adapter"], "info": pipe.infos[s.name]}
                                for s, c in zip(pipe.stages, spec["stages"])]},
        "input": {"audio": str(Path(audio_file).resolve()), "duration_s": round(duration, 3),
                  "sample": ref["id"] if ref else None},
        "mode": mode, "repeat": repeat,
        "sampler": {"interval_ms": interval_ms, "ticks": len(raw),
                    "self_cpu_ms": round(trailer.get("self_cpu_ns", 0) / 1e6, 1),
                    "note": "the sampler is a separate process; its CPU is not in any stage"},
        "load": report.load_summary(spans, order, smp, names),
        "runs": runs,
        "output": {"text": final["text"], "asr_text": final["asr_text"], "pieces": final["pieces"],
                   "vad_segments": final.get("vad_segments")},
        "quality": quality.score(ref["text"], final["asr_text"], final["text"]) if ref else None,
        "spans": report.rel_spans(spans, tr.t_start),
        "events": [{**e, "t_ms": round((e.pop("t") - tr.t_start) / 1e6, 3)} for e in tr.events],
        "samples": smp.series() if smp else [],
        "thread_names": names,
    }
    return rep
