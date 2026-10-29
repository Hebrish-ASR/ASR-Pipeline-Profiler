"""Turn spans + resource samples into the report: per-stage resources, junctions, latency.

All times in the report are milliseconds from the profile's start (`t_ms`, `t0_ms`, `t1_ms`).
CPU is given in **cores** (1.0 = one core fully busy; the machine has `host.cores`), because a
percentage of the whole machine hides a single-threaded stage and a percentage of one core
reads oddly above 100.

Definitions (docs/concepts.md has the long form):

- **stage resources**: over the stage's `stage` spans in a run. `cpu_avg_cores` is exact (process
  CPU time at the span ends / wall time); `cpu_max_cores` is the busiest sampler interval inside
  those spans (null if every span was shorter than one interval). `threads_active`: threads that
  averaged at least 0.1 core during the stage, from the sampler's per-thread times.
- **junction** (between stages A and B): from the end of A's last `infer` span to the start of B's
  first `infer` span: the time no model of either stage was computing while data moved from one
  to the other. Split into the `glue` spans inside it, by name, and `unaccounted` (orchestration,
  span bookkeeping, Python between calls).
- **gaps** (inside a stage): the same, between consecutive `infer` spans of one stage (between
  the ASR's pieces, between the paragraph model's windows).
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _r(x, n=2):
    return None if x is None else round(x, n)


class Samples:
    def __init__(self, raw: list[dict], t_start: int):
        self.raw = raw
        self.t0 = t_start
        # interval records: (t_prev, t, cores, thr, rss, gpu)
        self.iv = []
        for a, b in zip(raw, raw[1:]):
            dt = b["t"] - a["t"]
            if dt > 0:
                # summed from per-thread deltas, not the difference of process totals: a thread that
                # exits takes its CPU time out of the total and would make the interval negative
                self.iv.append((a["t"], b["t"], sum(b["thr"].values()) / dt, b["thr"], b["rss"], b.get("gpu")))

    def within(self, windows: list[tuple[int, int]]):
        """Intervals whose midpoint falls in any window."""
        out = []
        for iv in self.iv:
            mid = (iv[0] + iv[1]) // 2
            if any(a <= mid <= b for a, b in windows):
                out.append(iv)
        return out

    def series(self) -> list[dict]:
        out = []
        for a, b, cores, thr, rss, g in self.iv:
            rec = {"t_ms": _r((b - self.t0) / 1e6, 1), "cores": _r(cores, 3), "rss_mb": _r(rss / 2**20, 1),
                   "threads": {str(k): _r(v / (b - a), 3) for k, v in thr.items() if v / (b - a) >= 0.01}}
            if g:
                rec["gpu"] = g
            out.append(rec)
        return out


def _resources(spans, smp: Samples | None, names: dict) -> dict:
    wall = sum(s["t1"] - s["t0"] for s in spans)
    cpu = sum(s["cpu1"] - s["cpu0"] for s in spans)
    out = {"wall_ms": _r(wall / 1e6, 3), "calls": len(spans),
           "cpu_avg_cores": _r(cpu / wall, 3) if wall else None,
           "cpu_max_cores": None, "threads_active": None, "top_threads": [],
           "rss_max_mb": None, "gpu_busy_avg": None, "gpu_busy_max": None, "vram_max_mb": None,
           "samples": 0}
    if not smp:
        return out
    ivs = smp.within([(s["t0"], s["t1"]) for s in spans])
    out["samples"] = len(ivs)
    if not ivs:
        return out
    out["cpu_max_cores"] = _r(max(iv[2] for iv in ivs), 3)
    out["rss_max_mb"] = _r(max(iv[4] for iv in ivs) / 2**20, 1)
    per = defaultdict(int)
    span_t = sum(iv[1] - iv[0] for iv in ivs)
    for iv in ivs:
        for tid, ns in iv[3].items():
            per[tid] += ns
    busy = {tid: ns / span_t for tid, ns in per.items()}
    out["threads_active"] = sum(1 for v in busy.values() if v >= 0.1)
    out["top_threads"] = [{"tid": int(t), "name": names.get(str(t), names.get(t, "?")), "cores": _r(v, 3)}
                          for t, v in sorted(busy.items(), key=lambda kv: -kv[1])[:6]]
    gb = [iv[5]["busy"] for iv in ivs if iv[5] and iv[5].get("busy") is not None]
    if gb:
        out["gpu_busy_avg"] = _r(statistics.fmean(gb), 1)
        out["gpu_busy_max"] = max(gb)
    vr = [iv[5].get("proc_vram_mb", iv[5].get("vram_mb")) for iv in ivs if iv[5]]
    vr = [v for v in vr if v is not None]
    if vr:
        out["vram_max_mb"] = max(vr)
    return out


def _rss_at(smp: Samples | None, t: int):
    if not smp or not smp.raw:
        return None
    best = min(smp.raw, key=lambda r: abs(r["t"] - t))
    return best["rss"]


def _gap_breakdown(glue, a: int, b: int) -> tuple[dict, int]:
    by = defaultdict(int)
    for g in glue:
        lo, hi = max(g["t0"], a), min(g["t1"], b)
        if hi > lo:
            by[f'{g["stage"]}.{g["name"]}'] += hi - lo
    return {k: _r(v / 1e6, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])}, sum(by.values())


def summarise_run(spans: list[dict], run: int, stage_order: list[str], smp, names, t_start) -> dict:
    rs = [s for s in spans if s["run"] == run and s["phase"] == "run"]
    root = next(s for s in rs if s["kind"] == "run")
    infer = defaultdict(list)
    for s in rs:
        if s["kind"] == "infer":
            infer[s["stage"]].append(s)
    glue = [s for s in rs if s["kind"] == "glue"]
    for v in infer.values():
        v.sort(key=lambda s: s["t0"])

    stages = {}
    for name in stage_order:
        st_spans = [s for s in rs if s["stage"] == name and s["kind"] == "stage"]
        res = _resources(st_spans, smp, names)
        inf = infer.get(name, [])
        res["infer_ms"] = _r(sum(s["t1"] - s["t0"] for s in inf) / 1e6, 3)
        res["infer_calls"] = len(inf)
        res["glue_ms"] = _r(sum(s["t1"] - s["t0"] for s in glue if s["stage"] == name) / 1e6, 3)
        # gaps only inside one turn of the stage (same parent span): in replay, the time between
        # two VAD feeds is waiting for audio, not a gap
        turns = defaultdict(list)
        for x in inf:
            turns[x["parent"]].append(x)
        gaps, g_by = [], defaultdict(float)
        for xs in turns.values():
            for a, b in zip(xs, xs[1:]):
                gaps.append(b["t0"] - a["t1"])
                for k, v in _gap_breakdown(glue, a["t1"], b["t0"])[0].items():
                    g_by[k] += v
        g_by = {k: _r(v, 3) for k, v in sorted(g_by.items(), key=lambda kv: -kv[1])}
        res["gaps"] = {"count": len(gaps), "total_ms": _r(sum(gaps) / 1e6, 3),
                       "mean_ms": _r(statistics.fmean(gaps) / 1e6, 3) if gaps else None,
                       "p50_ms": _r(_pct(gaps, 0.5) / 1e6, 3) if gaps else None,
                       "max_ms": _r(max(gaps) / 1e6, 3) if gaps else None,
                       "glue_ms": g_by}
        stages[name] = res

    # Junctions: walk every infer span in time order; each change of stage (A's call ends, B's
    # begins) is one crossing. Batch has one crossing per stage pair; replay has one per segment.
    seq = sorted((s for v in infer.values() for s in v), key=lambda s: s["t0"])
    crossings = defaultdict(list)
    replay = root["name"] == "replay"
    first = stage_order[0] if stage_order else None
    for a, b in zip(seq, seq[1:]):
        # in replay, going back to the first stage (the VAD's next chunk) is waiting for audio,
        # not a handoff of data
        if a["stage"] != b["stage"] and b["t0"] >= a["t1"] and not (replay and b["stage"] == first):
            crossings[f'{a["stage"]}→{b["stage"]}'].append((a["t1"], b["t0"], a["stage"], b["stage"]))
    if seq and not replay:                    # in replay the first call waits for audio
        crossings[f'input→{seq[0]["stage"]}'].append((root["t0"], seq[0]["t0"], "input", seq[0]["stage"]))
    if seq:
        last = max(seq, key=lambda s: s["t1"])
        crossings[f'{last["stage"]}→output'].append((last["t1"], root["t1"], last["stage"], "output"))
    junctions = []
    for label, xs in crossings.items():
        by = defaultdict(float)
        glue_ns = 0
        for a, b, *_ in xs:
            part, ns = _gap_breakdown(glue, a, b)
            glue_ns += ns
            for k, v in part.items():
                by[k] += v
        durs = [b - a for a, b, *_ in xs]
        junctions.append({"junction": label, "from": xs[0][2], "to": xs[0][3], "count": len(xs),
                          "ms": _r(sum(durs) / 1e6, 3), "mean_ms": _r(statistics.fmean(durs) / 1e6, 3),
                          "p50_ms": _r(_pct(durs, .5) / 1e6, 3), "max_ms": _r(max(durs) / 1e6, 3),
                          "glue_ms": {k: _r(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
                          "unaccounted_ms": _r((sum(durs) - glue_ns) / 1e6, 3),
                          "at_ms": [[_r((a - t_start) / 1e6, 3), _r((b - t_start) / 1e6, 3)] for a, b, *_ in xs[:500]]})
    edge = ("input", "output")
    junctions.sort(key=lambda j: j["at_ms"][0][0])  # first crossing first: pipeline order

    infer_all = sum(s["t1"] - s["t0"] for v in infer.values() for s in v)
    wall = root["t1"] - root["t0"]
    total = _resources([root], smp, names)
    return {"run": run, "wall_ms": _r(wall / 1e6, 3), "infer_ms": _r(infer_all / 1e6, 3),
            "not_inferring_ms": _r((wall - infer_all) / 1e6, 3),
            "junction_total_ms": _r(sum(j["ms"] for j in junctions if j["from"] not in edge
                                        and j["to"] not in edge), 3),
            "cpu_avg_cores": total["cpu_avg_cores"], "cpu_max_cores": total["cpu_max_cores"],
            "rss_max_mb": total["rss_max_mb"], "stages": stages, "junctions": junctions}


def summarise_replay(out: dict, t_start: int) -> dict:
    segs = out["segments"]
    rows = []
    for s in segs:
        rows.append({"segment": s["segment"], "start_s": _r(s["start_s"], 3), "end_s": _r(s["end_s"], 3),
                     "words": s["words"],
                     "vad_close_ms": _r((s["closed"] - s["speech_end_wall"]) / 1e6, 1),
                     "junction_ms": _r((s["asr_t0"] - s["closed"]) / 1e6, 3),
                     "asr_ms": _r((s["asr_t1"] - s["asr_t0"]) / 1e6, 1),
                     "latency_ms": _r((s["text_out"] - s["speech_end_wall"]) / 1e6, 1)})
    lat = [r["latency_ms"] for r in rows]

    def agg(xs):
        return {"mean": _r(statistics.fmean(xs), 1), "p50": _r(_pct(xs, .5), 1), "p95": _r(_pct(xs, .95), 1),
                "max": _r(max(xs), 1)} if xs else None
    return {"segments": rows, "latency_ms": agg(lat),
            "vad_close_ms": agg([r["vad_close_ms"] for r in rows]),
            "asr_ms": agg([r["asr_ms"] for r in rows]),
            "tail_ms": _r((out["final_text_wall"] - out["audio_end_wall"]) / 1e6, 1),
            "chunk_ms": out["chunk_ms"], "speed": out["speed"], "chunk_lag_ms": out["chunk_lag_ms"]}


def load_summary(spans, stage_order, smp, names) -> dict:
    out = {}
    for name in stage_order:
        top = [s for s in spans if s["stage"] == name and s["phase"] == "load" and s["kind"] == "stage"]
        if not top:
            continue
        r = _resources(top, smp, names)
        a, b = _rss_at(smp, top[0]["t0"]), _rss_at(smp, top[-1]["t1"])
        r["rss_delta_mb"] = _r((b - a) / 2**20, 1) if a is not None and b is not None else None
        r["steps"] = {s["name"]: _r((s["t1"] - s["t0"]) / 1e6, 3) for s in spans
                      if s["stage"] == name and s["phase"] == "load" and s["kind"] == "load"}
        out[name] = r
    return out


def rel_spans(spans: list[dict], t_start: int) -> list[dict]:
    out = []
    for s in spans:
        d = {k: v for k, v in s.items() if k not in ("t0", "t1", "cpu0", "cpu1")}
        d["t0_ms"] = _r((s["t0"] - t_start) / 1e6, 3)
        d["t1_ms"] = _r((s["t1"] - t_start) / 1e6, 3)
        d["ms"] = _r((s["t1"] - s["t0"]) / 1e6, 3)
        d["cpu_ms"] = _r((s["cpu1"] - s["cpu0"]) / 1e6, 3)
        out.append(d)
    return out
