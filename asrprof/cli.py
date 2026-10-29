"""asrprof: profile a multi-model local ASR pipeline.

    asrprof run PIPELINE.toml --audio take.wav [--sample read-60s] [--mode batch|replay]
                [--var key=value ...] [--repeat N] [--out report.json] [--json]
    asrprof show report.json          the text summary of a saved report
    asrprof samples [ID]              list the reading samples, or print one to read aloud
    asrprof record ID [--out take.wav]  read a sample aloud into the default microphone
    asrprof serve [--host H] [--port P]  HTTP API (docs/server.md)
    asrprof gui [report.json]         the desktop app

Reports are JSON (schema asrprof.report/1, docs/report-schema.md). `--json` prints the report
to stdout and the progress to stderr, for agents and scripts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import samples as samplesmod

DATA = Path.home() / ".local" / "share" / "asrprof"


def _fmt(v, unit="", nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def _table(rows: list[list], head: list[str]) -> str:
    cells = [head] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(head))]
    line = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(head), line(["-" * x for x in w])] + [line(r) for r in cells[1:]])


def summary_text(rep: dict) -> str:
    h, inp = rep["host"], rep["input"]
    g = h.get("gpu") or {}
    out = [f"{rep['pipeline']['name']}  ·  {rep['mode']}  ·  {inp['duration_s']:.1f} s audio"
           f"{'  ·  sample ' + inp['sample'] if inp['sample'] else ''}",
           f"{h['cpu']} ({h['cores']} threads), GPU: {g.get('name', 'no telemetry')}", ""]
    st_info = {s["name"]: s for s in rep["pipeline"]["stages"]}
    rows = []
    for name, ld in rep["load"].items():
        i = st_info[name]["info"]
        rows.append([name, st_info[name]["role"], i.get("provider", "-"), _fmt(i.get("threads")),
                     _fmt(ld["wall_ms"], " ms", 0), _fmt(ld.get("rss_delta_mb"), " MB")])
    out += ["Load", _table(rows, ["stage", "role", "provider", "threads", "load", "RSS +"]), ""]
    for run in rep["runs"]:
        out.append(f"Run {run['run'] + 1}: {run['wall_ms']:.0f} ms wall, RTF {run['rtf']:.3f}, models computing "
                   f"{run['infer_ms']:.0f} ms, not computing {run['not_inferring_ms']:.0f} ms; "
                   f"CPU avg {_fmt(run['cpu_avg_cores'], '', 2)} / max {_fmt(run['cpu_max_cores'], '', 2)} cores; "
                   f"peak RSS {_fmt(run['rss_max_mb'], ' MB')}")
        rows = []
        for name, s in run["stages"].items():
            rows.append([name, _fmt(s["wall_ms"], "", 1), _fmt(s["infer_ms"], "", 1), _fmt(s["glue_ms"], "", 1),
                         s["infer_calls"], _fmt(s["cpu_avg_cores"], "", 2), _fmt(s["cpu_max_cores"], "", 2),
                         _fmt(s["threads_active"]), _fmt(s["gpu_busy_avg"], "%"), _fmt(s["gpu_busy_max"], "%"),
                         _fmt(s["gaps"]["total_ms"], "", 2), _fmt(s["gaps"]["max_ms"], "", 2)])
        out += [_table(rows, ["stage", "wall ms", "infer ms", "glue ms", "calls", "CPU avg", "CPU max",
                              "threads", "GPU avg", "GPU max", "gaps ms", "gap max"]), ""]
        rows = []
        for j in run["junctions"]:
            top = ", ".join(f"{k} {v:.2f}" for k, v in list(j["glue_ms"].items())[:3])
            rows.append([j["junction"], j["count"], _fmt(j["ms"], "", 3), _fmt(j["mean_ms"], "", 3),
                         _fmt(j["max_ms"], "", 3), _fmt(j["unaccounted_ms"], "", 3), top or "-"])
        out += ["Junctions (ms)", _table(rows, ["junction", "n", "total", "mean", "max", "unaccounted", "glue inside"]), ""]
        if "replay" in run:
            rp = run["replay"]
            lat, vc, ar = rp["latency_ms"] or {}, rp["vad_close_ms"] or {}, rp["asr_ms"] or {}
            out += [f"Replay ({rp['speed']}x, {rp['chunk_ms']:.0f} ms chunks): {len(rp['segments'])} segments; "
                    f"end of speech → text: mean {_fmt(lat.get('mean'))} / p95 {_fmt(lat.get('p95'))} / max "
                    f"{_fmt(lat.get('max'))} ms (VAD closing {_fmt(vc.get('mean'))} ms, ASR {_fmt(ar.get('mean'))} ms "
                    f"mean); last chunk → final text {_fmt(rp['tail_ms'])} ms; worst chunk lag "
                    f"{_fmt(rp['chunk_lag_ms']['max'])} ms", ""]
    q = rep.get("quality")
    if q:
        p = q["paragraphs"]
        out += [f"Quality vs {rep['input']['sample']}: WER {q['asr']['wer']:.1%} ({q['asr']['errors']}/"
                f"{q['asr']['ref_words']}); paragraph breaks {p['hits']}/{p['ref_breaks']} found, "
                f"{p['hyp_breaks']} placed (P {p['precision']:.2f} R {p['recall']:.2f} F1 {p['f1']:.2f}, "
                f"±{p['tolerance_words']} words)", ""]
    out += ["Output", rep["output"]["text"]]
    return "\n".join(out)


def default_out(rep: dict) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return DATA / "reports" / f"{stamp}_{rep['pipeline']['name']}_{rep['mode']}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="asrprof", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="profile a pipeline on one recording")
    r.add_argument("pipeline")
    r.add_argument("--audio", required=True)
    r.add_argument("--sample", help="reading sample id (or .md path) the audio reads; enables quality checks")
    r.add_argument("--mode", choices=["batch", "replay"], default="batch")
    r.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--interval", type=float, default=20.0, help="sampler interval, ms (20)")
    r.add_argument("--chunk-ms", type=float, help="replay chunk size, ms (100)")
    r.add_argument("--speed", type=float, default=1.0, help="replay speed (1.0 = real time)")
    r.add_argument("--out", help="report path (default ~/.local/share/asrprof/reports/...)")
    r.add_argument("--json", action="store_true", help="print the report JSON to stdout")
    s = sub.add_parser("show", help="print the summary of a saved report")
    s.add_argument("report")
    sm = sub.add_parser("samples", help="list reading samples or print one")
    sm.add_argument("id", nargs="?")
    rc = sub.add_parser("record", help="read a sample aloud into the microphone")
    rc.add_argument("id")
    rc.add_argument("--out")
    sv = sub.add_parser("serve", help="HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    g = sub.add_parser("gui", help="desktop app")
    g.add_argument("report", nargs="?", help="a report to open")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .profile import run_profile
        log = (lambda m: print(m, file=sys.stderr)) if a.json else print
        rep = run_profile(a.pipeline, a.audio, a.mode, a.var, a.sample, a.repeat, a.interval,
                          a.chunk_ms, a.speed, log=log)
        out = Path(a.out) if a.out else default_out(rep)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        if a.json:
            json.dump(rep, sys.stdout, ensure_ascii=False)
            print()
        else:
            print(summary_text(rep))
            print(f"\nreport: {out}")
        return 0
    if a.cmd == "show":
        print(summary_text(json.loads(Path(a.report).read_text())))
        return 0
    if a.cmd == "samples":
        if a.id:
            print(samplesmod.get(a.id)["text"])
        else:
            for x in samplesmod.all_samples():
                print(f"{x['id']:12} {x.get('target_seconds', '?'):>4} s  {x['words']:>4} words  "
                      f"{x.get('paragraphs', '?')} paragraphs  {x.get('title', '')}")
        return 0
    if a.cmd == "record":
        from .record import record_cli
        return record_cli(a.id, a.out)
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port)
        return 0
    if a.cmd == "gui":
        from .gui.app import main as gui_main
        return gui_main(a.report)
    return 1
