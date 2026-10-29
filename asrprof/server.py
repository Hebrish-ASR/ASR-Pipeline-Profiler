"""HTTP API, for a server or a benchmarking agent on another machine.

    asrprof serve --host 0.0.0.0 --port 8765

    GET  /health                       {"ok": true, "version": ...}
    GET  /pipelines                    pipeline files in pipelines/
    GET  /samples                      reading samples (id, title, text)
    POST /runs?pipeline=hebrish-desktop&mode=batch&sample=read-60s&repeat=2&var=asr.threads=8
         body: the audio file (any format ffmpeg reads)
         -> the report JSON (schema asrprof.report/1) when the profile finishes
    GET  /runs                         saved reports (newest first)
    GET  /runs/<file>                  one saved report

One profile at a time (a second POST waits): two at once would share the CPU and spoil both.
Each runs in a child process (`asrprof run`), so the server's own work is outside the numbers.
No authentication: bind to localhost or a private network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from . import samples as samplesmod
from .cli import DATA

ROOT = Path(__file__).resolve().parent.parent
PIPELINES = ROOT / "pipelines"
REPORTS = DATA / "reports"
_one = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = f"asrprof/{__version__}"

    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._json(200, {"ok": True, "version": __version__})
        if u.path == "/pipelines":
            return self._json(200, [{"name": p.stem, "file": str(p)} for p in sorted(PIPELINES.glob("*.toml"))])
        if u.path == "/samples":
            return self._json(200, [{k: s.get(k) for k in ("id", "title", "target_seconds", "paragraphs", "words", "text")}
                                    for s in samplesmod.all_samples()])
        if u.path == "/runs":
            files = sorted(REPORTS.glob("*.json"), reverse=True) if REPORTS.is_dir() else []
            return self._json(200, [f.name for f in files])
        if u.path.startswith("/runs/"):
            f = REPORTS / Path(u.path[6:]).name
            if f.is_file():
                return self._json(200, json.loads(f.read_text()))
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/runs":
            return self._json(404, {"error": "not found"})
        q = parse_qs(u.query)
        name = q.get("pipeline", [""])[0]
        pipe = PIPELINES / f"{Path(name).name}.toml"
        if not pipe.is_file():
            return self._json(400, {"error": f"no pipeline {name!r}"})
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return self._json(400, {"error": "send the audio as the request body"})
        with tempfile.TemporaryDirectory(prefix="asrprof-") as tmp:
            audio = Path(tmp) / "input"
            audio.write_bytes(self.rfile.read(n))
            mode = q.get("mode", ["batch"])[0]
            REPORTS.mkdir(parents=True, exist_ok=True)
            out = REPORTS / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{pipe.stem}_{mode}.json"
            args = [sys.executable, "-m", "asrprof", "run", str(pipe), "--audio", str(audio), "--mode", mode,
                    "--repeat", q.get("repeat", ["1"])[0], "--out", str(out)]
            for key in ("sample", "speed", "interval", "chunk-ms"):
                if key in q:
                    args += [f"--{key}", q[key][0]]
            for v in q.get("var", []):
                args += ["--var", v]
            env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            with _one:
                p = subprocess.run(args, capture_output=True, text=True, env=env)
            if p.returncode:
                return self._json(500, {"error": "profile failed", "stderr": p.stderr[-4000:]})
            return self._json(200, json.loads(out.read_text()))


def serve(host: str, port: int):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"asrprof {__version__} on http://{host}:{port}  (pipelines: {PIPELINES})")
    srv.serve_forever()
