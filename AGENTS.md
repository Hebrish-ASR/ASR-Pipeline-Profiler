# Agent notes

Profiler for multi-model local ASR pipelines. Public repo (Hebrish-ASR/ASR-Pipeline-Profiler):
write everything for a stranger; no machine names, account ids or credentials.

- Start with `README.md`, then `docs/concepts.md` (what every number means and how it is
  measured). Changing a metric means changing `docs/concepts.md` and `docs/report-schema.md` in the
  same commit; bump `SCHEMA` in `asrprof/__init__.py` for a breaking change to the report.
- Layout: `asrprof/trace.py` (spans), `sampler.py` (separate-process /proc + GPU sampler),
  `gpu.py`, `pipeline.py` (TOML, audio, batch/replay), `report.py` (stage resources, junctions,
  replay latency), `quality.py`, `profile.py` (one profile end to end), `cli.py`, `server.py`,
  `paths.py` (bundled vs `~/.config/asrprof` pipelines and samples, data dirs), `gui/` (PySide6;
  `style.py` is the Hebrish brand palette, light and dark), `data/` (pipelines, samples, icon, font), `adapters/` (role base classes, Hebrish adapters, `python` text adapter).
- Setup: `uv venv --system-site-packages --python /usr/bin/python3 && uv pip install -e '.[hebrish,test]'`.
  Tests (no models): `.venv/bin/python -m pytest -q tests`.
- The Hebrish adapters must follow the app's code (Hebrish-ASR/Hebrish-App `desktop/hebrish/`
  `engine.py`, `vad.py`, `paragraphs.py`). When the app changes how it cuts, pads or tokenises,
  update the adapter and say which app version it matches.
- GUI checks run offscreen (`QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`), never on a
  live desktop. Screenshots for the docs are in `docs/img/`, made that way.
- Findings from real runs go in `docs/findings/YYYY-MM-DD-<pipeline>.md`: setup with versions,
  input (say if it is synthetic), commands, numbers, and each conclusion marked measured or
  inferred.
- Package: `./build-deb.sh` -> `dist/asrprof_<version>-1_amd64.deb` (bundles sherpa-onnx and
  onnxruntime in `/usr/lib/asrprof`, everything else from apt; pinned to the build's python3
  minor). Temp files go under `build/` because `/tmp` can be short of quota. Check an installed
  copy with `asrprof pipelines` and a short `asrprof run`.
- Releases: small-project tier, tags `v1.0`, `v1.1`, … (`asrprof.__version__` matches). None cut yet.
