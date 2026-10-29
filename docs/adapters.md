# Writing an adapter

A stage in a pipeline file names an adapter: a builtin (`asrprof/adapters/__init__.py`,
`BUILTIN`) or `package.module:Class` importable from the working directory or `PYTHONPATH`.
Every key of the stage other than `name`, `role` and `adapter` is in `self.cfg`, with `[vars]` as
defaults.

```python
from asrprof.adapters import AsrAdapter

class MyAsr(AsrAdapter):
    def load(self):
        with self.loading("session"):          # timed as a load step
            self.model = make_model(self.cfg["model"], threads=int(self.cfg.get("threads", 4)))
        with self.loading("warmup"):
            self.model.run(silence())

    def info(self):                            # goes in the report as the stage's info
        return {"model": self.cfg["model"], "provider": "cpu", "threads": self.cfg.get("threads", 4)}

    def transcribe(self, audio, vad):          # batch: 16 kHz float32 mono, VadResult or None
        with self.glue("features"):
            feats = features(audio)
        with self.infer("decode"):
            return [self.model.run(feats)]

    def transcribe_segment(self, samples):     # replay: one closed segment
        return " ".join(self.transcribe(samples, None))
```

| role | class | methods |
| --- | --- | --- |
| `vad` | `VadAdapter` | `detect(audio) -> VadResult`; for replay `stream_reset()`, `stream_feed(chunk) -> [Segment]`, `stream_flush() -> [Segment]` |
| `asr` | `AsrAdapter` | `transcribe(audio, vad) -> [str]`; for replay `transcribe_segment(samples) -> str` |
| `text` | `TextAdapter` | `process(text) -> str` |

Rules of thumb:

- Put each model call in `self.infer(...)`, and nothing else in it. Put the work around it in
  `self.glue(...)` with a name that says what it is. Junctions are only as good as this split.
- Name glue after the step in your own code (`normalize`, `split_points`, `wordpiece`), so a
  junction's breakdown points at a function.
- Span keyword arguments go into the span's `meta` (`piece=i`, `tokens=n`) and show in the
  timeline's tooltips. A span object can be updated inside the block: `sp.meta["words"] = n`.
- `VadResult.extra` carries whatever the next stage can reuse (the Hebrish VAD passes its
  per-frame speech mask to the ASR, as the app does).
- Report the execution provider in `info()["provider"]`. That is how the report answers "did this
  stage run on the GPU at all".

The Hebrish adapters (`asrprof/adapters/hebrish.py`) are the full worked example, including a
model-free stage (`hebrish.energy`) and a token classifier split into tokenising, per-window ORT
runs and merging. Model-free test adapters are in `tests/fake_adapters.py`.
