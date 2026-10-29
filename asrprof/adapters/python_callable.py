"""A text stage from any Python function: `callable = "package.module:function"`, taking and
returning a string. The whole call is one `infer` span, so use it for a stage whose model call
you don't need to separate from its glue (or write a full adapter, docs/adapters.md).
Optional `load = "package.module:function"` runs once, timed as the stage's load span."""

from __future__ import annotations

import importlib

from . import TextAdapter


def _resolve(spec: str):
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


class PythonText(TextAdapter):
    def load(self):
        self.fn = _resolve(self.cfg["callable"])
        if self.cfg.get("load"):
            with self.loading("load"):
                _resolve(self.cfg["load"])()

    def info(self):
        return {"callable": self.cfg["callable"], "provider": self.cfg.get("provider", "unknown")}

    def process(self, text):
        with self.infer("call"):
            return self.fn(text)
