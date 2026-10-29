"""Where things live. Bundled pipelines and samples ship inside the package (`asrprof/data/`);
your own go in `~/.config/asrprof/pipelines/` and `~/.config/asrprof/samples/` and win on a
name clash. Reports and recordings go in `~/.local/share/asrprof/`."""

from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
PKG_PARENT = PKG.parent                       # on PYTHONPATH for child processes
BUNDLED = PKG / "data"
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "asrprof"
DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "asrprof"
REPORTS = DATA / "reports"
RECORDINGS = DATA / "recordings"


def _dirs(kind: str) -> list[Path]:
    return [CONFIG / kind, BUNDLED / kind]


def pipelines() -> dict[str, Path]:
    """name -> file; user files override bundled ones of the same name."""
    out: dict[str, Path] = {}
    for d in reversed(_dirs("pipelines")):
        if d.is_dir():
            out.update({f.stem: f for f in sorted(d.glob("*.toml"))})
    return dict(sorted(out.items()))


def resolve_pipeline(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_file():
        return p
    found = pipelines().get(spec.removesuffix(".toml"))
    if found:
        return found
    raise FileNotFoundError(f"no pipeline {spec!r} (have: {', '.join(pipelines()) or 'none'}; "
                            f"add your own in {CONFIG / 'pipelines'})")


def sample_dirs() -> list[Path]:
    return [d for d in _dirs("samples") if d.is_dir()]


def child_env(env: dict | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = str(PKG_PARENT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env
