"""Reading samples: `asrprof/data/samples/<id>.md` and `~/.config/asrprof/samples/<id>.md`.
YAML-ish front matter, then the text to read with blank lines between paragraphs (the reference
for paragraph checks)."""

from __future__ import annotations

from pathlib import Path

from .paths import sample_dirs


def _parse(path: Path) -> dict:
    raw = path.read_text()
    meta, body = {}, raw
    if raw.startswith("---"):
        head, _, body = raw[3:].partition("\n---")
        for line in head.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    meta.setdefault("id", path.stem)
    meta["text"] = body.strip()
    meta["words"] = len(meta["text"].split())
    meta["path"] = str(path)
    return meta


def all_samples() -> list[dict]:
    seen: dict[str, dict] = {}
    for d in reversed(sample_dirs()):            # user samples override bundled ones by id
        for p in d.glob("*.md"):
            s = _parse(p)
            seen[s["id"]] = s
    return sorted(seen.values(), key=lambda s: int(s.get("target_seconds", 0)))


def get(sample: str) -> dict:
    p = Path(sample)
    if p.is_file():
        return _parse(p)
    for s in all_samples():
        if s["id"] == sample:
            return s
    raise KeyError(f"no sample {sample!r} (have: {', '.join(s['id'] for s in all_samples())})")
