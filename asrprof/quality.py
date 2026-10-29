"""Output checks against a reading sample's reference text: word error rate, and paragraph
breaks (precision / recall / F1) after aligning hypothesis words to reference words."""

from __future__ import annotations

import difflib
import re

_WORD = re.compile(r"[\w']+")


def words(text: str) -> list[str]:
    return [w.lower().strip("'") for w in _WORD.findall(text)]


def paragraph_starts(text: str) -> list[int]:
    """Word indices (in `words(text)`) that start a paragraph, the first one excluded."""
    out, n = [], 0
    for i, para in enumerate(p for p in re.split(r"\n\s*\n", text) if words(p)):
        if i:
            out.append(n)
        n += len(words(para))
    return out


def wer(ref: list[str], hyp: list[str]) -> dict:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    errors = d[len(hyp)]
    return {"wer": round(errors / max(1, len(ref)), 4), "errors": errors,
            "ref_words": len(ref), "hyp_words": len(hyp)}


def paragraphs(ref_text: str, hyp_text: str, tolerance: int = 2) -> dict:
    """A hypothesis break counts as a hit if it lands within `tolerance` reference words of a
    reference break (each reference break matched once)."""
    ref, hyp = words(ref_text), words(hyp_text)
    rb, hb = paragraph_starts(ref_text), paragraph_starts(hyp_text)
    sm = difflib.SequenceMatcher(a=hyp, b=ref, autojunk=False)
    mapping = {}
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            mapping[a + k] = b + k
    mapped = []
    for h in hb:
        near = [i for i in range(h, len(hyp)) if i in mapping]
        mapped.append(mapping[near[0]] - (near[0] - h) if near else len(ref))
    left, hits = set(rb), 0
    for m in mapped:
        best = min(left, key=lambda r: abs(r - m), default=None)
        if best is not None and abs(best - m) <= tolerance:
            hits += 1
            left.discard(best)
    p = hits / len(hb) if hb else (1.0 if not rb else 0.0)
    r = hits / len(rb) if rb else 1.0
    return {"ref_breaks": len(rb), "hyp_breaks": len(hb), "hits": hits, "tolerance_words": tolerance,
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "ref_at": rb, "hyp_at_mapped": mapped}


def score(ref_text: str, asr_text: str, final_text: str) -> dict:
    return {"asr": wer(words(ref_text), words(asr_text)),
            "final": wer(words(ref_text), words(final_text)),
            "paragraphs": paragraphs(ref_text, final_text)}
