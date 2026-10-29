"""Adapters for the Hebrish desktop app (https://github.com/Hebrish-ASR/Hebrish-App, `desktop/`).

They import the app's own `hebrish` package and call its functions, so what is profiled is the
app's code path, split into spans at the points where the app hands data from one step to the
next. Where the app does several things in one call, the adapter calls the same functions one
at a time (noted per method), with the app's constants.

Finding the package (stage key `hebrish_src`, usually set once in `[vars]`): that folder (the
repo's `desktop/`) if given, else an importable `hebrish`, else the .deb's `/usr/lib/hebrish`.

Model folders (`model`): a version folder (holds `tokens.txt` or `model_int8.onnx`) or a line
folder (`.../models/sabich`), resolved like the app does: the version named in
`<data>/state.json` `active`, else the newest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import AsrAdapter, Segment, TextAdapter, VadAdapter, VadResult


def import_hebrish(src: str | None):
    if src:
        p = Path(src).expanduser()
        if not (p / "hebrish" / "engine.py").is_file():
            raise FileNotFoundError(f"hebrish_src {p} has no hebrish/engine.py")
        sys.path.insert(0, str(p))
    try:
        import hebrish
    except ImportError:
        if not Path("/usr/lib/hebrish/hebrish").is_dir():
            raise ImportError("can't find the Hebrish desktop package: set hebrish_src to the "
                              "Hebrish-App repo's desktop/ folder, or install the hebrish .deb")
        sys.path.insert(0, "/usr/lib/hebrish")
        import hebrish
    return hebrish


def resolve_model(path: str | Path, marker: str) -> Path:
    p = Path(path).expanduser()
    if (p / marker).is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"model folder {p} does not exist")
    versions = [d for d in p.iterdir() if (d / marker).is_file()]
    if not versions:
        raise FileNotFoundError(f"no {marker} in {p} or its subfolders")
    try:
        active = json.loads((p.parent.parent / "state.json").read_text())["active"][p.name]
        if (p / active / marker).is_file():
            return p / active
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return max(versions, key=lambda d: d.stat().st_mtime)


def _size_mb(folder: Path, pattern: str = "*.onnx") -> float:
    return round(sum(f.stat().st_size for f in folder.glob(pattern)) / 2**20, 1)


def _segments(mask: np.ndarray, frame: int) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i * frame, j * frame))
            i = j
        else:
            i += 1
    return out


class _Hebrish:
    def _import(self):
        h = import_hebrish(self.cfg.get("hebrish_src"))
        from hebrish import engine
        self.h, self.E = h, engine
        return h


class SileroVad(_Hebrish, VadAdapter):
    """The app's Silero VAD (`hebrish.vad.Silero`, sherpa-onnx's VoiceActivityDetector).

    Batch: `Silero.speech_frames`, the call `engine._speech_frames` makes when Voice detection is
    on. Replay: the same model and settings fed chunk by chunk; the app's live insertion cuts on
    the energy rule instead, so replay is the "VAD-segmented streaming" pattern, not the app's
    live mode. Stage keys: `threads` (1), `model` (the bundled file), `stream_gain` (1.0;
    batch scales each take to a fixed peak, a stream can't know its peak)."""

    def load(self):
        h = self._import()
        from hebrish import vad
        self.vad = vad
        self.path = Path(self.cfg.get("model") or vad.BUNDLED).expanduser()
        self.threads = int(self.cfg.get("threads", 1))
        with self.loading("session"):
            self.sil = vad.Silero(self.path, self.threads)
            noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
            self.sil.speech_frames(noise, self.E.FRAME)          # creates the detector
        self.gain = float(self.cfg.get("stream_gain", 1.0))

    def info(self):
        return {"model": str(self.path), "model_mb": round(self.path.stat().st_size / 2**20, 2),
                "provider": "cpu", "threads": self.threads, "runtime": "sherpa-onnx",
                "hebrish": getattr(self.h, "__version__", "?"), "window": self.vad.WINDOW}

    def detect(self, audio):
        with self.infer("speech_frames", audio_s=len(audio) / 16000):
            sp = self.sil.speech_frames(audio, self.E.FRAME)
        with self.glue("segments"):
            segs = _segments(sp, self.E.FRAME)
        return VadResult(segs, {"speech_frames": sp})

    def stream_reset(self):
        import sherpa_onnx
        self.det = sherpa_onnx.VoiceActivityDetector(self.sil._cfg, buffer_size_in_seconds=600)
        self.buf = np.zeros(0, np.float32)

    def _drain(self) -> list[Segment]:
        out = []
        while not self.det.empty():
            seg = self.det.front
            s = np.array(seg.samples, dtype=np.float32) / (self.gain or 1)
            out.append(Segment(seg.start, seg.start + len(s), s))
            self.det.pop()
        return out

    def stream_feed(self, chunk):
        w = self.vad.WINDOW
        with self.glue("buffer"):
            self.buf = np.concatenate([self.buf, chunk * self.gain])
            n = len(self.buf) // w * w
            todo, self.buf = self.buf[:n], self.buf[n:]
        with self.infer("feed", windows=n // w):
            for i in range(0, n, w):
                self.det.accept_waveform(todo[i:i + w])
        with self.glue("segments"):
            return self._drain()

    def stream_flush(self):
        with self.infer("flush"):
            self.det.flush()
        return self._drain()


class EnergyVad(_Hebrish, VadAdapter):
    """The app's energy rule (`engine._speech_frames` with Silero off): no model, so its whole
    turn is one `infer` span standing in for a model call."""

    def load(self):
        self._import()
        self.E.silero = None

    def info(self):
        return {"model": None, "provider": "cpu", "runtime": "numpy",
                "hebrish": getattr(self.h, "__version__", "?")}

    def detect(self, audio):
        with self.infer("energy_rule"):
            sp = self.E._speech_frames(audio)
        with self.glue("segments"):
            return VadResult(_segments(sp, self.E.FRAME), {"speech_frames": sp})


class Parakeet(_Hebrish, AsrAdapter):
    """Parakeet TDT through sherpa-onnx, as `engine.Engine`. The recognizer is built with
    `Engine.load`'s arguments (the provider is a stage key here; the app always uses "cpu") and
    the warm-up is its own span. `transcribe` is `Engine.transcribe_chunks` one step at a time:
    shorten long silences (if a VAD ran), normalise, stretch pauses, cut at pauses
    (`split_points`), then pad and decode each piece.

    Stage keys: `model` (required), `threads` (4), `provider` ("cpu"), `feature_dim` (128),
    `streaming` (false; true for Shakshuka)."""

    def load(self):
        self._import()
        import sherpa_onnx
        E = self.E
        self.dir = resolve_model(self.cfg["model"], "tokens.txt")
        self.threads = int(self.cfg.get("threads", 4))
        self.provider = self.cfg.get("provider", "cpu")
        self.streaming = bool(self.cfg.get("streaming", False))
        files = dict(encoder=str(self.dir / "encoder.int8.onnx"), decoder=str(self.dir / "decoder.int8.onnx"),
                     joiner=str(self.dir / "joiner.int8.onnx"), tokens=str(self.dir / "tokens.txt"),
                     decoding_method="greedy_search", num_threads=self.threads, sample_rate=16000,
                     feature_dim=int(self.cfg.get("feature_dim", 128)), provider=self.provider)
        with self.loading("recognizer"):
            if self.streaming:
                rec = sherpa_onnx.OnlineRecognizer.from_transducer(**files)
            else:
                rec = sherpa_onnx.OfflineRecognizer.from_transducer(model_type="nemo_transducer", **files)
        self.eng = E.Engine()
        self.eng._rec, self.eng._dir, self.eng.streaming, self.eng.threads = rec, self.dir, self.streaming, self.threads
        with self.loading("warmup"):
            self.eng.transcribe(np.zeros(16000, np.float32), record=False)
        self.pad = np.zeros(int(E.TAIL_PAD_S * 16000), np.float32)

    def info(self):
        return {"model": str(self.dir), "model_mb": _size_mb(self.dir), "provider": self.provider,
                "threads": self.threads, "runtime": "sherpa-onnx", "streaming": self.streaming,
                "hebrish": getattr(self.h, "__version__", "?")}

    def _decode(self, x, **meta):
        with self.infer("decode", audio_s=round(len(x) / 16000, 3), **meta) as sp:
            text, tokens = self.eng._decode_piece(x)
            sp.meta.update(tokens=tokens, words=len(text.split()))
        return text

    def transcribe(self, audio, vad):
        E = self.E
        sp = vad.extra.get("speech_frames") if vad else None
        if sp is not None:
            with self.glue("has_speech"):
                speech = E.has_speech(audio, None, sp)
            if not speech:
                return []
            with self.glue("shorten_silences"):
                audio = E.shorten_silences(audio, None, sp)
        with self.glue("normalize"):
            x = E.normalize(audio)
        if not self.streaming:
            with self.glue("stretch_pauses"):
                x = E.stretch_pauses(x)
        with self.glue("split_points", audio_s=round(len(x) / 16000, 3)):
            cuts = E.split_points(x)
        out = []
        for i, (a, b) in enumerate(cuts):
            with self.glue("pad"):
                piece = np.concatenate([x[a:b], self.pad])
            text = self._decode(piece, piece=i)
            if text:
                out.append(text)
        return out

    def transcribe_segment(self, samples):
        E = self.E
        with self.glue("normalize"):
            x = E.normalize(samples)
            if not self.streaming:
                x = E.stretch_pauses(x)
            x = np.concatenate([x, self.pad])
        return self._decode(x)


class Paragraphs(_Hebrish, TextAdapter):
    """The Paragraph model (`hebrish.paragraphs.Breaker`: pure-Python WordPiece, then a token
    classifier in ONNX Runtime). `process` is `Breaker.apply` + `Breaker.probs` one step at a time
    so tokenising, each window's ONNX run and the merge are separate spans.

    Stage keys: `model` (required, the `paragraph-breaks` folder), `threads` (0 = ORT default),
    `threshold` (the manifest's)."""

    def load(self):
        self._import()
        from hebrish import paragraphs as P
        self.P = P
        self.dir = resolve_model(self.cfg["model"], P.MODEL_FILE)
        self.threads = int(self.cfg.get("threads", 0))
        with self.loading("tokenizer"):
            thr = self.cfg.get("threshold")
            self.b = P.Breaker(self.dir, None if thr is None else float(thr), self.threads)
        with self.loading("session"):
            self.sess = self.b.session()
        with self.loading("warmup"):
            self.b.probs("warm up the session".split())

    def info(self):
        import onnxruntime as ort
        return {"model": str(self.dir), "model_mb": _size_mb(self.dir), "runtime": f"onnxruntime {ort.__version__}",
                "provider": ",".join(p.replace("ExecutionProvider", "").lower() for p in self.sess.get_providers()),
                "threads": self.threads or "default", "threshold": self.b.threshold}

    def process(self, text):
        b, P = self.b, self.P
        words = text.split()
        if not words:
            return text
        with self.glue("wordpiece", words=len(words)):
            pieces = b.word_pieces(words)
            wins = P.windows(pieces)
        best: dict[int, tuple[float, float]] = {}
        for k, (w0, w1) in enumerate(wins):
            with self.glue("window_ids"):
                ids, pos = [b.tok.cls], []
                for w in range(w0, w1):
                    pos.append((len(ids), w))
                    ids += pieces[w]
                ids.append(b.tok.sep)
                a = np.array([ids], dtype=np.int64)
            with self.infer("onnx_run", window=k, tokens=len(ids)):
                logits = self.sess.run(["logits"], {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]
            with self.glue("merge"):
                e = np.exp(logits - logits.max(-1, keepdims=True))
                pr = e[:, 1] / e.sum(-1)
                mid = (len(ids) - 1) / 2
                for p, w in pos:
                    c = -abs(p - mid)
                    if w not in best or c > best[w][0]:
                        best[w] = (c, float(pr[p]))
        with self.glue("join"):
            probs = [0.0] + [best[i][1] for i in range(1, len(words))]
            out = []
            for w, p in zip(words, probs):
                out.append(("\n\n" if p >= b.threshold else " ") + w if out else w)
            res = "".join(out)
        return res if res.split() == words else text
