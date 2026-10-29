"""Record a reading sample from the default microphone: 16 kHz mono 16-bit WAV."""

from __future__ import annotations

import datetime as dt
import threading
import wave
from pathlib import Path

import numpy as np

from . import samples as samplesmod

RECORDINGS = Path.home() / ".local" / "share" / "asrprof" / "recordings"


def new_path(sample_id: str) -> Path:
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    return RECORDINGS / f"{sample_id}_{dt.datetime.now():%Y-%m-%d_%H%M%S}.wav"


def save_wav(path: Path, audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


class Recorder:
    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.level = 0.0
        self._lock = threading.Lock()
        self.stream = None

    def _cb(self, data, frames, t, status):
        x = data[:, 0].copy()
        with self._lock:
            self.chunks.append(x)
        self.level = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0

    def start(self):
        import sounddevice as sd
        self.chunks = []
        self.stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32", callback=self._cb)
        self.stream.start()

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(c) for c in self.chunks) / 16000

    def stop(self) -> np.ndarray:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self._lock:
            return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def record_cli(sample_id: str, out: str | None) -> int:
    s = samplesmod.get(sample_id)
    path = Path(out) if out else new_path(s["id"])
    print(f"\n{s.get('title', s['id'])} (about {s.get('target_seconds', '?')} s)\n\n{s['text']}\n")
    input("Press Enter, then read the text aloud. Press Enter again when done. ")
    r = Recorder()
    r.start()
    input("Recording… ")
    audio = r.stop()
    save_wav(path, audio)
    print(f"{len(audio) / 16000:.1f} s saved to {path}")
    return 0
