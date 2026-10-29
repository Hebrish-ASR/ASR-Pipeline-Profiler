"""Model-free adapters for tests: each 'model call' is a timed busy loop."""

import time

import numpy as np

from asrprof.adapters import AsrAdapter, Segment, TextAdapter, VadAdapter, VadResult


def busy(ms):
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


class Vad(VadAdapter):
    def info(self):
        return {"provider": "cpu"}

    def detect(self, audio):
        with self.infer("detect"):
            busy(5)
        with self.glue("segments"):
            busy(1)
        return VadResult([(0, len(audio))])

    def stream_reset(self):
        self.pos, self.start = 0, 0

    def stream_feed(self, chunk):
        with self.infer("feed"):
            busy(0.5)
        self.pos += len(chunk)
        if self.pos - self.start >= 8000:              # a segment every 0.5 s
            seg = Segment(self.start, self.pos, np.zeros(self.pos - self.start, np.float32))
            self.start = self.pos
            return [seg]
        return []


class Asr(AsrAdapter):
    def transcribe(self, audio, vad):
        out = []
        for i in range(3):
            with self.glue("pad"):
                busy(2)
            with self.infer("decode"):
                busy(10)
            out.append(f"piece {i}")
        return out

    def transcribe_segment(self, samples):
        with self.infer("decode"):
            busy(5)
        return "word"


class Upper(TextAdapter):
    def process(self, text):
        with self.glue("prep"):
            busy(3)
        with self.infer("upper"):
            busy(4)
        return text.upper()
