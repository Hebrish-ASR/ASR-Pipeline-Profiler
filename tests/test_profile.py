from pathlib import Path

import numpy as np
import pytest

from asrprof import quality
from asrprof.cli import summary_text
from asrprof.pipeline import load_pipeline
from asrprof.profile import run_profile
from asrprof.record import save_wav

HERE = Path(__file__).parent


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    save_wav(p, np.zeros(16000 * 2, np.float32))
    return p


def test_vars_and_overrides():
    spec = load_pipeline(HERE / "fake.toml", ["x=7", "upper.extra=true"])
    assert spec["stages"][1]["threads"] == "7"
    assert spec["stages"][2]["extra"] is True


def test_batch_junctions(wav):
    rep = run_profile(str(HERE / "fake.toml"), str(wav), "batch", log=lambda m: None)
    run = rep["runs"][0]
    j = {x["junction"]: x for x in run["junctions"]}
    assert list(j) == ["input→vad", "vad→asr", "asr→upper", "upper→output"]
    # vad→asr holds the vad's 1 ms segments glue and the asr's 2 ms pad glue
    assert 3 <= j["vad→asr"]["ms"] < 15
    assert set(j["vad→asr"]["glue_ms"]) == {"vad.segments", "asr.pad"}
    assert j["asr→upper"]["glue_ms"].get("upper.prep", 0) >= 2.5
    asr = run["stages"]["asr"]
    assert asr["infer_calls"] == 3 and asr["gaps"]["count"] == 2
    assert asr["gaps"]["glue_ms"]["asr.pad"] >= 3.5
    assert 0.8 < asr["cpu_avg_cores"] < 1.3          # one busy thread
    assert rep["output"]["text"] == "PIECE 0 PIECE 1 PIECE 2"
    assert "Junctions" in summary_text(rep)


def test_replay(wav):
    rep = run_profile(str(HERE / "fake.toml"), str(wav), "replay", speed=4, log=lambda m: None)
    rp = rep["runs"][0]["replay"]
    assert len(rp["segments"]) == 4
    assert all(s["latency_ms"] >= s["asr_ms"] for s in rp["segments"])
    labels = [x["junction"] for x in rep["runs"][0]["junctions"]]
    assert "asr→vad" not in labels                    # back to listening is not a junction
    assert any(x["junction"] == "vad→asr" and x["count"] == 4 for x in rep["runs"][0]["junctions"])


def test_quality_paragraphs():
    ref = "one two three four.\n\nfive six seven eight.\n\nnine ten"
    hyp = "one two three four five.\n\nsix seven eight nine ten"
    q = quality.paragraphs(ref, hyp, tolerance=1)
    assert q["ref_breaks"] == 2 and q["hyp_breaks"] == 1 and q["hits"] == 1
    assert quality.wer(quality.words(ref), quality.words(hyp))["errors"] == 0
