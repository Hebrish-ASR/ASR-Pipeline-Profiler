# Hebrish desktop pipeline: first profile (2026-10-29)

**Setup.** `hebrish-desktop` (`asrprof/data/pipelines/hebrish-desktop.toml`) as shipped: Silero VAD (1 thread) → Sabich v2.2
(Parakeet TDT 0.6B int8, sherpa-onnx, 4 threads, CPU) → Paragraph model r5b (ONNX Runtime,
default threads). Hebrish desktop source at its 1.011 release. Laptop: AMD Ryzen 7 5700U
(8 cores / 16 threads), Radeon iGPU (amdgpu sysfs telemetry), Ubuntu, Python 3.14,
sherpa-onnx 2.0.1, onnxruntime 1.31.1 (the app's .deb bundles sherpa-onnx 1.13.8: these numbers
are for the newer runtime).

**Input.** The `read-120s` sample spoken by **espeak-ng** (95.3 s), not a person. It is fine for
timing and resources; it says nothing about model accuracy (robotic synthetic speech: WER 24%
batch). A human recording of each sample is still to be done.

Commands:

```bash
asrprof run hebrish-desktop --audio read-120s-tts.wav --sample read-120s --repeat 3
asrprof run hebrish-desktop --audio read-120s-tts.wav --sample read-120s --mode replay
```

## Load

| stage | load | of which | RSS added |
| --- | ---: | --- | ---: |
| vad | 318 ms | session 164 ms | 63.5 MB |
| asr | 4,137 ms | recognizer 3,848 ms, warm-up 288 ms | 892.2 MB |
| paragraphs | 420 ms | session 349 ms, tokenizer 10 ms, warm-up 7 ms | 49.0 MB |

## Batch, warm (run 3 of 3)

12,052 ms wall for 95.3 s of audio (RTF 0.127). Models were computing for 12,032 ms; the pipeline
was not computing for 20 ms in total.

| stage | wall ms | CPU avg (cores) | CPU peak | threads busy |
| --- | ---: | ---: | ---: | ---: |
| vad | 522 | 1.01 | 1.03 | 1 |
| asr (16 pieces) | 11,331 | 4.40 | 10.86 | 4 |
| paragraphs | 199 | 15.00 | 16.16 | 22 |

| junction | ms | mostly |
| --- | ---: | --- |
| vad→asr | 13.4 | normalize 4.2, stretch_pauses 4.1, split_points 3.3, shorten_silences 1.2 |
| asr→paragraphs | 3.4 | WordPiece tokenising 3.2 (pure Python) |
| paragraphs→output | 0.5 | merge and join |
| between the 16 ASR pieces | 2.9 total, 0.31 max | padding 1.6 |

Run 1 (cold) differed: vad 1,130 ms instead of 522 (the app's Silero wrapper re-creates its
detector when a take is longer than 60 s, and the load phase's ORT pools were still spinning,
CPU peak 15.6 cores during a 1-thread stage), vad→asr 24.0 ms instead of 13.4.

## Replay (1x, 100 ms chunks)

16 segments. End of speech → text: mean 1,426 ms, p95 2,084 ms, max 2,252 ms, made of VAD
closing (mean 293 ms) and ASR decode (mean 1,133 ms); the junction between them is under a
millisecond of orchestration plus the segment copy below. From the last chunk to the final,
paragraphed text: 1,278 ms. With the ASR on the processing thread, chunks were picked up as much
as 1,894 ms late while a long segment decoded.

## What it shows

1. **Junctions are small next to inference, and almost all named glue.** 13 ms between VAD and
   ASR on 95 s of audio, 3.4 ms before the paragraph model; unaccounted time under 0.4 ms each.
   There is no handoff problem to fix in batch mode. (Measured.)
2. **The paragraph model takes every core.** ONNX Runtime's default `intra_op_num_threads`
   uses all 16 logical CPUs: 15 cores for 200 ms. Short, but on a machine also running a
   dictation hotkey or a desktop it is a full-machine spike each time. Setting
   `paragraphs.threads` is the lever to measure next. (Measured; impact on the rest of the
   desktop not measured.)
3. **ASR uses more threads than it was given at the start of each piece.** With `threads = 4`,
   the first ~80 ms of each decode ran about 10 threads at a full core each, then settled to 4
   (per-thread sampler data, run 1 of the 60-second take). Likely ONNX Runtime pools of
   sherpa-onnx's separate encoder, decoder and joiner sessions spinning while another runs.
   (Measured threads; cause inferred, not confirmed.)
4. **Spinning pools spill into the next stage.** In replay, the VAD's chunk feeds just after an
   ASR decode show 10 busy threads although the VAD has 1. The extra threads are the ASR's.
   Per-stage CPU in a tight pipeline needs the per-thread view to be read correctly. (Measured
   thread counts; attribution inferred from thread ids.)
5. **Copying a VAD segment out of sherpa-onnx costs milliseconds.** `vad.segments` in replay,
   which turns each segment's samples (a Python list) into a NumPy array, took 2.1–15.5 ms per
   segment, 109 ms over 16 segments: most of the `vad→asr` junction. (Measured.)
6. **Replay latency is dominated by the decode, then by the VAD's silence wait.** Speeding up
   the ASR or decoding on a second thread is where latency would come from; the junctions are
   not. (Measured split.)
7. **GPU unused.** Every stage runs on the CPU provider; the iGPU's device-wide busy % stayed at
   0–8% (the display). (Measured; expected, since the app passes `provider="cpu"`.)
