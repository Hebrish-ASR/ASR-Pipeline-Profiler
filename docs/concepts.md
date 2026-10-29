# Concepts and method

## Spans

The profiled pipeline runs in one process. Adapters wrap their work in **spans**, each with a
start and end on `CLOCK_MONOTONIC` (`time.monotonic_ns()`) and the process CPU time at both ends
(`time.process_time_ns()`). Span kinds:

| kind | what it is |
| --- | --- |
| `run` | one pass of the whole pipeline over the input (`batch` or `replay`) |
| `stage` | one turn of one stage: VAD on the take, ASR on the take (or one segment, in replay), one text stage |
| `infer` | a model computing: a sherpa-onnx decode, an ONNX Runtime `run`, a batch of VAD windows |
| `glue` | work between model calls that is not a model: normalising, cutting, padding, tokenising, copying arrays |
| `load` | making a model ready: session creation, tokenizer, warm-up |

Which code is `infer` and which is `glue` is the adapter's choice, made by reading the
pipeline's code. When a library call mixes the two (sherpa-onnx's VAD scales its input inside the
same call that runs the model), the whole call is `infer` and the adapter's docstring says so.

## Junctions

A **junction** is the time between two stages during which no model is computing: from the end
of stage A's last `infer` span to the start of stage B's first. It is where data is handed on:
arrays are converted, audio is normalised and cut, text is tokenised, sessions are created
lazily, Python runs between calls. Each junction lists the `glue` spans that fall inside it, by
name, and the **unaccounted** remainder (orchestration and bookkeeping outside any named step).

Junctions are found by walking every `infer` span of a run in time order; each change of stage
is one crossing. In batch mode there is one crossing per pair of stages. In replay there is one
per segment (`vad→asr` each time the VAD closes a segment); a return to the first stage (the
VAD's next chunk) is waiting for audio, not a handoff, and is left out. The edges `input→first`
and `last→output` are reported too (batch only for `input→`).

**Gaps** are the same thing inside one turn of a stage: between consecutive `infer` spans with
the same parent (the ASR's pieces of one take, the paragraph model's windows).

## CPU

- **Average**: exact. Process CPU time over the span's wall time, in **cores** (1.0 = one core
  busy for the whole span). It covers every thread in the process, including runtime thread
  pools, and in replay the feeder thread.
- **Peak**: from the sampler, the busiest interval whose midpoint falls inside the stage's spans.
  Empty when every span was shorter than one interval (20 ms by default, `--interval`).
- **Threads busy**: threads averaging at least 0.1 core over the stage's sampled intervals, from
  per-thread on-CPU time. `top_threads` in the report lists the busiest.

The **sampler** is a separate process (`asrprof.sampler`) that reads, every interval,
`/proc/PID/task/*/schedstat` (nanoseconds on CPU per thread), `/proc/PID/statm` (resident memory)
and the GPU. Being outside, it never competes for the profiled process's GIL and its own CPU is
in no stage (`sampler.self_cpu_ms` in the report).

Busy-waiting counts as CPU. ONNX Runtime's thread pools spin for a while after a run before they
sleep (`session.intra_op.allow_spinning`, on by default), so a stage that follows an ORT stage can
show CPU and busy threads that belong to the previous stage's pool. The per-thread view is how to
tell: the extra threads are the earlier stage's.

## Memory

Resident set size (RSS) of the process: per stage the peak inside its spans, per load the change
from before to after. RSS includes memory-mapped model files once their pages are touched.

## GPU

| backend | what it gives |
| --- | --- |
| NVML (`pip install nvidia-ml-py`) | utilisation % of the device, device memory, **this process's** memory |
| amdgpu sysfs | busy % of the **whole device** (the desktop's drawing included), VRAM and GTT used |

"GPU offload" for a stage is first the execution provider its adapter reports (`provider`: `cpu`,
`cuda`, …); then whether the GPU actually got busier during its spans. On a shared, device-wide
counter, a few percent is noise from the display.

## Modes

**batch**: the whole recording at once, like "transcribe this file": VAD on the take, ASR on the
take (cut into pieces by the pipeline's own rule), then each text stage on the joined text.

**replay**: the recording fed at real time (or `--speed` times faster) in `chunk_ms` chunks from a
feeder thread, as a microphone would deliver it. The processing thread feeds each chunk to the
VAD; each segment the VAD closes is transcribed straight away; the text stages run on the joined
text after the last chunk. Per segment:

- **VAD closing**: from the moment the segment's last audio arrived to the moment the VAD closed
  it (its minimum-silence wait, plus any backlog).
- **junction**: VAD closed → ASR started.
- **ASR**: the segment's decode.
- **speech end → text**: all of it, from the end of speech in the audio to text out.

**Chunk lag** is how late the processing thread picked chunks up: with ASR on the same thread,
it grows while a long segment decodes.

## Runs

`--repeat N` runs the pipeline N times after one load. Run 1 is often slower: first-call
allocations, lazily grown buffers (the Hebrish VAD re-creates its detector when a take is longer
than 60 s), and spinning pools left over from the load phase's warm-ups.
