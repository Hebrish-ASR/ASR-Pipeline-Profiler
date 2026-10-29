# Report format (`asrprof.report/1`)

One JSON object. Times are milliseconds from the profile's start unless named otherwise; CPU is
in cores (1.0 = one core busy).

| key | contents |
| --- | --- |
| `schema`, `tool`, `created` | `"asrprof.report/1"`, `"asrprof 1.0"`, UTC ISO time |
| `host` | `os`, `python`, `cpu`, `cores` (logical), `ram_gb`, `gpu` (backend, name, `attributable`, totals, or null), `packages` (runtime versions) |
| `pipeline` | `name`, `file`, `description`, `vars` (after expansion and `--var`), `stages`: `[{name, role, adapter, info}]`, `info` from the adapter (`model`, `model_mb`, `provider`, `threads`, `runtime`, …) |
| `input` | `audio`, `duration_s`, `sample` |
| `mode`, `repeat` | `"batch"`/`"replay"`, number of runs |
| `sampler` | `interval_ms`, `ticks`, `self_cpu_ms` |
| `load` | per stage: `wall_ms`, `cpu_avg_cores`, `cpu_max_cores`, `threads_active`, `top_threads`, `rss_max_mb`, `rss_delta_mb`, `steps` (`{load step: ms}`), GPU fields |
| `runs[]` | per run, below |
| `output` | `text` (after the text stages), `asr_text`, `pieces`, `vad_segments` (`[[start_s, end_s]]`, batch) |
| `quality` | with a sample: `asr` and `final` (`wer`, `errors`, `ref_words`, `hyp_words`), `paragraphs` (`ref_breaks`, `hyp_breaks`, `hits`, `precision`, `recall`, `f1`, `tolerance_words`, `ref_at`, `hyp_at_mapped`) |
| `spans[]` | every span: `id`, `parent`, `stage`, `name`, `kind`, `phase`, `run`, `thread`, `t0_ms`, `t1_ms`, `ms`, `cpu_ms`, `meta` |
| `events[]` | replay points in time: `text_out` per segment |
| `samples[]` | sampler intervals: `t_ms` (interval end), `cores`, `rss_mb`, `threads` (`{tid: cores}`, ≥ 0.01), `gpu` |
| `thread_names` | `{tid: comm}` |

`runs[]`:

| key | contents |
| --- | --- |
| `run`, `wall_ms`, `rtf` | index from 0, wall time, wall / audio duration |
| `infer_ms`, `not_inferring_ms` | time inside any model call, and the rest |
| `junction_total_ms` | sum of the stage-to-stage junctions (edges excluded) |
| `cpu_avg_cores`, `cpu_max_cores`, `rss_max_mb` | over the whole run |
| `stages` | `{name: {wall_ms, calls, infer_ms, infer_calls, glue_ms, cpu_avg_cores, cpu_max_cores, threads_active, top_threads, rss_max_mb, gpu_busy_avg, gpu_busy_max, vram_max_mb, samples, gaps: {count, total_ms, mean_ms, p50_ms, max_ms, glue_ms}}}` |
| `junctions[]` | `{junction: "a→b", from, to, count, ms (total), mean_ms, p50_ms, max_ms, glue_ms: {stage.step: ms}, unaccounted_ms, at_ms: [[t0, t1]]}` in pipeline order |
| `replay` | replay only: `segments[]` (`start_s`, `end_s`, `words`, `vad_close_ms`, `junction_ms`, `asr_ms`, `latency_ms`), `latency_ms`/`vad_close_ms`/`asr_ms` (`mean`, `p50`, `p95`, `max`), `tail_ms`, `chunk_ms`, `speed`, `chunk_lag_ms` (`max`, `mean`, `chunks`) |

A field is `null` when it was not measured (no sampler interval inside a short span, no GPU
telemetry), never 0 for "unknown".
