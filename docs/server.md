# Server and agents

```bash
asrprof serve --host 0.0.0.0 --port 8765
```

| request | returns |
| --- | --- |
| `GET /health` | `{"ok": true, "version": "1.0"}` |
| `GET /pipelines` | the pipeline files in `pipelines/` |
| `GET /samples` | the reading samples, with their text |
| `POST /runs?pipeline=NAME&...` with the audio as the body | the report, when the profile finishes |
| `GET /runs`, `GET /runs/FILE` | saved reports |

`POST /runs` query parameters: `pipeline` (a file stem in `pipelines/`, required), `mode`
(`batch`/`replay`), `sample`, `repeat`, `speed`, `interval`, `chunk-ms`, and `var` (repeatable,
as `--var`).

```bash
curl -s -X POST --data-binary @take.wav \
  'http://localhost:8765/runs?pipeline=hebrish-desktop&sample=read-60s&repeat=2&var=asr.threads=8' \
  | jq '.runs[-1] | {wall_ms, rtf, junctions: [.junctions[] | {junction, ms}]}'
```

Profiles run one at a time (two at once would share the CPU and spoil both), each in a child
`asrprof run` process. There is no authentication: bind to localhost or a private network.

## For an agent

The CLI suits an agent better than the server on the same machine: `asrprof run ... --json`
prints the report to stdout, progress to stderr. A typical sweep:

```bash
for t in 1 2 4 8; do
  asrprof run pipelines/hebrish-desktop.toml --audio take.wav --sample read-60s --repeat 3 \
    --var asr.threads=$t --json > threads-$t.json
done
jq -s 'map({t: .pipeline.stages[1].info.threads, wall: .runs[-1].wall_ms, asr_cpu: .runs[-1].stages.asr.cpu_avg_cores})' threads-*.json
```

Use the last run of each report (`.runs[-1]`) for warm numbers; run 1 is cold.
