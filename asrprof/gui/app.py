"""The desktop app: pick a pipeline and its variables, read a sample aloud (or open a
recording), run the profile in a child process (`asrprof run`, so the GUI's own CPU is never in
the numbers), and read the result as a timeline and tables."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import __version__
from .. import samples as samplesmod
from ..cli import DATA, summary_text
from .timeline import Timeline

ROOT = Path(__file__).resolve().parents[2]
PIPELINES = ROOT / "pipelines"


def _f(v, nd=1, unit=""):
    if v is None:
        return "–"
    return f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def _table(head: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(head))
    t.setHorizontalHeaderLabels(head)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fill(t: QTableWidget, rows: list[list]):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            if j and str(v).replace(".", "", 1).replace("%", "").replace("–", "").replace("-", "").isdigit():
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(i, j, it)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"ASR Pipeline Profiler {__version__}")
        self.resize(1400, 900)
        self.rep = None
        self.proc: QProcess | None = None
        self.recorder = None
        self.out_path: Path | None = None

        # --- left: setup ---
        left = QWidget()
        lv = QVBoxLayout(left)
        pg = QGroupBox("Pipeline")
        pf = QFormLayout(pg)
        self.pipeline = QComboBox()
        for f in sorted(PIPELINES.glob("*.toml")):
            self.pipeline.addItem(f.stem, str(f))
        browse = QPushButton("Open…")
        browse.clicked.connect(self.browse_pipeline)
        row = QHBoxLayout()
        row.addWidget(self.pipeline, 1)
        row.addWidget(browse)
        pf.addRow(row)
        self.vars = QTableWidget(0, 2)
        self.vars.setHorizontalHeaderLabels(["variable", "value"])
        self.vars.verticalHeader().setVisible(False)
        self.vars.horizontalHeader().setStretchLastSection(True)
        self.vars.setToolTip("[vars] and stage keys (stage.key). Edited values are passed as --var.")
        pf.addRow(self.vars)
        lv.addWidget(pg, 3)
        self.pipeline.currentIndexChanged.connect(self.load_vars)

        sg = QGroupBox("Input")
        sf = QFormLayout(sg)
        self.sample = QComboBox()
        self.sample.addItem("(none: no quality check)", None)
        for s in samplesmod.all_samples():
            self.sample.addItem(f'{s["id"]} · {s.get("title", "")} · ~{s.get("target_seconds", "?")} s', s["id"])
        self.sample.currentIndexChanged.connect(self.show_sample)
        sf.addRow("Sample", self.sample)
        self.reading = QTextBrowser()
        f = self.reading.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        self.reading.setFont(f)
        sf.addRow(self.reading)
        self.audio = QLineEdit()
        self.audio.setPlaceholderText("a recording (any format ffmpeg reads)")
        ab = QPushButton("Open…")
        ab.clicked.connect(self.browse_audio)
        self.rec = QPushButton("● Record")
        self.rec.clicked.connect(self.toggle_record)
        row = QHBoxLayout()
        row.addWidget(self.audio, 1)
        row.addWidget(ab)
        row.addWidget(self.rec)
        sf.addRow("Audio", row)
        self.rec_label = QLabel("")
        sf.addRow(self.rec_label)
        lv.addWidget(sg, 4)

        rg = QGroupBox("Run")
        rf = QFormLayout(rg)
        self.mode = QComboBox()
        self.mode.addItem("batch: the whole recording at once", "batch")
        self.mode.addItem("replay: fed in real time, as from a microphone", "replay")
        rf.addRow("Mode", self.mode)
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 20)
        self.repeat.setValue(2)
        self.repeat.setToolTip("Runs after one load. Run 1 is often cold (first-call allocations).")
        rf.addRow("Runs", self.repeat)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 20)
        self.speed.setValue(1.0)
        self.speed.setSuffix("× real time")
        rf.addRow("Replay speed", self.speed)
        self.interval = QSpinBox()
        self.interval.setRange(5, 500)
        self.interval.setValue(20)
        self.interval.setSuffix(" ms")
        rf.addRow("Sampler interval", self.interval)
        self.go = QPushButton("Run profile")
        self.go.setDefault(True)
        self.go.clicked.connect(self.run_profile)
        rf.addRow(self.go)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        rf.addRow(self.log)
        lv.addWidget(rg, 3)

        # --- right: results ---
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        self.headline = QLabel("No report yet")
        self.headline.setWordWrap(True)
        top.addWidget(self.headline, 1)
        self.run_pick = QComboBox()
        self.run_pick.currentIndexChanged.connect(self.show_run)
        top.addWidget(QLabel("Run"))
        top.addWidget(self.run_pick)
        self.show_load = QCheckBox("Show load phase")
        self.show_load.toggled.connect(lambda _: self.show_run(self.run_pick.currentIndex()))
        top.addWidget(self.show_load)
        rv.addLayout(top)
        self.tabs = QTabWidget()
        self.timeline = Timeline()
        self.tabs.addTab(self.timeline, "Timeline")
        self.stage_t = _table(["stage", "role", "provider", "threads set", "wall ms", "model ms", "glue ms",
                               "calls", "CPU avg (cores)", "CPU max", "threads busy", "GPU avg", "GPU max",
                               "VRAM max MB", "RSS max MB", "gaps ms", "gap max ms"])
        self.load_t = _table(["stage", "load ms", "RSS + MB", "CPU avg", "steps"])
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.addWidget(QLabel("<b>Run</b> (model ms: time inside model calls; glue: work between them; "
                            "threads busy: threads averaging ≥ 0.1 core)"))
        sl.addWidget(self.stage_t, 2)
        sl.addWidget(QLabel("<b>Load</b>"))
        sl.addWidget(self.load_t, 1)
        self.tabs.addTab(sw, "Stages")
        self.junc_t = _table(["junction", "n", "total ms", "mean ms", "max ms", "unaccounted ms", "glue inside (ms)"])
        jw = QWidget()
        jl = QVBoxLayout(jw)
        jl.addWidget(QLabel("A <b>junction</b> runs from the end of one stage's last model call to the start "
                            "of the next stage's first: the time no model is computing while data is handed "
                            "on. <i>Unaccounted</i> is the part not inside a named glue step (orchestration, "
                            "Python between calls)."))
        jl.itemAt(0).widget().setWordWrap(True)
        jl.addWidget(self.junc_t)
        self.tabs.addTab(jw, "Junctions")
        self.lat_t = _table(["segment", "audio s", "words", "VAD closing ms", "junction ms", "ASR ms",
                             "speech end → text ms"])
        self.tabs.addTab(self.lat_t, "Replay latency")
        self.output = QTextBrowser()
        self.tabs.addTab(self.output, "Output")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.summary.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.summary, "Summary")
        rv.addWidget(self.tabs, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([430, 970])
        self.setCentralWidget(split)

        m = self.menuBar().addMenu("&File")
        a = QAction("Open report…", self)
        a.triggered.connect(self.open_report)
        m.addAction(a)
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)

        self.tick = QTimer(self)
        self.tick.timeout.connect(self._rec_tick)
        self.load_vars()
        self.show_sample()

    # setup -----------------------------------------------------------------------------------
    def browse_pipeline(self):
        f, _ = QFileDialog.getOpenFileName(self, "Pipeline", str(PIPELINES), "Pipeline (*.toml)")
        if f:
            self.pipeline.addItem(Path(f).stem, f)
            self.pipeline.setCurrentIndex(self.pipeline.count() - 1)

    def load_vars(self):
        path = self.pipeline.currentData()
        self.vars.setRowCount(0)
        self._defaults = {}
        if not path:
            return
        doc = tomllib.loads(Path(path).read_text())
        rows = list(doc.get("vars", {}).items())
        for st in doc.get("stages", []):
            rows += [(f'{st["name"]}.{k}', v) for k, v in st.items() if k not in ("name", "role", "adapter")]
        self.vars.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ki = QTableWidgetItem(k)
            ki.setFlags(ki.flags() & ~Qt.ItemIsEditable)
            self.vars.setItem(i, 0, ki)
            self.vars.setItem(i, 1, QTableWidgetItem(str(v)))
            self._defaults[k] = str(v)

    def overrides(self) -> list[str]:
        out = []
        for i in range(self.vars.rowCount()):
            k, v = self.vars.item(i, 0).text(), self.vars.item(i, 1).text()
            if self._defaults.get(k) != v:
                out.append(f"{k}={v}")
        return out

    def show_sample(self):
        sid = self.sample.currentData()
        if not sid:
            self.reading.setPlainText("Pick a reading sample to read aloud, or profile any recording "
                                      "without one (no quality checks then).")
            return
        s = samplesmod.get(sid)
        paras = "".join(f"<p>{p}</p>" for p in s["text"].split("\n\n"))
        self.reading.setHtml(f'<h3>{s.get("title", sid)}</h3><p><i>About {s.get("target_seconds", "?")} seconds, '
                             f'{s["words"]} words, {s.get("paragraphs", "?")} paragraphs. Read at your normal '
                             f'pace and pause naturally between paragraphs.</i></p>{paras}')

    def browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(self, "Recording", str(DATA / "recordings"),
                                           "Audio (*.wav *.flac *.mp3 *.ogg *.opus *.m4a);;All (*)")
        if f:
            self.audio.setText(f)

    def toggle_record(self):
        from ..record import Recorder, new_path, save_wav
        if self.recorder is None:
            try:
                self.recorder = Recorder()
                self.recorder.start()
            except Exception as e:
                self.recorder = None
                QMessageBox.warning(self, "Microphone", f"Can't record: {e}")
                return
            self.rec.setText("■ Stop")
            self.tick.start(200)
        else:
            audio = self.recorder.stop()
            self.recorder = None
            self.tick.stop()
            self.rec.setText("● Record")
            path = new_path(self.sample.currentData() or "take")
            save_wav(path, audio)
            self.audio.setText(str(path))
            self.rec_label.setText(f"{len(audio) / 16000:.1f} s saved to {path}")

    def _rec_tick(self):
        if self.recorder:
            lvl = self.recorder.level
            bar = "█" * min(30, int(lvl * 300))
            self.rec_label.setText(f"Recording {self.recorder.seconds:.1f} s  {bar}")

    # running ---------------------------------------------------------------------------------
    def run_profile(self):
        if self.proc:
            self.proc.kill()
            return
        audio = self.audio.text().strip()
        if not audio or not Path(audio).expanduser().is_file():
            QMessageBox.information(self, "Input", "Record or open a recording first.")
            return
        import datetime as dt
        self.out_path = DATA / "reports" / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{self.pipeline.currentText()}_{self.mode.currentData()}.json"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        args = ["-m", "asrprof", "run", self.pipeline.currentData(), "--audio", str(Path(audio).expanduser()),
                "--mode", self.mode.currentData(), "--repeat", str(self.repeat.value()),
                "--interval", str(self.interval.value()), "--speed", str(self.speed.value()),
                "--out", str(self.out_path)]
        if self.sample.currentData():
            args += ["--sample", self.sample.currentData()]
        for o in self.overrides():
            args += ["--var", o]
        self.log.clear()
        self.log.appendPlainText("asrprof " + " ".join(args[2:]))
        self.proc = QProcess(self)
        env = self.proc.processEnvironment().systemEnvironment()
        env.insert("PYTHONPATH", str(ROOT) + os.pathsep + env.value("PYTHONPATH", ""))
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.go.setText("Stop")
        self.proc.start(sys.executable, args)

    def _read(self):
        txt = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in txt.splitlines():
            if line.startswith(("loading", "run ", "Traceback", "  File", "report:")) or "Error" in line:
                self.log.appendPlainText(line)

    def _done(self, code, status):
        self.proc = None
        self.go.setText("Run profile")
        if code == 0 and self.out_path and self.out_path.is_file():
            self.set_report(json.loads(self.out_path.read_text()))
            self.log.appendPlainText(f"report: {self.out_path}")
        else:
            self.log.appendPlainText(f"failed (exit {code})")

    def open_report(self):
        f, _ = QFileDialog.getOpenFileName(self, "Report", str(DATA / "reports"), "Report (*.json)")
        if f:
            self.set_report(json.loads(Path(f).read_text()))

    # results ---------------------------------------------------------------------------------
    def set_report(self, rep: dict):
        self.rep = rep
        h = rep["host"]
        g = (h.get("gpu") or {}).get("name", "no GPU telemetry")
        self.headline.setText(f'<b>{rep["pipeline"]["name"]}</b> · {rep["mode"]} · '
                              f'{rep["input"]["duration_s"]:.1f} s audio · {h["cpu"]} ({h["cores"]} threads) · {g}')
        self.run_pick.blockSignals(True)
        self.run_pick.clear()
        for r in rep["runs"]:
            self.run_pick.addItem(f'{r["run"] + 1}: {r["wall_ms"]:.0f} ms')
        self.run_pick.blockSignals(False)
        info = {s["name"]: s for s in rep["pipeline"]["stages"]}
        rows = []
        for name, ld in rep["load"].items():
            steps = ", ".join(f"{k} {v:.0f}" for k, v in ld["steps"].items())
            rows.append([name, _f(ld["wall_ms"], 0), _f(ld.get("rss_delta_mb")), _f(ld["cpu_avg_cores"], 2), steps])
        _fill(self.load_t, rows)
        self.summary.setPlainText(summary_text(rep))
        q = rep.get("quality")
        html = ""
        if q:
            p = q["paragraphs"]
            html += (f'<p><b>WER</b> {q["asr"]["wer"]:.1%} ({q["asr"]["errors"]} errors / {q["asr"]["ref_words"]} '
                     f'words) · <b>paragraph breaks</b> {p["hits"]} of {p["ref_breaks"]} found, {p["hyp_breaks"]} '
                     f'placed (F1 {p["f1"]:.2f}, ±{p["tolerance_words"]} words)</p>')
        paras = "".join(f"<p>{x}</p>" for x in rep["output"]["text"].split("\n\n"))
        html += f"<h3>Final text</h3>{paras}<h3>ASR pieces</h3><ol>"
        html += "".join(f"<li>{x}</li>" for x in rep["output"]["pieces"]) + "</ol>"
        self.output.setHtml(html)
        self._info = info
        self.show_run(0)

    def show_run(self, i: int):
        if not self.rep or i < 0:
            return
        run = self.rep["runs"][i]
        self.timeline.set_report(self.rep, i, self.show_load.isChecked())
        rows = []
        for name, s in run["stages"].items():
            inf = self._info[name]["info"]
            rows.append([name, self._info[name]["role"], inf.get("provider", "–"), _f(inf.get("threads")),
                         _f(s["wall_ms"]), _f(s["infer_ms"]), _f(s["glue_ms"], 2), s["infer_calls"],
                         _f(s["cpu_avg_cores"], 2), _f(s["cpu_max_cores"], 2), _f(s["threads_active"]),
                         _f(s["gpu_busy_avg"], 1, "%"), _f(s["gpu_busy_max"], 0, "%"), _f(s["vram_max_mb"]),
                         _f(s["rss_max_mb"]), _f(s["gaps"]["total_ms"], 3), _f(s["gaps"]["max_ms"], 3)])
        _fill(self.stage_t, rows)
        rows = []
        for j in run["junctions"]:
            rows.append([j["junction"], j["count"], _f(j["ms"], 3), _f(j["mean_ms"], 3), _f(j["max_ms"], 3),
                         _f(j["unaccounted_ms"], 3), ", ".join(f"{k} {v:.3f}" for k, v in j["glue_ms"].items())])
        _fill(self.junc_t, rows)
        rp = run.get("replay")
        _fill(self.lat_t, [[r["segment"], f'{r["start_s"]:.2f}–{r["end_s"]:.2f}', r["words"], _f(r["vad_close_ms"]),
                            _f(r["junction_ms"], 3), _f(r["asr_ms"]), _f(r["latency_ms"])]
                           for r in (rp["segments"] if rp else [])])
        self.tabs.setTabEnabled(3, bool(rp))
        self.headline.setToolTip(f'run {i + 1}: {run["wall_ms"]:.0f} ms, RTF {run["rtf"]:.3f}')


def main(report: str | None = None) -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("asrprof")
    w = Main()
    if report:
        w.set_report(json.loads(Path(report).read_text()))
    w.show()
    return app.exec()
