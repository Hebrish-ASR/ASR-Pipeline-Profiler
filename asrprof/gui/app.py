"""The desktop app: pick a pipeline and its variables, read a sample aloud (or open a
recording), run the profile in a child process (`asrprof run`, so the GUI's own CPU is never in
the numbers), and read the result as headline figures, a timeline and tables."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tomllib
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import __version__
from .. import samples as samplesmod
from ..cli import summary_text
from ..paths import BUNDLED, CONFIG, REPORTS, RECORDINGS, child_env, pipelines
from . import style
from .timeline import Timeline

ICON = BUNDLED / "icons" / "asrprof.svg"


def _f(v, nd=1, unit=""):
    if v is None:
        return "–"
    return f"{v:,.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def _table(head: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(head))
    t.setHorizontalHeaderLabels(head)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setFocusPolicy(Qt.NoFocus)
    t.verticalHeader().setDefaultSectionSize(30)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setHighlightSections(False)
    return t


def _numeric(v) -> bool:
    s = str(v).replace(",", "").replace("%", "").replace("–", "").strip()
    return s.replace(".", "", 1).lstrip("-").isdigit() or s == ""


def _fill(t: QTableWidget, rows: list[list], bold_first: bool = True):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            if j and _numeric(v):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if j == 0 and bold_first:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            t.setItem(i, j, it)


def _card() -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    v = QVBoxLayout(f)
    v.setContentsMargins(16, 14, 16, 16)
    v.setSpacing(10)
    return f, v


def _step(n: int, text: str) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    num = QLabel(str(n))
    num.setObjectName("stepnum")
    lbl = QLabel(text)
    lbl.setObjectName("step")
    h.addWidget(num)
    h.addWidget(lbl, 1)
    return w


class Kpi(QFrame):
    def __init__(self, label: str, tip: str):
        super().__init__()
        self.setObjectName("kpi")
        self.setToolTip(tip)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(1)
        self.value = QLabel("–")
        self.value.setObjectName("kpiValue")
        lab = QLabel(label)
        lab.setObjectName("kpiLabel")
        self.note = QLabel("")
        self.note.setObjectName("kpiNote")
        v.addWidget(lab)
        v.addWidget(self.value)
        v.addWidget(self.note)

    def set(self, value: str, note: str = ""):
        self.value.setText(value)
        self.note.setText(note)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASR Pipeline Profiler")
        self.setWindowIcon(QIcon(str(ICON)))
        self.resize(1500, 940)
        self.rep = None
        self.proc: QProcess | None = None
        self.recorder = None
        self.out_path: Path | None = None
        t = style.theme

        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- header ---
        header = QFrame()
        header.setObjectName("header")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 12, 18, 12)
        logo = QLabel()
        logo.setPixmap(QIcon(str(ICON)).pixmap(QSize(42, 42)))
        hh.addWidget(logo)
        tv = QVBoxLayout()
        tv.setSpacing(0)
        title = QLabel("ASR Pipeline Profiler")
        title.setObjectName("title")
        if t:
            title.setFont(QFont(t.display, 17, QFont.Bold))
        self.subtitle = QLabel("Where the time and the compute go in a multi-model local speech pipeline")
        self.subtitle.setObjectName("subtitle")
        tv.addWidget(title)
        tv.addWidget(self.subtitle)
        hh.addLayout(tv, 1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("muted")
        hh.addWidget(ver, 0, Qt.AlignTop)
        outer.addWidget(header)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(14, 14, 14, 14)
        outer.addWidget(body, 1)

        # --- left: setup, three steps ---
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(12)

        c1, v1 = _card()
        v1.addWidget(_step(1, "Pipeline"))
        self.pipeline = QComboBox()
        for n, f in pipelines().items():
            self.pipeline.addItem(n, str(f))
        browse = QPushButton("Open…")
        browse.clicked.connect(self.browse_pipeline)
        row = QHBoxLayout()
        row.addWidget(self.pipeline, 1)
        row.addWidget(browse)
        v1.addLayout(row)
        self.vars = QTableWidget(0, 2)
        self.vars.setHorizontalHeaderLabels(["variable", "value"])
        self.vars.verticalHeader().setVisible(False)
        self.vars.verticalHeader().setDefaultSectionSize(26)
        self.vars.setAlternatingRowColors(True)
        self.vars.setShowGrid(False)
        self.vars.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.vars.horizontalHeader().setStretchLastSection(True)
        self.vars.setMinimumHeight(150)
        self.vars.setToolTip("Pipeline [vars] and stage keys (stage.key). Double-click a value to edit; "
                             "changed values are passed as --var.")
        v1.addWidget(self.vars, 1)
        lv.addWidget(c1, 1)
        self.pipeline.currentIndexChanged.connect(self.load_vars)

        c2, v2 = _card()
        v2.addWidget(_step(2, "Input"))
        self.sample = QComboBox()
        self.sample.addItem("No sample: any recording, no quality check", None)
        for s in samplesmod.all_samples():
            self.sample.addItem(f'{s.get("title", s["id"])}  ·  ~{s.get("target_seconds", "?")} s  ·  '
                                f'{s.get("paragraphs", "?")} paragraphs', s["id"])
        self.sample.currentIndexChanged.connect(self.show_sample)
        v2.addWidget(self.sample)
        self.reading = QTextBrowser()
        self.reading.setObjectName("reading")
        f = self.reading.font()
        f.setPointSizeF(f.pointSizeF() * 1.2)
        self.reading.setFont(f)
        self.reading.setVisible(False)          # the text is shown large on the Read aloud tab
        v2.addWidget(self.reading)
        self.sample_meta = QLabel("")
        self.sample_meta.setObjectName("muted")
        self.sample_meta.setWordWrap(True)
        v2.addWidget(self.sample_meta)
        self.rec = QPushButton("●  Record")
        self.rec.setObjectName("record")
        self.rec.clicked.connect(self.toggle_record)
        ab = QPushButton("Open recording…")
        ab.clicked.connect(self.browse_audio)
        row = QHBoxLayout()
        row.addWidget(self.rec, 1)
        row.addWidget(ab, 1)
        v2.addLayout(row)
        self.audio = QLineEdit()
        self.audio.setPlaceholderText("Record yourself reading the sample, or open any recording")
        v2.addWidget(self.audio)
        self.rec_label = QLabel("")
        self.rec_label.setObjectName("muted")
        self.rec_label.setVisible(False)
        v2.addWidget(self.rec_label)
        lv.addWidget(c2, 0)

        c3, v3 = _card()
        v3.addWidget(_step(3, "Run"))
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        self.mode = QComboBox()
        self.mode.addItem("Batch: the whole recording at once", "batch")
        self.mode.addItem("Replay: fed in real time, like a microphone", "replay")
        form.addRow("Mode", self.mode)
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 20)
        self.repeat.setValue(2)
        self.repeat.setToolTip("Runs after one load. Run 1 is often cold (first-call allocations).")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.25, 20)
        self.speed.setValue(1.0)
        self.speed.setSuffix(" ×")
        self.speed.setEnabled(False)
        self.speed.setToolTip("Replay speed: 1 × is real time")
        self.mode.currentIndexChanged.connect(lambda _: self.speed.setEnabled(self.mode.currentData() == "replay"))
        self.interval = QSpinBox()
        self.interval.setRange(5, 500)
        self.interval.setValue(20)
        self.interval.setSuffix(" ms")
        self.interval.setToolTip("How often the sampler reads CPU, memory and GPU")
        row = QHBoxLayout()
        row.setSpacing(14)
        for lbl, w in (("Runs", self.repeat), ("Replay speed", self.speed), ("Sampling", self.interval)):
            col = QVBoxLayout()
            col.setSpacing(3)
            l = QLabel(lbl)
            l.setObjectName("muted")
            col.addWidget(l)
            col.addWidget(w)
            row.addLayout(col, 1)
        form.addRow(row)
        v3.addLayout(form)
        self.go = QPushButton("Run profile")
        self.go.setObjectName("primary")
        self.go.setDefault(True)
        self.go.clicked.connect(self.run_profile)
        v3.addWidget(self.go)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setPlaceholderText("Progress appears here")
        self.log.setMaximumHeight(64)
        self.log.setVisible(False)                 # shown once a run starts
        v3.addWidget(self.log)
        lv.addWidget(c3, 0)

        # --- right: results ---
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(12)
        top = QHBoxLayout()
        self.headline = QLabel("No report yet. Set up a run on the left, or File → Open report.")
        self.headline.setObjectName("muted")
        self.headline.setWordWrap(True)
        top.addWidget(self.headline, 1)
        self.run_pick = QComboBox()
        self.run_pick.setMinimumWidth(150)
        self.run_pick.currentIndexChanged.connect(self.show_run)
        rl = QLabel("Run")
        rl.setObjectName("muted")
        top.addWidget(rl)
        top.addWidget(self.run_pick)
        self.show_load = QCheckBox("Show load phase")
        self.show_load.toggled.connect(lambda _: self.show_run(self.run_pick.currentIndex()))
        top.addWidget(self.show_load)
        rv.addLayout(top)

        kp = QGridLayout()
        kp.setSpacing(10)
        self.kpi = {
            "wall": Kpi("Pipeline time", "Wall time of the selected run"),
            "rtf": Kpi("Real-time factor", "Wall time ÷ audio duration (below 1 is faster than real time)"),
            "junc": Kpi("Junctions", "Time between stages with no model computing, summed"),
            "cpu": Kpi("CPU", "Average and peak cores over the run"),
            "rss": Kpi("Peak memory", "Peak resident memory of the process during the run"),
            "q": Kpi("Quality", "Against the reading sample: word error rate and paragraph-break F1"),
        }
        for i, k in enumerate(self.kpi.values()):
            kp.addWidget(k, 0, i)
        rv.addLayout(kp)

        self.tabs = QTabWidget()
        self.timeline = Timeline()
        self.tabs.addTab(self.timeline, "Timeline")
        self.stage_t = _table(["stage", "role", "provider", "threads", "wall", "model", "glue", "calls",
                               "CPU avg", "CPU peak", "busy thr.", "GPU avg", "GPU peak", "VRAM MB", "RSS MB",
                               "gaps", "gap max"])
        self.load_t = _table(["stage", "load ms", "RSS added MB", "CPU avg", "steps (ms)"])
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(12, 12, 12, 12)
        self.stage_note = lab = QLabel("<b>Run.</b> Times in ms. Model: inside model calls; glue: work between them; gaps: "
                     "between one stage's own model calls. CPU in cores (1.0 = one core). Busy thr.: threads "
                     "averaging ≥ 0.1 core. Threads: as configured.")
        lab.setWordWrap(True)
        sl.addWidget(lab)
        sl.addWidget(self.stage_t, 2)
        sl.addWidget(QLabel("<b>Load.</b> Making each model ready, once per profile."))
        sl.addWidget(self.load_t, 1)
        self.tabs.addTab(sw, "Stages")
        self.junc_t = _table(["junction", "times", "total ms", "mean ms", "max ms", "unaccounted ms",
                              "glue inside (ms)"])
        jw = QWidget()
        jl = QVBoxLayout(jw)
        jl.setContentsMargins(12, 12, 12, 12)
        jlab = QLabel("A <b>junction</b> runs from the end of one stage's last model call to the start of the "
                      "next stage's first: the time no model is computing while data is handed on. "
                      "<i>Unaccounted</i> is the part outside any named glue step (orchestration, Python "
                      "between calls).")
        jlab.setWordWrap(True)
        jl.addWidget(jlab)
        jl.addWidget(self.junc_t)
        self.tabs.addTab(jw, "Junctions")
        self.lat_t = _table(["segment", "audio s", "words", "VAD closing ms", "junction ms", "ASR ms",
                             "speech end → text ms"])
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(12, 12, 12, 12)
        self.lat_label = QLabel("")
        self.lat_label.setWordWrap(True)
        ll.addWidget(self.lat_label)
        ll.addWidget(self.lat_t)
        self.tabs.addTab(lw, "Replay latency")
        self.output = QTextBrowser()
        self.tabs.addTab(self.output, "Output")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.summary.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.summary, "Summary")
        # a large copy of the sample to read from while recording
        self.prompter = QTextBrowser()
        self.prompter.setObjectName("prompter")
        pf = self.prompter.font()
        pf.setPointSizeF(pf.pointSizeF() * 1.7)
        self.prompter.setFont(pf)
        self.tabs.addTab(self.prompter, "Read aloud")
        rv.addWidget(self.tabs, 1)

        # the setup column scrolls on short screens instead of forcing the window taller
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([440, 1060])
        split.setStretchFactor(1, 1)
        bl.addWidget(split)
        self.setCentralWidget(root)

        m = self.menuBar().addMenu("&File")
        for text, slot, key in (("Open report…", self.open_report, "Ctrl+O"),
                                ("Open recording…", self.browse_audio, "Ctrl+Shift+O"),
                                ("Run profile", self.run_profile, "Ctrl+R"), (None, None, None),
                                ("Quit", self.close, "Ctrl+Q")):
            if text is None:
                m.addSeparator()
                continue
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(slot)
            m.addAction(a)
        self.statusBar().showMessage(f"Reports are saved in {REPORTS}")

        self.tick = QTimer(self)
        self.tick.timeout.connect(self._rec_tick)
        self.load_vars()
        self.show_sample()
        self.lat_tab = lw
        self.tabs.setTabEnabled(self.tabs.indexOf(lw), False)
        self.tabs.setCurrentWidget(self.prompter)

    # setup -----------------------------------------------------------------------------------
    def browse_pipeline(self):
        f, _ = QFileDialog.getOpenFileName(self, "Pipeline", str(CONFIG / "pipelines"), "Pipeline (*.toml)")
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
        self.subtitle.setText(doc.get("description") or "Where the time and the compute go in a multi-model "
                              "local speech pipeline")
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
            self.reading.setHtml("<p>Pick a reading sample to read aloud: about 30, 60 or 120 seconds, "
                                 "with paragraphs, so the paragraph stage can be checked too.</p>"
                                 "<p>Or open any recording and profile it without a quality check.</p>")
            self.sample_meta.setText("Profiling any recording: no word error rate or paragraph check.")
            self.prompter.setHtml("<div style='margin: 18px 40px; color: gray'><p>Pick a reading sample "
                                  "and it appears here in large type, to read from while you record.</p></div>")
            return
        s = samplesmod.get(sid)
        paras = "".join(f"<p style='line-height:140%'>{p}</p>" for p in s["text"].split("\n\n"))
        self.prompter.setHtml(f'<div style="margin: 18px 40px"><h2>{s.get("title", sid)}</h2>'
                              + "".join(f"<p style='line-height:160%; margin-bottom: 22px'>{p}</p>"
                                        for p in s["text"].split("\n\n")) + "</div>")
        self.sample_meta.setText(f'About {s.get("target_seconds", "?")} seconds · {s["words"]} words · '
                                 f'{s.get("paragraphs", "?")} paragraphs. The text is on the <b>Read aloud</b> '
                                 f'tab: press Record, read at your normal pace, pause between paragraphs.')
        if not self.rep:
            self.tabs.setCurrentWidget(self.prompter)
        self.reading.setHtml(f'<h3>{s.get("title", sid)}</h3><p style="color:gray"><i>About '
                             f'{s.get("target_seconds", "?")} seconds · {s["words"]} words · '
                             f'{s.get("paragraphs", "?")} paragraphs. Read at your normal pace and pause '
                             f'between paragraphs.</i></p>{paras}')

    def browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(self, "Recording", str(RECORDINGS),
                                           "Audio (*.wav *.flac *.mp3 *.ogg *.opus *.m4a);;All (*)")
        if f:
            self.audio.setText(f)

    def _rec_style(self, on: bool):
        self.rec.setProperty("recording", "true" if on else "false")
        self.rec.style().unpolish(self.rec)
        self.rec.style().polish(self.rec)

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
            self.rec.setText("■  Stop recording")
            self._rec_style(True)
            self.rec_label.setVisible(True)
            self.tabs.setCurrentWidget(self.prompter)
            self.prompter.verticalScrollBar().setValue(0)
            self.tick.start(150)
        else:
            audio = self.recorder.stop()
            self.recorder = None
            self.tick.stop()
            self.rec.setText("●  Record")
            self._rec_style(False)
            path = new_path(self.sample.currentData() or "take")
            save_wav(path, audio)
            self.audio.setText(str(path))
            self.rec_label.setText(f"Saved {len(audio) / 16000:.1f} s · {path.name}")

    def _rec_tick(self):
        if self.recorder:
            n = min(24, int(self.recorder.level * 240))
            self.rec_label.setText(f"Recording  {self.recorder.seconds:5.1f} s   " + "▮" * n + "▯" * (24 - n))

    # running ---------------------------------------------------------------------------------
    def run_profile(self):
        if self.proc:
            self.proc.kill()
            return
        audio = self.audio.text().strip()
        if not audio or not Path(audio).expanduser().is_file():
            QMessageBox.information(self, "Input", "Record yourself reading a sample, or open a recording, first.")
            return
        self.out_path = REPORTS / (f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{self.pipeline.currentText()}_"
                                   f"{self.mode.currentData()}.json")
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
        self.log.setVisible(True)
        self.proc = QProcess(self)
        env = QProcessEnvironment()
        for k, v in child_env().items():
            env.insert(k, v)
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.go.setText("Stop")
        self.progress.setVisible(True)
        self.statusBar().showMessage("Profiling… the numbers come from a separate process, so the window "
                                     "stays out of them")
        self.proc.start(sys.executable, args)

    def _read(self):
        txt = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in txt.splitlines():
            if line.startswith(("loading", "run ", "Traceback", "  File")) or "Error" in line:
                self.log.appendPlainText(line)

    def _done(self, code, status):
        self.proc = None
        self.go.setText("Run profile")
        self.progress.setVisible(False)
        if code == 0 and self.out_path and self.out_path.is_file():
            self.set_report(json.loads(self.out_path.read_text()))
            self.log.appendPlainText("done")
            self.statusBar().showMessage(f"Report saved: {self.out_path}")
        else:
            self.log.appendPlainText(f"failed (exit {code})")
            self.statusBar().showMessage("The profile failed: see the log")

    def open_report(self):
        f, _ = QFileDialog.getOpenFileName(self, "Report", str(REPORTS), "Report (*.json)")
        if f:
            self.set_report(json.loads(Path(f).read_text()))
            self.statusBar().showMessage(f)

    # results ---------------------------------------------------------------------------------
    def set_report(self, rep: dict):
        self.rep = rep
        h = rep["host"]
        g = (h.get("gpu") or {}).get("name", "no GPU telemetry")
        self.headline.setText(f'<b>{rep["pipeline"]["name"]}</b> · {rep["mode"]} · '
                              f'{rep["input"]["duration_s"]:.1f} s of audio · {h["cpu"]} ({h["cores"]} threads) · {g}')
        self.subtitle.setText(rep["pipeline"].get("description") or self.subtitle.text())
        self.run_pick.blockSignals(True)
        self.run_pick.clear()
        for r in rep["runs"]:
            cold = " (cold)" if r["run"] == 0 and len(rep["runs"]) > 1 else ""
            self.run_pick.addItem(f'Run {r["run"] + 1}{cold} · {r["wall_ms"] / 1000:.2f} s')
        self.run_pick.blockSignals(False)
        self._info = {s["name"]: s for s in rep["pipeline"]["stages"]}
        rows = []
        for name, ld in rep["load"].items():
            steps = "   ".join(f"{k} {v:,.0f}" for k, v in ld["steps"].items())
            rows.append([name, _f(ld["wall_ms"], 0), _f(ld.get("rss_delta_mb")), _f(ld["cpu_avg_cores"], 2), steps])
        _fill(self.load_t, rows)
        self.summary.setPlainText(summary_text(rep))
        q = rep.get("quality")
        html = ""
        if q:
            p = q["paragraphs"]
            html += (f'<p><b>Word error rate</b> {q["asr"]["wer"]:.1%} ({q["asr"]["errors"]} errors in '
                     f'{q["asr"]["ref_words"]} words) &nbsp;·&nbsp; <b>Paragraph breaks</b> {p["hits"]} of '
                     f'{p["ref_breaks"]} found, {p["hyp_breaks"]} placed (F1 {p["f1"]:.2f}, within '
                     f'{p["tolerance_words"]} words)</p>')
            self.kpi["q"].set(f'{q["asr"]["wer"]:.1%}', f'WER · paragraphs F1 {p["f1"]:.2f}')
        else:
            self.kpi["q"].set("–", "no reading sample")
        paras = "".join(f"<p style='line-height:140%'>{x}</p>" for x in rep["output"]["text"].split("\n\n"))
        html += f"<h3>Final text</h3>{paras}<h3>ASR pieces</h3><ol>"
        html += "".join(f"<li>{x}</li>" for x in rep["output"]["pieces"]) + "</ol>"
        self.output.setHtml(html)
        if self.tabs.currentWidget() is self.prompter:
            self.tabs.setCurrentWidget(self.timeline)
        last = len(rep["runs"]) - 1
        self.run_pick.blockSignals(True)
        self.run_pick.setCurrentIndex(last)
        self.run_pick.blockSignals(False)
        self.show_run(last)

    def show_run(self, i: int):
        if not self.rep or i < 0:
            return
        run = self.rep["runs"][i]
        self.timeline.set_report(self.rep, i, self.show_load.isChecked())
        cores = self.rep["host"]["cores"]
        wall = run["wall_ms"] or 1
        self.kpi["wall"].set(f'{run["wall_ms"] / 1000:,.2f} s', f'models computing {run["infer_ms"] / wall:.1%}')
        rtf = run["rtf"]
        self.kpi["rtf"].set(f'{rtf:.3f}', f'{1 / rtf:.1f}× faster than real time' if rtf and rtf < 1
                            else "not faster than real time")
        self.kpi["junc"].set(f'{run["junction_total_ms"]:,.1f} ms', f'{run["junction_total_ms"] / wall:.2%} of the run')
        self.kpi["cpu"].set(f'{_f(run["cpu_avg_cores"], 1)} cores', f'peak {_f(run["cpu_max_cores"], 1)} of {cores}')
        self.kpi["rss"].set(f'{_f(run["rss_max_mb"], 0)} MB', "resident, whole process")
        rows = []
        for name, s in run["stages"].items():
            inf = self._info[name]["info"]
            rows.append([name, self._info[name]["role"], inf.get("provider", "–"), _f(inf.get("threads")),
                         _f(s["wall_ms"]), _f(s["infer_ms"]), _f(s["glue_ms"], 2), s["infer_calls"],
                         _f(s["cpu_avg_cores"], 2), _f(s["cpu_max_cores"], 2), _f(s["threads_active"]),
                         _f(s["gpu_busy_avg"], 1, "%"), _f(s["gpu_busy_max"], 0, "%"), _f(s["vram_max_mb"], 0),
                         _f(s["rss_max_mb"], 0), _f(s["gaps"]["total_ms"], 2), _f(s["gaps"]["max_ms"], 2)])
        _fill(self.stage_t, rows)
        g = self.rep["host"].get("gpu")
        base = self.stage_note.text().split(" <i>GPU")[0]
        self.stage_note.setText(base + (" <i>GPU and VRAM are for the whole device: this GPU can't report per "
                                        "process.</i>" if g and not g.get("attributable") else ""))
        rows = []
        for j in run["junctions"]:
            rows.append([j["junction"], j["count"], _f(j["ms"], 3), _f(j["mean_ms"], 3), _f(j["max_ms"], 3),
                         _f(j["unaccounted_ms"], 3), "   ".join(f"{k} {v:.2f}" for k, v in j["glue_ms"].items())])
        _fill(self.junc_t, rows)
        rp = run.get("replay")
        _fill(self.lat_t, [[r["segment"] + 1, f'{r["start_s"]:.2f} – {r["end_s"]:.2f}', r["words"],
                            _f(r["vad_close_ms"]), _f(r["junction_ms"], 3), _f(r["asr_ms"]), _f(r["latency_ms"])]
                           for r in (rp["segments"] if rp else [])])
        if rp:
            lat = rp["latency_ms"] or {}
            self.lat_label.setText(
                f'<b>End of speech → text</b>: mean {_f(lat.get("mean"), 0)} ms, p95 {_f(lat.get("p95"), 0)} ms, '
                f'max {_f(lat.get("max"), 0)} ms. Last chunk → final text {_f(rp["tail_ms"], 0)} ms. '
                f'Processing fell behind the audio by up to {_f(rp["chunk_lag_ms"]["max"], 0)} ms '
                f'({rp["speed"]}× speed, {rp["chunk_ms"]:.0f} ms chunks).')
        self.tabs.setTabEnabled(self.tabs.indexOf(self.lat_tab), bool(rp))


def main(report: str | None = None) -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("asrprof")
    app.setApplicationDisplayName("ASR Pipeline Profiler")
    app.setDesktopFileName("asrprof")
    app.setWindowIcon(QIcon(str(ICON)))
    style.setup(app)
    w = Main()
    if report:
        w.set_report(json.loads(Path(report).read_text()))
    w.show()
    return app.exec()
