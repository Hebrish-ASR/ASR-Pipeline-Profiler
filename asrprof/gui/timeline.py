"""The timeline: one lane per stage (stage turns light, model calls solid, glue grey), a lane
of junctions, then CPU (cores) and GPU (busy %) traces on the same time axis.

Wheel zooms around the pointer, drag pans, double-click shows everything. Hovering a span or
junction shows its numbers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

PALETTE = ["#2f7fd1", "#d1772f", "#3aa35b", "#9a4fc4", "#c43f5d", "#2fa7a3", "#8a8a2f"]
GLUE = QColor("#9aa0a6")
JUNCTION = QColor("#e0282e")
CPU = QColor("#2f7fd1")
GPU = QColor("#3aa35b")

LANE_H, LANE_GAP, LEFT, TOP, PLOT_H = 26, 6, 110, 24, 90


class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(360)
        self.rep = None
        self.run = 0
        self.show_load = False
        self.t0, self.t1 = 0.0, 1.0
        self._drag = None
        self._hits: list[tuple[QRectF, str]] = []

    # data ------------------------------------------------------------------------------------
    def set_report(self, rep: dict | None, run: int = 0, show_load: bool = False):
        self.rep, self.run, self.show_load = rep, run, show_load
        self.reset_view()

    def _spans(self):
        if not self.rep:
            return []
        return [s for s in self.rep["spans"]
                if (s["phase"] == "run" and s["run"] == self.run) or (self.show_load and s["phase"] == "load")]

    def _extent(self):
        sp = self._spans()
        if not sp:
            return 0.0, 1.0
        return min(s["t0_ms"] for s in sp), max(s["t1_ms"] for s in sp)

    def reset_view(self):
        a, b = self._extent()
        pad = (b - a) * 0.01 or 1
        self.t0, self.t1 = a - pad, b + pad
        self.update()

    def stages(self) -> list[str]:
        return [s["name"] for s in self.rep["pipeline"]["stages"]] if self.rep else []

    # geometry --------------------------------------------------------------------------------
    def x(self, t):
        return LEFT + (t - self.t0) / (self.t1 - self.t0) * (self.width() - LEFT - 12)

    def t_at(self, x):
        return self.t0 + (x - LEFT) / (self.width() - LEFT - 12) * (self.t1 - self.t0)

    # painting --------------------------------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), self.palette().base())
        self._hits = []
        if not self.rep:
            p.drawText(self.rect(), Qt.AlignCenter, "Run a profile or open a report")
            return
        fg = self.palette().text().color()
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))
        stages = self.stages() + ["pipeline"]
        spans = self._spans()
        lanes = {name: TOP + i * (LANE_H + LANE_GAP) for i, name in enumerate(stages)}
        jy = TOP + len(stages) * (LANE_H + LANE_GAP)
        cpu_y = jy + LANE_H + 28
        gpu_y = cpu_y + PLOT_H + 28
        self._axis(p, fg, small)

        p.setFont(small)
        for i, name in enumerate(stages):
            y = lanes[name]
            p.setPen(fg)
            p.drawText(QRectF(4, y, LEFT - 10, LANE_H), Qt.AlignVCenter | Qt.AlignRight, name)
        p.drawText(QRectF(4, jy, LEFT - 10, LANE_H), Qt.AlignVCenter | Qt.AlignRight, "junctions")

        colour = {n: QColor(PALETTE[i % len(PALETTE)]) for i, n in enumerate(stages)}
        order = {"stage": 0, "load": 0, "run": 0, "glue": 1, "infer": 2}
        for s in sorted(spans, key=lambda s: order.get(s["kind"], 3)):
            if s["stage"] not in lanes:
                continue
            x0, x1 = self.x(s["t0_ms"]), self.x(s["t1_ms"])
            if x1 < LEFT or x0 > self.width():
                continue
            x0 = max(x0, LEFT)
            y = lanes[s["stage"]]
            c = QColor(colour[s["stage"]])
            if s["kind"] in ("stage", "run", "load"):
                c.setAlpha(55 if s["kind"] != "load" else 35)
                r = QRectF(x0, y, max(1.0, x1 - x0), LANE_H)
            elif s["kind"] == "glue":
                c = GLUE
                r = QRectF(x0, y + LANE_H * 0.55, max(1.0, x1 - x0), LANE_H * 0.45)
            else:
                r = QRectF(x0, y + 3, max(1.0, x1 - x0), LANE_H * 0.5)
            p.fillRect(r, c)
            if s["kind"] in ("infer", "glue", "load") and r.width() > 60:
                p.setPen(Qt.white if s["kind"] == "infer" else fg)
                p.drawText(r.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft,
                           f'{s["name"]} {s["ms"]:.1f} ms')
            meta = ", ".join(f"{k}={v}" for k, v in s.get("meta", {}).items())
            cores = s["cpu_ms"] / s["ms"] if s["ms"] else 0
            self._hits.append((r, f'<b>{s["stage"]} · {s["name"]}</b> ({s["kind"]}, {s["phase"]})<br>'
                                  f'{s["ms"]:.3f} ms at {s["t0_ms"]:.1f} ms<br>CPU {s["cpu_ms"]:.1f} ms '
                                  f'(avg {cores:.2f} cores)' + (f"<br>{meta}" if meta else "")))

        run = self.rep["runs"][self.run]
        last_label = -1e9
        if run:
            for j in sorted(run["junctions"], key=lambda j: j["at_ms"][0][0]):
                for a, b in j["at_ms"]:
                    x0, x1 = self.x(a), self.x(b)
                    if x1 < LEFT or x0 > self.width():
                        continue
                    r = QRectF(max(x0, LEFT), jy + 4, max(2.0, x1 - x0), LANE_H - 8)
                    p.fillRect(r, JUNCTION)
                    p.setPen(fg)
                    lbl = f"{b - a:.2f} ms" if b - a < 10 else f"{b - a:.0f} ms"
                    if r.center().x() - 30 > last_label:          # skip labels that would overlap
                        p.drawText(QRectF(r.center().x() - 40, jy + LANE_H - 2, 80, 14), Qt.AlignCenter, lbl)
                        last_label = r.center().x() + 30
                    glue = "<br>".join(f"{k}: {v:.3f} ms" for k, v in j["glue_ms"].items())
                    self._hits.append((r.adjusted(-3, -3, 3, 3),
                                       f'<b>junction {j["junction"]}</b><br>{b - a:.3f} ms<br>{glue}'
                                       f'<br>unaccounted (all {j["count"]}): {j["unaccounted_ms"]:.3f} ms'))

        self._plot(p, fg, small, cpu_y, "cpu", "CPU cores", self.rep["host"]["cores"] or 1, CPU)
        if any("gpu" in s for s in self.rep["samples"]):
            g = self.rep["host"].get("gpu") or {}
            self._plot(p, fg, small, gpu_y, "gpu", "GPU busy %" + ("" if g.get("attributable") else " (device)"),
                       100, GPU)

    def _axis(self, p, fg, font):
        p.setFont(font)
        span = self.t1 - self.t0
        step = 10 ** max(-1, int(f"{span / 8:e}".split("e")[1]))
        for m in (1, 2, 5, 10):
            if span / (step * m) <= 10:
                step *= m
                break
        t = (self.t0 // step) * step
        pen = QPen(self.palette().mid().color())
        while t <= self.t1:
            x = self.x(t)
            if x >= LEFT:
                p.setPen(pen)
                p.drawLine(QPointF(x, TOP - 4), QPointF(x, self.height()))
                p.setPen(fg)
                lbl = f"{t / 1000:.2f} s" if step >= 100 else f"{t:.1f} ms"
                p.drawText(QPointF(x + 2, 12), lbl)
            t += step

    def _plot(self, p, fg, font, y, key, label, ymax, colour):
        p.setPen(fg)
        p.drawText(QRectF(4, y, LEFT - 10, 30), Qt.AlignTop | Qt.AlignRight | Qt.TextWordWrap, label)
        p.drawText(QRectF(4, y + PLOT_H - 14, LEFT - 10, 14), Qt.AlignRight, "0")
        p.drawText(QRectF(4, y + 30, LEFT - 10, 14), Qt.AlignRight, f"top {ymax}")
        p.setPen(QPen(self.palette().mid().color()))
        p.drawRect(QRectF(LEFT, y, self.width() - LEFT - 12, PLOT_H))
        path = QPainterPath()
        started = False
        for s in self.rep["samples"]:
            v = s.get("cores") if key == "cpu" else (s.get("gpu") or {}).get("busy")
            if v is None:
                continue
            x = self.x(s["t_ms"])
            if x < LEFT or x > self.width():
                continue
            yy = y + PLOT_H - min(1.0, v / ymax) * PLOT_H
            if not started:
                path.moveTo(x, yy)
                started = True
            else:
                path.lineTo(x, yy)
        p.setPen(QPen(colour, 1.4))
        p.drawPath(path)

    # interaction -----------------------------------------------------------------------------
    def wheelEvent(self, ev):
        f = 0.8 if ev.angleDelta().y() > 0 else 1.25
        t = self.t_at(ev.position().x())
        self.t0, self.t1 = t - (t - self.t0) * f, t + (self.t1 - t) * f
        self.update()

    def mousePressEvent(self, ev):
        self._drag = (ev.position().x(), self.t0, self.t1)

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        self.reset_view()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag:
            x0, a, b = self._drag
            dt = (pos.x() - x0) / (self.width() - LEFT - 12) * (b - a)
            self.t0, self.t1 = a - dt, b - dt
            self.update()
            return
        for r, tip in reversed(self._hits):
            if r.contains(pos):
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()
