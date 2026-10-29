"""The timeline: one lane per stage (stage turns light, model calls solid, glue grey), a lane
of junctions, then CPU (cores) and GPU (busy %) traces on the same time axis.

Wheel zooms around the pointer, drag pans, double-click shows everything. Hovering a span or
junction shows its numbers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from . import style

LANE_H, LANE_GAP, LEFT, TOP = 28, 8, 116, 30


def _colours():
    t = style.theme
    c = t.c if t else style.LIGHT
    stages = t.stages if t else style.STAGES_LIGHT
    return c, stages


class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(420)
        self.rep = None
        self.run = 0
        self.show_load = False
        self.t0, self.t1 = 0.0, 1.0
        self.origin = 0.0
        self.plot_h = 96
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
        # the axis counts from the start of the run (or of the profile, with the load phase shown)
        self.origin = 0.0 if self.show_load or not self.rep else next(
            (s["t0_ms"] for s in self.rep["spans"] if s["kind"] == "run" and s["run"] == self.run), a)
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
        pal, stage_cols = _colours()
        p.fillRect(self.rect(), QColor(pal["surface"]))
        self._hits = []
        if not self.rep:
            p.setPen(QColor(pal["muted"]))
            p.drawText(self.rect(), Qt.AlignCenter, "The timeline of a run appears here.\n"
                       "Lanes per stage, model calls in colour, glue in grey, junctions in red,\n"
                       "with CPU and GPU underneath on the same clock.")
            return
        fg = QColor(pal["ink"])
        self.GLUE, self.JUNCTION = QColor(pal["glue"]), QColor(pal["junction"])
        self.grid = QColor(pal["grid"])
        self.muted = QColor(pal["muted"])
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))
        stages = self.stages() + ["pipeline"]
        spans = self._spans()
        lanes = {name: TOP + i * (LANE_H + LANE_GAP) for i, name in enumerate(stages)}
        jy = TOP + len(stages) * (LANE_H + LANE_GAP)
        has_gpu = any("gpu" in s for s in self.rep["samples"])
        free = self.height() - (jy + LANE_H + 30) - 60
        self.plot_h = PLOT_H = max(80, min(220, free / (2 if has_gpu else 1) - (26 if has_gpu else 0)))
        cpu_y = jy + LANE_H + 30
        gpu_y = cpu_y + PLOT_H + 26
        self._axis(p, fg, small)

        colour = {n: QColor(stage_cols[i % len(stage_cols)]) for i, n in enumerate(stages)}
        colour["pipeline"] = QColor(pal["muted"])
        bold = QFont(small)
        bold.setBold(True)
        for i, name in enumerate(stages):
            y = lanes[name]
            p.setPen(Qt.NoPen)
            p.setBrush(colour[name])
            p.drawEllipse(QRectF(10, y + LANE_H / 2 - 4, 8, 8))
            p.setFont(bold)
            p.setPen(fg)
            p.drawText(QRectF(22, y, LEFT - 30, LANE_H), Qt.AlignVCenter | Qt.AlignLeft, name)
        p.setPen(QColor(pal["junction"]))
        p.drawText(QRectF(22, jy, LEFT - 30, LANE_H), Qt.AlignVCenter | Qt.AlignLeft, "junctions")
        p.setFont(small)
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
                c.setAlpha(40 if s["kind"] != "load" else 28)
                r = QRectF(x0, y, max(1.0, x1 - x0), LANE_H)
            elif s["kind"] == "glue":
                c = self.GLUE
                r = QRectF(x0, y + LANE_H * 0.62, max(1.0, x1 - x0), LANE_H * 0.3)
            else:
                r = QRectF(x0, y + 4, max(1.0, x1 - x0), LANE_H * 0.52)
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            rad = min(4.0, r.width() / 2, r.height() / 2)
            p.drawRoundedRect(r, rad, rad)
            if s["kind"] in ("infer", "glue", "load") and r.width() > 60:
                p.setPen(QColor("#ffffff") if s["kind"] == "infer" and not (style.theme and style.theme.dark)
                         else QColor("#0c1412") if s["kind"] == "infer" else fg)
                txt = p.fontMetrics().elidedText(f'{s["name"]} {s["ms"]:.1f} ms', Qt.ElideRight, int(r.width() - 8))
                p.drawText(r.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, txt)
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
                    r = QRectF(max(x0, LEFT), jy + 3, max(3.0, x1 - x0), LANE_H - 12)
                    p.setPen(Qt.NoPen)
                    p.setBrush(self.JUNCTION)
                    p.drawRoundedRect(r, 1.5, 1.5)
                    p.setPen(fg)
                    lbl = f"{b - a:.2f} ms" if b - a < 10 else f"{b - a:.0f} ms"
                    if r.center().x() - 30 > last_label:          # skip labels that would overlap
                        p.drawText(QRectF(r.center().x() - 40, jy + LANE_H - 9, 80, 14), Qt.AlignCenter, lbl)
                        last_label = r.center().x() + 30
                    glue = "<br>".join(f"{k}: {v:.3f} ms" for k, v in j["glue_ms"].items())
                    self._hits.append((r.adjusted(-3, -3, 3, 3),
                                       f'<b>junction {j["junction"]}</b><br>{b - a:.3f} ms<br>{glue}'
                                       f'<br>unaccounted (all {j["count"]}): {j["unaccounted_ms"]:.3f} ms'))

        self._plot(p, fg, small, cpu_y, "cpu", "CPU cores", self.rep["host"]["cores"] or 1, QColor(pal["accent"]))
        bottom = cpu_y + PLOT_H
        if has_gpu:
            g = self.rep["host"].get("gpu") or {}
            self._plot(p, fg, small, gpu_y, "gpu", "GPU busy %" + ("" if g.get("attributable") else "\n(whole device)"),
                       100, QColor(pal["gold"]))
            bottom = gpu_y + PLOT_H
        self._legend(p, small, bottom + 22, colour, stages)

    def _legend(self, p, font, y, colour, stages):
        p.setFont(font)
        x = LEFT
        first = QColor(colour[stages[0]]) if stages else self.muted
        soft = QColor(first)
        soft.setAlpha(40)
        items = [("model call", first, 0.52), ("glue", self.GLUE, 0.3), ("stage turn", soft, 1.0),
                 ("junction", self.JUNCTION, 0.52)]
        for label, col, h in items:
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(QRectF(x, y + (14 - 14 * h) / 2, 22, 14 * h), 2, 2)
            p.setPen(self.muted)
            p.drawText(QPointF(x + 28, y + 11), label)
            x += 28 + p.fontMetrics().horizontalAdvance(label) + 22
        p.drawText(QPointF(x + 10, y + 11), "Scroll to zoom · drag to pan · double-click to fit · hover for details")

    def _axis(self, p, fg, font):
        p.setFont(font)
        span = self.t1 - self.t0
        step = 10 ** max(-1, int(f"{span / 8:e}".split("e")[1]))
        for m in (1, 2, 5, 10):
            if span / (step * m) <= 10:
                step *= m
                break
        t = self.origin + ((self.t0 - self.origin) // step) * step
        pen = QPen(self.grid)
        while t <= self.t1:
            x = self.x(t)
            if LEFT <= x <= self.width() - 12:
                p.setPen(pen)
                p.drawLine(QPointF(x, TOP - 4), QPointF(x, self.height()))
                p.setPen(self.muted)
                rel = t - self.origin
                if abs(rel) < step / 1000:
                    rel = 0.0
                lbl = f"{rel / 1000:.2f} s" if step >= 100 else f"{rel:.1f} ms"
                if x + 3 + p.fontMetrics().horizontalAdvance(lbl) < self.width() - 12:
                    p.drawText(QPointF(x + 3, 16), lbl)
            t += step

    def _plot(self, p, fg, font, y, key, label, ymax, colour):
        PLOT_H = self.plot_h
        p.setPen(fg)
        f = QFont(font)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(10, y, LEFT - 18, 34), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, label)
        p.setFont(font)
        p.setPen(self.muted)
        p.drawText(QRectF(4, y + PLOT_H - 14, LEFT - 10, 14), Qt.AlignRight, "0")
        p.drawText(QRectF(4, y - 2, LEFT - 10, 14), Qt.AlignRight, f"{ymax}")
        box = QRectF(LEFT, y, self.width() - LEFT - 12, PLOT_H)
        p.setPen(QPen(self.grid))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(box, 4, 4)
        p.drawLine(QPointF(LEFT, y + PLOT_H / 2), QPointF(box.right(), y + PLOT_H / 2))
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
        if started:
            area = QPainterPath(path)
            area.lineTo(path.currentPosition().x(), y + PLOT_H)
            area.lineTo(path.elementAt(0).x, y + PLOT_H)
            area.closeSubpath()
            fill = QColor(colour)
            fill.setAlpha(45)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPath(area)
        p.setPen(QPen(colour, 1.6))
        p.setBrush(Qt.NoBrush)
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
