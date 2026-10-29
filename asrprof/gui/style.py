"""Look and feel. Colours are the Hebrish brand tokens (Hebrish-ASR/Hebrish-Mgmt `brand/tokens.json`,
light and dark), chosen at start-up from the system palette, so the window follows the desktop's
light or dark theme. Display face: Fraunces (OFL, bundled)."""

from __future__ import annotations

from dataclasses import dataclass

from ..paths import BUNDLED

LIGHT = dict(bg="#fbf8f2", surface="#ffffff", band="#f1ece1", ink="#1d2422", muted="#5b6662", line="#e2dccf",
             accent="#0f6b5c", accent_hover="#0c5a4d", accent_ink="#ffffff", accent_soft="#d8efe7",
             gold="#c8901c", red="#d62828", junction="#e0282e", glue="#a3a9a6", grid="#ece6da")
DARK = dict(bg="#111615", surface="#18201e", band="#151c1a", ink="#e8ece9", muted="#9aa7a2", line="#2a3532",
            accent="#3fbf9f", accent_hover="#35a88b", accent_ink="#0c1412", accent_soft="#163a31",
            gold="#e0b14a", red="#ef4b4b", junction="#ff5a4e", glue="#6f7a76", grid="#222b29")
# stage lanes, in pipeline order: brand green first, then colours that stay apart in both themes
STAGES_LIGHT = ["#0f6b5c", "#c8701c", "#3b6fb6", "#8a4fb8", "#b8475f", "#4f8a2f"]
STAGES_DARK = ["#3fbf9f", "#e9a04a", "#6f9fe6", "#b78ae0", "#e07a91", "#8cc464"]


@dataclass
class Theme:
    dark: bool
    c: dict
    stages: list
    display: str


theme: Theme | None = None


def _display_family() -> str:
    from PySide6.QtGui import QFontDatabase
    fid = QFontDatabase.addApplicationFont(str(BUNDLED / "fonts" / "Fraunces.ttf"))
    fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
    return fams[0] if fams else "serif"


def setup(app) -> Theme:
    """Pick light or dark from the system palette, set the palette and style sheet."""
    global theme
    from PySide6.QtGui import QColor, QPalette
    dark = app.palette().color(QPalette.Window).lightness() < 128
    c = DARK if dark else LIGHT
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "surface"), (QPalette.AlternateBase, "band"),
                      (QPalette.Text, "ink"), (QPalette.WindowText, "ink"), (QPalette.Button, "surface"),
                      (QPalette.ButtonText, "ink"), (QPalette.Highlight, "accent"),
                      (QPalette.HighlightedText, "accent_ink"), (QPalette.ToolTipBase, "surface"),
                      (QPalette.ToolTipText, "ink"), (QPalette.Mid, "line"), (QPalette.PlaceholderText, "muted")):
        pal.setColor(role, QColor(c[key]))
    app.setPalette(pal)
    theme = Theme(dark, c, STAGES_DARK if dark else STAGES_LIGHT, _display_family())
    app.setStyleSheet(qss(c))
    return theme


def qss(c: dict) -> str:
    return f"""
    QMainWindow, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {c['bg']}; }}
    QWidget {{ color: {c['ink']}; }}
    QFrame#card {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 12px; }}
    QFrame#header {{ background: {c['surface']}; border-bottom: 1px solid {c['line']}; }}
    QLabel#title {{ font-size: 21px; font-weight: 700; }}
    QLabel#subtitle, QLabel#muted {{ color: {c['muted']}; }}
    QLabel#step {{ font-size: 14px; font-weight: 600; }}
    QLabel#stepnum {{ background: {c['accent_soft']}; color: {c['accent']}; border-radius: 11px;
                      min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
                      font-weight: 700; qproperty-alignment: AlignCenter; }}
    QFrame#kpi {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 10px; }}
    QLabel#kpiValue {{ font-size: 20px; font-weight: 700; }}
    QLabel#kpiLabel {{ color: {c['muted']}; font-size: 11px; }}
    QLabel#kpiNote {{ color: {c['muted']}; font-size: 10px; }}
    QPushButton {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 8px;
                   padding: 6px 12px; }}
    QPushButton:hover {{ border-color: {c['accent']}; }}
    QPushButton#primary {{ background: {c['accent']}; color: {c['accent_ink']}; border: none;
                           font-weight: 600; padding: 9px 16px; font-size: 14px; }}
    QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
    QPushButton#record {{ color: {c['red']}; font-weight: 600; }}
    QPushButton#record[recording="true"] {{ background: {c['red']}; color: white; border: none; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {c['surface']}; border: 1px solid {c['line']};
                   border-radius: 7px; padding: 4px 8px; min-height: 22px; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c['accent']}; }}
    QTableWidget {{ background: {c['surface']}; alternate-background-color: {c['band']}; border: 1px solid {c['line']};
                    border-radius: 8px; gridline-color: transparent; selection-background-color: {c['accent_soft']};
                    selection-color: {c['ink']}; }}
    QTableWidget::item {{ padding: 2px 5px; }}
    QHeaderView::section {{ background: {c['band']}; color: {c['muted']}; border: none;
                            border-bottom: 1px solid {c['line']}; padding: 6px 5px; font-weight: 600; }}
    QTabWidget::pane {{ border: 1px solid {c['line']}; border-radius: 10px; background: {c['surface']}; top: -1px; }}
    QTabBar::tab {{ background: transparent; padding: 8px 16px; margin-right: 2px; color: {c['muted']};
                    border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {c['ink']}; border-bottom: 2px solid {c['accent']}; font-weight: 600; }}
    QTabBar::tab:disabled {{ color: {c['line']}; }}
    QTextBrowser, QPlainTextEdit {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 8px;
                                    padding: 6px; }}
    QTextBrowser#reading {{ background: {c['band']}; border: none; padding: 12px; }}
    QTextBrowser#prompter {{ background: {c['surface']}; border: none; padding: 8px; }}
    QProgressBar {{ background: {c['band']}; border: none; border-radius: 3px; max-height: 6px; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
    QSplitter::handle {{ background: transparent; width: 8px; }}
    QToolTip {{ background: {c['surface']}; color: {c['ink']}; border: 1px solid {c['line']}; padding: 6px; }}
    QStatusBar {{ color: {c['muted']}; }}
    """
