from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QStackedWidget, QProgressBar,
    QComboBox, QSizePolicy, QFileDialog, QMessageBox,
    QDialog, QCheckBox, QScrollArea, QRadioButton, QButtonGroup,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap, QColor, QPainter

import config


def apply_theme(app, dark=False):
    import config as _cfg
    _cfg.IS_DARK_MODE = dark

    if dark:
        _cfg.PRIMARY       = _cfg.DARK_PRIMARY
        _cfg.PRIMARY_LIGHT = _cfg.DARK_PRIMARY_LIGHT
        _cfg.BG            = _cfg.DARK_BG
        _cfg.WHITE         = _cfg.DARK_WHITE
        _cfg.TEXT_DARK     = _cfg.DARK_TEXT_DARK
        _cfg.TEXT_MED      = _cfg.DARK_TEXT_MED
        _cfg.TEXT_GRAY     = _cfg.DARK_TEXT_GRAY
        _cfg.BORDER        = _cfg.DARK_BORDER
    else:
        _cfg.PRIMARY       = "#1B5E35"
        _cfg.PRIMARY_LIGHT = "#C8E0D0"
        _cfg.BG            = "#F8F9FA"
        _cfg.WHITE         = "#FFFFFF"
        _cfg.TEXT_DARK     = "#1A1A2E"
        _cfg.TEXT_MED      = "#374151"
        _cfg.TEXT_GRAY     = "#6B7280"
        _cfg.BORDER        = "#E5E7EB"

    PRIMARY       = _cfg.PRIMARY
    PRIMARY_LIGHT = _cfg.PRIMARY_LIGHT
    BG            = _cfg.BG
    WHITE         = _cfg.WHITE
    TEXT_DARK     = _cfg.TEXT_DARK
    TEXT_MED      = _cfg.TEXT_MED
    TEXT_GRAY     = _cfg.TEXT_GRAY
    BORDER        = _cfg.BORDER

    alt_row   = "#141420" if dark else "#FAFAFA"
    sel_row   = "#1A2E24" if dark else "#D4EBE1"
    header_bg = "#111118" if dark else "#F5F5F5"
    scroll_handle = "#444466" if dark else "#D1D5DB"
    input_bg  = WHITE
    btn_cancel_bg = "#1C1C2E" if dark else "#F3F4F6"

    qss = f"""
QWidget, QDialog, QMainWindow {{
    background-color: {BG};
    color: {TEXT_DARK};
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QFrame {{
    background-color: {BG};
    color: {TEXT_DARK};
}}
QLabel {{
    background-color: transparent;
    color: {TEXT_DARK};
}}
QLineEdit {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    background-color: {WHITE};
    color: {TEXT_DARK};
}}
QLineEdit:focus {{
    border: 1.5px solid {PRIMARY};
}}
QPushButton {{
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    background-color: {WHITE};
    color: {TEXT_DARK};
}}
QPushButton:hover {{
    background-color: {PRIMARY_LIGHT};
    color: {PRIMARY};
}}
QPushButton:pressed {{
    padding: 9px 15px 7px 17px;
    background-color: {PRIMARY};
    color: #ffffff;
}}
QPushButton[class="primary"]:hover {{
    background-color: {PRIMARY};
    color: #ffffff;
    opacity: 0.88;
}}
QComboBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    background-color: {WHITE};
    color: {TEXT_DARK};
}}
QComboBox QAbstractItemView {{
    background-color: {WHITE};
    color: {TEXT_DARK};
    border: 1px solid {BORDER};
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: {TEXT_DARK};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QTableWidget {{
    border: none;
    outline: none;
    background-color: {WHITE};
    color: {TEXT_DARK};
    gridline-color: transparent;
    alternate-background-color: {BG};
}}
QTableWidget::item {{
    padding: 6px 10px;
    border: none;
    color: {TEXT_DARK};
    background-color: transparent;
}}
QTableWidget::item:selected {{
    background-color: {PRIMARY_LIGHT};
    color: {TEXT_DARK};
}}
QHeaderView::section {{
    background-color: {BG};
    color: {TEXT_GRAY};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 600;
}}
QScrollBar:vertical {{
    border: none;
    background-color: {BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background-color: {BG};
}}
QProgressBar {{
    background-color: {BORDER};
    border-radius: 4px;
    border: none;
}}
QProgressBar::chunk {{
    background-color: {PRIMARY};
    border-radius: 4px;
}}
QCheckBox {{
    color: {TEXT_DARK};
    spacing: 8px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {WHITE};
}}
QCheckBox::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
QRadioButton {{
    color: {TEXT_DARK};
    spacing: 8px;
    background-color: transparent;
}}
QRadioButton::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
QMessageBox {{
    background-color: {BG};
    color: {TEXT_DARK};
}}
QStackedWidget {{
    background-color: {BG};
}}
"""
    app.setStyleSheet(qss)
    for widget in app.topLevelWidgets():
        widget.setStyleSheet(f"background-color:{BG};")
        force_theme_on_all_widgets(widget)
        widget.update()


def force_theme_on_all_widgets(root_widget):
    from PyQt5.QtGui import QPalette, QColor
    import config as _cfg

    BG        = _cfg.BG
    WHITE     = _cfg.WHITE
    TEXT_DARK = _cfg.TEXT_DARK
    TEXT_MED  = _cfg.TEXT_MED
    TEXT_GRAY = _cfg.TEXT_GRAY
    BORDER    = _cfg.BORDER
    PRIMARY   = _cfg.PRIMARY
    PRIMARY_LIGHT = _cfg.PRIMARY_LIGHT

    # Stylesheet string replacements: baked hex values → current theme values.
    # Handles both bg/text/border in either direction (light↔dark).
    replacements = [
        # Backgrounds — light variants → BG
        ("background-color:#F8F9FA",  f"background-color:{BG}"),
        ("background-color: #F8F9FA", f"background-color:{BG}"),
        ("background:#F8F9FA",        f"background:{BG}"),
        ("background: #F8F9FA",       f"background:{BG}"),
        ("background-color:#F5F5F5",  f"background-color:{BG}"),
        ("background:#F5F5F5",        f"background:{BG}"),
        ("background-color:#FAFAFA",  f"background-color:{BG}"),
        ("background:#FAFAFA",        f"background:{BG}"),
        ("background-color:#F3F4F6",  f"background-color:{BG}"),
        ("background:#F3F4F6",        f"background:{BG}"),
        # Backgrounds — dark variant → BG
        ("background-color:#0A0A0F",  f"background-color:{BG}"),
        ("background:#0A0A0F",        f"background:{BG}"),
        # White/surface — both directions
        ("background-color:#FFFFFF",  f"background-color:{WHITE}"),
        ("background:#FFFFFF",        f"background:{WHITE}"),
        ("background-color:#111118",  f"background-color:{WHITE}"),
        ("background:#111118",        f"background:{WHITE}"),
        ("background:white",          f"background:{WHITE}"),
        ("background: white",         f"background:{WHITE}"),
        # Text — both directions
        ("color:#1A1A2E",  f"color:{TEXT_DARK}"),
        ("color: #1A1A2E", f"color:{TEXT_DARK}"),
        ("color:#F1F1F5",  f"color:{TEXT_DARK}"),
        ("color:#374151",  f"color:{TEXT_MED}"),
        ("color:#C4C4D4",  f"color:{TEXT_MED}"),
        ("color:#6B7280",  f"color:{TEXT_GRAY}"),
        ("color:#7777AA",  f"color:{TEXT_GRAY}"),
        # Borders — both directions
        ("border:1px solid #E5E7EB",        f"border:1px solid {BORDER}"),
        ("border: 1px solid #E5E7EB",       f"border:1px solid {BORDER}"),
        ("border:1px solid #222235",        f"border:1px solid {BORDER}"),
        ("border-bottom:1px solid #E5E7EB", f"border-bottom:1px solid {BORDER}"),
        ("border-bottom: 1px solid #E5E7EB",f"border-bottom:1px solid {BORDER}"),
        ("border-bottom:1px solid #222235", f"border-bottom:1px solid {BORDER}"),
        # PRIMARY — both directions
        ("background:#1B5E35",  f"background:{PRIMARY}"),
        ("background:#4ecf8e",  f"background:{PRIMARY}"),
        ("background:#166E42",  f"background:{PRIMARY}"),
        # PRIMARY_LIGHT — both directions
        ("background:#C8E0D0",  f"background:{PRIMARY_LIGHT}"),
        ("background:#0D2B18",  f"background:{PRIMARY_LIGHT}"),
    ]

    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT_DARK))
    palette.setColor(QPalette.Base,            QColor(WHITE))
    palette.setColor(QPalette.AlternateBase,   QColor(BG))
    palette.setColor(QPalette.Text,            QColor(TEXT_DARK))
    palette.setColor(QPalette.Button,          QColor(WHITE))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT_DARK))
    palette.setColor(QPalette.Highlight,       QColor(PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

    def patch(w):
        w.setPalette(palette)
        w.setAutoFillBackground(True)
        ss = w.styleSheet()
        if ss:
            for old, new in replacements:
                if old in ss:
                    ss = ss.replace(old, new)
            w.setStyleSheet(ss)
        w.update()

    patch(root_widget)
    for child in root_widget.findChildren(QWidget):
        patch(child)
    root_widget.repaint()


def h_sep():
    import config as _cfg
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"background:{_cfg.BORDER}; max-height:1px; border:none;")
    return line


def make_badge(text):
    import config as _cfg
    bg, fg = _cfg.BADGE_MAP.get(text, ("#E5E7EB", _cfg.TEXT_MED))
    outer = QWidget()
    outer.setStyleSheet("background:transparent;")
    ol = QHBoxLayout(outer)
    ol.setContentsMargins(8, 4, 8, 4)
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setFixedHeight(26)
    lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    lbl.setStyleSheet(
        f"background:{bg}; color:{fg}; border-radius:4px;"
        f"padding:2px 10px; font-size:12px; font-weight:600;"
    )
    ol.addWidget(lbl)
    return outer


def make_stat_card(title, value, value_color=None):
    import config as _cfg
    if value_color is None:
        value_color = _cfg.TEXT_DARK
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background:{_cfg.WHITE}; border:1px solid {_cfg.BORDER}; border-radius:8px; }}"
    )
    frame.setFixedHeight(82)
    frame.setMinimumWidth(130)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 12, 16, 12)
    lay.setSpacing(2)
    v = QLabel(str(value))
    v.setStyleSheet(
        f"font-size:26px; font-weight:800; color:{value_color}; border:none; background:transparent;"
    )
    t = QLabel(title)
    t.setStyleSheet(f"font-size:11px; color:{_cfg.TEXT_GRAY}; border:none; background:transparent;")
    lay.addWidget(v)
    lay.addWidget(t)
    return frame


def make_avatar(initials, size=44, bg=None, fg=None):
    import config as _cfg
    if bg is None:
        bg = _cfg.PRIMARY
    if fg is None:
        fg = _cfg.WHITE
    lbl = QLabel(initials)
    lbl.setFixedSize(size, size)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background:{bg}; color:{fg}; border-radius:{size // 2}px;"
        f"font-size:{size // 2 - 2}px; font-weight:800;"
    )
    return lbl


def make_sidebar_base(initials, name, role):
    import config as _cfg
    sidebar = QFrame()
    sidebar.setFixedWidth(200)
    sidebar.setStyleSheet(
        f"QFrame {{ background:{_cfg.WHITE}; border-right:1px solid {_cfg.BORDER}; border-top:none;"
        f"border-left:none; border-bottom:none; }}"
    )
    lay = QVBoxLayout(sidebar)
    lay.setContentsMargins(16, 24, 16, 16)
    lay.setSpacing(6)

    av = make_avatar(initials)
    name_lbl = QLabel(name)
    name_lbl.setAlignment(Qt.AlignCenter)
    name_lbl.setStyleSheet(
        f"font-weight:700; color:{_cfg.TEXT_DARK}; font-size:13px; background:transparent; border:none;"
    )
    role_lbl = QLabel(role)
    role_lbl.setAlignment(Qt.AlignCenter)
    role_lbl.setStyleSheet(
        f"color:{_cfg.TEXT_GRAY}; font-size:11px; background:transparent; border:none;"
    )
    lay.addWidget(av, alignment=Qt.AlignCenter)
    lay.addWidget(name_lbl)
    lay.addWidget(role_lbl)
    lay.addSpacing(10)
    lay.addWidget(h_sep())
    lay.addSpacing(6)
    return sidebar, lay


def make_table(cols, rows_data, col_widths=None, stretch_col=1):
    import config as _cfg
    tbl = QTableWidget(len(rows_data), len(cols))
    tbl.setStyleSheet(f"QTableWidget {{ background-color:{_cfg.WHITE}; color:{_cfg.TEXT_DARK}; }} QTableWidget::item {{ color:{_cfg.TEXT_DARK}; }} QHeaderView::section {{ background-color:{_cfg.BG}; color:{_cfg.TEXT_GRAY}; border-bottom:1px solid {_cfg.BORDER}; }}")
    tbl.setHorizontalHeaderLabels(cols)
    tbl.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    if col_widths:
        for ci, w in col_widths.items():
            if ci != stretch_col:
                tbl.setColumnWidth(ci, w)
    tbl.verticalHeader().setVisible(False)
    tbl.setAlternatingRowColors(True)
    tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
    tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    tbl.setFocusPolicy(Qt.NoFocus)
    tbl.setShowGrid(False)
    return tbl


class AnimatedButton(QPushButton):
    """QPushButton with smooth hover color transition."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._hover_progress = 0.0   # 0.0 = normal, 1.0 = fully hovered
        self._hovered = False
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, event):
        self._hovered = True
        if not self._anim_timer.isActive():
            self._anim_timer.start(16)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        if not self._anim_timer.isActive():
            self._anim_timer.start(16)
        super().leaveEvent(event)

    def _tick(self):
        speed = 0.12
        if self._hovered:
            self._hover_progress = min(1.0, self._hover_progress + speed)
        else:
            self._hover_progress = max(0.0, self._hover_progress - speed)

        if (self._hovered and self._hover_progress >= 1.0) or \
           (not self._hovered and self._hover_progress <= 0.0):
            self._anim_timer.stop()

        self._apply_hover_style()

    def _apply_hover_style(self):
        pass  # Styling is handled via QSS :hover pseudo-class below
