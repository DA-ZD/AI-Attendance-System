# ui_login.py
# Contains: AppWindow, LoginWindow, CourseCard, MainWindow

import os
import ast
import bcrypt
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFrame, QDialog, QLineEdit, QStackedWidget,
    QMessageBox, QApplication, QGraphicsDropShadowEffect, QSizePolicy,
    QProgressBar, QFileDialog, QHeaderView, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient, QIcon

import config
from config import (
    BG, WHITE, PRIMARY, PRIMARY_LIGHT, BORDER,
    TEXT_DARK, TEXT_MED, TEXT_GRAY,
    IS_DARK_MODE, DB_CONNECTED, db,
    get_db,
)
from ui_theme import (
    apply_theme, force_theme_on_all_widgets,
    make_badge, make_sidebar_base, make_table,
)
from ui_session import LiveSessionWindow, choose_camera_dialog, ImageAttendanceWorker


class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIAS — AI Attendance System")
        import os
        from PyQt5.QtGui import QIcon
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aias_icon.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        screen = QApplication.primaryScreen().geometry()
        self.resize(960, 640)
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        # Splash state
        self._showing_splash = True
        self._sp_opacity     = 0.0
        self._sp_scale       = 0.7
        self._sp_progress    = 0.0
        self._sp_scan_y      = 0.25
        self._sp_scan_dir    = 1
        self._sp_corner_op   = 1.0
        self._sp_corner_dir  = -1
        self._sp_ring_scale  = [1.0, 1.0, 1.0]
        self._sp_ring_dir    = [1, 1, 1]
        self._sp_dot_scale   = [1.0, 1.0]
        self._sp_dot_dir     = [1, -1]
        self._sp_done        = False
        self._sp_fade_out    = 1.0

        # Particle system
        import random, math
        self._particles = []
        for _ in range(100):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(0.0003, 0.0012)
            self._particles.append({
                "px":          random.uniform(0.0, 1.0),
                "py":          random.uniform(0.0, 1.0),
                "vx":          math.cos(angle) * speed,
                "vy":          math.sin(angle) * speed,
                "size":        random.uniform(1.5, 3.5),
                "opacity":     random.uniform(0.2, 0.75),
                "pulse":       random.uniform(0, 6.28),
                "pulse_speed": random.uniform(0.015, 0.045),
            })

        # Login page
        self._login_page = LoginWindow(embedded=True)
        self._login_page.login_success.connect(self._on_login_success)

        # Single-window navigation stack
        self._stack = QStackedWidget(self)
        self._stack.setStyleSheet("background:transparent;")
        self._stack.addWidget(self._login_page)
        self._stack.hide()

        self._overlay_opacity = 1.0
        self._overlay_active  = False

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

        self.setStyleSheet("background:#0a0f0a;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def resizeEvent(self, event):
        self._stack.setGeometry(0, 0, self.width(), self.height())

    def _tick(self):
        if not self._sp_done:
            # Animate splash
            if self._sp_opacity < 1.0:
                self._sp_opacity = min(1.0, self._sp_opacity + 0.04)
            if self._sp_scale < 1.0:
                self._sp_scale = min(1.0, self._sp_scale + 0.02)
            self._sp_progress = min(100.0, self._sp_progress + 0.9)

            self._sp_scan_y += 0.01 * self._sp_scan_dir
            if self._sp_scan_y >= 0.72: self._sp_scan_dir = -1
            elif self._sp_scan_y <= 0.22: self._sp_scan_dir = 1

            self._sp_corner_op += 0.05 * self._sp_corner_dir
            if self._sp_corner_op <= 0.2: self._sp_corner_dir = 1
            elif self._sp_corner_op >= 1.0: self._sp_corner_dir = -1

            speeds = [0.008, 0.006, 0.004]
            for i in range(3):
                self._sp_ring_scale[i] += speeds[i] * self._sp_ring_dir[i]
                if self._sp_ring_scale[i] >= 1.06: self._sp_ring_dir[i] = -1
                elif self._sp_ring_scale[i] <= 0.96: self._sp_ring_dir[i] = 1

            for i in range(2):
                self._sp_dot_scale[i] += 0.06 * self._sp_dot_dir[i]
                if self._sp_dot_scale[i] >= 1.5: self._sp_dot_dir[i] = -1
                elif self._sp_dot_scale[i] <= 0.6: self._sp_dot_dir[i] = 1

            if self._sp_progress >= 100.0:
                QTimer.singleShot(2500, self._start_fade_out)
                self._sp_done = True

            # Update particles
            import math
            for pt in self._particles:
                pt["px"]    += pt["vx"]
                pt["py"]    += pt["vy"]
                pt["pulse"] += pt["pulse_speed"]
                pt["opacity"] = 0.3 + 0.5 * abs(math.sin(pt["pulse"]))
                # Wrap around using percentages
                if pt["px"] < 0.0: pt["px"] += 1.0
                if pt["px"] > 1.0: pt["px"] -= 1.0
                if pt["py"] < 0.0: pt["py"] += 1.0
                if pt["py"] > 1.0: pt["py"] -= 1.0

            self.update()
        else:
            # Fading out
            if self._sp_fade_out > 0.0:
                import math
                self._sp_fade_out = max(0.0, self._sp_fade_out - 0.007)
                self.update()
            else:
                if not hasattr(self, '_splash_finished'):
                    self._splash_finished = True
                    self._finish_splash()
                # Keep particles running in background
                import math
                for pt in self._particles:
                    pt["px"]    += pt["vx"]
                    pt["py"]    += pt["vy"]
                    pt["pulse"] += pt["pulse_speed"]
                    pt["opacity"] = 0.3 + 0.5 * abs(math.sin(pt["pulse"]))
                    if pt["px"] < 0.0: pt["px"] += 1.0
                    if pt["px"] > 1.0: pt["px"] -= 1.0
                    if pt["py"] < 0.0: pt["py"] += 1.0
                    if pt["py"] > 1.0: pt["py"] -= 1.0
                self.update()

    def _start_fade_out(self):
        self._showing_splash = False

    def _finish_splash(self):
        self.setStyleSheet("background:#0a0f0a;")
        self._stack.setGeometry(0, 0, self.width(), self.height())
        self._stack.setCurrentWidget(self._login_page)
        self._stack.show()
        self._stack.raise_()
        # Start overlay fade
        self._overlay_opacity = 1.0
        self._overlay_active  = True
        self._fade_in_step    = 0.0
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._fade_in_login)
        self._fade_timer.start(16)

    def _on_login_success(self, role, username, fullname):
        from ui_admin import AdminPanelWindow
        try:
            if role == "instructor":
                import re as _re2
                instr = db["Instructors"].find_one({"Username": {"$regex": f"^{_re2.escape(username)}$", "$options": "i"}})
                assigned = instr.get("AssignedSections", []) if instr else []
                new_page = MainWindow(username, fullname, assigned, embedded=True)
            else:
                new_page = AdminPanelWindow(embedded=True)
        except Exception as e:
            print(f"[AIAS] login error: {e}")
            return

        # Remove old pages except login
        while self._stack.count() > 1:
            widget = self._stack.widget(1)
            self._stack.removeWidget(widget)
            widget.deleteLater()

        self._stack.addWidget(new_page)
        self._stack.setCurrentWidget(new_page)
        self.setStyleSheet(f"background:{BG};")
        new_page.update()

    def _fade_in_login(self):
        self._fade_in_step = min(1.0, self._fade_in_step + 0.008)
        t = self._fade_in_step
        # Cubic ease-in-out
        if t < 0.5:
            eased = 4 * t * t * t
        else:
            eased = 1 - ((-2 * t + 2) ** 3) / 2
        # Overlay goes from 1.0 (black) to 0.0 (transparent)
        self._overlay_opacity = 1.0 - eased
        self.update()
        if self._fade_in_step >= 1.0:
            self._fade_timer.stop()
            self._overlay_active  = False
            self._overlay_opacity = 0.0
            self.update()

    def paintEvent(self, event):
        # Draw black overlay during login fade-in
        if self._overlay_active:
            from PyQt5.QtGui import QPainter, QColor
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setOpacity(self._overlay_opacity)
            p.setBrush(QColor("#0a0f0a"))
            p.setPen(Qt.NoPen)
            p.drawRect(0, 0, self.width(), self.height())
            p.end()
            return

        if not self._showing_splash and self._sp_fade_out <= 0.0:
            # Still draw particles behind login
            if self._overlay_active:
                return
            from PyQt5.QtGui import QPainter, QColor
            import math
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QColor("#0a0f0a"))
            p.setPen(Qt.NoPen)
            p.drawRect(0, 0, self.width(), self.height())
            W, H = self.width(), self.height()
            for pt in self._particles:
                px = pt["px"] * W
                py = pt["py"] * H
                size = pt["size"]
                p.setOpacity(pt["opacity"] * 0.4)
                p.setBrush(QColor("#1B5E35"))
                p.setPen(Qt.NoPen)
                p.drawEllipse(int(px-size*2), int(py-size*2), int(size*4), int(size*4))
                p.setOpacity(pt["opacity"] * 0.6)
                p.setBrush(QColor("#4ecf8e"))
                p.drawEllipse(int(px-size*0.5), int(py-size*0.5), int(size), int(size))
            p.end()
            return

        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QRadialGradient
        from PyQt5.QtCore import QRectF, QPointF

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        fade = self._sp_fade_out if not self._showing_splash else 1.0
        p.setOpacity(self._sp_opacity * fade)

        W, H = self.width(), self.height()
        cx, cy = W // 2, H // 2 - 30

        # Background
        p.setBrush(QColor("#0a0f0a"))
        p.setPen(Qt.NoPen)
        p.drawRect(0, 0, W, H)

        # Draw particles
        import math
        for pt in self._particles:
            pt_opacity = pt["opacity"] * self._sp_opacity * fade
            if pt_opacity <= 0:
                continue
            # Convert percentage to actual pixels
            px = pt["px"] * W
            py = pt["py"] * H
            size = pt["size"]
            # Outer glow
            p.setOpacity(pt_opacity * 0.3)
            p.setBrush(QColor("#1B5E35"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(
                int(px - size * 2), int(py - size * 2),
                int(size * 4), int(size * 4)
            )
            # Inner bright dot
            p.setOpacity(pt_opacity)
            p.setBrush(QColor("#4ecf8e"))
            p.drawEllipse(
                int(px - size * 0.5), int(py - size * 0.5),
                int(size), int(size)
            )
        p.setOpacity(self._sp_opacity * fade)

        # Rings
        ring_sizes = [340, 440, 540]
        ring_ops   = [0.18, 0.10, 0.06]
        for i, (sz, op) in enumerate(zip(ring_sizes, ring_ops)):
            scaled = sz * self._sp_ring_scale[i]
            pen = QPen(QColor("#1B5E35"))
            pen.setWidthF(1.0)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.setOpacity(self._sp_opacity * fade * op)
            from PyQt5.QtCore import QRectF
            p.drawEllipse(QRectF(cx - scaled/2, cy - scaled/2, scaled, scaled))

        p.setOpacity(self._sp_opacity * fade)

        # Logo circle
        zoom = 1.0 + (1.0 - self._sp_fade_out) * 0.30
        logo_r = 100 * self._sp_scale * zoom
        grad = QRadialGradient(cx, cy - 20, logo_r)
        grad.setColorAt(0, QColor("#1B5E35"))
        grad.setColorAt(1, QColor("#0d3520"))
        p.setBrush(QBrush(grad))
        pen2 = QPen(QColor("#2d8a57"))
        pen2.setWidthF(2.0)
        p.setPen(pen2)
        from PyQt5.QtCore import QRectF
        p.drawEllipse(QRectF(cx - logo_r, cy - logo_r, logo_r*2, logo_r*2))

        # Dashed ring
        pen3 = QPen(QColor(180, 220, 200, 60))
        pen3.setWidthF(1.2)
        pen3.setStyle(Qt.DashLine)
        p.setPen(pen3)
        p.setBrush(Qt.NoBrush)
        outer_r = logo_r + 8
        p.drawEllipse(QRectF(cx - outer_r, cy - outer_r, outer_r*2, outer_r*2))

        # User head
        head_r  = 28 * self._sp_scale
        head_cx = cx
        head_cy = cy - 22 * self._sp_scale
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(180, 220, 200, 195))
        p.drawEllipse(QRectF(head_cx - head_r, head_cy - head_r, head_r*2, head_r*2))

        # Scan line
        scan_y = head_cy - head_r + (head_r*2 * self._sp_scan_y)
        p.setClipRect(QRectF(head_cx - head_r, head_cy - head_r, head_r*2, head_r*2).toRect())
        sp = QPen(QColor(255, 255, 255, 210))
        sp.setWidthF(1.5)
        p.setPen(sp)
        from PyQt5.QtCore import QPointF
        p.drawLine(QPointF(head_cx - head_r + 2, scan_y), QPointF(head_cx + head_r - 2, scan_y))
        p.setClipping(False)

        # Dots
        for i, (dx, dy) in enumerate([(head_cx - head_r*0.3, head_cy + head_r*0.05),
                                       (head_cx + head_r*0.3, head_cy + head_r*0.05)]):
            ds = 3.5 * self._sp_dot_scale[i] * self._sp_scale
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(13, 53, 32, 220))
            p.drawEllipse(QRectF(dx-ds, dy-ds, ds*2, ds*2))

        # Body
        body_w = 74 * self._sp_scale
        body_h = 30 * self._sp_scale
        body_y = head_cy + head_r + 3 * self._sp_scale
        p.setBrush(QColor(180, 220, 200, 165))
        p.setPen(Qt.NoPen)
        p.drawChord(QRectF(cx - body_w/2, body_y, body_w, body_h*2), 0, 180*16)

        # Corners
        cc = QColor(255, 255, 255, int(255 * self._sp_corner_op))
        cp = QPen(cc)
        cp.setWidthF(2.5)
        cp.setCapStyle(Qt.RoundCap)
        p.setPen(cp)
        p.setBrush(Qt.NoBrush)
        cw  = 16 * self._sp_scale
        fx1 = head_cx - head_r - 10 * self._sp_scale
        fx2 = head_cx + head_r + 10 * self._sp_scale
        fy1 = head_cy - head_r - 10 * self._sp_scale
        fy2 = body_y + body_h + 4 * self._sp_scale
        p.drawLine(QPointF(fx1, fy1+cw), QPointF(fx1, fy1)); p.drawLine(QPointF(fx1, fy1), QPointF(fx1+cw, fy1))
        p.drawLine(QPointF(fx2-cw, fy1), QPointF(fx2, fy1)); p.drawLine(QPointF(fx2, fy1), QPointF(fx2, fy1+cw))
        p.drawLine(QPointF(fx1, fy2-cw), QPointF(fx1, fy2)); p.drawLine(QPointF(fx1, fy2), QPointF(fx1+cw, fy2))
        p.drawLine(QPointF(fx2-cw, fy2), QPointF(fx2, fy2)); p.drawLine(QPointF(fx2, fy2), QPointF(fx2, fy2-cw))

        # AIAS text
        text_y = body_y + body_h + 18 * self._sp_scale
        f1 = QFont("Segoe UI", int(18 * self._sp_scale), QFont.Bold)
        p.setFont(f1)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(cx-80, text_y, 160, 26), Qt.AlignCenter, "AIAS")
        f2 = QFont("Segoe UI", int(6 * self._sp_scale))
        f2.setLetterSpacing(QFont.AbsoluteSpacing, 2.5)
        p.setFont(f2)
        p.setPen(QColor(200, 230, 210, 180))
        p.drawText(QRectF(cx-90, text_y+24, 180, 18), Qt.AlignCenter, "AI ATTENDANCE SYSTEM")

        # Title
        title_y = cy + logo_r + 28
        p.setOpacity(self._sp_opacity * (fade ** 3) * min(1.0, self._sp_progress / 30))
        f3 = QFont("Segoe UI", 22, QFont.Bold)
        p.setFont(f3)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(0, title_y, W, 36), Qt.AlignCenter, "AI Attendance System")
        f4 = QFont("Segoe UI", 9)
        f4.setLetterSpacing(QFont.AbsoluteSpacing, 3.5)
        p.setFont(f4)
        p.setPen(QColor(180, 220, 200, 140))
        p.drawText(QRectF(0, title_y+40, W, 20), Qt.AlignCenter, "POWERED BY FACE RECOGNITION")

        # Loading bar
        p.setOpacity(self._sp_opacity * (fade ** 3) * min(1.0, self._sp_progress / 20))
        bar_w = 240; bar_h = 3
        bar_x = (W - bar_w) / 2
        bar_y = title_y + 76
        p.setBrush(QColor(255, 255, 255, 18))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)
        fill_w = bar_w * (self._sp_progress / 100)
        lg = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        lg.setColorAt(0, QColor("#1B5E35"))
        lg.setColorAt(1, QColor("#4ecf8e"))
        p.setBrush(QBrush(lg))
        p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2, 2)
        f5 = QFont("Segoe UI", 8)
        f5.setLetterSpacing(QFont.AbsoluteSpacing, 2)
        p.setFont(f5)
        p.setPen(QColor(180, 220, 200, 90))
        p.drawText(QRectF(0, bar_y+10, W, 16), Qt.AlignCenter, "INITIALIZING...")

        p.end()


class LoginWindow(QWidget):
    login_success = pyqtSignal(str, str, str)  # role, username, fullname

    def __init__(self, embedded=False):
        super().__init__()
        self._embedded = embedded
        if not embedded:
            self.setWindowTitle("AIAS — Login")
            self.setMinimumSize(600, 680)
            self.setStyleSheet("background:transparent;")
        self._role = "instructor"
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setContentsMargins(40, 40, 40, 40)

        # Card
        card = QFrame()
        card.setFixedWidth(400)
        card.setMaximumHeight(560)
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        card.setStyleSheet(
            f"QFrame {{ background:{config.WHITE}; border-radius:16px; }}"
        )
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 40))
        card.setGraphicsEffect(shadow)

        lay = QVBoxLayout(card)
        lay.setContentsMargins(32, 32, 32, 32)
        lay.setSpacing(0)

        # 1. Icon + Name row
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        icon_lbl = QLabel("AI")
        icon_lbl.setFixedSize(48, 48)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background:{config.PRIMARY}; color:{config.WHITE}; border-radius:10px;"
            "font-size:14px; font-weight:700; border:none;"
        )

        name_col = QVBoxLayout()
        name_col.setSpacing(2)
        app_name = QLabel("AIAS")
        app_name.setStyleSheet(
            f"font-size:22px; font-weight:700; color:{config.TEXT_DARK};"
            "background:transparent; border:none;"
        )
        app_sub = QLabel("AI Attendance System")
        app_sub.setStyleSheet(
            f"font-size:12px; color:{config.TEXT_GRAY}; background:transparent; border:none;"
        )
        name_col.addWidget(app_name)
        name_col.addWidget(app_sub)

        header_row.addWidget(icon_lbl)
        header_row.addLayout(name_col)
        header_row.addStretch()
        lay.addLayout(header_row)
        lay.addSpacing(24)

        # 2. Badge
        badge_lbl = QLabel("Qassim University")
        badge_lbl.setFixedHeight(26)
        badge_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        badge_lbl.setStyleSheet(
            f"background:{config.PRIMARY_LIGHT}; color:{config.PRIMARY}; border-radius:12px;"
            "padding:4px 12px; font-size:11px; font-weight:600; border:none;"
        )
        lay.addWidget(badge_lbl)
        lay.addSpacing(20)

        # 3. Welcome text
        welcome_lbl = QLabel("Welcome back")
        welcome_lbl.setStyleSheet(
            f"font-size:26px; font-weight:700; color:{config.TEXT_DARK};"
            "background:transparent; border:none;"
        )
        signin_sub = QLabel("Sign in to access the dashboard")
        signin_sub.setStyleSheet(
            f"font-size:13px; color:{config.TEXT_GRAY}; background:transparent; border:none;"
        )
        lay.addWidget(welcome_lbl)
        lay.addWidget(signin_sub)
        lay.addSpacing(24)

        # 4. Role toggle
        toggle_frame = QFrame()
        toggle_frame.setFixedHeight(44)
        toggle_frame.setStyleSheet(
            f"QFrame {{ background:{config.BG}; border-radius:10px; border:none; }}"
        )
        toggle_lay = QHBoxLayout(toggle_frame)
        toggle_lay.setContentsMargins(4, 4, 4, 4)
        toggle_lay.setSpacing(4)
        self._btn_instructor = QPushButton("Instructor")
        self._btn_admin = QPushButton("Admin")
        for btn in (self._btn_instructor, self._btn_admin):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._btn_instructor.clicked.connect(lambda: self._set_role("instructor"))
        self._btn_admin.clicked.connect(lambda: self._set_role("admin"))
        toggle_lay.addWidget(self._btn_instructor)
        toggle_lay.addWidget(self._btn_admin)
        lay.addWidget(toggle_frame)
        lay.addSpacing(20)
        self._apply_toggle_style()

        # 5. Username
        u_lbl = QLabel("Username")
        u_lbl.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{config.TEXT_DARK};"
            "background:transparent; border:none;"
        )
        self.username_field = QLineEdit()
        self.username_field.setPlaceholderText("Enter your username")
        self.username_field.setFixedHeight(48)
        self.username_field.setStyleSheet(
            f"QLineEdit {{ background:{config.WHITE}; color:{config.TEXT_DARK}; border:1.5px solid {config.BORDER}; "
            "border-radius:10px; padding:12px 16px; font-size:14px; }"
            f"QLineEdit:focus {{ border:1.5px solid {config.PRIMARY}; }}"
        )
        lay.addWidget(u_lbl)
        lay.addSpacing(6)
        self.username_field.returnPressed.connect(self._login)
        lay.addWidget(self.username_field)
        lay.addSpacing(2)
        self.err_username = QLabel("")
        self.err_username.setStyleSheet(
            "color:#DC2626; font-size:11px; background:transparent; border:none;"
        )
        self.err_username.setFixedHeight(14)
        lay.addWidget(self.err_username)

        # 6. Password
        p_lbl = QLabel("Password")
        p_lbl.setStyleSheet(
            f"font-size:13px; font-weight:600; color:{config.TEXT_DARK};"
            "background:transparent; border:none;"
        )
        self.password_field = QLineEdit()
        self.password_field.setPlaceholderText("Enter your password")
        self.password_field.setEchoMode(QLineEdit.Password)
        self.password_field.setStyleSheet(
            f"QLineEdit {{ background:transparent; color:{config.TEXT_DARK}; border:none; "
            "padding:0; font-size:14px; }"
        )
        self.password_field.returnPressed.connect(self._login)

        self._pw_frame = QFrame()
        self._pw_frame.setFixedHeight(48)
        self._pw_frame.setStyleSheet(
            f"QFrame {{ background:{config.WHITE}; border:1.5px solid {config.BORDER}; border-radius:10px; }}"
        )

        pw_inner = QHBoxLayout(self._pw_frame)
        pw_inner.setContentsMargins(16, 0, 4, 0)
        pw_inner.setSpacing(0)
        pw_inner.addWidget(self.password_field, stretch=1)

        toggle_btn = QPushButton("SHOW")
        toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {config.PRIMARY_LIGHT};
                color: {config.PRIMARY};
                border: none;
                border-left: 1px solid {config.BORDER};
                border-radius: 0px;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                font-size: 12px;
                font-weight: 700;
                padding: 0px 10px;
                min-width: 52px;
            }}
            QPushButton:hover {{
                background: {config.PRIMARY};
                color: {config.WHITE};
            }}
        """)
        toggle_btn.setCursor(Qt.PointingHandCursor)

        def _toggle_pw():
            if self.password_field.echoMode() == QLineEdit.Password:
                self.password_field.setEchoMode(QLineEdit.Normal)
                toggle_btn.setText("HIDE")
            else:
                self.password_field.setEchoMode(QLineEdit.Password)
                toggle_btn.setText("SHOW")

        toggle_btn.clicked.connect(_toggle_pw)
        pw_inner.addWidget(toggle_btn)

        self.password_field.installEventFilter(self)

        lay.addWidget(p_lbl)
        lay.addSpacing(6)
        lay.addWidget(self._pw_frame)
        lay.addSpacing(2)
        self.err_password = QLabel("")
        self.err_password.setStyleSheet(
            "color:#DC2626; font-size:11px; background:transparent; border:none;"
        )
        self.err_password.setFixedHeight(14)
        lay.addWidget(self.err_password)
        lay.addSpacing(8)

        # 7. Sign In button
        self._sign_btn = QPushButton("Sign In")
        self._sign_btn.setFixedHeight(48)
        self._sign_btn.setStyleSheet(
            f"QPushButton {{ background:{config.PRIMARY}; color:{config.WHITE}; border-radius:10px; "
            f"height:48px; font-size:15px; font-weight:700; }}"
            f"QPushButton:hover {{ background:{config.PRIMARY}; color:{config.WHITE}; }}"
            f"QPushButton:pressed {{ background:{config.PRIMARY}; color:{config.WHITE}; }}"
        )
        self._sign_btn.setCursor(Qt.PointingHandCursor)
        self._sign_btn.clicked.connect(self._login)
        lay.addWidget(self._sign_btn)
        lay.addSpacing(12)

        # 8. Error label
        self.err_label = QLabel("")
        self.err_label.setAlignment(Qt.AlignCenter)
        self.err_label.setStyleSheet(
            "color:#DC2626; font-size:12px; background:transparent; border:none;"
        )
        self.err_label.setFixedHeight(18)
        lay.addWidget(self.err_label)

        if not config.DB_CONNECTED:
            db_warn = QFrame()
            db_warn.setStyleSheet(
                "QFrame { background:#FEF3C7; border:1.5px solid #F59E0B; border-radius:8px; }"
            )
            db_warn_lay = QHBoxLayout(db_warn)
            db_warn_lay.setContentsMargins(12, 10, 12, 10)
            db_warn_lay.setSpacing(8)
            warn_icon = QLabel("⚠️")
            warn_icon.setStyleSheet("font-size:16px; background:transparent; border:none;")
            warn_text = QLabel("Database offline — please start MongoDB and restart the app.")
            warn_text.setStyleSheet(
                "font-size:12px; font-weight:600; color:#92400E; background:transparent; border:none;"
            )
            warn_text.setWordWrap(True)
            db_warn_lay.addWidget(warn_icon)
            db_warn_lay.addWidget(warn_text, stretch=1)
            db_warn.setFixedWidth(400)
            outer.addWidget(db_warn, alignment=Qt.AlignCenter)
            outer.addSpacing(8)

        outer.addWidget(card, alignment=Qt.AlignCenter)

    def _set_role(self, role):
        self._role = role
        self._apply_toggle_style()

    def _apply_toggle_style(self):
        active_ss = (
            f"background:{config.WHITE}; color:{config.PRIMARY}; border:1.5px solid {config.PRIMARY};"
            "border-radius:8px; font-size:13px; font-weight:600;"
        )
        inactive_ss = (
            f"background:{config.BG}; color:{config.TEXT_GRAY}; border:none; border-radius:8px;"
            "font-size:13px; font-weight:500;"
        )
        if self._role == "instructor":
            self._btn_instructor.setStyleSheet(active_ss)
            self._btn_admin.setStyleSheet(inactive_ss)
        else:
            self._btn_instructor.setStyleSheet(inactive_ss)
            self._btn_admin.setStyleSheet(active_ss)

    def _login(self):
        self._sign_btn.setEnabled(False)
        self._sign_btn.setText("Signing in...")
        self.err_label.setText("")
        self.err_username.setText("")
        self.err_password.setText("")
        QApplication.processEvents()
        try:
            u = self.username_field.text().strip()
            p = self.password_field.text()
            if not DB_CONNECTED:
                self.err_label.setText("Database not connected.")
                return
            try:
                doc = db["Instructors"].find_one({"Username": {"$regex": f"^{u}$", "$options": "i"}})
                if not doc:
                    self.err_password.setText("Invalid username or password.")
                    return

                stored = doc.get("Password", "")
                if isinstance(stored, str):
                    # Migrate plain-text password to bcrypt on first login (Bug 2)
                    if stored == p:
                        hashed = bcrypt.hashpw(p.encode("utf-8"), bcrypt.gensalt())
                        db["Instructors"].update_one(
                            {"Username": u}, {"$set": {"Password": hashed}}
                        )
                        pw_ok = True
                    else:
                        pw_ok = False
                else:
                    pw_ok = bcrypt.checkpw(p.encode("utf-8"), stored)

                if not pw_ok:
                    self.err_password.setText("Invalid username or password.")
                    return

                role = doc.get("Role", "instructor")

                # Enforce toggle selection matches DB role
                selected = self._role  # "admin" or "instructor"
                if selected == "admin" and role != "admin":
                    self.err_label.setText("This account does not have Admin privileges.")
                    return
                if selected == "instructor" and role == "admin":
                    self.err_label.setText("Please select 'Admin' to log in as Administrator.")
                    return

                if role == "admin":
                    self.login_success.emit("admin", "", "")
                else:
                    fullname = doc.get("FullName", u)
                    self.login_success.emit("instructor", u, fullname)
            except Exception as e:
                print(f"Exception: {e}")
                self.err_label.setText("Database error. Please try again.")
        finally:
            self._sign_btn.setEnabled(True)
            self._sign_btn.setText("Sign In")

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if obj is self.password_field:
            if event.type() == QEvent.FocusIn:
                self._pw_frame.setStyleSheet(
                    f"QFrame {{ background:{config.WHITE}; border:1.5px solid {config.PRIMARY}; border-radius:10px; }}"
                )
            elif event.type() == QEvent.FocusOut:
                self._pw_frame.setStyleSheet(
                    f"QFrame {{ background:{config.WHITE}; border:1.5px solid {config.BORDER}; border-radius:10px; }}"
                )
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        from PyQt5.QtCore import Qt
        if event.key() == Qt.Key_Escape:
            pass
        else:
            super().keyPressEvent(event)


class CourseCard(QFrame):
    def __init__(self, code, name, schedule, active=False, n_students=0):
        super().__init__()
        self.setFixedHeight(86)
        self.setCursor(Qt.PointingHandCursor)
        self._active = active
        self._apply_style()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        top = QLabel(f"<b>{code}</b> — {name}")
        top.setStyleSheet(
            f"font-size:12px; color:{TEXT_DARK}; background:transparent; border:none;"
        )
        bot = QLabel(schedule)
        bot.setStyleSheet(
            f"font-size:11px; color:{TEXT_GRAY}; background:transparent; border:none;"
        )
        count_lbl = QLabel(f"\U0001f465 {n_students} students")
        count_lbl.setStyleSheet(
            f"font-size:11px; color:{TEXT_GRAY}; background:transparent; border:none;"
        )
        lay.addWidget(top)
        lay.addWidget(bot)
        lay.addWidget(count_lbl)

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                f"QFrame {{ background:{PRIMARY_LIGHT}; border:1.5px solid {PRIMARY};"
                f"border-radius:8px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame {{ background:{WHITE}; border:1px solid {BORDER};"
                f"border-radius:8px; }}"
            )

    def set_active(self, val):
        self._active = val
        self._apply_style()


class MainWindow(QWidget):
    def __init__(self, username, fullname, sections, embedded=False):
        super().__init__()
        self._embedded = embedded
        self.instructor_data = {
            "username": username,
            "fullname": fullname,
            "assigned_sections": sections,
        }
        self._courses_data = []
        if not embedded:
            self.setWindowTitle("AIAS — Course Selection")
            import os
            from PyQt5.QtGui import QIcon
            _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aias_icon.ico")
            if os.path.exists(_icon_path):
                self.setWindowIcon(QIcon(_icon_path))
            self.setMinimumSize(1100, 700)
            self.resize(1100, 700)
        self.setStyleSheet(f"background:{BG};")
        self._active_idx = 0
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        assigned = self.instructor_data.get("assigned_sections", [])
        fullname = self.instructor_data.get("fullname", "Instructor")

        import ast
        if isinstance(assigned, str):
            try:
                assigned = ast.literal_eval(assigned)
            except Exception:
                assigned = [assigned]

        if DB_CONNECTED:
            try:
                import re as _re
                _patterns = [_re.compile(f"^{_re.escape(s)}$", _re.IGNORECASE) for s in assigned]
                self._courses_data = list(
                    db["Courses"].find({"SectionID": {"$in": _patterns}})
                ) if assigned else []
            except Exception:
                self._courses_data = []

        parts = fullname.split()
        initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else fullname[:2].upper()
        sidebar, s_lay = make_sidebar_base(initials, fullname, "Instructor")

        dash_btn = QPushButton("📊  Dashboard")
        dash_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY}; color:#ffffff; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:#145a30; color:#ffffff; }}"
        )
        dash_btn.setCursor(Qt.PointingHandCursor)
        dash_btn.clicked.connect(self._show_dashboard)
        s_lay.addWidget(dash_btn)
        s_lay.addSpacing(8)

        sec_lbl = QLabel("MY COURSES")
        sec_lbl.setStyleSheet(
            f"font-size:10px; font-weight:700; color:{TEXT_GRAY}; letter-spacing:1px;"
            f"background:transparent; border:none;"
        )
        s_lay.addWidget(sec_lbl)
        s_lay.addSpacing(4)

        self._course_cards = []
        for i, course in enumerate(self._courses_data):
            code = str(course.get("SectionID", ""))
            name = course.get("CourseTitle", "")
            sched_raw = course.get("Schedule", "")
            if isinstance(sched_raw, dict):
                sched = f"{sched_raw.get(chr(68) + 'ay', '')} {sched_raw.get('StartTime', '')}–{sched_raw.get('EndTime', '')}"
            else:
                sched = str(sched_raw)
            section_id = course.get("SectionID", "")
            enrolled_ids = course.get("EnrolledStudents", [])
            try:
                n_students = db["Students"].count_documents({"StudentID": {"$in": enrolled_ids}}) if enrolled_ids and DB_CONNECTED else 0
            except Exception:
                n_students = 0
            card = CourseCard(code, name, sched, active=(i == 0), n_students=n_students)
            card.mousePressEvent = lambda e, idx=i: self._select_course(idx)
            self._course_cards.append(card)
            s_lay.addWidget(card)

        s_lay.addStretch()

        history_btn = QPushButton("Session History")
        history_btn.setToolTip("View all past attendance sessions")
        history_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{PRIMARY}; color:#ffffff; }}"
            f"QPushButton:pressed {{ background:{PRIMARY}; color:#ffffff; padding:9px 11px 7px 13px; }}"
        )
        history_btn.setCursor(Qt.PointingHandCursor)
        history_btn.clicked.connect(self._open_history)
        s_lay.addWidget(history_btn)
        s_lay.addSpacing(4)

        analytics_btn = QPushButton("Analytics")
        analytics_btn.setToolTip("View attendance statistics and at-risk students")
        analytics_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{PRIMARY}; color:#ffffff; }}"
            f"QPushButton:pressed {{ background:{PRIMARY}; color:#ffffff; padding:9px 11px 7px 13px; }}"
        )
        analytics_btn.setCursor(Qt.PointingHandCursor)
        analytics_btn.clicked.connect(self._open_analytics)
        s_lay.addWidget(analytics_btn)
        s_lay.addSpacing(4)

        email_btn = QPushButton("Email Settings")
        email_btn.setToolTip("Configure email settings for sending reports")
        email_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{PRIMARY}; color:#ffffff; }}"
            f"QPushButton:pressed {{ background:{PRIMARY}; color:#ffffff; padding:9px 11px 7px 13px; }}"
        )
        email_btn.setCursor(Qt.PointingHandCursor)
        email_btn.clicked.connect(self._open_email_settings)
        s_lay.addWidget(email_btn)
        s_lay.addSpacing(6)

        change_pass_btn = QPushButton("🔒  Change Password")
        change_pass_btn.setToolTip("Change your login password")
        change_pass_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{PRIMARY}; color:#ffffff; }}"
            f"QPushButton:pressed {{ background:{PRIMARY}; color:#ffffff; padding:9px 11px 7px 13px; }}"
        )
        change_pass_btn.setCursor(Qt.PointingHandCursor)
        change_pass_btn.clicked.connect(self._change_password)
        s_lay.addWidget(change_pass_btn)
        s_lay.addSpacing(4)

        dark_btn = QPushButton("🌙  Dark Mode" if not IS_DARK_MODE else "☀  Light Mode")
        dark_btn.setStyleSheet(
            f"background:{'#2A3D35' if IS_DARK_MODE else PRIMARY_LIGHT}; "
            f"color:{'#4ecf8e' if IS_DARK_MODE else PRIMARY}; "
            "border-radius:6px; padding:8px 12px; font-size:12px; font-weight:600;"
        )
        dark_btn.setCursor(Qt.PointingHandCursor)
        dark_btn.clicked.connect(lambda: self._toggle_dark_mode(dark_btn))
        s_lay.addWidget(dark_btn)
        s_lay.addSpacing(4)

        logout_btn = QPushButton("Logout")
        logout_btn.setToolTip("Sign out and return to login screen")
        logout_btn.setStyleSheet(
            "QPushButton { background:#DC2626; color:white; border-radius:6px; "
            "padding:8px 12px; font-size:12px; font-weight:700; }"
            "QPushButton:hover { background:#B91C1C; color:white; }"
            "QPushButton:pressed { background:#991B1B; color:white; }"
        )
        logout_btn.setCursor(Qt.PointingHandCursor)
        logout_btn.clicked.connect(self._logout)
        s_lay.addWidget(logout_btn)

        main_w = QWidget()
        main_w.setStyleSheet(f"background:{BG};")
        m_lay = QVBoxLayout(main_w)
        m_lay.setContentsMargins(32, 28, 32, 24)
        m_lay.setSpacing(18)

        self._title_lbl = QLabel("")
        self._title_lbl.setStyleSheet(
            f"font-size:20px; font-weight:800; color:{TEXT_DARK};"
        )
        self._sub_lbl = QLabel("")
        self._sub_lbl.setStyleSheet(f"font-size:13px; color:{TEXT_GRAY};")
        m_lay.addWidget(self._title_lbl)
        m_lay.addWidget(self._sub_lbl)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        enrolled_card = QFrame()
        enrolled_card.setStyleSheet(
            f"QFrame {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:8px; }}"
        )
        enrolled_card.setFixedHeight(82)
        enrolled_card.setMinimumWidth(130)
        ec_lay = QVBoxLayout(enrolled_card)
        ec_lay.setContentsMargins(16, 12, 16, 12)
        ec_lay.setSpacing(2)
        self._enrolled_val = QLabel("0")
        self._enrolled_val.setStyleSheet(
            f"font-size:26px; font-weight:800; color:{PRIMARY}; border:none; background:transparent;"
        )
        ec_lbl = QLabel("Enrolled")
        ec_lbl.setStyleSheet(
            f"font-size:11px; color:{TEXT_GRAY}; border:none; background:transparent;"
        )
        ec_lay.addWidget(self._enrolled_val)
        ec_lay.addWidget(ec_lbl)
        stats_row.addWidget(enrolled_card)

        def _make_updatable_card(title, initial):
            f = QFrame()
            f.setStyleSheet(f"QFrame {{ background:{WHITE}; border:1px solid {BORDER}; border-radius:8px; }}")
            f.setFixedHeight(82)
            f.setMinimumWidth(130)
            fl = QVBoxLayout(f)
            fl.setContentsMargins(16, 12, 16, 12)
            fl.setSpacing(2)
            vl = QLabel(str(initial))
            vl.setStyleSheet(f"font-size:26px; font-weight:800; color:{TEXT_DARK}; border:none; background:transparent;")
            tl = QLabel(title)
            tl.setStyleSheet(f"font-size:11px; color:{TEXT_GRAY}; border:none; background:transparent;")
            fl.addWidget(vl)
            fl.addWidget(tl)
            return f, vl

        sessions_card, self._sessions_held_lbl = _make_updatable_card("Sessions held", 0)
        last_card,     self._last_session_lbl  = _make_updatable_card("Last session",  "—")
        stats_row.addWidget(sessions_card)
        stats_row.addWidget(last_card)
        stats_row.addStretch()
        m_lay.addLayout(stats_row)

        self._tbl = make_table(
            ["Student ID", "Full Name", "Status"],
            [],
            col_widths={0: 140, 2: 140},
            stretch_col=1,
        )
        m_lay.addWidget(self._tbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        img_att_btn = QPushButton("📷  Attendance from Image")
        img_att_btn.setToolTip("Upload an image and recognize faces to mark attendance")
        img_att_btn.setFixedHeight(40)
        img_att_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY_LIGHT}; color:{PRIMARY}; border-radius:8px; "
            f"padding:12px 24px; font-size:14px; font-weight:700; }}"
            f"QPushButton:hover {{ background:{PRIMARY}; color:white; }}"
            f"QPushButton:pressed {{ background:#0d3d20; color:white; }}"
        )
        img_att_btn.setCursor(Qt.PointingHandCursor)
        img_att_btn.clicked.connect(self._attendance_from_image)
        btn_row.addWidget(img_att_btn)
        btn_row.addSpacing(10)
        start_btn = QPushButton("▶  Start session")
        start_btn.setToolTip("Start a live face-recognition attendance session")
        start_btn.setFixedHeight(40)
        start_btn.setStyleSheet(
            f"QPushButton {{ background:{PRIMARY}; color:white; border-radius:8px; "
            f"padding:12px 24px; font-size:14px; font-weight:700; }}"
            f"QPushButton:hover {{ background:#145a30; color:white; }}"
            f"QPushButton:pressed {{ background:#0d3d20; color:white; }}"
        )
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.clicked.connect(self._start_session)
        btn_row.addWidget(start_btn)
        m_lay.addLayout(btn_row)

        if not self._courses_data:
            for w in [self._tbl, start_btn]:
                w.setVisible(False)
            no_course_lbl = QLabel("No courses assigned to your account.\nPlease contact the admin.")
            no_course_lbl.setAlignment(Qt.AlignCenter)
            no_course_lbl.setStyleSheet(
                f"font-size:15px; color:{TEXT_GRAY}; background:transparent;"
            )
            m_lay.insertWidget(2, no_course_lbl)

        root.addWidget(sidebar)
        root.addWidget(main_w)

        if self._courses_data:
            self._select_course(0)

    def _select_course(self, idx):
        for i, c in enumerate(self._course_cards):
            c.set_active(i == idx)
        self._active_idx = idx
        course = self._courses_data[idx]
        code = str(course.get("SectionID", ""))
        name = course.get("CourseTitle", "")
        sched_raw = course.get("Schedule", "")
        if isinstance(sched_raw, dict):
            sched = f"{sched_raw.get('Day', '')} {sched_raw.get('StartTime', '')}–{sched_raw.get('EndTime', '')}"
        else:
            sched = str(sched_raw)
        enrolled_ids = course.get("EnrolledStudents", [])
        students = []
        if DB_CONNECTED and enrolled_ids:
            try:
                students = list(db["Students"].find({"StudentID": {"$in": enrolled_ids}}))
            except Exception as _e:
                print(f"[AIAS] warning: {_e}")
        self._title_lbl.setText(name)
        self._sub_lbl.setText(f"{code} · {sched} · {len(students)} students")
        self._enrolled_val.setText(str(len(students)))
        self._tbl.setRowCount(len(students))
        for r, stu in enumerate(students):
            self._tbl.setItem(r, 0, QTableWidgetItem(str(stu.get("StudentID", ""))))
            self._tbl.setItem(r, 1, QTableWidgetItem(stu.get("FullName", "")))
            self._tbl.setCellWidget(r, 2, make_badge("Enrolled"))
            self._tbl.setRowHeight(r, 44)

        if DB_CONNECTED:
            try:
                n_sessions = db["Sessions"].count_documents({"SectionID": code, "Status": "completed"})
                last_sess  = db["Sessions"].find_one({"SectionID": code, "Status": "completed"}, sort=[("EndTime", -1)])
                self._sessions_held_lbl.setText(str(n_sessions))
                self._last_session_lbl.setText(last_sess["EndTime"].strftime("%Y-%m-%d") if last_sess else "—")
            except Exception as _e:
                print(f"[AIAS] warning: {_e}")

    def _start_session(self):
        if not self._courses_data:
            return
        course     = self._courses_data[self._active_idx]
        session_id = None
        enrolled_students = []

        if DB_CONNECTED:
            try:
                enrolled_ids      = course.get("EnrolledStudents", [])
                enrolled_students = list(db["Students"].find({"StudentID": {"$in": enrolled_ids}}))
            except Exception as e:
                print(f"DB error: {e}")

        # Rec 2: warn if no student has a face embedding
        if enrolled_students:
            n_with_face = sum(1 for s in enrolled_students if s.get("FaceEmbedding"))
            if n_with_face == 0:
                reply = QMessageBox.warning(
                    self, "No Face Data",
                    f"None of the {len(enrolled_students)} enrolled students have face embeddings.\n"
                    "Attendance will need to be entered manually.\n\nProceed anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        if DB_CONNECTED:
            try:
                session_doc = {
                    "CourseID":           course.get("CourseID", course.get("SectionID")),
                    "SectionID":          course.get("SectionID"),
                    "InstructorID":       self.instructor_data.get("username", ""),
                    "StartTime":          datetime.now(),
                    "EndTime":            None,
                    "Status":             "active",
                    "AttendanceInterval": 5,
                }
                result     = db["Sessions"].insert_one(session_doc)
                session_id = result.inserted_id
            except Exception as e:
                print(f"DB error: {e}")

        cam_idx = choose_camera_dialog(self)
        if cam_idx is None:
            return  # user cancelled

        start_btn = self.sender()
        if start_btn:
            start_btn.setText("▶  Starting...")
            start_btn.setEnabled(False)
            QApplication.processEvents()

        self.hide()
        self._live = LiveSessionWindow(self, session_id, enrolled_students, course, camera_index=cam_idx)
        self._live.show()

        if start_btn:
            start_btn.setText("▶  Start session")
            start_btn.setEnabled(True)

    def _attendance_from_image(self):
        """Instructor: pick an image file, detect faces, mark attendance for the active course."""
        if not self._courses_data:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Attendance Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return

        course = self._courses_data[self._active_idx]
        enrolled_students = []
        if DB_CONNECTED:
            try:
                enrolled_ids      = course.get("EnrolledStudents", [])
                enrolled_students = list(db["Students"].find({"StudentID": {"$in": enrolled_ids}}))
            except Exception as e:
                QMessageBox.critical(self, "DB Error", str(e))
                return

        if not enrolled_students:
            QMessageBox.warning(self, "No Students", "No enrolled students found for this course.")
            return

        n_with_face = sum(1 for s in enrolled_students if s.get("FaceEmbedding"))
        if n_with_face == 0:
            QMessageBox.warning(
                self, "No Face Data",
                "None of the enrolled students have face embeddings uploaded.\n"
                "Please ask the admin to upload student photos first."
            )
            return

        progress_dlg = QDialog(self)
        progress_dlg.setWindowTitle("Processing Image...")
        progress_dlg.setFixedWidth(400)
        progress_dlg.setStyleSheet(f"background:{BG};")
        progress_dlg.setWindowFlags(progress_dlg.windowFlags() & ~Qt.WindowCloseButtonHint)

        pd_lay = QVBoxLayout(progress_dlg)
        pd_lay.setContentsMargins(24, 24, 24, 24)
        pd_lay.setSpacing(12)

        pd_title = QLabel("Analyzing Image...")
        pd_title.setStyleSheet(f"font-size:15px; font-weight:700; color:{TEXT_DARK};")
        pd_lay.addWidget(pd_title)

        pd_sub = QLabel("Running face detection and recognition. This may take a few seconds.")
        pd_sub.setStyleSheet(f"font-size:12px; color:{TEXT_GRAY};")
        pd_sub.setWordWrap(True)
        pd_lay.addWidget(pd_sub)

        pd_bar = QProgressBar()
        pd_bar.setRange(0, 0)
        pd_bar.setFixedHeight(8)
        pd_lay.addWidget(pd_bar)

        progress_dlg.setModal(True)
        progress_dlg.show()
        QApplication.processEvents()

        self._img_worker = ImageAttendanceWorker(path, enrolled_students)

        def _on_finished(recognized):
            progress_dlg.close()
            self._show_image_attendance_results(course, enrolled_students, recognized, path)

        def _on_error(msg):
            progress_dlg.close()
            QMessageBox.critical(self, "Processing Error", msg)

        self._img_worker.finished.connect(_on_finished)
        self._img_worker.error.connect(_on_error)
        self._img_worker.start()

    def _show_image_attendance_results(self, course, enrolled_students, recognized, image_path):
        """Show results dialog and allow instructor to submit image-based attendance."""
        section_id   = str(course.get("SectionID", ""))
        course_title = course.get("CourseTitle", "")

        dlg = QDialog(self)
        dlg.setWindowTitle("Image Attendance Results")
        dlg.setMinimumWidth(580)
        dlg.setStyleSheet(f"background:{BG};")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        title_lbl = QLabel(f"Attendance Results — {section_id}  {course_title}")
        title_lbl.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_DARK};")
        lay.addWidget(title_lbl)

        img_name  = os.path.basename(image_path)
        n_present = len(recognized)
        n_total   = len(enrolled_students)
        sub_lbl = QLabel(
            f"Image: {img_name}   •   {n_present} recognized / {n_total} enrolled"
        )
        sub_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_GRAY};")
        lay.addWidget(sub_lbl)

        tbl = QTableWidget()
        tbl.setColumnCount(3)
        tbl.setHorizontalHeaderLabels(["Student ID", "Full Name", "Status"])
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionMode(QAbstractItemView.NoSelection)
        tbl.verticalHeader().setVisible(False)
        tbl.setStyleSheet(
            f"QTableWidget {{ border:1px solid {BORDER}; border-radius:8px; background:{WHITE}; }}"
            f"QHeaderView::section {{ background:{WHITE}; color:{TEXT_MED}; font-weight:600; "
            f"border:none; border-bottom:1px solid {BORDER}; padding:6px; }}"
        )

        rows = []
        for stu in enrolled_students:
            sid  = str(stu.get("StudentID", ""))
            name = stu.get("FullName", "")
            rows.append((sid, name, "Present" if sid in recognized else "Absent"))

        tbl.setRowCount(len(rows))
        for r, (sid, name, status) in enumerate(rows):
            tbl.setItem(r, 0, QTableWidgetItem(sid))
            tbl.setItem(r, 1, QTableWidgetItem(name))
            status_item = QTableWidgetItem(status)
            status_item.setForeground(
                QColor("#22C55E") if status == "Present" else QColor("#EF4444")
            )
            tbl.setItem(r, 2, status_item)

        tbl.setFixedHeight(min(40 * len(rows) + 42, 360))
        lay.addWidget(tbl)

        note_lbl = QLabel(
            "Review the results, then click \"Submit Attendance\" to save to the database."
        )
        note_lbl.setStyleSheet(f"font-size:11px; color:{TEXT_GRAY};")
        note_lbl.setWordWrap(True)
        lay.addWidget(note_lbl)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            f"background:{WHITE}; color:{TEXT_MED}; border:1px solid {BORDER}; "
            "border-radius:6px; padding:8px 20px; font-weight:600;"
        )
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(dlg.reject)

        submit_btn = QPushButton("Submit Attendance")
        submit_btn.setStyleSheet(
            f"background:{PRIMARY}; color:white; border-radius:6px; "
            "padding:8px 20px; font-weight:700;"
        )
        submit_btn.setCursor(Qt.PointingHandCursor)

        def _submit():
            if not DB_CONNECTED:
                QMessageBox.warning(dlg, "No Database", "Database is not connected.")
                return
            try:
                now = datetime.now()
                session_doc = {
                    "CourseID":    course.get("CourseID", section_id),
                    "SectionID":   section_id,
                    "InstructorID": self.instructor_data.get("username", ""),
                    "StartTime":   now,
                    "EndTime":     now,
                    "Status":      "completed",
                    "Source":      "image_upload",
                    "ImageFile":   os.path.basename(image_path),
                }
                result  = db["Sessions"].insert_one(session_doc)
                sess_id = result.inserted_id

                for sid, name, status in rows:
                    db["AttendanceLogs"].insert_one({
                        "SessionID":   sess_id,
                        "StudentID":   sid,
                        "Status":      status,
                        "FirstSeenAt": now if status == "Present" else None,
                        "LastSeenAt":  now if status == "Present" else None,
                    })

                QMessageBox.information(
                    dlg, "Attendance Saved",
                    f"Session created.\n"
                    f"{n_present} marked Present, {n_total - n_present} marked Absent."
                )
                dlg.accept()
                self._select_course(self._active_idx)
            except Exception as e:
                QMessageBox.critical(dlg, "DB Error", f"Could not save attendance:\n{e}")

        submit_btn.clicked.connect(_submit)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(submit_btn)
        lay.addLayout(btn_row)

        dlg.exec_()

    def _show_dashboard(self):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QLabel, QFrame, QScrollArea, QWidget)
        from PyQt5.QtCore import Qt
        import datetime

        dlg = QDialog(self)
        dlg.setWindowTitle("Dashboard")
        dlg.setMinimumSize(820, 580)
        dlg.setStyleSheet(f"background:{BG};")

        import os
        from PyQt5.QtGui import QIcon
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aias_icon.ico")
        if os.path.exists(_icon_path):
            dlg.setWindowIcon(QIcon(_icon_path))

        main_lay = QVBoxLayout(dlg)
        main_lay.setContentsMargins(28, 24, 28, 24)
        main_lay.setSpacing(20)

        # Title
        _full_name = self.instructor_data.get("fullname", "Instructor")
        _username  = self.instructor_data.get("username", "")
        title = QLabel("📊  Instructor Dashboard")
        title.setStyleSheet(f"font-size:22px; font-weight:800; color:{TEXT_DARK}; background:transparent;")
        sub = QLabel(f"Welcome, {_full_name}  ·  {datetime.date.today().strftime('%B %d, %Y')}")
        sub.setStyleSheet(f"font-size:13px; color:{TEXT_GRAY}; background:transparent;")
        main_lay.addWidget(title)
        main_lay.addWidget(sub)

        # Load data from DB
        try:
            db = get_db()
            # Get all sections for this instructor
            instructor = db["Instructors"].find_one({"Username": _username})
            sections = instructor.get("Sections", []) if instructor else []

            total_students = 0
            total_sessions = 0
            total_present  = 0
            total_records  = 0
            absent_count   = {}

            for sec_id in sections:
                # Students in section
                n_students = db["Students"].count_documents({"Sections": sec_id})
                total_students += n_students

                # Sessions for this section
                sessions = list(db["AttendanceSessions"].find({"SectionID": sec_id}))
                total_sessions += len(sessions)

                for session in sessions:
                    records = session.get("Records", [])
                    for rec in records:
                        total_records += 1
                        status = rec.get("Status", "")
                        if status in ("Present", "Late"):
                            total_present += 1
                        elif status == "Absent":
                            sid = rec.get("StudentID", "")
                            absent_count[sid] = absent_count.get(sid, 0) + 1

            attendance_rate = (total_present / total_records * 100) if total_records > 0 else 0

            # Top absentees
            top_absent = sorted(absent_count.items(), key=lambda x: x[1], reverse=True)[:5]
            top_absent_data = []
            for sid, cnt in top_absent:
                student = db["Students"].find_one({"StudentID": sid})
                name = student.get("FullName", sid) if student else sid
                top_absent_data.append((name, cnt))

        except Exception as e:
            print(f"[AIAS] Dashboard error: {e}")
            sections = []
            total_students = total_sessions = attendance_rate = 0
            top_absent_data = []

        # Stats cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        def make_stat_card(value, label, color):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background:{WHITE}; border-radius:12px; "
                f"border-left:4px solid {color}; }}"
            )
            card.setMinimumHeight(90)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(20, 14, 20, 14)
            v_lbl = QLabel(str(value))
            v_lbl.setStyleSheet(f"font-size:28px; font-weight:800; color:{color}; background:transparent;")
            l_lbl = QLabel(label)
            l_lbl.setStyleSheet(f"font-size:12px; color:{TEXT_GRAY}; background:transparent;")
            lay.addWidget(v_lbl)
            lay.addWidget(l_lbl)
            return card

        cards_row.addWidget(make_stat_card(total_students, "Total Students", PRIMARY))
        cards_row.addWidget(make_stat_card(total_sessions, "Sessions Held", "#2563EB"))
        cards_row.addWidget(make_stat_card(f"{attendance_rate:.1f}%", "Attendance Rate", "#16A34A"))
        cards_row.addWidget(make_stat_card(len(sections), "My Courses", "#9333EA"))
        main_lay.addLayout(cards_row)

        # Attendance rate bar
        bar_frame = QFrame()
        bar_frame.setStyleSheet(f"QFrame {{ background:{WHITE}; border-radius:12px; }}")
        bar_lay = QVBoxLayout(bar_frame)
        bar_lay.setContentsMargins(20, 16, 20, 16)
        bar_title = QLabel("Overall Attendance Rate")
        bar_title.setStyleSheet(f"font-size:14px; font-weight:700; color:{TEXT_DARK}; background:transparent;")
        bar_lay.addWidget(bar_title)

        from PyQt5.QtWidgets import QProgressBar
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(int(attendance_rate))
        color = PRIMARY if attendance_rate >= 75 else "#F59E0B" if attendance_rate >= 50 else "#DC2626"
        progress.setStyleSheet(f"""
            QProgressBar {{ background:#E5E7EB; border-radius:8px; height:18px; text-align:center; color:{TEXT_DARK}; font-weight:600; }}
            QProgressBar::chunk {{ background:{color}; border-radius:8px; }}
        """)
        bar_lay.addWidget(progress)
        main_lay.addWidget(bar_frame)

        # Most absent students
        if top_absent_data:
            absent_frame = QFrame()
            absent_frame.setStyleSheet(f"QFrame {{ background:{WHITE}; border-radius:12px; }}")
            absent_lay = QVBoxLayout(absent_frame)
            absent_lay.setContentsMargins(20, 16, 20, 16)
            absent_lay.setSpacing(10)

            absent_title = QLabel("⚠️  Most Absent Students")
            absent_title.setStyleSheet(f"font-size:14px; font-weight:700; color:{TEXT_DARK}; background:transparent;")
            absent_lay.addWidget(absent_title)

            for name, cnt in top_absent_data:
                row = QHBoxLayout()
                name_lbl = QLabel(f"• {name}")
                name_lbl.setStyleSheet(f"font-size:13px; color:{TEXT_DARK}; background:transparent;")
                cnt_lbl = QLabel(f"{cnt} absences")
                cnt_lbl.setStyleSheet(f"font-size:13px; color:#DC2626; font-weight:600; background:transparent;")
                row.addWidget(name_lbl)
                row.addStretch()
                row.addWidget(cnt_lbl)
                absent_lay.addLayout(row)

            main_lay.addWidget(absent_frame)

        main_lay.addStretch()
        dlg.exec_()

    def _open_history(self):
        from ui_reports import SessionHistoryWindow
        self.hide()
        self._history_win = SessionHistoryWindow(self)
        self._history_win.show()

    def _open_analytics(self):
        from ui_reports import AnalyticsWindow
        self.hide()
        self._analytics_win = AnalyticsWindow(self)
        self._analytics_win.show()

    def _open_email_settings(self):
        univ_email = ""
        if DB_CONNECTED:
            try:
                instr = db["Instructors"].find_one(
                    {"Username": self.instructor_data.get("username", "")}
                )
                if instr:
                    univ_email = instr.get("UniversityEmail", "")
            except Exception as e:
                print(f"[AIAS] Email settings load error: {e}")

        dlg = QDialog(self)
        dlg.setWindowTitle("Email Settings")
        dlg.setFixedWidth(440)
        dlg.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        t = QLabel("Email Settings")
        t.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_DARK};")
        lay.addWidget(t)

        if univ_email:
            status_html = (
                f"<b style='color:#22C55E;'>✔ Configured</b><br>"
                f"<span style='color:#374151;'>Send to: {univ_email}</span>"
            )
            status_color = "#F0FDF4"
            border_color = "#86EFAC"
        else:
            status_html = (
                "<b style='color:#F59E0B;'>⚠ Not configured</b><br>"
                "<span style='color:#6B7280;'>You'll be prompted to set up email the first time you send a report.</span>"
            )
            status_color = "#FFFBEB"
            border_color = "#FCD34D"

        status_lbl = QLabel()
        status_lbl.setText(status_html)
        status_lbl.setStyleSheet(
            f"background:{status_color}; border:1px solid {border_color};"
            "border-radius:8px; padding:12px;"
        )
        status_lbl.setWordWrap(True)
        lay.addWidget(status_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"background:{WHITE}; color:{TEXT_MED}; border:1px solid {BORDER}; border-radius:6px; padding:8px 16px;"
        )
        close_btn.clicked.connect(dlg.reject)

        reset_btn = QPushButton("Reset / Clear All")
        reset_btn.setStyleSheet(
            "background:#FEE2E2; color:#DC2626; border-radius:6px; padding:8px 14px; font-weight:600;"
        )
        reset_btn.clicked.connect(lambda: dlg.done(2))

        update_btn = QPushButton("Update Settings")
        update_btn.setStyleSheet(
            f"background:{PRIMARY}; color:white; border-radius:6px; padding:8px 16px; font-weight:700;"
        )
        update_btn.clicked.connect(dlg.accept)

        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(update_btn)
        lay.addLayout(btn_row)

        result = dlg.exec_()

        if result == QDialog.Accepted:
            self._open_email_settings_edit(univ_email)
        elif result == 2:
            reply = QMessageBox.question(
                self, "Confirm Reset",
                "Clear all saved email credentials?\n"
                "You'll need to re-enter them the next time you send a report.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes and DB_CONNECTED:
                try:
                    db["Instructors"].update_one(
                        {"Username": self.instructor_data.get("username", "")},
                        {"$unset": {"UniversityEmail": ""}},
                    )
                    QMessageBox.information(self, "Cleared", "Email credentials have been cleared.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))

    def _open_email_settings_edit(self, current_univ_email=""):
        dlg = QDialog(self)
        dlg.setWindowTitle("Update Email Settings")
        dlg.setFixedWidth(420)
        dlg.setStyleSheet(f"background:{BG};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(12)

        t = QLabel("Update Email Settings")
        t.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_DARK};")
        lay.addWidget(t)

        sub = QLabel("Enter the university email address to send reports to.")
        sub.setStyleSheet(f"font-size:12px; color:{TEXT_GRAY};")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        lbl = QLabel("Send reports to:")
        lbl.setStyleSheet(f"font-size:12px; font-weight:600; color:{TEXT_MED};")
        univ_field = QLineEdit()
        univ_field.setPlaceholderText("name@university.edu")
        univ_field.setFixedHeight(38)
        univ_field.setStyleSheet(
            f"border:1px solid {BORDER}; border-radius:6px; padding:6px 10px; background:{WHITE}; color:{TEXT_DARK};"
        )
        univ_field.setText(current_univ_email)
        lay.addWidget(lbl)
        lay.addWidget(univ_field)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            f"background:{WHITE}; color:{TEXT_MED}; border:1px solid {BORDER}; border-radius:6px; padding:8px 16px;"
        )
        cancel_btn.clicked.connect(dlg.reject)
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            f"background:{PRIMARY}; color:white; border-radius:6px; padding:8px 20px; font-weight:700;"
        )
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            return

        univ_email = univ_field.text().strip()
        if not univ_email:
            QMessageBox.warning(self, "Missing Field", "Please enter a destination email address.")
            return

        if DB_CONNECTED:
            try:
                db["Instructors"].update_one(
                    {"Username": self.instructor_data.get("username", "")},
                    {"$set": {"UniversityEmail": univ_email}},
                )
                QMessageBox.information(self, "Saved", "Email settings saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")

    def _toggle_dark_mode(self, btn):
        from ui_theme import apply_theme, force_theme_on_all_widgets
        apply_theme(QApplication.instance(), dark=not config.IS_DARK_MODE)
        force_theme_on_all_widgets(self)
        btn.setText("☀  Light Mode" if config.IS_DARK_MODE else "🌙  Dark Mode")
        btn.setStyleSheet(
            f"QPushButton {{ background:{config.PRIMARY_LIGHT}; color:{config.PRIMARY}; border-radius:6px; "
            f"padding:8px 12px; font-size:12px; font-weight:600; }}"
            f"QPushButton:hover {{ background:{config.PRIMARY}; color:{config.WHITE}; }}"
            f"QPushButton:pressed {{ background:{config.PRIMARY}; color:{config.WHITE}; padding:9px 11px 7px 13px; }}"
        )
        self.update()
        self.repaint()

    def _change_password(self):
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                                     QLineEdit, QPushButton, QHBoxLayout)
        import bcrypt

        dlg = QDialog(self)
        dlg.setWindowTitle("Change Password")
        dlg.setStyleSheet(f"background:{BG};")
        dlg.setMinimumWidth(340)

        lay = QVBoxLayout(dlg)
        lay.setSpacing(14)
        lay.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Change Password")
        title.setStyleSheet(f"font-size:16px; font-weight:700; color:{TEXT_DARK};")
        lay.addWidget(title)

        def make_field(label_text, placeholder):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"font-size:12px; font-weight:600; color:{TEXT_DARK};")
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            field.setEchoMode(QLineEdit.Password)
            field.setStyleSheet(
                f"border:1px solid {BORDER}; border-radius:8px; padding:9px 12px; "
                f"background:{WHITE}; color:{TEXT_DARK}; font-size:13px;"
            )
            lay.addWidget(lbl)
            lay.addWidget(field)
            return field

        current_field = make_field("Current Password", "Enter current password")
        new_field     = make_field("New Password", "Enter new password")
        confirm_field = make_field("Confirm New Password", "Re-enter new password")

        err_lbl = QLabel("")
        err_lbl.setStyleSheet("font-size:11px; color:#DC2626;")
        err_lbl.setWordWrap(True)
        lay.addWidget(err_lbl)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            f"background:{WHITE}; color:{TEXT_MED}; border:1px solid {BORDER}; "
            "border-radius:6px; padding:8px 16px; font-weight:600;"
        )
        cancel_btn.clicked.connect(dlg.reject)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(
            f"background:{PRIMARY}; color:white; border-radius:6px; "
            "padding:8px 16px; font-weight:600;"
        )

        def do_change():
            current  = current_field.text().strip()
            new_pass = new_field.text().strip()
            confirm  = confirm_field.text().strip()

            if not current or not new_pass or not confirm:
                err_lbl.setText("Please fill in all fields.")
                return
            if new_pass != confirm:
                err_lbl.setText("New passwords do not match.")
                return
            if len(new_pass) < 6:
                err_lbl.setText("Password must be at least 6 characters.")
                return

            # Verify current password
            username = self.instructor_data.get("username", "")
            user = db["Instructors"].find_one({"Username": username})
            if not user:
                err_lbl.setText("User not found.")
                return

            stored = user.get("Password", "")
            try:
                if isinstance(stored, str):
                    stored = stored.encode()
                if not bcrypt.checkpw(current.encode(), stored):
                    err_lbl.setText("Current password is incorrect.")
                    return
            except Exception:
                err_lbl.setText("Error verifying password.")
                return

            # Save new password
            new_hash = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
            db["Instructors"].update_one(
                {"Username": username},
                {"$set": {"Password": new_hash.decode()}}
            )
            dlg.accept()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Success", "Password changed successfully.")

        save_btn.clicked.connect(do_change)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)
        dlg.exec_()

    def _logout(self):
        for widget in QApplication.instance().topLevelWidgets():
            if isinstance(widget, AppWindow):
                widget._login_page.username_field.clear()
                widget._login_page.password_field.clear()
                widget._stack.setCurrentWidget(widget._login_page)
                widget.setStyleSheet("background:#0a0f0a;")
                break
        # Remove this page from stack
        parent_stack = self.parent()
        if parent_stack:
            try:
                stack = parent_stack.parent()
                if hasattr(stack, 'removeWidget'):
                    stack.removeWidget(self)
            except Exception as e:
                print(f"[AIAS] Logout stack cleanup error: {e}")
        self.deleteLater()

    def closeEvent(self, event):
        # Stop any active live session worker when main window closes
        try:
            if hasattr(self, '_live') and self._live is not None:
                try:
                    if hasattr(self._live, '_worker') and self._live._worker is not None:
                        self._live._worker.stop()
                        self._live._worker.quit()
                        self._live._worker.wait(2000)
                except Exception as e:
                    print(f"[AIAS] closeEvent worker stop error: {e}")
                try:
                    self._live.close()
                except Exception as e:
                    print(f"[AIAS] closeEvent live close error: {e}")
        except Exception as e:
            print(f"[AIAS] closeEvent error: {e}")
        event.accept()
