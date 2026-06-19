import sys
import socket
from datetime import datetime

# torch must be imported before PyQt5 — PyQt5's DLL changes break torch on Windows
from config import DB_CONNECTED, db, GLOBAL_QSS, _ensure_admin_exists
from ai_models import AIModelLoader

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

from ui_theme import apply_theme
from ui_login import AppWindow

if __name__ == "__main__":
    try:
        _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _lock_socket.bind(('localhost', 47200))
    except OSError:
        print("[AIAS] Another instance is already running. Exiting.")
        sys.exit(0)

    # Bug 3 / Rec 1: mark any stuck active sessions as completed on startup
    if DB_CONNECTED:
        try:
            _cleanup = db["Sessions"].update_many(
                {"Status": "active"},
                {"$set": {"Status": "completed", "EndTime": datetime.now()}},
            )
            if _cleanup.modified_count:
                print(f"[AIAS] Cleaned up {_cleanup.modified_count} stuck active session(s).")
        except Exception as _ce:
            print(f"[AIAS] Session cleanup error: {_ce}")

    # Bug 1 / Rec 4: ensure admin account exists in MongoDB
    _ensure_admin_exists()

    app = QApplication(sys.argv)
    apply_theme(app, dark=False)

    import os
    from PyQt5.QtGui import QIcon
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aias_icon.ico")
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    # Load AI models BEFORE showing any window
    # Use a blocking approach with event loop processing so UI stays responsive
    _models_loaded = [False]
    _models_ok = [True]

    def _on_models_done(ok):
        _models_ok[0] = ok
        _models_loaded[0] = True
        if not ok:
            print("[AIAS] Warning: AI models failed to load.")

    _AI_LOADER_THREAD = AIModelLoader()
    _AI_LOADER_THREAD.done.connect(_on_models_done)
    _AI_LOADER_THREAD.start()

    # Wait for models to finish — process events so app doesn't freeze
    while not _models_loaded[0]:
        app.processEvents()
        QThread.msleep(50)

    if not _models_ok[0]:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(None, "AIAS Warning",
            "AI models failed to load. Face recognition will not work.\n"
            "Please check that all model files are present.")

    # Models ready — now show app with splash
    app_win = AppWindow()
    app_win.show()

    sys.exit(app.exec_())
