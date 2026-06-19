import sys
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["ORT_LOG_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "0"

# Add torch's lib dir to DLL search path before PyQt5 imports alter it.
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.origin:
        _torch_lib = os.path.join(os.path.dirname(_spec.origin), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
except Exception as _e:
    print(f"[AIAS] warning: {_e}")

try:
    import cv2 as _cv2_pre
    import onnxruntime as _ort_pre
    import torch as _torch_pre
    from ultralytics import YOLO as _YOLO_pre
    from insightface.app import FaceAnalysis as _FA_pre
    print("[AIAS] AI libraries pre-loaded")
except Exception as _pre_e:
    print(f"[AIAS] AI pre-load warning: {_pre_e}")

import pymongo
import bcrypt
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as _F
try:
    import pytorch_lightning as _pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "AIAS_DB")
try:
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.server_info()
    db = client[DB_NAME]
    DB_CONNECTED = True
except Exception:
    client = None
    db = None
    DB_CONNECTED = False

if DB_CONNECTED:
    try:
        db.command("collMod", "AttendanceLogs", validator={}, validationLevel="off")
        print("[AIAS] AttendanceLogs validator disabled")
    except Exception as _e:
        print(f"[AIAS] warning: {_e}")

SMTP_SERVER   = os.getenv("SMTP_SERVER", "smtp-relay.brevo.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_LOGIN    = os.getenv("SMTP_LOGIN", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", "")

LATE_CUTOFF_MINUTES        = 10
EARLY_LEAVE_CUTOFF_MINUTES = 15
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.55"))

def _load_settings():
    global LATE_CUTOFF_MINUTES, EARLY_LEAVE_CUTOFF_MINUTES, RECOGNITION_THRESHOLD
    if not DB_CONNECTED:
        return
    try:
        doc = db["Settings"].find_one({"_id": "global"})
        if doc:
            LATE_CUTOFF_MINUTES        = doc.get("LateCutoffMinutes",       10)
            EARLY_LEAVE_CUTOFF_MINUTES = doc.get("EarlyLeaveCutoffMinutes", 15)
            RECOGNITION_THRESHOLD      = doc.get("RecognitionThreshold",    0.55)
    except Exception as e:
        print(f"[AIAS] Settings load error: {e}")

_load_settings()


def _ensure_admin_exists():
    if not DB_CONNECTED:
        return
    try:
        if not db["Instructors"].find_one({"Username": "admin"}):
            import secrets, string
            _chars = string.ascii_letters + string.digits
            _rand_pw = ''.join(secrets.choice(_chars) for _ in range(12))
            hashed = bcrypt.hashpw(_rand_pw.encode(), bcrypt.gensalt())
            print(f"[AIAS] Default admin account created. Password: {_rand_pw}")
            print(f"[AIAS] Please change this password after first login.")
            db["Instructors"].insert_one({
                "Username": "admin",
                "FullName": "Administrator",
                "Password": hashed,
                "Role": "admin",
                "AssignedSections": [],
            })
            print("[AIAS] Default admin account created.")
    except Exception as e:
        print(f"[AIAS] Admin setup error: {e}")


def get_db():
    """Returns the global db object."""
    return db


PRIMARY = "#1B5E35"
PRIMARY_LIGHT = "#C8E0D0"
BG = "#F8F9FA"
WHITE = "#FFFFFF"
TEXT_DARK = "#1A1A2E"
TEXT_MED = "#374151"
TEXT_GRAY = "#6B7280"
BORDER = "#E5E7EB"

# Dark mode color palette
DARK_PRIMARY       = "#4ecf8e"
DARK_PRIMARY_LIGHT = "#0D2B18"
DARK_BG            = "#0A0A0F"
DARK_WHITE         = "#111118"
DARK_TEXT_DARK     = "#F1F1F5"
DARK_TEXT_MED      = "#C4C4D4"
DARK_TEXT_GRAY     = "#7777AA"
DARK_BORDER        = "#222235"

IS_DARK_MODE = False

BADGE_MAP = {
    "Present": ("#EAF3DE", "#27500A"),
    "Late": ("#FAEEDA", "#633806"),
    "Absent": ("#FCEBEB", "#791F1F"),
    "Early Leave": ("#E6F1FB", "#0C447C"),
    "Enrolled": ("#EAF3DE", "#27500A"),
    "Ready": ("#EAF3DE", "#27500A"),
    "No photo": ("#FAEEDA", "#633806"),
}

COURSES  = []
STUDENTS = []

GLOBAL_QSS = f"""
QWidget {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}}
QLineEdit {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    background: {WHITE};
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
}}
QTableWidget {{
    border: none;
    outline: none;
    background: {WHITE};
    alternate-background-color: #FAFAFA;
    selection-background-color: #D4EBE1;
    selection-color: {TEXT_DARK};
    gridline-color: transparent;
}}
QTableWidget::item {{
    padding: 6px 10px;
    border: none;
}}
QHeaderView::section {{
    background: #F5F5F5;
    color: {TEXT_GRAY};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 600;
}}
QScrollBar:vertical {{
    border: none;
    background: {BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #D1D5DB;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QComboBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    background: {WHITE};
    color: {TEXT_DARK};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QProgressBar {{
    background: #E5E7EB;
    border-radius: 4px;
    border: none;
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 4px;
}}
"""
