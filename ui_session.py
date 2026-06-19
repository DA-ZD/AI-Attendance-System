import os
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import config
from config import (
    DB_CONNECTED, db,
    LATE_CUTOFF_MINUTES, EARLY_LEAVE_CUTOFF_MINUTES, RECOGNITION_THRESHOLD,
    SMTP_SERVER, SMTP_PORT,
    PRIMARY, PRIMARY_LIGHT, BG, WHITE, TEXT_DARK, TEXT_MED, TEXT_GRAY, BORDER,
    BADGE_MAP,
)
from ai_models import (
    _ensure_ai_models, apply_clahe, upscale_face_crop, compute_dynamic_threshold,
    _gfpgan_model,
)
from ui_theme import (
    h_sep, make_badge, make_stat_card,
    make_avatar, make_sidebar_base, make_table,
)


def get_available_cameras(max_check=5):
    """Returns a list of (index, label) for all available cameras."""
    import cv2 as _cv2
    cameras = []
    for i in range(max_check):
        cap = _cv2.VideoCapture(i, _cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            cameras.append((i, f"Camera {i}" if i > 0 else f"Camera {i} (Default)"))
            cap.release()
    return cameras

def choose_camera_dialog(parent=None):
    """Shows a dialog to let the user pick a camera. Returns selected index or 0."""
    cameras = get_available_cameras()
    if len(cameras) <= 1:
        return 0  # only one camera, use it directly

    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
    dlg = QDialog(parent)
    dlg.setWindowTitle("Select Camera")
    dlg.setStyleSheet(f"background:{config.BG};")
    dlg.setMinimumWidth(300)

    lay = QVBoxLayout(dlg)
    lay.setSpacing(14)
    lay.setContentsMargins(24, 24, 24, 24)

    title = QLabel("Select Camera")
    title.setStyleSheet(f"font-size:15px; font-weight:700; color:{config.TEXT_DARK};")
    lay.addWidget(title)

    sub = QLabel("Choose which camera to use for face recognition:")
    sub.setStyleSheet(f"font-size:12px; color:{config.TEXT_GRAY};")
    sub.setWordWrap(True)
    lay.addWidget(sub)

    combo = QComboBox()
    combo.setStyleSheet(
        f"border:1px solid {config.BORDER}; border-radius:6px; padding:6px 10px; "
        f"background:{config.WHITE}; color:{config.TEXT_DARK}; font-size:13px;"
    )
    for idx, label in cameras:
        combo.addItem(label, idx)
    lay.addWidget(combo)

    btn_row = QHBoxLayout()
    cancel_btn = QPushButton("Cancel")
    cancel_btn.setStyleSheet(
        f"background:{config.WHITE}; color:{config.TEXT_MED}; border:1px solid {config.BORDER}; "
        "border-radius:6px; padding:8px 16px; font-weight:600;"
    )
    cancel_btn.clicked.connect(dlg.reject)

    confirm_btn = QPushButton("Start Session")
    confirm_btn.setStyleSheet(
        f"background:{config.PRIMARY}; color:white; border-radius:6px; "
        "padding:8px 16px; font-weight:600;"
    )
    confirm_btn.clicked.connect(dlg.accept)

    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(confirm_btn)
    lay.addLayout(btn_row)

    result = dlg.exec_()
    if result == QDialog.Accepted:
        return combo.currentData()
    return None  # user cancelled


class _ProcessingWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, frame, cv2_mod, np_mod, yolo, arcface, parent_worker):
        super().__init__()
        self._frame   = frame
        self._cv2     = cv2_mod
        self._np      = np_mod
        self._yolo    = yolo
        self._arcface = arcface
        self._worker  = parent_worker

    def run(self):
        try:
            result = self._worker._process_frame(
                self._frame, self._cv2, self._np, self._yolo, self._arcface
            )
            self.result_ready.emit(result)
        except Exception as e:
            print(f"[AIAS] ProcessingWorker error: {e}")
            self.result_ready.emit({})


def _detect_faces_multiscale(_cv2, _np, _yolo, frame, conf=0.25):
    """
    Multi-scale face detection: runs YOLO at 1x, 2x, and 4x zoom.
    4x zoom catches faces at 5-7 meters from a ceiling-mounted camera.
    All coordinates mapped back to original frame space.
    """
    h, w = frame.shape[:2]
    all_boxes = []

    # ── 1. Native resolution ──────────────────────────────────────────
    try:
        res = _yolo(frame, verbose=False, imgsz=640, conf=conf)
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            all_boxes.append((x1, y1, x2, y2))
    except Exception:
        pass

    # ── 2. 2x upscale ─────────────────────────────────────────────────
    try:
        up2 = _cv2.resize(frame, (w * 2, h * 2))
        res2 = _yolo(up2, verbose=False, imgsz=640, conf=conf)
        for box in res2[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append((
                int(bx1 / 2), int(by1 / 2),
                int(bx2 / 2), int(by2 / 2)
            ))
    except Exception:
        pass

    # ── 3. 4x upscale — catches faces at 5-7 meters ───────────────────
    try:
        up4 = _cv2.resize(frame, (w * 4, h * 4))
        res4 = _yolo(up4, verbose=False, imgsz=640, conf=conf)
        for box in res4[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append((
                int(bx1 / 4), int(by1 / 4),
                int(bx2 / 4), int(by2 / 4)
            ))
    except Exception:
        pass

    if not all_boxes:
        return []

    # ── 4. IoU deduplication ──────────────────────────────────────────
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    kept = []
    used = [False] * len(all_boxes)
    for i, box_a in enumerate(all_boxes):
        if used[i]:
            continue
        for j in range(i + 1, len(all_boxes)):
            if not used[j] and _iou(box_a, all_boxes[j]) > 0.35:
                used[j] = True
        kept.append(box_a)
        used[i] = True

    return kept


class CaptureWorker(QThread):
    recognized  = pyqtSignal(list)
    frame_ready = pyqtSignal(QImage, dict)
    camera_ok   = pyqtSignal(bool)

    def __init__(self, interval_minutes, session_id, enrolled_students, camera_index=0):
        super().__init__()
        self.interval_minutes   = interval_minutes
        self.session_id         = session_id
        self.enrolled_students  = enrolled_students
        self._camera_index      = camera_index
        self._running           = True
        self._emb_buffer        = {}
        self._buffer_size       = 1
        self._frame_count       = 0
        self._attendance        = {}
        self._processing_worker = None
        self._recognizing       = False
        self._interval_ms       = 5 * 60 * 1000
        self._timer_reset       = False
        self._capture_active    = False
        self._score_accumulator    = {}
        self._cumulative_threshold = 0.20
        self._last_det_boxes   = []          # cached boxes from last detection
        self._last_det_time    = 0           # timestamp of last detection (ms)
        self._det_interval_ms  = 1500        # run detection every 1500ms
        self._detecting        = False       # prevent overlapping detection threads
        self._recognition_results = __import__('queue').Queue()  # thread-safe result passing
        self._upload_queue = __import__('queue').Queue()         # image paths to process
        self._upload_results = __import__('queue').Queue()       # results from image processing

    def set_interval(self, ms):
        """Change capture interval in milliseconds while running."""
        self._interval_ms = ms

    def reset_timer(self):
        """Reset capture timer to zero — next capture fires after interval."""
        self._timer_reset = True

    def run(self):
        import time

        try:
            import cv2 as _cv2
            import numpy as _np
        except Exception as e:
            print(f"[AIAS] cv2/numpy not available: {e}")
            return

        _yolo, _arcface, _ai_ok = _ensure_ai_models(det_size=(1280, 1280))
        if _ai_ok:
            print("[AIAS] CaptureWorker using pre-loaded AI models")

        try:
            cap = _cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                cap = None
        except Exception as e:
            print(f"[AIAS] Camera init failed: {e}")
            cap = None

        self.camera_ok.emit(cap is not None)

        last_capture = int(time.time() * 1000)

        while self._running:
            # Process any pending image upload requests in this thread (ONNX-safe)
            try:
                while not self._upload_queue.empty():
                    _img_path = self._upload_queue.get_nowait()
                    try:
                        import cv2 as _cv2u
                        import numpy as _npu
                        _imgu = _cv2u.imread(_img_path)
                        if _imgu is not None and _ai_ok:
                            _boxes = _detect_faces_multiscale(_cv2u, _npu, _yolo, _imgu, conf=0.25)
                            _recognized = {}
                            for (_ux1, _uy1, _ux2, _uy2) in _boxes:
                                try:
                                    _ux1,_uy1,_ux2,_uy2 = int(_ux1),int(_uy1),int(_ux2),int(_uy2)
                                    if _ux2<=_ux1 or _uy2<=_uy1: continue
                                    _fc = _imgu[_uy1:_uy2,_ux1:_ux2]
                                    _fc = upscale_face_crop(_fc)
                                    _fc = apply_clahe(_fc)
                                    _fc_rgb = _cv2u.cvtColor(_fc, _cv2u.COLOR_BGR2RGB)
                                    _faces = _arcface.get(_fc_rgb)
                                    if not _faces: _faces = _arcface.get(_imgu)
                                    if not _faces: continue
                                    for _face in _faces:
                                        _emb = _face.embedding
                                        if _emb is None: continue
                                        _n = _npu.linalg.norm(_emb)
                                        if _n == 0: continue
                                        _ne = _emb / _n
                                        _best_id, _best_score, _best_name = None, 0.0, None
                                        for _stu in self.enrolled_students:
                                            _db = _stu.get("Embedding", _stu.get("FaceEmbedding", []))
                                            if not _db: continue
                                            _st = _npu.array(_db, dtype=_npu.float32)
                                            if _st.ndim != 1 or _st.shape[0] != _ne.shape[0]: continue
                                            _sn = _npu.linalg.norm(_st)
                                            if _sn < 1e-6: continue
                                            _sc = float(_npu.dot(_ne, _st/_sn))
                                            if _sc > _best_score:
                                                _best_score = _sc
                                                _best_id = str(_stu.get("StudentID",""))
                                                _best_name = _stu.get("FullName","")
                                        if _best_id and _best_score >= 0.22 and _best_id not in _recognized:
                                            _recognized[_best_id] = {"name": _best_name, "similarity": _best_score}
                                except Exception: continue
                            self._upload_results.put(_recognized)
                        else:
                            self._upload_results.put({})
                    except Exception as _ue:
                        print(f"[AIAS] Upload processing error: {_ue}")
                        self._upload_results.put({})
            except Exception: pass

            if cap:
                ret, frame = cap.read()
            else:
                ret = False

            if ret:
                # ── Draw face bounding boxes on every frame ──
                if _ai_ok:
                    # ── Run detection in background every 500ms ──
                    _now_ms = int(time.time() * 1000)
                    if (not self._detecting and not self._recognizing and
                            _now_ms - self._last_det_time >= self._det_interval_ms):
                        self._detecting = True
                        self._last_det_time = _now_ms
                        _det_frame = frame.copy()
                        def _run_det(_f=_det_frame):
                            try:
                                boxes = _detect_faces_multiscale(_cv2, _np, _yolo, _f, conf=0.25)
                                self._last_det_boxes = boxes
                            except Exception:
                                pass
                            finally:
                                self._detecting = False
                        _threading.Thread(target=_run_det, daemon=True).start()

                    # ── Draw cached boxes on current frame (non-blocking) ──
                    for (_bx1, _by1, _bx2, _by2) in self._last_det_boxes:
                        try:
                            _cv2.rectangle(frame, (_bx1, _by1), (_bx2, _by2), (0, 220, 100), 2)
                            _cv2.putText(
                                frame, "Detecting...",
                                (_bx1, max(_by1 - 8, 12)),
                                _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 220, 100), 1, _cv2.LINE_AA
                            )
                        except Exception:
                            pass

                rgb  = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

                # Handle timer reset from user clicking "Start Capture"
                if self._timer_reset:
                    last_capture = int(time.time() * 1000)
                    self._timer_reset = False
                    self._capture_active = True

                interval_ms = self._interval_ms
                now_ms = int(time.time() * 1000)
                if self._capture_active and now_ms - last_capture >= interval_ms and _ai_ok:
                    last_capture = now_ms
                    # Non-blocking recognition — pass current frame directly
                    if not self._recognizing:
                        self._recognizing = True
                        _frame_copy = frame.copy()  # snapshot current frame — no cap.read() conflict
                        _yolo_ref   = _yolo
                        _arc_ref    = _arcface

                        def _run_recognition(_f=_frame_copy):
                            try:
                                result = self._process_frame(_f, _cv2, _np, _yolo_ref, _arc_ref)
                                if result:
                                    self._recognition_results.put(list(result.keys()))
                            except Exception as e:
                                print(f"[AIAS] Recognition error: {e}")
                            finally:
                                self._recognizing = False

                        _threading.Thread(target=_run_recognition, daemon=True).start()
                else:
                    self.frame_ready.emit(qimg, {})

            self._frame_count += 1
            if cap:
                self.msleep(33)
            else:
                self.msleep(500)

        if cap:
            cap.release()

    def _process_frame(self, frame, _cv2, _np, _yolo, _arcface):
        import ai_models as _aim
        recognized_this_frame = {}
        ms_boxes = _detect_faces_multiscale(_cv2, _np, _yolo, frame, conf=0.25)
        if not ms_boxes:
            return recognized_this_frame

        for (x1, y1, x2, y2) in ms_boxes:
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop_raw = frame[y1:y2, x1:x2]
                # Apply GFPGAN enhancement if available
                try:
                    if _aim._gfpgan_model is not None and face_crop_raw.shape[0] > 10 and face_crop_raw.shape[1] > 10:
                        _, _, gfpgan_output = _aim._gfpgan_model.enhance(
                            face_crop_raw,
                            has_aligned=False,
                            only_center_face=True,
                            paste_back=True
                        )
                        if gfpgan_output is not None:
                            face_crop_raw = gfpgan_output
                except Exception:
                    pass

                face_crop = upscale_face_crop(face_crop_raw)
                face_crop = apply_clahe(face_crop)
                face_crop_rgb = _cv2.cvtColor(face_crop, _cv2.COLOR_BGR2RGB)

                try:
                    dynamic_thresh = compute_dynamic_threshold(face_crop_raw)
                except Exception as e:
                    print(f"[AIAS] Dynamic threshold error: {e}")
                    dynamic_thresh = config.RECOGNITION_THRESHOLD

                faces = _arcface.get(face_crop_rgb)
                if not faces:
                    faces = _arcface.get(frame)
                if not faces:
                    continue

                for face in faces:
                    try:
                        embedding = face.embedding
                        if embedding is None:
                            continue

                        norm = _np.linalg.norm(embedding)
                        if norm == 0:
                            continue
                        normed_emb = embedding / norm

                        # position key based on YOLO bbox center in frame coords
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        pos_key = f"{int(cx/50)}_{int(cy/50)}"

                        # accumulate embedding in buffer
                        if pos_key not in self._emb_buffer:
                            self._emb_buffer[pos_key] = []
                        self._emb_buffer[pos_key].append(normed_emb)

                        # wait until buffer is full before matching
                        if len(self._emb_buffer[pos_key]) < self._buffer_size:
                            continue

                        avg_emb = _np.mean(self._emb_buffer[pos_key], axis=0)
                        avg_emb = avg_emb / _np.linalg.norm(avg_emb)
                        self._emb_buffer[pos_key] = []  # reset for next window

                        best_match = None
                        best_score = 0.0
                        best_name  = None

                        for stu in self.enrolled_students:
                            db_emb = stu.get("Embedding", stu.get("FaceEmbedding", []))
                            if not db_emb:
                                continue
                            stored = _np.array(db_emb, dtype=_np.float32)
                            if stored.ndim != 1 or stored.shape[0] != avg_emb.shape[0]:
                                continue
                            stored_norm = _np.linalg.norm(stored)
                            if stored_norm < 1e-6:
                                continue
                            s_buf = float(_np.dot(avg_emb, stored / stored_norm))
                            stu_id = str(stu.get("StudentID", ""))

                            s_final = s_buf

                            if s_final > best_score:
                                best_score = s_final
                                best_match = stu.get("StudentID")
                                best_name  = stu.get("FullName", "")

                        print(f"[AIAS] Face {pos_key}: score={best_score:.3f} thresh={dynamic_thresh:.3f}")
                        # Instant recognition — no accumulation needed
                        INSTANT_THRESHOLD = 0.22

                        if best_match is not None and best_score >= INSTANT_THRESHOLD:
                            sid_str = str(best_match)
                            if sid_str in self._attendance:
                                continue
                            matched = True
                            print(f"[AIAS] Matched {sid_str}: score={best_score:.3f} ✅")
                        else:
                            matched = False

                        if matched:
                            sid = str(best_match)
                            recognized_this_frame[sid] = {
                                "name":       best_name,
                                "similarity": best_score,
                                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                            }
                            # Draw green box with student name on the recognition frame
                            _cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
                            _label = str(best_name).split()[0] if best_name else ""
                            if _label:
                                _cv2.putText(
                                    frame, _label,
                                    (x1, max(y1 - 8, 12)),
                                    _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 220, 100), 2, _cv2.LINE_AA
                                )
                    except Exception as _fe:
                        print(f"[AIAS] face error: {_fe}")
                        continue

            except Exception as _be:
                print(f"[AIAS] box error: {_be}")
                continue

        return recognized_this_frame

    def stop(self):
        self._running = False
        self._score_accumulator.clear()
        self.wait()


def _augment_image(img_bgr):
    """
    Generate 8 augmented variants of a BGR face image.
    Fast, in-memory only — no disk writes.
    Returns a list of BGR images (the 8 augmented variants, NOT including the original).
    """
    import cv2
    import numpy as np
    import random

    variants = []
    h, w = img_bgr.shape[:2]

    # 1. Horizontal flip
    variants.append(cv2.flip(img_bgr, 1))

    # 2. Brightness +30
    bright = np.clip(img_bgr.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    variants.append(bright)

    # 3. Brightness -30
    dark = np.clip(img_bgr.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    variants.append(dark)

    # 4. Slight rotation +10 degrees
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)
    variants.append(cv2.warpAffine(img_bgr, M, (w, h)))

    # 5. Slight rotation -10 degrees
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -10, 1.0)
    variants.append(cv2.warpAffine(img_bgr, M, (w, h)))

    # 6. Gaussian blur (simulate soft focus)
    variants.append(cv2.GaussianBlur(img_bgr, (3, 3), 0))

    # 7. Contrast adjustment (CLAHE already applied later, so light stretch here)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    variants.append(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))

    # 8. Small random crop + resize back
    crop_margin = max(4, int(min(h, w) * 0.08))
    y1 = random.randint(0, crop_margin)
    x1 = random.randint(0, crop_margin)
    y2 = h - random.randint(0, crop_margin)
    x2 = w - random.randint(0, crop_margin)
    if y2 > y1 + 10 and x2 > x1 + 10:
        cropped = img_bgr[y1:y2, x1:x2]
        variants.append(cv2.resize(cropped, (w, h)))
    else:
        variants.append(cv2.flip(img_bgr, 0))  # fallback: vertical flip

    return variants  # always exactly 8 items


class EmbeddingWorker(QThread):
    """Worker thread to extract face embedding without blocking the UI."""
    finished = pyqtSignal(list)   # embedding list
    error    = pyqtSignal(str)    # error message

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

    def run(self):
        try:
            import cv2
            import numpy as np

            yolo, arcface, ok = _ensure_ai_models(det_size=(640, 640))
            if not ok:
                self.error.emit("AI models failed to load. Check that insightface and ultralytics are installed.")
                return

            embeddings = []
            for image_path in self.image_paths:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img = apply_clahe(img)

                results = yolo(img, verbose=False)
                boxes = results[0].boxes
                if len(boxes) == 0:
                    continue

                x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy().astype(int)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop = img[y1:y2, x1:x2]
                face_crop = upscale_face_crop(face_crop, target_size=112)

                faces = arcface.get(face_crop)
                if not faces:
                    faces = arcface.get(img)
                if not faces:
                    continue

                raw = faces[0].embedding
                embeddings.append(raw / np.linalg.norm(raw))

                # ── Auto-augmentation (in-memory, no disk writes) ──
                # For each original photo, generate 8 variants and embed them
                aug_variants = _augment_image(face_crop)
                for aug_img in aug_variants:
                    try:
                        aug_clahe = apply_clahe(aug_img)
                        aug_faces = arcface.get(aug_clahe)
                        if not aug_faces:
                            aug_faces = arcface.get(aug_img)
                        if aug_faces:
                            aug_raw = aug_faces[0].embedding
                            embeddings.append(aug_raw / np.linalg.norm(aug_raw))
                    except Exception:
                        pass  # skip failed augmented variants silently

            if not embeddings:
                self.error.emit("No face detected in any of the selected photos.")
                return

            avg = np.mean(np.array(embeddings), axis=0)
            embedding = (avg / np.linalg.norm(avg)).tolist()
            self.finished.emit(embedding)

        except Exception as e:
            self.error.emit(str(e))


class BatchEmbedWorker(QThread):
    """Processes a root folder where each subfolder name = StudentID.
    Embeds all images per student and saves to MongoDB."""
    progress = pyqtSignal(str)          # status message per student
    finished = pyqtSignal(int, int)     # (processed, skipped)

    def __init__(self, root_folder, data_folder):
        super().__init__()
        self.root_folder = root_folder
        self.data_folder = data_folder  # where to copy images

    def run(self):
        import cv2
        import numpy as np
        import shutil
        import ai_models as _aim

        yolo, arcface, ok = _ensure_ai_models(det_size=(640, 640))
        if not ok:
            self.progress.emit("ERROR: AI models failed to load.")
            self.finished.emit(0, 0)
            return

        processed = skipped = 0
        try:
            entries = [e for e in os.scandir(self.root_folder) if e.is_dir()]
        except Exception as e:
            self.progress.emit(f"ERROR reading folder: {e}")
            self.finished.emit(0, 0)
            return

        for entry in entries:
            sid = entry.name.strip()
            if not sid:
                continue

            student = db["Students"].find_one({"StudentID": sid}) if DB_CONNECTED else None
            if student is None:
                self.progress.emit(f"⚠ Skipped '{sid}' — not found in database")
                skipped += 1
                continue

            images = [
                f for f in os.listdir(entry.path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not images:
                self.progress.emit(f"⚠ Skipped '{sid}' — no images in folder")
                skipped += 1
                continue

            def _embed_one(img_file):
                try:
                    img_path = os.path.join(entry.path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        return [], None
                    img = apply_clahe(img)
                    results = yolo(img, verbose=False)
                    boxes = results[0].boxes
                    if len(boxes) == 0:
                        return [], None
                    x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy().astype(int)
                    face_crop = img[y1:y2, x1:x2]
                    face_crop = upscale_face_crop(face_crop, target_size=112)
                    faces = arcface.get(face_crop)
                    if not faces:
                        faces = arcface.get(img)
                    if not faces:
                        return [], None
                    raw = faces[0].embedding
                    embs = [raw / np.linalg.norm(raw)]

                    # ── Auto-augmentation (in-memory, no disk writes) ──
                    aug_variants = _augment_image(face_crop)
                    for aug_img in aug_variants:
                        try:
                            aug_clahe = apply_clahe(aug_img)
                            aug_faces = arcface.get(aug_clahe)
                            if not aug_faces:
                                aug_faces = arcface.get(aug_img)
                            if aug_faces:
                                aug_raw = aug_faces[0].embedding
                                embs.append(aug_raw / np.linalg.norm(aug_raw))
                        except Exception:
                            pass

                    # Copy image to Data folder
                    dest_dir = os.path.join(self.data_folder, sid)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest = os.path.join(dest_dir, img_file)
                    if os.path.abspath(img_path) != os.path.abspath(dest):
                        try:
                            shutil.copy2(img_path, dest)
                        except Exception as e:
                            print(f"[AIAS] Copy error for {img_file}: {e}")
                    return embs, os.path.abspath(dest)
                except Exception:
                    return [], None

            with ThreadPoolExecutor(max_workers=4) as ex:
                results_parallel = list(ex.map(_embed_one, images))
            all_embeddings = [emb for r in results_parallel for emb in r[0]]
            dest_paths = [r[1] for r in results_parallel if r[1] is not None]

            if not all_embeddings:
                self.progress.emit(f"⚠ Skipped '{sid}' — no face detected in any image")
                skipped += 1
                continue

            mean_emb = np.mean(np.array(all_embeddings), axis=0)
            norm_emb = (mean_emb / np.linalg.norm(mean_emb)).tolist()

            if DB_CONNECTED:
                db["Students"].update_one(
                    {"StudentID": sid},
                    {"$set": {"FaceEmbedding": norm_emb, "ImagePaths": dest_paths}}
                )

            self.progress.emit(f"✔ '{sid}' — embedded from {len(all_embeddings)} image(s)")
            processed += 1

        self.finished.emit(processed, skipped)


class ImageAttendanceWorker(QThread):
    """Process a single image through YOLO+ArcFace and return recognized student IDs."""
    finished = pyqtSignal(dict, object)
    error    = pyqtSignal(str)

    def __init__(self, image_path, enrolled_students):
        super().__init__()
        self.image_path        = image_path
        self.enrolled_students = enrolled_students

    def run(self):
        try:
            import cv2
            import numpy as np
        except Exception as e:
            self.error.emit(f"cv2/numpy not available: {e}")
            return

        import ai_models as _aim

        # Use already-loaded global models directly — do NOT call _ensure_ai_models
        # from a background thread while CaptureWorker may be stopping, as this
        # causes a lock conflict / silent crash on Windows.
        _yolo    = _aim.yolo_model
        _arcface = _aim.arcface_model
        if _yolo is None or _arcface is None:
            self.error.emit("AI models are not loaded yet. Please try again.")
            return

        try:
            img = cv2.imread(self.image_path)
        except Exception as _read_err:
            import traceback
            print(f"[AIAS] ImageAttendanceWorker crash: {_read_err}")
            traceback.print_exc()
            self.error.emit(str(_read_err))
            return
        if img is None:
            self.error.emit(f"Could not read image: {self.image_path}")
            return

        ms_boxes = _detect_faces_multiscale(cv2, np, _yolo, img, conf=0.25)
        recognized = {}

        for (x1, y1, x2, y2) in ms_boxes:
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_crop_raw = img[y1:y2, x1:x2]
                try:
                    if _aim._gfpgan_model is not None and face_crop_raw.shape[0] > 10:
                        _, _, gfpgan_out = _aim._gfpgan_model.enhance(
                            face_crop_raw, has_aligned=False,
                            only_center_face=True, paste_back=True
                        )
                        if gfpgan_out is not None:
                            face_crop_raw = gfpgan_out
                except Exception:
                    pass

                face_crop = upscale_face_crop(face_crop_raw)
                face_crop = apply_clahe(face_crop)
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                faces = _arcface.get(face_crop_rgb)
                if not faces:
                    faces = _arcface.get(img)
                if not faces:
                    continue

                for face in faces:
                    emb = face.embedding
                    if emb is None:
                        continue
                    norm = np.linalg.norm(emb)
                    if norm == 0:
                        continue
                    normed_emb = emb / norm

                    best_match = None
                    best_score = 0.0
                    best_name  = None
                    for stu in self.enrolled_students:
                        db_emb = stu.get("Embedding", stu.get("FaceEmbedding", []))
                        if not db_emb:
                            continue
                        stored = np.array(db_emb, dtype=np.float32)
                        if stored.ndim != 1 or stored.shape[0] != normed_emb.shape[0]:
                            continue
                        stored_norm = np.linalg.norm(stored)
                        if stored_norm < 1e-6:
                            continue
                        score = float(np.dot(normed_emb, stored / stored_norm))
                        if score > best_score:
                            best_score = score
                            best_match = str(stu.get("StudentID", ""))
                            best_name  = stu.get("FullName", "")

                    INSTANT_THRESHOLD = 0.22
                    if best_match and best_score >= INSTANT_THRESHOLD and best_match not in recognized:
                        recognized[best_match] = {"name": best_name, "similarity": best_score}
            except Exception as e:
                print(f"[AIAS] Image attendance face error: {e}")

        self.finished.emit(recognized, ms_boxes)


class LiveSessionWindow(QWidget):
    def __init__(self, main_win, session_id=None, enrolled_students=None, course_data=None, camera_index=0):
        super().__init__()
        self.main_win          = main_win
        self.session_id        = session_id
        self.enrolled_students = enrolled_students or []
        self.course_data       = course_data or {}
        self._attendance       = {}
        self._session_start    = datetime.now()
        self._worker           = None
        self._interval_minutes = 5
        self._camera_index     = camera_index
        self._capture_running  = False
        self._session_ended    = False
        self.setWindowTitle("AIAS — Live Session")
        import os
        from PyQt5.QtGui import QIcon
        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aias_icon.ico")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        self.setMinimumSize(1100, 700)
        self.resize(1100, 700)
        self.setStyleSheet(f"background:{config.BG};")
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        instr    = getattr(self.main_win, "instructor_data", {})
        fullname = instr.get("fullname", "Dr. Ahmed")
        parts    = fullname.split()
        initials = (parts[0][0] + parts[-1][0]).upper() if len(parts) >= 2 else fullname[:2].upper()
        sidebar, s_lay = make_sidebar_base(initials, fullname, "Instructor")

        self._cam_preview = QLabel("Camera feed")
        self._cam_preview.setFixedHeight(120)
        self._cam_preview.setAlignment(Qt.AlignCenter)
        self._cam_preview.setStyleSheet(
            f"background:#D1D5DB; border-radius:8px; border:none;"
            f"color:{config.TEXT_GRAY}; font-size:12px;"
        )
        s_lay.addWidget(self._cam_preview)
        s_lay.addSpacing(12)

        # ── Lecture duration ──────────────────────────
        lec_label = QLabel("Lecture duration")
        lec_label.setStyleSheet("font-size:12px; color: #888; font-weight:500;")

        lec_h_layout = QHBoxLayout()
        lec_h_layout.setSpacing(4)

        self._lec_hours = QLineEdit("1")
        self._lec_hours.setFixedWidth(45)
        self._lec_hours.setFixedHeight(36)
        self._lec_hours.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lec_hours.setStyleSheet("""
            QLineEdit {
                background: #1a1a2e; color: white;
                border: 1px solid #2a2a4a; border-radius: 8px;
                font-size: 16px; font-weight: bold;
            }
        """)

        colon_lbl = QLabel(":")
        colon_lbl.setStyleSheet("color: white; font-size:18px; font-weight:bold;")
        colon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lec_mins = QLineEdit("30")
        self._lec_mins.setFixedWidth(45)
        self._lec_mins.setFixedHeight(36)
        self._lec_mins.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lec_mins.setStyleSheet("""
            QLineEdit {
                background: #1a1a2e; color: white;
                border: 1px solid #2a2a4a; border-radius: 8px;
                font-size: 16px; font-weight: bold;
            }
        """)

        from PyQt5.QtGui import QIntValidator as _IVLec
        self._lec_hours.setValidator(_IVLec(0, 23))
        self._lec_mins.setValidator(_IVLec(0, 59))

        lec_h_layout.addWidget(self._lec_hours)
        lec_h_layout.addWidget(colon_lbl)
        lec_h_layout.addWidget(self._lec_mins)
        lec_h_layout.addStretch()

        lec_container = QVBoxLayout()
        lec_container.setSpacing(6)
        lec_container.addWidget(lec_label)
        lec_container.addLayout(lec_h_layout)

        s_lay.addLayout(lec_container)
        s_lay.addSpacing(12)

        # ── Capture interval ──────────────────────────
        interval_label = QLabel("Capture interval")
        interval_label.setStyleSheet("font-size:12px; color: #888; font-weight:500;")

        self._interval_input = QLineEdit("5")
        self._interval_input.setFixedHeight(38)
        self._interval_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._interval_input.setStyleSheet("""
            QLineEdit {
                background: #1a1a2e;
                color: white;
                border: 1px solid #2a2a4a;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 1px solid #1B5E35;
            }
        """)
        from PyQt5.QtGui import QIntValidator
        self._interval_input.setValidator(QIntValidator(1, 9999))

        self._interval_unit = "M"
        unit_layout = QHBoxLayout()
        unit_layout.setSpacing(6)
        self._unit_buttons = {}
        for unit, label in [("S", "S"), ("M", "M"), ("H", "H")]:
            btn = QPushButton(label)
            btn.setFixedHeight(34)
            btn.setCheckable(True)
            btn.setChecked(unit == "M")
            btn.setStyleSheet("""
                QPushButton {
                    background: #1a1a2e;
                    color: #888;
                    border: 1px solid #2a2a4a;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background: #1B5E35;
                    color: white;
                    border: 1px solid #1B5E35;
                }
                QPushButton:hover:!checked {
                    background: #2a2a4a;
                    color: white;
                }
            """)
            self._unit_buttons[unit] = btn
            unit_layout.addWidget(btn)

        def _set_unit(u):
            self._interval_unit = u
            for k, b in self._unit_buttons.items():
                b.setChecked(k == u)
            _update_interval()

        def _update_interval():
            try:
                val = int(self._interval_input.text() or "5")
            except ValueError:
                val = 5
            multipliers = {"S": 1000, "M": 60000, "H": 3600000}
            ms = val * multipliers.get(self._interval_unit, 60000)
            if hasattr(self, '_worker') and self._worker:
                self._worker.set_interval(ms)

        for unit, btn in self._unit_buttons.items():
            btn.clicked.connect(lambda _, u=unit: _set_unit(u))
        self._interval_input.textChanged.connect(lambda _: _update_interval())

        self._btn_start_capture = QPushButton("▶  Start Capture")
        self._btn_start_capture.setFixedHeight(42)
        self._btn_start_capture.setStyleSheet("""
            QPushButton {
                background: #1B5E35;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2E7D4F;
            }
            QPushButton:pressed {
                background: #145228;
            }
        """)
        self._btn_start_capture.clicked.connect(self._start_capture_timer)

        interval_container = QVBoxLayout()
        interval_container.setSpacing(6)
        interval_container.addWidget(interval_label)
        interval_container.addWidget(self._interval_input)
        interval_container.addLayout(unit_layout)
        interval_container.addWidget(self._btn_start_capture)

        s_lay.addLayout(interval_container)
        self._start_capture(5)
        s_lay.addSpacing(10)

        # ── Upload Image attendance ───────────────────
        self._upload_img_btn = QPushButton("📷  Upload Image")
        self._upload_img_btn.setFixedHeight(40)
        self._upload_img_btn.setToolTip(
            "Upload a photo to recognize faces and mark attendance instantly"
        )
        self._upload_img_btn.setStyleSheet("""
            QPushButton {
                background: #1a3a2a;
                color: #4ecf8e;
                border: 1px solid #2a5a3a;
                border-radius: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2E7D4F;
                color: white;
            }
            QPushButton:pressed {
                background: #145228;
                color: white;
            }
            QPushButton:disabled {
                background: #111118;
                color: #555;
            }
        """)
        self._upload_img_btn.setCursor(Qt.PointingHandCursor)
        self._upload_img_btn.clicked.connect(self._upload_image_attendance)
        s_lay.addWidget(self._upload_img_btn)

        s_lay.addStretch()

        end_btn = QPushButton("End session")
        end_btn.setFixedHeight(40)
        end_btn.setStyleSheet(
            "background:#DC2626; color:white; border-radius:6px;"
            "padding:8px 16px; font-size:13px; font-weight:700;"
        )
        end_btn.setCursor(Qt.PointingHandCursor)
        end_btn.clicked.connect(self._end_session)
        s_lay.addWidget(end_btn)

        main_w = QWidget()
        main_w.setStyleSheet(f"background:{config.BG};")
        m_lay = QVBoxLayout(main_w)
        m_lay.setContentsMargins(32, 28, 32, 24)
        m_lay.setSpacing(18)

        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame {{ background:{config.PRIMARY_LIGHT}; border:1.5px solid {config.PRIMARY}; border-radius:8px; }}"
        )
        bar_lay = QVBoxLayout(bar)
        bar_lay.setContentsMargins(16, 8, 16, 8)
        bar_lay.setSpacing(4)

        self._lbl_lecture_timer = QLabel("Lecture ends in   --:--")
        self._lbl_lecture_timer.setStyleSheet(
            "font-size:22px; font-weight:bold; color:#1B5E35; background:transparent; border:none;"
        )
        self._lbl_lecture_timer.setAlignment(Qt.AlignCenter)

        self._lbl_capture_timer = QLabel("Next capture in   --:--")
        self._lbl_capture_timer.setStyleSheet(
            "font-size:14px; color:#888; background:transparent; border:none;"
        )
        self._lbl_capture_timer.setAlignment(Qt.AlignCenter)

        bar_lay.addWidget(self._lbl_lecture_timer)
        bar_lay.addWidget(self._lbl_capture_timer)
        m_lay.addWidget(bar)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)

        def _make_live_card(title, color):
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame {{ background:{config.WHITE}; border:1px solid {config.BORDER}; border-radius:8px; }}"
            )
            frame.setFixedHeight(82)
            frame.setMinimumWidth(130)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(16, 12, 16, 12)
            fl.setSpacing(2)
            v = QLabel("0")
            v.setStyleSheet(
                f"font-size:26px; font-weight:800; color:{color}; border:none; background:transparent;"
            )
            t = QLabel(title)
            t.setStyleSheet(f"font-size:11px; color:{config.TEXT_GRAY}; border:none; background:transparent;")
            fl.addWidget(v)
            fl.addWidget(t)
            return frame, v

        present_card, self._present_val = _make_live_card("Present", "#22C55E")
        late_card,    self._late_val    = _make_live_card("Late",    "#F59E0B")
        absent_card,  self._absent_val  = _make_live_card("Absent",  "#EF4444")
        stats_row.addWidget(present_card)
        stats_row.addWidget(late_card)
        stats_row.addWidget(absent_card)
        stats_row.addStretch()
        m_lay.addLayout(stats_row)

        # Rec 3: persistent camera-unavailable banner (hidden until camera fails)
        self._cam_banner = QFrame()
        self._cam_banner.setStyleSheet(
            "QFrame { background:#FFFBEB; border:1px solid #FCD34D; border-radius:8px; }"
        )
        self._cam_banner.setFixedHeight(44)
        self._cam_banner.setVisible(False)
        _ban_lay = QHBoxLayout(self._cam_banner)
        _ban_lay.setContentsMargins(16, 0, 16, 0)
        _ban_lbl = QLabel("⚠  Camera unavailable — attendance must be entered manually.")
        _ban_lbl.setStyleSheet(
            "font-size:12px; font-weight:600; color:#92400E; background:transparent; border:none;"
        )
        _ban_lay.addWidget(_ban_lbl)
        m_lay.addWidget(self._cam_banner)

        self._live_tbl = make_table(
            ["Student ID", "Full Name", "First seen", "Last seen", "Status", ""],
            self.enrolled_students,
            col_widths={0: 140, 2: 100, 3: 100, 4: 130, 5: 70},
            stretch_col=1,
        )
        m_lay.addWidget(self._live_tbl)

        root.addWidget(sidebar)
        root.addWidget(main_w)

        self._remaining_seconds = self._interval_minutes * 60
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

        self._lecture_remaining = 0
        self._lecture_timer_qt  = QTimer()
        self._lecture_timer_qt.setInterval(1000)
        self._lecture_timer_qt.timeout.connect(self._tick_lecture_timer)

        self._capture_countdown_remaining = 0
        self._capture_countdown_qt = QTimer()
        self._capture_countdown_qt.setInterval(1000)
        self._capture_countdown_qt.timeout.connect(self._tick_capture_timer)

        self._refresh_table()
        self._start_capture(5)

    def _start_capture(self, interval_minutes):
        if self._worker:
            self._worker.stop()
        self._worker = CaptureWorker(interval_minutes, self.session_id, self.enrolled_students, camera_index=self._camera_index)
        self._worker.recognized.connect(self._on_recognized)
        self._worker.frame_ready.connect(self._update_preview)
        self._worker.camera_ok.connect(self._on_camera_status)
        self._worker.start()

        # ── Poll recognition queue from main thread every 200ms ──
        if not hasattr(self, '_recognition_poll_timer'):
            from PyQt5.QtCore import QTimer
            self._recognition_poll_timer = QTimer(self)
            self._recognition_poll_timer.timeout.connect(self._poll_recognition_queue)
        self._recognition_poll_timer.start(50)

    def _poll_recognition_queue(self):
        """Drain the recognition results queue from the main thread."""
        if not self._worker:
            return
        try:
            q = self._worker._recognition_results
            found = False
            while not q.empty():
                sids = q.get_nowait()
                self._on_recognized(sids)
                found = True
            if found:
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
        except Exception:
            pass

    def _tick(self):
        if self._remaining_seconds > 0:
            self._remaining_seconds -= 1
        else:
            self._remaining_seconds = self._interval_minutes * 60

    def _tick_lecture_timer(self):
        if self._lecture_remaining > 0:
            self._lecture_remaining -= 1
            h = self._lecture_remaining // 3600
            m = (self._lecture_remaining % 3600) // 60
            s = self._lecture_remaining % 60
            self._lbl_lecture_timer.setText(f"Lecture ends in   {h}:{m:02d}:{s:02d}")
        else:
            self._lecture_timer_qt.stop()
            self._lbl_lecture_timer.setText("Lecture ended")

    def _tick_capture_timer(self):
        if self._capture_countdown_remaining > 0:
            self._capture_countdown_remaining -= 1
            s_total = self._capture_countdown_remaining
            m = s_total // 60
            s = s_total % 60
            self._lbl_capture_timer.setText(f"Next capture in   {m}:{s:02d}")
        else:
            self._lbl_capture_timer.setText("Capturing...")
            # Reset countdown for next capture
            try:
                val = int(self._interval_input.text() or "5")
            except Exception:
                val = 5
            multipliers = {"S": 1, "M": 60, "H": 3600}
            self._capture_countdown_remaining = val * multipliers.get(
                getattr(self, "_interval_unit", "M"), 60
            )

    def _on_recognized(self, student_ids):
        now     = datetime.now()
        elapsed = (now - self._session_start).total_seconds() / 60

        for sid in student_ids:
            sid = str(sid)
            if sid not in self._attendance:
                status = "Present" if elapsed <= config.LATE_CUTOFF_MINUTES else "Late"
                self._attendance[sid] = {
                    "status":        status,
                    "first_seen":    now.strftime("%H:%M"),
                    "first_seen_dt": now,
                    "last_seen":     now.strftime("%H:%M"),
                    "last_seen_dt":  now,
                }
                if DB_CONNECTED:
                    try:
                        db["AttendanceLogs"].update_one(
                            {"SessionID": self.session_id, "StudentID": sid},
                            {"$set": {
                                "SessionID": self.session_id,
                                "StudentID": sid,
                                "Status":    status,
                                "FirstSeenAt": now,
                                "LastSeenAt":  now,
                            }},
                            upsert=True,
                        )
                    except Exception as e:
                        print(f"DB error: {e}")
            else:
                self._attendance[sid]["last_seen"]    = now.strftime("%H:%M")
                self._attendance[sid]["last_seen_dt"] = now
                if DB_CONNECTED:
                    try:
                        db["AttendanceLogs"].update_one(
                            {"SessionID": self.session_id, "StudentID": sid},
                            {"$set": {"LastSeenAt": now}},
                        )
                    except Exception as e:
                        print(f"DB error: {e}")

        for sid, data in self._attendance.items():
            if sid not in student_ids:
                last = data.get("last_seen_dt") or datetime.strptime(
                    data["last_seen"], "%H:%M"
                ).replace(year=now.year, month=now.month, day=now.day)
                if (now - last).total_seconds() / 60 >= config.EARLY_LEAVE_CUTOFF_MINUTES:
                    self._attendance[sid]["status"] = "Early Leave"

        self._refresh_table()

    def _update_preview(self, qimg, recognized):
        pixmap = QPixmap.fromImage(qimg)
        self._cam_preview.setPixmap(
            pixmap.scaled(
                self._cam_preview.width(), self._cam_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )

    def _refresh_table(self):
        n = len(self.enrolled_students)
        self._live_tbl.setRowCount(n)
        n_present = n_late = n_absent = 0
        for r, stu in enumerate(self.enrolled_students):
            sid  = str(stu.get("StudentID", ""))
            name = stu.get("FullName", "")
            att    = self._attendance.get(sid, {})
            first  = att.get("first_seen", "—")
            last   = att.get("last_seen",  "—")
            status = att.get("status",     "Absent")
            if status in ("Present", "Early Leave"):
                n_present += 1
            elif status == "Late":
                n_late += 1
            elif status == "Absent":
                n_absent += 1
            self._live_tbl.setItem(r, 0, QTableWidgetItem(sid))
            self._live_tbl.setItem(r, 1, QTableWidgetItem(name))
            self._live_tbl.setItem(r, 2, QTableWidgetItem(first))
            self._live_tbl.setItem(r, 3, QTableWidgetItem(last))
            self._live_tbl.setCellWidget(r, 4, make_badge(status))
            edit_btn = QPushButton("Edit")
            edit_btn.setFixedHeight(28)
            edit_btn.setStyleSheet(
                f"background:#E6F1FB; color:#0C447C; border-radius:5px;"
                "font-size:11px; font-weight:600;"
            )
            edit_btn.setCursor(Qt.PointingHandCursor)
            edit_btn.clicked.connect(lambda _, s=sid: self._edit_attendance(s))
            self._live_tbl.setCellWidget(r, 5, edit_btn)
            self._live_tbl.setRowHeight(r, 44)
        self._present_val.setText(str(n_present))
        self._late_val.setText(str(n_late))
        self._absent_val.setText(str(n_absent))

    def _set_interval(self, mins):
        self._interval_minutes = mins
        self._remaining_seconds = mins * 60
        self._start_capture(mins)

    def _start_capture_timer(self):
        if self._capture_running:
            # Stop capture
            self._capture_running = False
            self._btn_start_capture.setText("▶  Start Capture")
            self._btn_start_capture.setStyleSheet("""
                QPushButton {
                    background: #1B5E35; color: white;
                    border: none; border-radius: 10px;
                    font-size: 13px; font-weight: bold;
                }
                QPushButton:hover { background: #2E7D4F; }
            """)
            if hasattr(self, '_worker') and self._worker:
                self._worker._capture_active = False
            self._lecture_timer_qt.stop()
            self._capture_countdown_qt.stop()
            self._lbl_lecture_timer.setText("Lecture ends in   --:--")
            self._lbl_capture_timer.setText("Next capture in   --:--")
        else:
            # Start capture
            self._capture_running = True
            self._btn_start_capture.setText("⏹  Stop Capture")
            self._btn_start_capture.setStyleSheet("""
                QPushButton {
                    background: #8B0000; color: white;
                    border: none; border-radius: 10px;
                    font-size: 13px; font-weight: bold;
                }
                QPushButton:hover { background: #A00000; }
            """)
            try:
                val = int(self._interval_input.text() or "5")
            except ValueError:
                val = 5
            multipliers = {"S": 1000, "M": 60000, "H": 3600000}
            ms = val * multipliers.get(self._interval_unit, 60000)
            if hasattr(self, '_worker') and self._worker:
                self._worker.set_interval(ms)
                self._worker.reset_timer()
                print(f"[AIAS] Capture timer started: every {val}{self._interval_unit}")
            # Start lecture timer
            try:
                lec_h = int(self._lec_hours.text() or "1")
                lec_m = int(self._lec_mins.text() or "30")
            except Exception:
                lec_h, lec_m = 1, 30
            self._lecture_remaining = lec_h * 3600 + lec_m * 60
            self._lecture_timer_qt.start()
            # Start capture countdown
            multipliers_s = {"S": 1, "M": 60, "H": 3600}
            self._capture_countdown_remaining = val * multipliers_s.get(self._interval_unit, 60)
            self._capture_countdown_qt.start()

    def _upload_image_attendance(self):
        """Pick an image, run YOLO+ArcFace, show boxes on preview, then mark attendance."""
        # Stop worker BEFORE opening file dialog — QFileDialog + CaptureWorker = crash on Windows
        if self._worker is not None:
            self._worker._running = False
            self._worker._capture_active = False
            try:
                self._worker.quit()
                self._worker.wait(3000)
            except Exception:
                pass
            self._worker = None

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Attendance Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )

        if not path:
            self._start_capture(self._interval_minutes)
            return

        # Show uploaded image in camera preview area immediately
        try:
            from PyQt5.QtGui import QPixmap
            _px = QPixmap(path)
            if not _px.isNull():
                self._cam_preview.setPixmap(
                    _px.scaled(
                        self._cam_preview.width(), self._cam_preview.height(),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation,
                    )
                )
                self._cam_preview.setText("")
        except Exception:
            pass

        self._upload_img_btn.setEnabled(False)
        self._upload_img_btn.setText("⏳  Processing...")
        QApplication.processEvents()

        self._img_att_worker = ImageAttendanceWorker(path, self.enrolled_students)

        def _on_finished(recognized, ms_boxes):
            # Draw boxes on image: green = recognized, red = unknown
            try:
                import cv2
                from PyQt5.QtGui import QPixmap, QImage
                img = cv2.imread(path)
                if img is not None:
                    recognized_ids = set(recognized.keys())
                    # Build a map of box -> recognized student
                    box_results = []
                    for (x1, y1, x2, y2) in (ms_boxes or []):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        box_results.append((x1, y1, x2, y2))

                    # Color each box based on recognition result
                    for i, (x1, y1, x2, y2) in enumerate(box_results):
                        # Check if any recognized student's bbox matches this box
                        is_recognized = False
                        label = "Unknown"
                        for sid, data in recognized.items():
                            bbox = data.get("bbox", [])
                            if bbox and abs(bbox[0]-x1) < 20 and abs(bbox[1]-y1) < 20:
                                is_recognized = True
                                label = data.get("name", sid).split()[0]
                                break
                        color = (0, 200, 80) if is_recognized else (0, 0, 220)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, max(y1-8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
                    self._cam_preview.setPixmap(
                        QPixmap.fromImage(qimg).scaled(
                            self._cam_preview.width(), self._cam_preview.height(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation,
                        )
                    )
            except Exception as e:
                print(f"[AIAS] Box draw error: {e}")

            self._upload_img_btn.setEnabled(True)
            self._upload_img_btn.setText("📷  Upload Image")
            self._start_capture(self._interval_minutes)
            if recognized:
                self._on_recognized(list(recognized.keys()))
            else:
                QMessageBox.information(
                    self, "No Faces Found",
                    "No enrolled students were recognized in the uploaded image."
                )

        def _on_error(msg):
            self._upload_img_btn.setEnabled(True)
            self._upload_img_btn.setText("📷  Upload Image")
            self._start_capture(self._interval_minutes)
            QMessageBox.critical(self, "Processing Error", msg)

        self._img_att_worker.finished.connect(_on_finished)
        self._img_att_worker.error.connect(_on_error)
        self._img_att_worker.start()

    def _on_camera_status(self, ok):
        if not ok:
            self._cam_preview.setText("⚠ No camera\ndetected")
            self._cam_preview.setStyleSheet(
                "background:#FEE2E2; border-radius:8px; border:1px solid #FCA5A5;"
                "color:#DC2626; font-size:11px; font-weight:600;"
            )
            self._cam_banner.setVisible(True)

    def _edit_attendance(self, sid):
        name = next(
            (s.get("FullName", sid) for s in self.enrolled_students
             if str(s.get("StudentID", "")) == sid),
            sid,
        )
        current_status = self._attendance.get(sid, {}).get("status", "Absent")
        current_note   = self._attendance.get(sid, {}).get("note", "")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Attendance — {name}")
        dlg.setFixedWidth(340)
        dlg.setStyleSheet(f"background:{config.BG};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        title_lbl = QLabel(f"Override status for:\n{name}")
        title_lbl.setStyleSheet(f"font-size:14px; font-weight:700; color:{config.TEXT_DARK};")
        title_lbl.setWordWrap(True)
        lay.addWidget(title_lbl)

        group = QButtonGroup(dlg)
        buttons = {}
        for status in ["Present", "Late", "Early Leave", "Absent"]:
            rb = QRadioButton(status)
            rb.setStyleSheet(f"font-size:13px; color:{config.TEXT_MED};")
            if status == current_status:
                rb.setChecked(True)
            group.addButton(rb)
            buttons[status] = rb
            lay.addWidget(rb)

        note_lbl = QLabel("Note (optional)")
        note_lbl.setStyleSheet(f"font-size:12px; font-weight:600; color:{config.TEXT_MED};")
        note_field = QLineEdit()
        note_field.setPlaceholderText("e.g. Medical excuse, arrived late by bus…")
        note_field.setFixedHeight(36)
        note_field.setText(current_note)
        lay.addWidget(note_lbl)
        lay.addWidget(note_field)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(
            f"background:{config.WHITE}; color:{config.TEXT_MED}; border:1px solid {config.BORDER}; border-radius:6px; padding:7px 14px;"
        )
        cancel.clicked.connect(dlg.reject)
        save = QPushButton("Save")
        save.setStyleSheet(
            f"background:{config.PRIMARY}; color:white; border-radius:6px;"
            "padding:7px 14px; font-weight:700;"
        )
        save.clicked.connect(dlg.accept)
        btn_row.addWidget(cancel)
        btn_row.addStretch()
        btn_row.addWidget(save)
        lay.addLayout(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            return

        new_status = next(
            (s for s, rb in buttons.items() if rb.isChecked()), current_status
        )
        new_note = note_field.text().strip()
        if new_status == current_status and new_note == current_note:
            return

        now = datetime.now()
        if sid not in self._attendance:
            self._attendance[sid] = {
                "status":        new_status,
                "first_seen":    now.strftime("%H:%M"),
                "first_seen_dt": now,
                "last_seen":     now.strftime("%H:%M"),
                "last_seen_dt":  now,
                "note":          new_note,
            }
        else:
            self._attendance[sid]["status"] = new_status
            self._attendance[sid]["note"]   = new_note

        if DB_CONNECTED and self.session_id:
            try:
                db["AttendanceLogs"].update_one(
                    {"SessionID": self.session_id, "StudentID": sid},
                    {"$set": {"Status": new_status, "Note": new_note}},
                    upsert=True,
                )
            except Exception as e:
                print(f"[AIAS] Override error: {e}")

        self._refresh_table()

    def _end_session(self):
        if self._session_ended:
            return
        self._session_ended = True
        from ui_reports import ReportWindow
        self._timer.stop()
        if self._worker:
            self._worker.stop()
        if hasattr(self, '_recognition_poll_timer'):
            self._recognition_poll_timer.stop()

        # Apply Early Leave to any Present/Late student not seen within the cutoff window
        _now = datetime.now()
        for _sid, _data in self._attendance.items():
            if _data.get("status") in ("Present", "Late"):
                _last = _data.get("last_seen_dt")
                if _last and (_now - _last).total_seconds() / 60 >= config.EARLY_LEAVE_CUTOFF_MINUTES:
                    self._attendance[_sid]["status"] = "Early Leave"

        if DB_CONNECTED and self.session_id:
            # Flush final statuses for all students (Early Leave may have just changed some)
            for sid, data in self._attendance.items():
                try:
                    db["AttendanceLogs"].update_one(
                        {"SessionID": self.session_id, "StudentID": sid},
                        {"$set": {"Status": data.get("status", "Absent"),
                                  "Note":   data.get("note", "")}},
                    )
                except Exception as e:
                    print(f"DB error: {e}")
            for stu in self.enrolled_students:
                sid = str(stu.get("StudentID", ""))
                if sid not in self._attendance:
                    try:
                        db["AttendanceLogs"].update_one(
                            {"SessionID": self.session_id, "StudentID": sid},
                            {"$set": {
                                "SessionID": self.session_id,
                                "StudentID": sid,
                                "Status":    "Absent",
                                "FirstSeenAt": None,
                                "LastSeenAt":  None,
                            }},
                            upsert=True,
                        )
                    except Exception as e:
                        print(f"DB error: {e}")
            try:
                db["Sessions"].update_one(
                    {"_id": self.session_id},
                    {"$set": {"EndTime": datetime.now(), "Status": "completed"}},
                )
            except Exception as e:
                print(f"DB error: {e}")

        # Auto-send email silently using per-instructor saved credentials
        _auto_email_ok = None
        if DB_CONNECTED:
            try:
                instr_username = getattr(self.main_win, "instructor_data", {}).get("username", "")
                instr_doc = db["Instructors"].find_one({"Username": instr_username})
                if instr_doc:
                    to_email = instr_doc.get("UniversityEmail", "")
                    if to_email:
                        _auto_email_ok = self._auto_send_report(to_email)
            except Exception as e:
                print(f"[AIAS] Auto-email error: {e}")
                _auto_email_ok = False

        self.hide()
        self._report = ReportWindow(
            self.main_win,
            session_id=self.session_id,
            attendance=self._attendance,
            enrolled_students=self.enrolled_students,
            course_data=self.course_data,
            auto_email_ok=_auto_email_ok,
        )
        self._report.show()

    def _auto_send_report(self, to_email):
        course_title = self.course_data.get("CourseTitle", "Attendance")
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        _success = False
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Attendance Report"
            ws.append(["Student ID", "Full Name", "First Seen", "Last Seen", "Status", "Note"])
            for stu in self.enrolled_students:
                sid = str(stu.get("StudentID", ""))
                rec = self._attendance.get(sid, {})
                first = rec.get("first_seen") or "—"
                last  = rec.get("last_seen")  or "—"
                ws.append([
                    sid,
                    stu.get("FullName", ""),
                    first,
                    last,
                    rec.get("status", "Absent"),
                    rec.get("note",   ""),
                ])
            wb.save(tmp.name)

            att_statuses = {str(s.get("StudentID","")): self._attendance.get(str(s.get("StudentID","")), {}).get("status", "Absent")
                            for s in self.enrolled_students}
            n_present = sum(1 for v in att_statuses.values() if v == "Present")
            n_late    = sum(1 for v in att_statuses.values() if v == "Late")
            n_early   = sum(1 for v in att_statuses.values() if v == "Early Leave")
            n_absent  = sum(1 for v in att_statuses.values() if v == "Absent")

            msg = MIMEMultipart()
            msg["From"]    = config.SMTP_FROM
            msg["To"]      = to_email
            msg["Subject"] = (
                f"Attendance Report — {course_title} — "
                f"{datetime.now().strftime('%Y-%m-%d')}"
            )
            body = (
                f"Attendance report for {course_title}.\n\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"Total: {len(self.enrolled_students)}  "
                f"Present: {n_present}  Late: {n_late}  "
                f"Early Leave: {n_early}  Absent: {n_absent}\n\n"
                "Sent automatically by AIAS."
            )
            msg.attach(MIMEText(body, "plain"))
            with open(tmp.name, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            safe = "".join(c for c in course_title if c.isalnum() or c in "-_ ")
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="Attendance_{safe}.xlsx"',
            )
            msg.attach(part)

            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.SMTP_LOGIN, config.SMTP_PASSWORD)
                server.send_message(msg)
            print(f"[AIAS] Auto-report sent to {to_email}")
            _success = True
        except Exception as e:
            print(f"[AIAS] Auto-send failed: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except Exception as e:
                print(f"[AIAS] Auto-report temp file cleanup error: {e}")
        return _success

    def closeEvent(self, event):
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "End Session",
            "Are you sure you want to end the attendance session?\nAll recorded attendance will be saved.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Stop worker thread cleanly before closing
            if self._worker is not None:
                try:
                    self._worker.stop()
                except Exception as e:
                    print(f"[AIAS] Session closeEvent worker stop error: {e}")
                try:
                    self._worker.quit()
                    self._worker.wait(2000)  # wait max 2 seconds
                except Exception as e:
                    print(f"[AIAS] Session closeEvent worker quit error: {e}")
                self._worker = None
            event.accept()
        else:
            event.ignore()
