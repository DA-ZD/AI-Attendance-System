import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as _F
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import pytorch_lightning as _pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

from config import DB_CONNECTED, db
import threading
_model_lock = threading.RLock()

AI_LOADED     = False
yolo_model    = None
_gfpgan_model = None
arcface_model = None
_sr_model      = None

_AI_LOADER_THREAD = None   # kept alive so Qt doesn't GC it


# ── Real-ESRGAN ───────────────────────────────────────────────────────────────

class _RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.c1=nn.Conv2d(nf,gc,3,1,1); self.c2=nn.Conv2d(nf+gc,gc,3,1,1)
        self.c3=nn.Conv2d(nf+2*gc,gc,3,1,1); self.c4=nn.Conv2d(nf+3*gc,gc,3,1,1)
        self.c5=nn.Conv2d(nf+4*gc,nf,3,1,1)
        self.act=nn.LeakyReLU(0.2,inplace=True)
    def forward(self,x):
        x1=self.act(self.c1(x)); x2=self.act(self.c2(torch.cat((x,x1),1)))
        x3=self.act(self.c3(torch.cat((x,x1,x2),1))); x4=self.act(self.c4(torch.cat((x,x1,x2,x3),1)))
        return self.c5(torch.cat((x,x1,x2,x3,x4),1))*0.2+x

class _RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.r1=_RDB(nf,gc); self.r2=_RDB(nf,gc); self.r3=_RDB(nf,gc)
    def forward(self,x): return self.r3(self.r2(self.r1(x)))*0.2+x

class _RRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cf=nn.Conv2d(3,64,3,1,1)
        self.body=nn.Sequential(*[_RRDB(64) for _ in range(23)])
        self.cb=nn.Conv2d(64,64,3,1,1)
        self.u1=nn.Conv2d(64,64,3,1,1); self.u2=nn.Conv2d(64,64,3,1,1)
        self.hr=nn.Conv2d(64,64,3,1,1); self.cl=nn.Conv2d(64,3,3,1,1)
        self.act=nn.LeakyReLU(0.2,inplace=True)
    def forward(self,x):
        f=self.cf(x); f=f+self.cb(self.body(f))
        f=self.act(self.u1(_F.interpolate(f,scale_factor=2,mode='nearest')))
        f=self.act(self.u2(_F.interpolate(f,scale_factor=2,mode='nearest')))
        return self.cl(self.act(self.hr(f)))

def _load_realesrgan(model_path="RealESRGAN_x4plus.pth"):
    """Load Real-ESRGAN model. Returns model or None if unavailable."""
    if not os.path.exists(model_path):
        return None
    try:
        import re
        model = _RRDBNet()
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = raw.get("params_ema", raw.get("params", raw))

        # Remap checkpoint keys to match our _RRDBNet attribute names
        key_map = {
            "conv_first": "cf",
            "conv_body":  "cb",
            "conv_up1":   "u1",
            "conv_up2":   "u2",
            "conv_hr":    "hr",
            "conv_last":  "cl",
        }
        remapped = {}
        for k, v in sd.items():
            new_k = k
            for old, new in key_map.items():
                if new_k.startswith(old + "."):
                    new_k = new + "." + new_k[len(old) + 1:]
                    break
            new_k = re.sub(
                r'body\.(\d+)\.rdb(\d+)\.conv(\d+)',
                lambda m: f'body.{m.group(1)}.r{m.group(2)}.c{m.group(3)}',
                new_k
            )
            remapped[new_k] = v

        model.load_state_dict(remapped, strict=True)
        model.eval()
        print("[AIAS] Real-ESRGAN loaded")
        return model
    except Exception as e:
        print(f"[AIAS] Real-ESRGAN load failed: {e}")
        return None

def _load_gfpgan():
    """Load GFPGAN face enhancement model."""
    try:
        from gfpgan import GFPGANer
        import os
        model_path = "GFPGANv1.4.pth"
        if not os.path.exists(model_path):
            import urllib.request
            print("[AIAS] Downloading GFPGAN model...")
            url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            urllib.request.urlretrieve(url, model_path)
        model = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        print("[AIAS] GFPGAN loaded")
        return model
    except Exception as e:
        print(f"[AIAS] GFPGAN load failed: {e}")
        return None

def _sr_upscale(sr_model, img_bgr, target_size=80):
    """Upscale small face crop using Real-ESRGAN."""
    import cv2, numpy as np
    if sr_model is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if min(h, w) >= target_size:
        return img_bgr
    try:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            out = sr_model(t).squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
        out = (out * 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[AIAS] SR upscale error: {e}")
        return img_bgr


def _ensure_ai_models(det_size=(640, 640)):
    """Load YOLO + ArcFace into globals if not already loaded. Returns (yolo, arcface, ok)."""
    with _model_lock:
        global yolo_model, arcface_model, AI_LOADED, _sr_model, _gfpgan_model
        if AI_LOADED and yolo_model is not None and arcface_model is not None:
            # Re-prepare with requested det_size (lightweight operation)
            try:
                arcface_model.prepare(ctx_id=-1, det_size=det_size)
            except Exception:
                pass
            if _sr_model is None:
                _sr_model = _load_realesrgan("RealESRGAN_x4plus.pth")
            if _gfpgan_model is None:
                _gfpgan_model = _load_gfpgan()
            return yolo_model, arcface_model, True
        try:
            import logging
            logging.getLogger("insightface").setLevel(logging.ERROR)
            logging.getLogger("onnxruntime").setLevel(logging.ERROR)
            from ultralytics import YOLO
            from insightface.app import FaceAnalysis
            import os
            _yolo_path = "yolov11l-face.pt" if os.path.exists("yolov11l-face.pt") else "yolov8n-face.pt"
            yolo_model = YOLO(_yolo_path)
            print(f"[AIAS] Using YOLO model: {_yolo_path}")
            try:
                arcface_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            except TypeError:
                arcface_model = FaceAnalysis(name="buffalo_l")
            arcface_model.prepare(ctx_id=-1, det_size=det_size)
            AI_LOADED = True
            print(f"[AIAS] AI models loaded (det_size={det_size})")
            if _sr_model is None:
                _sr_model = _load_realesrgan("RealESRGAN_x4plus.pth")
            if _gfpgan_model is None:
                _gfpgan_model = _load_gfpgan()
            return yolo_model, arcface_model, True
        except Exception as e:
            print(f"[AIAS] AI model load failed: {e}")
            return None, None, False


def apply_clahe(img_bgr):
    """
    Apply adaptive CLAHE to a BGR face image.
    Automatically adjusts clip limit based on image brightness.
    Returns BGR image.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 60:
        clip_limit = 4.0
    elif brightness < 110:
        clip_limit = 2.5
    elif brightness < 170:
        clip_limit = 1.5
    else:
        clip_limit = 1.0

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def upscale_face_crop(img_bgr, target_size=112):
    """
    Upscale a small face crop to target_size using bicubic interpolation.
    Only upscales if the face is smaller than target_size.
    Returns BGR image.
    """
    import cv2
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)

    if min_dim >= target_size:
        return img_bgr

    scale = target_size / min_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def compute_dynamic_threshold(face_crop_bgr):
    """
    Compute a dynamic recognition threshold based on face image quality.
    Returns a float threshold between 0.28 and 0.52.

    Lower quality → lower threshold (try harder to recognize)
    Higher quality → higher threshold (be more strict)
    """
    import cv2
    import numpy as np

    h, w = face_crop_bgr.shape[:2]
    face_area = h * w

    # ── Factor 1: Face size (distance proxy) ──────────────────────────
    if face_area >= 150 * 150:
        size_score = 1.0
    elif face_area >= 80 * 80:
        size_score = 0.75
    elif face_area >= 40 * 40:
        size_score = 0.5
    elif face_area >= 20 * 20:
        size_score = 0.3
    else:
        size_score = 0.15

    # ── Factor 2: Sharpness (motion blur / focus) ──────────────────────
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var >= 500:
        sharp_score = 1.0
    elif laplacian_var >= 200:
        sharp_score = 0.8
    elif laplacian_var >= 80:
        sharp_score = 0.6
    elif laplacian_var >= 30:
        sharp_score = 0.4
    else:
        sharp_score = 0.2

    # ── Factor 3: Brightness ───────────────────────────────────────────
    brightness = np.mean(gray)

    if 80 <= brightness <= 200:
        bright_score = 1.0
    elif 50 <= brightness < 80:
        bright_score = 0.7
    elif 200 < brightness <= 230:
        bright_score = 0.8
    elif brightness < 50:
        bright_score = 0.4
    else:
        bright_score = 0.3

    # ── Combined quality score (weighted) ─────────────────────────────
    quality = (size_score * 0.5) + (sharp_score * 0.3) + (bright_score * 0.2)

    # ── Map quality to threshold ───────────────────────────────────────
    threshold = 0.22 + 0.20 * quality
    return float(np.clip(threshold, 0.22, 0.42))


class AIModelLoader(QThread):
    """Loads YOLO + ArcFace once at startup in a background thread."""
    done = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

    def run(self):
        with _model_lock:
            _, _, ok = _ensure_ai_models(det_size=(640, 640))
        self.done.emit(ok)
