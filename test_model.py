"""
AIAS Model Test Script
Full test: YOLO + GFPGAN + Real-ESRGAN + ArcFace + MongoDB face matching
Usage: py -3.11 test_model.py --image test.jpg
       py -3.11 test_model.py --image test.jpg --threshold 0.35
"""

import cv2
import numpy as np
import sys
import os
import time
import argparse

# ── Setup ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to test image")
parser.add_argument("--threshold", type=float, default=None, help="Override recognition threshold (default: dynamic)")
args = parser.parse_args()

if not os.path.exists(args.image):
    print(f"[ERROR] Image not found: {args.image}")
    sys.exit(1)

print("=" * 60)
print("  AIAS - Full Model Test")
print("=" * 60)

# ── Load Libraries ───────────────────────────────────────────────────────────
print("\n[1/6] Loading libraries...")
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ── MongoDB Connection ────────────────────────────────────────────────────────
print("[1b] Connecting to MongoDB...")
face_db = {}  # {student_id: {"name": str, "embedding": np.array}}
try:
    import pymongo
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    mongo_client.server_info()
    mongo_db = mongo_client["AIAS_DB"]

    students = list(mongo_db["Students"].find(
        {"FaceEmbedding": {"$exists": True}},
        {"StudentID": 1, "FullName": 1, "FaceEmbedding": 1}
    ))
    for stu in students:
        sid = str(stu.get("StudentID", ""))
        name = stu.get("FullName", "Unknown")
        emb = stu.get("FaceEmbedding", [])
        if emb and len(emb) == 512:
            arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                face_db[sid] = {"name": name, "embedding": arr / norm}

    print(f"  -> MongoDB connected | {len(face_db)} students with embeddings loaded")
except Exception as e:
    print(f"  -> MongoDB: failed - {e}")
    print("  -> Running without face matching (embedding extraction only)")

# ── Load Models ──────────────────────────────────────────────────────────────
print("[2/6] Loading models...")

# YOLO
yolo_path = "yolov11l-face.pt" if os.path.exists("yolov11l-face.pt") else "yolov8n-face.pt"
print(f"  -> YOLO: {yolo_path}")
yolo = YOLO(yolo_path)

# ArcFace
print("  -> ArcFace: buffalo_l")
arcface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
arcface.prepare(ctx_id=-1, det_size=(640, 640))

# GFPGAN
gfpgan_model = None
try:
    from gfpgan import GFPGANer
    if os.path.exists("GFPGANv1.4.pth"):
        gfpgan_model = GFPGANer(
            model_path="GFPGANv1.4.pth",
            upscale=2, arch='clean',
            channel_multiplier=2, bg_upsampler=None
        )
        print("  -> GFPGAN: loaded")
    else:
        print("  -> GFPGAN: file not found")
except Exception as e:
    print(f"  -> GFPGAN: failed - {e}")

# Real-ESRGAN
realesrgan_model = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import re

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
            f=self.act(self.u1(F.interpolate(f,scale_factor=2,mode='nearest')))
            f=self.act(self.u2(F.interpolate(f,scale_factor=2,mode='nearest')))
            return self.cl(self.act(self.hr(f)))

    if os.path.exists("RealESRGAN_x4plus.pth"):
        model = _RRDBNet()
        raw = torch.load("RealESRGAN_x4plus.pth", map_location="cpu", weights_only=False)
        sd = raw.get("params_ema", raw.get("params", raw))
        key_map = {"conv_first":"cf","conv_body":"cb","conv_up1":"u1","conv_up2":"u2","conv_hr":"hr","conv_last":"cl"}
        remapped = {}
        for k, v in sd.items():
            new_k = k
            for old, new in key_map.items():
                if new_k.startswith(old + "."):
                    new_k = new + "." + new_k[len(old)+1:]; break
            new_k = re.sub(r'body\.(\d+)\.rdb(\d+)\.conv(\d+)',
                lambda m: f'body.{m.group(1)}.r{m.group(2)}.c{m.group(3)}', new_k)
            remapped[new_k] = v
        model.load_state_dict(remapped, strict=True)
        model.eval()
        realesrgan_model = model
        print("  -> Real-ESRGAN: loaded")
    else:
        print("  -> Real-ESRGAN: file not found")
except Exception as e:
    print(f"  -> Real-ESRGAN: failed - {e}")

# ── Read Image ───────────────────────────────────────────────────────────────
print(f"\n[3/6] Reading image: {args.image}")
img_orig = cv2.imread(args.image)
if img_orig is None:
    print("[ERROR] Failed to read image")
    sys.exit(1)
h, w = img_orig.shape[:2]
print(f"  -> Size: {w}x{h} px")

# ── YOLO Face Detection ──────────────────────────────────────────────────────
print("\n[4/6] Detecting faces with YOLO...")
t0 = time.time()

# Multi-zoom detection
all_faces = []
zoom_levels = [1, 2, 4]
for zoom in zoom_levels:
    if zoom > 1:
        zh, zw = h // zoom, w // zoom
        cx, cy = w // 2, h // 2
        x1z = max(0, cx - zw // 2); y1z = max(0, cy - zh // 2)
        x2z = min(w, cx + zw // 2); y2z = min(h, cy + zh // 2)
        frame_zoom = img_orig[y1z:y2z, x1z:x2z]
    else:
        frame_zoom = img_orig

    results = yolo(frame_zoom, verbose=False, conf=0.25)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            if zoom > 1:
                x1 = x1z + x1; y1 = y1z + y1
                x2 = x1z + x2; y2 = y1z + y2
            all_faces.append({"box": [x1, y1, x2, y2], "conf": conf, "zoom": zoom})

# Remove duplicates (NMS)
def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0

unique_faces = []
for f in sorted(all_faces, key=lambda x: -x["conf"]):
    skip = any(iou(f["box"], u["box"]) > 0.5 for u in unique_faces)
    if not skip:
        unique_faces.append(f)

yolo_time = time.time() - t0
print(f"  -> Faces detected: {len(unique_faces)}")
print(f"  -> Detection time: {yolo_time*1000:.0f}ms")

# ── Analyze Each Face ────────────────────────────────────────────────────────
print(f"\n[5/6] Analyzing faces...")

result_img = img_orig.copy()
face_results = []

for i, face_info in enumerate(unique_faces):
    x1, y1, x2, y2 = face_info["box"]
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    face_crop = img_orig[y1:y2, x1:x2]
    fh, fw = face_crop.shape[:2]

    print(f"\n  -- Face #{i+1} ---------------------------")
    print(f"     YOLO confidence: {face_info['conf']:.3f}")
    print(f"     Size: {fw}x{fh} px")

    # Image quality metrics
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    print(f"     Sharpness (Laplacian): {sharpness:.1f}")
    print(f"     Brightness: {brightness:.1f}")

    # Dynamic recognition threshold
    face_area = fh * fw
    if face_area >= 150*150: size_score = 1.0
    elif face_area >= 80*80: size_score = 0.75
    elif face_area >= 40*40: size_score = 0.5
    elif face_area >= 20*20: size_score = 0.3
    else: size_score = 0.15
    if sharpness >= 500: sharp_score = 1.0
    elif sharpness >= 200: sharp_score = 0.8
    elif sharpness >= 80: sharp_score = 0.6
    elif sharpness >= 30: sharp_score = 0.4
    else: sharp_score = 0.2
    if 80 <= brightness <= 200: bright_score = 1.0
    elif 50 <= brightness < 80: bright_score = 0.7
    elif 200 < brightness <= 230: bright_score = 0.8
    elif brightness < 50: bright_score = 0.4
    else: bright_score = 0.3
    quality = (size_score*0.5)+(sharp_score*0.3)+(bright_score*0.2)
    threshold = float(np.clip(0.22 + 0.20*quality, 0.22, 0.42))
    print(f"     Face quality: {quality:.2f} | Threshold: {threshold:.3f}")

    # Real-ESRGAN first (upscale small faces before enhancement)
    enhanced_face = face_crop.copy()
    if realesrgan_model is not None and min(fh, fw) < 200:
        try:
            import torch
            t_sr = time.time()
            img_t = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            t_in = torch.from_numpy(img_t.transpose(2,0,1)).unsqueeze(0)
            with torch.no_grad():
                out = realesrgan_model(t_in).squeeze(0).clamp(0,1).numpy().transpose(1,2,0)
            enhanced_face = cv2.cvtColor((out*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            print(f"     Real-ESRGAN: done ({(time.time()-t_sr)*1000:.0f}ms)")
        except Exception as e:
            print(f"     Real-ESRGAN: failed - {e}")

    # GFPGAN enhancement
    if gfpgan_model is not None:
        try:
            t_gfp = time.time()
            # استخدم enhanced_face (بعد Real-ESRGAN) وليس face_crop الصغير
            gfp_h, gfp_w = enhanced_face.shape[:2]
            print(f"     GFPGAN input: {gfp_w}x{gfp_h}px")
            if min(gfp_h, gfp_w) < 256:
                scale = 256 / min(gfp_h, gfp_w)
                gfp_in = cv2.resize(enhanced_face, (int(gfp_w * scale), int(gfp_h * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                gfp_in = enhanced_face.copy()
            print(f"     GFPGAN resized: {gfp_in.shape[1]}x{gfp_in.shape[0]}px")
            gfp_input = cv2.cvtColor(gfp_in, cv2.COLOR_BGR2RGB)
            cropped_faces, restored_imgs, restored_faces = gfpgan_model.enhance(
                gfp_input,
                has_aligned=False,
                only_center_face=True,
                paste_back=False,
                weight=0.5
            )
            print(f"     GFPGAN detected: {len(cropped_faces) if cropped_faces else 0} faces")
            # restored_faces = cropped faces only, restored_imgs = full image
            # use cropped_faces output (restored individual faces)
            result_faces = restored_faces if (restored_faces is not None and len(restored_faces) > 0) else None
            if result_faces is None and restored_imgs is not None and len(restored_imgs) > 0:
                result_faces = restored_imgs
            if result_faces is not None and len(result_faces) > 0:
                gfp_out = cv2.cvtColor(result_faces[0], cv2.COLOR_RGB2BGR)
                gfp_out = cv2.resize(gfp_out, (enhanced_face.shape[1], enhanced_face.shape[0]))
                enhanced_face = gfp_out
                print(f"     GFPGAN: done ({(time.time()-t_gfp)*1000:.0f}ms)")
            else:
                print(f"     GFPGAN: no face restored")
        except Exception as e:
            print(f"     GFPGAN: failed - {e}")

    # ArcFace embedding — crop مع padding أولاً، fallback للصورة الكاملة
    face_result = {"face": i+1, "quality": quality, "threshold": threshold, "conf": face_info["conf"]}
    try:
        t_arc = time.time()
        pad = 30
        x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad); y2p = min(h, y2 + pad)
        face_crop_padded = img_orig[y1p:y2p, x1p:x2p]

        faces_arc = arcface.get(face_crop_padded)
        if not faces_arc:
            # fallback: الصورة الكاملة، اختار الوجه الأقرب للـ box
            if not hasattr(arcface, '_full_faces_cache'):
                arcface._full_faces_cache = arcface.get(img_orig)
            faces_arc_full = arcface._full_faces_cache
            if faces_arc_full:
                def box_dist(f):
                    fb = f.bbox
                    cx_f = (fb[0] + fb[2]) / 2; cy_f = (fb[1] + fb[3]) / 2
                    cx_b = (x1 + x2) / 2; cy_b = (y1 + y2) / 2
                    return (cx_f - cx_b)**2 + (cy_f - cy_b)**2
                faces_arc = [min(faces_arc_full, key=box_dist)]

        if faces_arc:
            emb = faces_arc[0].embedding
            emb_norm = emb / np.linalg.norm(emb)
            print(f"     ArcFace: done  dim={len(emb_norm)} ({(time.time()-t_arc)*1000:.0f}ms)")

            # ── MongoDB Face Matching ─────────────────────────────────────
            if face_db:
                best_score = -1
                best_id = None
                best_name = None
                for sid, data in face_db.items():
                    score = float(np.dot(emb_norm, data["embedding"]))
                    if score > best_score:
                        best_score = score
                        best_id = sid
                        best_name = data["name"]

                match_threshold = args.threshold if args.threshold else threshold
                if best_score >= match_threshold:
                    match_label = f"{best_name}"
                    label_color = (0, 220, 0)
                    print(f"     [RECOGNIZED] {best_name} | ID: {best_id} | score: {best_score:.4f} | threshold: {match_threshold:.3f}")
                    face_result["match"] = f"{best_name} ({best_id})"
                    face_result["score"] = best_score
                else:
                    match_label = "UNKNOWN"
                    label_color = (0, 0, 255)
                    print(f"     [UNKNOWN] Best: {best_name} | score: {best_score:.4f} < threshold: {match_threshold:.3f}")
                    face_result["match"] = "Unknown"
                    face_result["score"] = best_score

                # رسم اسم الشخص تحت المربع مع خلفية
                (tw, th), _ = cv2.getTextSize(match_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(result_img, (x1, y2+2), (x1+tw+8, y2+th+12), (0,0,0), -1)
                cv2.putText(result_img, match_label, (x1+3, y2+th+6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)
                score_label = f"{best_score:.3f}"
                (sw, sh), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(result_img, (x1, y2+th+14), (x1+sw+6, y2+th+sh+22), (0,0,0), -1)
                cv2.putText(result_img, score_label, (x1+3, y2+th+sh+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
            else:
                print(f"     Face matching: skipped (no DB)")
        else:
            print(f"     ArcFace: no face detected")
    except Exception as e:
        print(f"     ArcFace: failed - {e}")

    face_results.append(face_result)

    # Draw bounding box + label فوق المربع
    color = (0, 255, 0)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
    label = f"#{i+1} {face_info['conf']:.2f}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result_img, (x1, y1-lh-8), (x1+lw+4, y1), (0,0,0), -1)
    cv2.putText(result_img, label, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save enhanced face
    enhanced_path = f"face_{i+1}_enhanced.jpg"
    cv2.imwrite(enhanced_path, enhanced_face)
    print(f"     Enhanced face saved: {enhanced_path}")

# ── Save Detection Result ────────────────────────────────────────────────────
print(f"\n[6/6] Saving results...")
output_path = "result_detection.jpg"
cv2.imwrite(output_path, result_img)
print(f"  -> Detection image saved: {output_path}")

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Faces detected: {len(unique_faces)}")
print(f"  DB students loaded: {len(face_db)}")
for r in face_results:
    match_info = ""
    if "match" in r:
        score_str = f" | score: {r['score']:.4f}" if "score" in r else ""
        match_info = f" | {r['match']}{score_str}"
    print(f"  Face #{r['face']}: YOLO={r['conf']:.2f} | quality={r['quality']:.2f} | threshold={r['threshold']:.3f}{match_info}")
print(f"\n  Output files:")
print(f"  -> {output_path} (full detection image)")
for i in range(len(unique_faces)):
    print(f"  -> face_{i+1}_enhanced.jpg (enhanced face {i+1})")
print("=" * 60)
