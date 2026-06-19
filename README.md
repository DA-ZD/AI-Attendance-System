# AIAS — AI Attendance System

An AI-powered, face-recognition-based classroom attendance system. AIAS replaces manual roll-call with a live camera feed that detects, recognizes, and logs student attendance automatically — with an admin dashboard for course/session management and reporting.

> 🎓 Graduation Project — Qassim University, Computer Science

## Overview

AIAS uses a multi-stage computer vision pipeline to recognize students in real time, even under non-ideal classroom conditions (low resolution, partial occlusion, poor lighting), and stores attendance records in MongoDB for instructors to review and export.

## Tech Stack

| Layer | Technology |
|---|---|
| Face Detection | YOLOv11 |
| Face Enhancement | Real-ESRGAN, GFPGAN |
| Face Recognition | ArcFace (InsightFace `buffalo_l`) / AdaFace |
| Desktop UI | PyQt5 |
| Database | MongoDB |
| Auth | bcrypt |
| Notifications | SMTP (email) |

## Key Features

- **Live recognition pipeline** — YOLOv11 detects faces from the camera feed; low-quality crops are upscaled/enhanced (Real-ESRGAN + GFPGAN) before being passed to the recognition model.
- **Adaptive quality handling** — dynamic confidence threshold and CLAHE contrast correction adjust automatically based on face size, sharpness, and brightness.
- **Admin dashboard** — manage courses, sections, students, and instructors.
- **Live session view** — real-time attendance marking during class with Present / Late / Absent / Early Leave status.
- **Reports** — attendance history and exportable session reports.
- **Secure login** — bcrypt password hashing, auto-provisioned default admin account on first run.
- **Light/Dark theme** — full UI theming support.

## Project Structure

```
├── main.py              # App entry point
├── config.py            # DB connection, env config, global styling
├── database.py          # MongoDB connection helpers
├── ai_models.py          # YOLO + ArcFace + Real-ESRGAN + GFPGAN pipeline
├── adaface_wrapper.py    # Alternative AdaFace recognition backend
├── ui_login.py           # Login window
├── ui_admin.py           # Admin panel (courses, students, instructors)
├── ui_session.py         # Live attendance session view
├── ui_reports.py         # Attendance reports
├── ui_theme.py           # Theming, shared UI components
└── run.bat               # Windows launcher
```

## Setup

### 1. Requirements
- Python 3.11
- MongoDB (local or remote instance)
- Windows (tested) — should also run on Linux/macOS with minor path adjustments

### 2. Install dependencies
```bash
pip install pyqt5 pymongo bcrypt torch torchvision ultralytics insightface opencv-python python-dotenv gfpgan
```

### 3. Download model weights
These are excluded from the repo due to size. Download and place them in the project root:

| File | Source |
|---|---|
| `yolov11l-face.pt` | [YOLOv11 face detection weights](https://github.com/akanametov/yolo-face) |
| `GFPGANv1.4.pth` | [GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases) |
| `RealESRGAN_x4plus.pth` | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) |

### 4. Configure environment
```bash
cp .env.example .env
# then edit .env with your MongoDB URI and SMTP credentials
```

### 5. Run
```bash
python main.py
# or on Windows:
run.bat
```

On first launch, a default admin account is created automatically — check the console output for the generated password, and change it after logging in.

## Results

Achieved ~89% recognition recall in classroom simulation testing across multiple students under live camera conditions.

## Roadmap

- Migration to a web-based architecture (FastAPI + Next.js)
- Cloud hosting for multi-classroom deployment
- Arabic-first UI and Hijri calendar support
- PDPL (Saudi data protection law) compliance

## Author

Ziyad — Computer Science, Qassim University
[GitHub](https://github.com/DA-ZD)
